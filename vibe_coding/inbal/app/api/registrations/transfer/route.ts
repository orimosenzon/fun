import { NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { prisma } from "@/lib/prisma";
import { slotAvailable } from "@/lib/slots";

export async function POST(req: Request) {
  const session = await getServerSession(authOptions);
  if (!session) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const { fromRegistrationId, toSessionId } = await req.json();

  const userEmail = (session.user as { email?: string }).email!;
  const user = await prisma.user.findUnique({ where: { email: userEmail } });
  if (!user) return NextResponse.json({ error: "User not found" }, { status: 404 });

  const fromReg = await prisma.sessionRegistration.findUnique({
    where: { id: fromRegistrationId },
    include: { session: true },
  });
  if (!fromReg) return NextResponse.json({ error: "Registration not found" }, { status: 404 });

  const isAdmin = (session.user as { role?: string }).role === "ADMIN";
  if (!isAdmin && fromReg.userId !== user.id) {
    return NextResponse.json({ error: "Forbidden" }, { status: 403 });
  }

  if (fromReg.status !== "REGISTERED") {
    return NextResponse.json({ error: "ניתן להעביר רק רישום פעיל" }, { status: 400 });
  }

  const toSession = await prisma.session.findUnique({
    where: { id: toSessionId },
    include: { registrations: { select: { slotType: true, status: true } } },
  });
  if (!toSession) return NextResponse.json({ error: "Session not found" }, { status: 404 });
  if (toSession.status !== "SCHEDULED") {
    return NextResponse.json({ error: "שיעור היעד אינו פעיל" }, { status: 400 });
  }

  // Check existing registration in target session
  const existingInTarget = await prisma.sessionRegistration.findUnique({
    where: { userId_sessionId: { userId: fromReg.userId, sessionId: toSessionId } },
  });
  if (existingInTarget && existingInTarget.status === "REGISTERED") {
    return NextResponse.json({ error: "כבר רשום לשיעור זה" }, { status: 409 });
  }

  // Check slot availability
  if (fromReg.slotType && !slotAvailable(toSession.registrations, fromReg.slotType as "WHEEL" | "NO_WHEEL")) {
    return NextResponse.json({ error: "אין מקום פנוי מהסוג המבוקש" }, { status: 409 });
  }

  // Atomic transfer
  const [newReg] = await prisma.$transaction(async (tx) => {
    let newRegistration;
    if (existingInTarget) {
      newRegistration = await tx.sessionRegistration.update({
        where: { id: existingInTarget.id },
        data: { status: "REGISTERED", slotType: fromReg.slotType, transferredFromId: fromReg.id },
      });
    } else {
      newRegistration = await tx.sessionRegistration.create({
        data: {
          userId: fromReg.userId,
          sessionId: toSessionId,
          status: "REGISTERED",
          slotType: fromReg.slotType,
          transferredFromId: fromReg.id,
        },
      });
    }

    await tx.sessionRegistration.update({
      where: { id: fromReg.id },
      data: { status: "TRANSFERRED", transferredToId: newRegistration.id },
    });

    return [newRegistration];
  });

  return NextResponse.json(newReg, { status: 201 });
}
