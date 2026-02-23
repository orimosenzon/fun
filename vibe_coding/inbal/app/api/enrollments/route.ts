import { NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { prisma } from "@/lib/prisma";
import { slotAvailable } from "@/lib/slots";

export async function POST(req: Request) {
  const session = await getServerSession(authOptions);
  if (!session || (session.user as { role: string }).role !== "ADMIN") {
    return NextResponse.json({ error: "Forbidden" }, { status: 403 });
  }

  const { userId, groupId, slotType } = await req.json();

  const existing = await prisma.groupEnrollment.findUnique({
    where: { userId_groupId: { userId, groupId } },
  });

  if (existing) {
    if (existing.status !== "ACTIVE") {
      const updated = await prisma.groupEnrollment.update({
        where: { userId_groupId: { userId, groupId } },
        data: { status: "ACTIVE", preferredSlotType: slotType ?? existing.preferredSlotType },
      });
      return NextResponse.json(updated);
    }
    return NextResponse.json({ error: "כבר רשום לקבוצה" }, { status: 409 });
  }

  const enrollment = await prisma.groupEnrollment.create({
    data: { userId, groupId, status: "ACTIVE", preferredSlotType: slotType ?? null },
  });

  const futureSessions = await prisma.session.findMany({
    where: { groupId, date: { gte: new Date() }, status: "SCHEDULED" },
    include: { registrations: { select: { slotType: true, status: true } } },
  });

  for (const s of futureSessions) {
    const canRegister = !slotType || slotAvailable(s.registrations, slotType);
    await prisma.sessionRegistration.upsert({
      where: { userId_sessionId: { userId, sessionId: s.id } },
      create: { userId, sessionId: s.id, status: "REGISTERED", slotType: canRegister ? slotType ?? null : null },
      update: { status: "REGISTERED", slotType: canRegister ? slotType ?? null : null },
    });
  }

  return NextResponse.json(enrollment, { status: 201 });
}

export async function PATCH(req: Request) {
  const session = await getServerSession(authOptions);
  if (!session || (session.user as { role: string }).role !== "ADMIN") {
    return NextResponse.json({ error: "Forbidden" }, { status: 403 });
  }

  const { userId, groupId, status } = await req.json();
  const enrollment = await prisma.groupEnrollment.update({
    where: { userId_groupId: { userId, groupId } },
    data: { status },
  });
  return NextResponse.json(enrollment);
}
