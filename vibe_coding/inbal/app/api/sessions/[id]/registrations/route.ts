import { NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { prisma } from "@/lib/prisma";

export async function POST(req: Request, { params }: { params: Promise<{ id: string }> }) {
  const session = await getServerSession(authOptions);
  if (!session) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  const { id: sessionId } = await params;
  const { userId, transferredFromId, note } = await req.json();

  const existing = await prisma.sessionRegistration.findUnique({
    where: { userId_sessionId: { userId, sessionId } },
  });

  let reg;
  if (existing) {
    reg = await prisma.sessionRegistration.update({
      where: { userId_sessionId: { userId, sessionId } },
      data: { status: "REGISTERED", transferredFromId, note },
    });
  } else {
    reg = await prisma.sessionRegistration.create({
      data: { userId, sessionId, status: "REGISTERED", transferredFromId, note },
    });
  }
  return NextResponse.json(reg, { status: 201 });
}
