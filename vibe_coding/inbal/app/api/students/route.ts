import { NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { prisma } from "@/lib/prisma";
import bcrypt from "bcryptjs";
import { slotAvailable } from "@/lib/slots";

export async function GET() {
  const session = await getServerSession(authOptions);
  if (!session || (session.user as { role: string }).role !== "ADMIN") {
    return NextResponse.json({ error: "Forbidden" }, { status: 403 });
  }

  const students = await prisma.user.findMany({
    where: { role: "STUDENT" },
    include: {
      enrollments: {
        where: { status: "ACTIVE" },
        include: { group: { select: { id: true, name: true } } },
      },
      payments: { orderBy: { date: "desc" }, take: 1 },
    },
    orderBy: { name: "asc" },
  });
  return NextResponse.json(students);
}

export async function POST(req: Request) {
  const session = await getServerSession(authOptions);
  if (!session || (session.user as { role: string }).role !== "ADMIN") {
    return NextResponse.json({ error: "Forbidden" }, { status: 403 });
  }

  const { name, email, phone, groupId, slotType } = await req.json();

  const existing = await prisma.user.findUnique({ where: { email } });
  if (existing) return NextResponse.json({ error: "אימייל כבר קיים" }, { status: 409 });

  const password = await bcrypt.hash("student123", 10);
  const user = await prisma.user.create({
    data: { name, email, phone, role: "STUDENT", password },
  });

  if (groupId) {
    await prisma.groupEnrollment.create({
      data: { userId: user.id, groupId, status: "ACTIVE", preferredSlotType: slotType ?? null },
    });
    const futureSessions = await prisma.session.findMany({
      where: { groupId, date: { gte: new Date() }, status: "SCHEDULED" },
      include: { registrations: { select: { slotType: true, status: true } } },
    });
    for (const s of futureSessions) {
      const canRegister = !slotType || slotAvailable(s.registrations, slotType);
      await prisma.sessionRegistration.create({
        data: { userId: user.id, sessionId: s.id, status: "REGISTERED", slotType: canRegister ? slotType ?? null : null },
      });
    }
  }

  return NextResponse.json(user, { status: 201 });
}
