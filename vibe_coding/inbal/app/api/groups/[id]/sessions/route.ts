import { NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { prisma } from "@/lib/prisma";

function nextWeekday(dayOfWeek: number, offsetWeeks: number): Date {
  const now = new Date();
  const day = now.getDay();
  const daysUntil = (dayOfWeek - day + 7) % 7 || 7;
  const d = new Date(now);
  d.setDate(d.getDate() + daysUntil + offsetWeeks * 7);
  return d;
}

function setTime(date: Date, time: string): Date {
  const [h, m] = time.split(":").map(Number);
  const d = new Date(date);
  d.setHours(h, m, 0, 0);
  return d;
}

export async function POST(req: Request, { params }: { params: Promise<{ id: string }> }) {
  const session = await getServerSession(authOptions);
  if (!session || (session.user as { role: string }).role !== "ADMIN") {
    return NextResponse.json({ error: "Forbidden" }, { status: 403 });
  }
  const { id } = await params;
  const { weeks = 8 } = await req.json().catch(() => ({}));

  const group = await prisma.group.findUnique({
    where: { id },
    include: { enrollments: { where: { status: "ACTIVE" } } },
  });
  if (!group) return NextResponse.json({ error: "Not found" }, { status: 404 });

  const created = [];
  for (let w = 0; w < weeks; w++) {
    const baseDate = nextWeekday(group.dayOfWeek, w);
    const sessionDate = setTime(baseDate, group.time);

    const startOfDay = new Date(sessionDate);
    startOfDay.setHours(0, 0, 0, 0);
    const endOfDay = new Date(sessionDate);
    endOfDay.setHours(23, 59, 59, 999);

    const existing = await prisma.session.findFirst({
      where: { groupId: id, date: { gte: startOfDay, lte: endOfDay } },
    });
    if (existing) continue;

    const newSession = await prisma.session.create({
      data: { groupId: id, date: sessionDate, status: "SCHEDULED" },
    });

    for (const enrollment of group.enrollments) {
      await prisma.sessionRegistration.create({
        data: { sessionId: newSession.id, userId: enrollment.userId, status: "REGISTERED" },
      });
    }
    created.push(newSession);
  }

  return NextResponse.json({ created: created.length, sessions: created });
}
