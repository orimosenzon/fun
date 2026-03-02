import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { redirect } from "next/navigation";
import { prisma } from "@/lib/prisma";
import { WHEEL_SLOTS, NO_WHEEL_SLOTS } from "@/lib/slots";
import WeekCalendar from "./WeekCalendar";

function getWeekBounds(weekParam?: string): { start: Date; end: Date } {
  let base: Date;
  if (weekParam) {
    base = new Date(weekParam);
    if (isNaN(base.getTime())) base = new Date();
  } else {
    base = new Date();
  }
  // Find Sunday of the week
  const day = base.getDay();
  const sunday = new Date(base);
  sunday.setDate(base.getDate() - day);
  sunday.setHours(0, 0, 0, 0);
  const saturday = new Date(sunday);
  saturday.setDate(sunday.getDate() + 6);
  saturday.setHours(23, 59, 59, 999);
  return { start: sunday, end: saturday };
}

export default async function MyPage({
  searchParams,
}: {
  searchParams: Promise<{ week?: string }>;
}) {
  const session = await getServerSession(authOptions);
  if (!session) redirect("/login");

  const { week } = await searchParams;
  const { start, end } = getWeekBounds(week);

  const user = await prisma.user.findUnique({
    where: { email: session.user!.email! },
  });
  if (!user) redirect("/login");

  // All sessions this week
  const sessions = await prisma.session.findMany({
    where: { date: { gte: start, lte: end }, status: "SCHEDULED" },
    include: {
      group: { select: { name: true, location: true } },
      registrations: {
        select: { id: true, userId: true, slotType: true, status: true },
      },
    },
    orderBy: { date: "asc" },
  });

  // Student's upcoming REGISTERED sessions (for transfer source selection)
  const myUpcomingRegistrations = await prisma.sessionRegistration.findMany({
    where: {
      userId: user.id,
      status: "REGISTERED",
      session: { date: { gte: new Date() }, status: "SCHEDULED" },
    },
    include: { session: { include: { group: { select: { name: true } } } } },
    orderBy: { session: { date: "asc" } },
  });

  // Standby entries for this user (for sessions shown this week)
  const sessionIds = sessions.map((s) => s.id);
  const myStandbyEntries = await prisma.standbyEntry.findMany({
    where: { userId: user.id, sessionId: { in: sessionIds } },
    select: { sessionId: true, slotType: true, notifiedAt: true, expiresAt: true, createdAt: true },
  });

  // Standby positions — count all entries per session sorted by createdAt
  const standbyPositionMap: Record<string, number> = {};
  for (const entry of myStandbyEntries) {
    const count = await prisma.standbyEntry.count({
      where: { sessionId: entry.sessionId, createdAt: { lte: entry.createdAt } },
    });
    standbyPositionMap[entry.sessionId] = count;
  }

  const sessionData = sessions.map((s) => {
    const activeRegs = s.registrations.filter((r) => r.status === "REGISTERED");
    const myReg = s.registrations.find((r) => r.userId === user.id);
    const myStandby = myStandbyEntries.find((e) => e.sessionId === s.id);

    return {
      id: s.id,
      date: s.date.toISOString(),
      groupName: s.group.name,
      groupLocation: s.group.location ?? null,
      myRegistration: myReg
        ? { id: myReg.id, slotType: myReg.slotType as string | null, status: myReg.status }
        : null,
      myStandby: myStandby
        ? {
            slotType: myStandby.slotType as string | null,
            notifiedAt: myStandby.notifiedAt?.toISOString() ?? null,
            expiresAt: myStandby.expiresAt?.toISOString() ?? null,
            position: standbyPositionMap[s.id] ?? 1,
          }
        : null,
      slots: {
        wheelTaken: activeRegs.filter((r) => r.slotType === "WHEEL").length,
        noWheelTaken: activeRegs.filter((r) => r.slotType === "NO_WHEEL").length,
        wheelTotal: WHEEL_SLOTS,
        noWheelTotal: NO_WHEEL_SLOTS,
      },
    };
  });

  const myRegsData = myUpcomingRegistrations.map((r) => ({
    id: r.id,
    sessionId: r.sessionId,
    sessionDate: r.session.date.toISOString(),
    groupName: r.session.group.name,
    slotType: r.slotType as string | null,
  }));

  return (
    <WeekCalendar
      sessions={sessionData}
      weekStart={start.toISOString()}
      myUpcomingRegistrations={myRegsData}
    />
  );
}
