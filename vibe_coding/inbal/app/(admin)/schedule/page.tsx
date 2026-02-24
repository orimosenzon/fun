import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { redirect } from "next/navigation";
import { prisma } from "@/lib/prisma";
import { WHEEL_SLOTS, NO_WHEEL_SLOTS } from "@/lib/slots";
import AdminWeekCalendar from "./AdminWeekCalendar";

function getWeekBounds(weekParam?: string): { start: Date; end: Date } {
  let base: Date;
  if (weekParam) {
    base = new Date(weekParam);
    if (isNaN(base.getTime())) base = new Date();
  } else {
    base = new Date();
  }
  const day = base.getDay();
  const sunday = new Date(base);
  sunday.setDate(base.getDate() - day);
  sunday.setHours(0, 0, 0, 0);
  const saturday = new Date(sunday);
  saturday.setDate(sunday.getDate() + 6);
  saturday.setHours(23, 59, 59, 999);
  return { start: sunday, end: saturday };
}

export default async function SchedulePage({
  searchParams,
}: {
  searchParams: Promise<{ week?: string }>;
}) {
  const session = await getServerSession(authOptions);
  if (!session || (session.user as { role?: string })?.role !== "ADMIN") redirect("/my");

  const { week } = await searchParams;
  const { start, end } = getWeekBounds(week);

  const sessions = await prisma.session.findMany({
    where: { date: { gte: start, lte: end } },
    include: {
      group: { select: { name: true, location: true } },
      registrations: {
        where: { status: "REGISTERED" },
        include: { user: { select: { id: true, name: true } } },
      },
    },
    orderBy: { date: "asc" },
  });

  const sessionData = sessions.map((s) => ({
    id: s.id,
    date: s.date.toISOString(),
    status: s.status,
    groupName: s.group.name,
    groupLocation: s.group.location ?? null,
    slots: {
      wheelTotal: WHEEL_SLOTS,
      noWheelTotal: NO_WHEEL_SLOTS,
      wheel: s.registrations
        .filter((r) => r.slotType === "WHEEL")
        .map((r) => r.user.name),
      noWheel: s.registrations
        .filter((r) => r.slotType === "NO_WHEEL")
        .map((r) => r.user.name),
      unassigned: s.registrations
        .filter((r) => r.slotType === null)
        .map((r) => r.user.name),
    },
  }));

  return (
    <AdminWeekCalendar
      sessions={sessionData}
      weekStart={start.toISOString()}
    />
  );
}
