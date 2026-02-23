"use client";

import { useRouter } from "next/navigation";
import StudentSessionCard from "./StudentSessionCard";

const DAYS_HE = ["ראשון", "שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת"];

export type SessionData = {
  id: string;
  date: string;
  groupName: string;
  groupLocation: string | null;
  myRegistration: { id: string; slotType: string | null; status: string } | null;
  slots: {
    wheelTaken: number;
    noWheelTaken: number;
    wheelTotal: number;
    noWheelTotal: number;
  };
};

export type MyRegistration = {
  id: string;
  sessionId: string;
  sessionDate: string;
  groupName: string;
  slotType: string | null;
};

interface Props {
  sessions: SessionData[];
  weekStart: string;
  myUpcomingRegistrations: MyRegistration[];
}

export default function WeekCalendar({ sessions, weekStart, myUpcomingRegistrations }: Props) {
  const router = useRouter();
  const weekStartDate = new Date(weekStart);

  function prevWeek() {
    const d = new Date(weekStartDate);
    d.setDate(d.getDate() - 7);
    router.push(`/my?week=${d.toISOString().slice(0, 10)}`);
  }

  function nextWeek() {
    const d = new Date(weekStartDate);
    d.setDate(d.getDate() + 7);
    router.push(`/my?week=${d.toISOString().slice(0, 10)}`);
  }

  const weekLabel = (() => {
    const end = new Date(weekStartDate);
    end.setDate(weekStartDate.getDate() + 6);
    const fmt = (d: Date) =>
      d.toLocaleDateString("he-IL", { day: "numeric", month: "short" });
    return `${fmt(weekStartDate)} – ${fmt(end)}`;
  })();

  // Group sessions by day of week (0=Sun … 6=Sat)
  const byDay: SessionData[][] = Array.from({ length: 7 }, () => []);
  for (const s of sessions) {
    const d = new Date(s.date);
    byDay[d.getDay()].push(s);
  }

  // Days that have sessions (or all 7 if none)
  const activeDays = byDay.some((d) => d.length > 0)
    ? byDay.map((sessions, idx) => ({ idx, sessions })).filter((d) => d.sessions.length > 0 || true)
    : [];

  const hasAnySessions = sessions.length > 0;

  return (
    <div className="space-y-4" dir="rtl">
      {/* Week navigation */}
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold text-stone-800">הלוח שלי</h1>
        <div className="flex items-center gap-3">
          <button
            onClick={prevWeek}
            className="p-2 rounded-lg hover:bg-stone-100 transition text-stone-600"
            aria-label="שבוע קודם"
          >
            ←
          </button>
          <span className="text-sm font-medium text-stone-700 min-w-[140px] text-center">
            {weekLabel}
          </span>
          <button
            onClick={nextWeek}
            className="p-2 rounded-lg hover:bg-stone-100 transition text-stone-600"
            aria-label="שבוע הבא"
          >
            →
          </button>
        </div>
      </div>

      {!hasAnySessions ? (
        <div className="bg-white rounded-xl border border-stone-200 p-10 text-center text-stone-400">
          <div className="text-4xl mb-2">📅</div>
          <div>אין שיעורים השבוע</div>
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
          {activeDays.map(({ idx, sessions: daySessions }) => {
            if (daySessions.length === 0) return null;
            const dayDate = new Date(weekStartDate);
            dayDate.setDate(weekStartDate.getDate() + idx);

            return (
              <div key={idx} className="space-y-2">
                {/* Day header */}
                <div className="text-xs font-semibold text-stone-500 uppercase tracking-wide px-1">
                  {DAYS_HE[idx]}{" "}
                  <span className="text-stone-400 font-normal">
                    {dayDate.toLocaleDateString("he-IL", { day: "numeric", month: "numeric" })}
                  </span>
                </div>
                {daySessions.map((s) => (
                  <StudentSessionCard
                    key={s.id}
                    session={s}
                    myUpcomingRegistrations={myUpcomingRegistrations}
                  />
                ))}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
