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
  const today = new Date();
  today.setHours(0, 0, 0, 0);

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
    const fmt = (d: Date) => d.toLocaleDateString("he-IL", { day: "numeric", month: "long" });
    return `${fmt(weekStartDate)} – ${fmt(end)}`;
  })();

  // Build 7-day array with dates
  const days = Array.from({ length: 7 }, (_, i) => {
    const d = new Date(weekStartDate);
    d.setDate(weekStartDate.getDate() + i);
    return d;
  });

  // Group sessions by day index (0=Sun)
  const byDay: SessionData[][] = Array.from({ length: 7 }, () => []);
  for (const s of sessions) {
    const d = new Date(s.date);
    byDay[d.getDay()].push(s);
  }

  return (
    <div className="space-y-3" dir="rtl">
      {/* Header: title + week nav */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <h1 className="text-xl font-bold text-stone-800">הלוח שלי</h1>
        <div className="flex items-center gap-2 bg-white border border-stone-200 rounded-xl px-2 py-1 shadow-sm">
          <button
            onClick={nextWeek}
            className="w-8 h-8 flex items-center justify-center rounded-lg hover:bg-stone-100 transition text-stone-600 font-bold"
          >
            &rsaquo;
          </button>
          <span className="text-sm font-medium text-stone-700 min-w-[180px] text-center">
            {weekLabel}
          </span>
          <button
            onClick={prevWeek}
            className="w-8 h-8 flex items-center justify-center rounded-lg hover:bg-stone-100 transition text-stone-600 font-bold"
          >
            &lsaquo;
          </button>
        </div>
      </div>

      {/* Calendar grid — horizontal scroll on mobile */}
      <div className="overflow-x-auto rounded-xl border border-stone-200 bg-white shadow-sm">
        <div className="min-w-[560px]">
          {/* Day headers */}
          <div className="grid grid-cols-7 border-b border-stone-200">
            {days.map((day, idx) => {
              const isToday = day.getTime() === today.getTime();
              return (
                <div
                  key={idx}
                  className={`px-2 py-3 text-center border-r border-stone-100 last:border-r-0 ${
                    isToday ? "bg-amber-50" : ""
                  }`}
                >
                  <div className={`text-xs font-semibold ${isToday ? "text-amber-700" : "text-stone-500"}`}>
                    {DAYS_HE[idx]}
                  </div>
                  <div
                    className={`mt-1 text-sm font-bold inline-flex w-7 h-7 items-center justify-center rounded-full ${
                      isToday
                        ? "bg-amber-500 text-white"
                        : "text-stone-700"
                    }`}
                  >
                    {day.getDate()}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Session rows */}
          <div className="grid grid-cols-7 min-h-[160px]">
            {days.map((_, idx) => {
              const isToday = days[idx].getTime() === today.getTime();
              return (
                <div
                  key={idx}
                  className={`border-r border-stone-100 last:border-r-0 p-1.5 space-y-1.5 ${
                    isToday ? "bg-amber-50/40" : ""
                  }`}
                >
                  {byDay[idx].length === 0 ? (
                    <div className="h-full flex items-center justify-center">
                      <span className="text-stone-200 text-xs">—</span>
                    </div>
                  ) : (
                    byDay[idx].map((s) => (
                      <StudentSessionCard
                        key={s.id}
                        session={s}
                        myUpcomingRegistrations={myUpcomingRegistrations}
                      />
                    ))
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Legend */}
      <div className="flex items-center gap-4 text-xs text-stone-400 px-1">
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-3 rounded-full bg-amber-500" /> המקום שלך
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-3 rounded-full bg-stone-400" /> תפוס
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-3 rounded-full border-2 border-green-400 bg-green-50" /> פנוי (ניתן להעברה)
        </span>
      </div>
    </div>
  );
}
