"use client";

import { useRouter } from "next/navigation";
import Link from "next/link";

const DAYS_HE = ["ראשון", "שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת"];

type AdminSessionData = {
  id: string;
  date: string;
  status: string;
  groupName: string;
  groupLocation: string | null;
  slots: {
    wheelTotal: number;
    noWheelTotal: number;
    wheel: string[];
    noWheel: string[];
    unassigned: string[];
  };
};

interface Props {
  sessions: AdminSessionData[];
  weekStart: string;
}

function AdminSessionCard({ session }: { session: AdminSessionData }) {
  const date = new Date(session.date);
  const timeStr = date.toLocaleTimeString("he-IL", { hour: "2-digit", minute: "2-digit" });
  const isCompleted = session.status === "COMPLETED";
  const isCancelled = session.status === "CANCELLED";
  const isDimmed = isCompleted || isCancelled;

  function SlotRow({ names, total, label }: { names: string[]; total: number; label: string }) {
    const free = total - names.length;
    return (
      <div>
        <div className="text-[10px] text-stone-400 mb-0.5">{label}</div>
        <div className="space-y-0.5">
          {names.map((name, i) => (
            <div key={i} className={`text-[10px] px-1.5 py-0.5 rounded font-medium truncate leading-tight ${isDimmed ? "bg-stone-100 text-stone-400" : "bg-amber-100 text-amber-800"}`}>
              {name}
            </div>
          ))}
          {Array.from({ length: free }).map((_, i) => (
            <div key={`f${i}`} className="text-[10px] border border-dashed border-stone-200 text-stone-300 px-1.5 py-0.5 rounded leading-tight">
              פנוי
            </div>
          ))}
        </div>
      </div>
    );
  }

  const wheelTaken = session.slots.wheel.length;
  const noWheelTaken = session.slots.noWheel.length;
  const totalTaken = wheelTaken + noWheelTaken + session.slots.unassigned.length;
  const totalSlots = session.slots.wheelTotal + session.slots.noWheelTotal;

  return (
    <Link
      href={`/sessions/${session.id}`}
      className={`block rounded-lg border overflow-hidden transition ${isDimmed ? "border-stone-100 bg-stone-50 opacity-60 hover:opacity-80" : "border-stone-200 bg-white hover:border-amber-300 hover:shadow-sm"}`}
    >
      {/* Card header */}
      <div className={`border-b px-2 py-1.5 ${isDimmed ? "bg-stone-100 border-stone-100" : "bg-stone-50 border-stone-100"}`}>
        <div className="font-semibold text-stone-800 text-xs truncate">{session.groupName}</div>
        <div className="text-[10px] text-stone-400 flex items-center justify-between">
          <span>{timeStr}</span>
          <div className="flex items-center gap-1">
            {isCompleted && <span className="text-stone-400">הושלם</span>}
            {isCancelled && <span className="text-red-400">בוטל</span>}
            <span className={`font-medium ${totalTaken >= totalSlots ? "text-red-500" : "text-green-600"}`}>
              {totalTaken}/{totalSlots}
            </span>
          </div>
        </div>
      </div>

      {/* Slots */}
      <div className="p-1.5 space-y-1.5">
        <div className="grid grid-cols-2 gap-1.5">
          <SlotRow
            names={session.slots.wheel}
            total={session.slots.wheelTotal}
            label="🎡 אובן"
          />
          <SlotRow
            names={session.slots.noWheel}
            total={session.slots.noWheelTotal}
            label="✋ ללא"
          />
        </div>
        {session.slots.unassigned.length > 0 && (
          <div>
            <div className="text-[10px] text-stone-400 mb-0.5">לא משוייך</div>
            <div className="flex flex-wrap gap-0.5">
              {session.slots.unassigned.map((name, i) => (
                <span key={i} className="text-[10px] bg-stone-100 text-stone-500 px-1 py-0.5 rounded">
                  {name}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </Link>
  );
}

export default function AdminWeekCalendar({ sessions, weekStart }: Props) {
  const router = useRouter();
  const weekStartDate = new Date(weekStart);
  const today = new Date();
  today.setHours(0, 0, 0, 0);

  function prevWeek() {
    const d = new Date(weekStartDate);
    d.setDate(d.getDate() - 7);
    router.push(`/schedule?week=${d.toISOString().slice(0, 10)}`);
  }

  function nextWeek() {
    const d = new Date(weekStartDate);
    d.setDate(d.getDate() + 7);
    router.push(`/schedule?week=${d.toISOString().slice(0, 10)}`);
  }

  const weekLabel = (() => {
    const end = new Date(weekStartDate);
    end.setDate(weekStartDate.getDate() + 6);
    const fmt = (d: Date) => d.toLocaleDateString("he-IL", { day: "numeric", month: "long" });
    return `${fmt(weekStartDate)} – ${fmt(end)}`;
  })();

  const days = Array.from({ length: 7 }, (_, i) => {
    const d = new Date(weekStartDate);
    d.setDate(weekStartDate.getDate() + i);
    return d;
  });

  const byDay: AdminSessionData[][] = Array.from({ length: 7 }, () => []);
  for (const s of sessions) {
    const d = new Date(s.date);
    byDay[d.getDay()].push(s);
  }

  return (
    <div className="space-y-3" dir="rtl">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <h1 className="text-2xl font-bold text-stone-800">לוח שיעורים</h1>
        <div className="flex items-center gap-2 bg-white border border-stone-200 rounded-xl px-2 py-1 shadow-sm">
          <button
            onClick={prevWeek}
            className="w-8 h-8 flex items-center justify-center rounded-lg hover:bg-stone-100 transition text-stone-600 font-bold text-lg"
          >
            &rsaquo;
          </button>
          <span className="text-sm font-medium text-stone-700 min-w-[180px] text-center">
            {weekLabel}
          </span>
          <button
            onClick={nextWeek}
            className="w-8 h-8 flex items-center justify-center rounded-lg hover:bg-stone-100 transition text-stone-600 font-bold text-lg"
          >
            &lsaquo;
          </button>
        </div>
      </div>

      {/* Calendar grid */}
      <div className="overflow-x-auto rounded-xl border border-stone-200 bg-white shadow-sm">
        <div className="min-w-[600px]">
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
                      isToday ? "bg-amber-500 text-white" : "text-stone-700"
                    }`}
                  >
                    {day.getDate()}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Session cells */}
          <div className="grid grid-cols-7 min-h-[200px]">
            {days.map((day, idx) => {
              const isToday = day.getTime() === today.getTime();
              return (
                <div
                  key={idx}
                  className={`border-r border-stone-100 last:border-r-0 p-1.5 space-y-1.5 ${
                    isToday ? "bg-amber-50/40" : ""
                  }`}
                >
                  {byDay[idx].length === 0 ? (
                    <div className="h-full flex items-start justify-center pt-4">
                      <span className="text-stone-200 text-xs">—</span>
                    </div>
                  ) : (
                    byDay[idx].map((s) => (
                      <AdminSessionCard key={s.id} session={s} />
                    ))
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
