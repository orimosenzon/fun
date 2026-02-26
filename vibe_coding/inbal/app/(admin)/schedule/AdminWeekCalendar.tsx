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
  view?: string;
  day?: string;
}

function toLocalDateStr(d: Date): string {
  return (
    d.getFullYear() +
    "-" +
    String(d.getMonth() + 1).padStart(2, "0") +
    "-" +
    String(d.getDate()).padStart(2, "0")
  );
}

// Compact card for weekly grid
function AdminSessionCard({ session }: { session: AdminSessionData }) {
  const date = new Date(session.date);
  const timeStr = date.toLocaleTimeString("he-IL", { hour: "2-digit", minute: "2-digit" });
  const isCompleted = session.status === "COMPLETED";
  const isCancelled = session.status === "CANCELLED";

  const wheelFree = session.slots.wheelTotal - session.slots.wheel.length;
  const noWheelFree = session.slots.noWheelTotal - session.slots.noWheel.length;
  const totalTaken = session.slots.wheel.length + session.slots.noWheel.length + session.slots.unassigned.length;
  const totalSlots = session.slots.wheelTotal + session.slots.noWheelTotal;
  const isFull = totalTaken >= totalSlots;

  return (
    <Link
      href={`/sessions/${session.id}`}
      className={`block rounded-lg border overflow-hidden transition hover:shadow-md ${
        isCancelled
          ? "border-red-200 bg-red-50"
          : isCompleted
          ? "border-stone-200 bg-stone-50"
          : "border-stone-200 bg-white hover:border-amber-300"
      }`}
    >
      <div className={`px-2 py-1.5 border-b ${
        isCancelled ? "bg-red-100 border-red-200" :
        isCompleted ? "bg-stone-100 border-stone-200" :
        "bg-stone-50 border-stone-100"
      }`}>
        <div className="text-xs font-bold text-stone-800 leading-tight">{session.groupName}</div>
        <div className="flex items-center justify-between mt-0.5 gap-1">
          <span className="text-[11px] text-stone-500">{timeStr}</span>
          <div className="flex items-center gap-1">
            {isCompleted && <span className="text-[10px] bg-stone-200 text-stone-600 px-1 rounded">הושלם</span>}
            {isCancelled && <span className="text-[10px] bg-red-200 text-red-700 px-1 rounded">בוטל</span>}
            <span className={`text-[11px] font-semibold ${isFull ? "text-red-500" : "text-green-600"}`}>
              {totalTaken}/{totalSlots}
            </span>
          </div>
        </div>
      </div>

      <div className="p-2 space-y-2">
        <div>
          <div className="text-[10px] font-semibold text-stone-400 mb-1">🎡 אובן ({session.slots.wheelTotal})</div>
          <div className="space-y-0.5">
            {session.slots.wheel.map((name, i) => (
              <div key={i} className="text-xs bg-amber-100 text-amber-900 px-1.5 py-0.5 rounded font-medium">
                {name}
              </div>
            ))}
            {wheelFree > 0 && (
              <div className="text-[10px] text-stone-400 border border-dashed border-stone-200 px-1.5 py-0.5 rounded text-center">
                {wheelFree} פנויים
              </div>
            )}
          </div>
        </div>

        <div>
          <div className="text-[10px] font-semibold text-stone-400 mb-1">✋ ללא אובן ({session.slots.noWheelTotal})</div>
          <div className="space-y-0.5">
            {session.slots.noWheel.map((name, i) => (
              <div key={i} className="text-xs bg-sky-100 text-sky-900 px-1.5 py-0.5 rounded font-medium">
                {name}
              </div>
            ))}
            {noWheelFree > 0 && (
              <div className="text-[10px] text-stone-400 border border-dashed border-stone-200 px-1.5 py-0.5 rounded text-center">
                {noWheelFree} פנויים
              </div>
            )}
          </div>
        </div>

        {session.slots.unassigned.length > 0 && (
          <div>
            <div className="text-[10px] font-semibold text-stone-400 mb-1">לא משוייך</div>
            <div className="space-y-0.5">
              {session.slots.unassigned.map((name, i) => (
                <div key={i} className="text-xs bg-stone-100 text-stone-600 px-1.5 py-0.5 rounded">
                  {name}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </Link>
  );
}

// Expanded card for daily view
function AdminDaySessionCard({ session }: { session: AdminSessionData }) {
  const date = new Date(session.date);
  const timeStr = date.toLocaleTimeString("he-IL", { hour: "2-digit", minute: "2-digit" });
  const isCompleted = session.status === "COMPLETED";
  const isCancelled = session.status === "CANCELLED";

  const wheelFree = session.slots.wheelTotal - session.slots.wheel.length;
  const noWheelFree = session.slots.noWheelTotal - session.slots.noWheel.length;
  const totalTaken = session.slots.wheel.length + session.slots.noWheel.length + session.slots.unassigned.length;
  const totalSlots = session.slots.wheelTotal + session.slots.noWheelTotal;
  const isFull = totalTaken >= totalSlots;

  return (
    <Link
      href={`/sessions/${session.id}`}
      className={`block rounded-xl border overflow-hidden transition hover:shadow-md ${
        isCancelled
          ? "border-red-200 bg-red-50"
          : isCompleted
          ? "border-stone-200 bg-stone-50"
          : "border-stone-200 bg-white hover:border-amber-300"
      }`}
    >
      {/* Header */}
      <div className={`px-4 py-3 border-b ${
        isCancelled ? "bg-red-100 border-red-200" :
        isCompleted ? "bg-stone-100 border-stone-200" :
        "bg-stone-50 border-stone-100"
      }`}>
        <div className="flex items-center justify-between">
          <div>
            <div className="text-base font-bold text-stone-800">{session.groupName}</div>
            {session.groupLocation && (
              <div className="text-sm text-stone-500 mt-0.5">{session.groupLocation}</div>
            )}
          </div>
          <div className="text-right">
            <div className="text-xl font-bold text-stone-700">{timeStr}</div>
            <div className="flex items-center gap-1 justify-end mt-1">
              {isCompleted && <span className="text-xs bg-stone-200 text-stone-600 px-2 py-0.5 rounded">הושלם</span>}
              {isCancelled && <span className="text-xs bg-red-200 text-red-700 px-2 py-0.5 rounded">בוטל</span>}
              <span className={`text-sm font-semibold ${isFull ? "text-red-500" : "text-green-600"}`}>
                {totalTaken}/{totalSlots} מקומות
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Slots */}
      <div className="p-4 grid grid-cols-2 gap-4">
        <div>
          <div className="text-sm font-semibold text-stone-500 mb-2">🎡 אובן ({session.slots.wheelTotal})</div>
          <div className="space-y-1">
            {session.slots.wheel.map((name, i) => (
              <div key={i} className="text-sm bg-amber-100 text-amber-900 px-2 py-1 rounded font-medium">
                {name}
              </div>
            ))}
            {wheelFree > 0 && (
              <div className="text-xs text-stone-400 border border-dashed border-stone-200 px-2 py-1 rounded text-center">
                {wheelFree} מקומות פנויים
              </div>
            )}
          </div>
        </div>

        <div>
          <div className="text-sm font-semibold text-stone-500 mb-2">✋ ללא אובן ({session.slots.noWheelTotal})</div>
          <div className="space-y-1">
            {session.slots.noWheel.map((name, i) => (
              <div key={i} className="text-sm bg-sky-100 text-sky-900 px-2 py-1 rounded font-medium">
                {name}
              </div>
            ))}
            {noWheelFree > 0 && (
              <div className="text-xs text-stone-400 border border-dashed border-stone-200 px-2 py-1 rounded text-center">
                {noWheelFree} מקומות פנויים
              </div>
            )}
          </div>
        </div>

        {session.slots.unassigned.length > 0 && (
          <div className="col-span-2">
            <div className="text-sm font-semibold text-stone-500 mb-2">לא משוייך</div>
            <div className="flex flex-wrap gap-1">
              {session.slots.unassigned.map((name, i) => (
                <div key={i} className="text-sm bg-stone-100 text-stone-600 px-2 py-1 rounded">
                  {name}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </Link>
  );
}

export default function AdminWeekCalendar({ sessions, weekStart, view: viewProp, day: dayProp }: Props) {
  const router = useRouter();
  const weekStartDate = new Date(weekStart);
  const today = new Date();
  today.setHours(0, 0, 0, 0);

  const isDayView = viewProp === "day";

  function getWeekStartStr(date: Date): string {
    const d = new Date(date);
    d.setDate(d.getDate() - d.getDay());
    d.setHours(0, 0, 0, 0);
    return toLocalDateStr(d);
  }

  // Selected day for daily view
  const selectedDayDate = (() => {
    if (dayProp) {
      const d = new Date(dayProp + "T00:00:00");
      if (!isNaN(d.getTime())) return d;
    }
    // default: today if in current week, else first day of week
    const weekEnd = new Date(weekStartDate);
    weekEnd.setDate(weekStartDate.getDate() + 6);
    if (today >= weekStartDate && today <= weekEnd) return new Date(today);
    return new Date(weekStartDate);
  })();

  const selectedDayIdx = selectedDayDate.getDay();
  const weekStr = getWeekStartStr(weekStartDate);

  function goToToday() {
    if (isDayView) {
      router.push(`/schedule?week=${getWeekStartStr(today)}&view=day&day=${toLocalDateStr(today)}`);
    } else {
      router.push(`/schedule?week=${toLocalDateStr(today)}`);
    }
  }

  function prevWeek() {
    const d = new Date(weekStartDate);
    d.setDate(d.getDate() - 7);
    router.push(`/schedule?week=${toLocalDateStr(d)}`);
  }

  function nextWeek() {
    const d = new Date(weekStartDate);
    d.setDate(d.getDate() + 7);
    router.push(`/schedule?week=${toLocalDateStr(d)}`);
  }

  function prevDay() {
    const d = new Date(selectedDayDate);
    d.setDate(d.getDate() - 1);
    router.push(`/schedule?week=${getWeekStartStr(d)}&view=day&day=${toLocalDateStr(d)}`);
  }

  function nextDay() {
    const d = new Date(selectedDayDate);
    d.setDate(d.getDate() + 1);
    router.push(`/schedule?week=${getWeekStartStr(d)}&view=day&day=${toLocalDateStr(d)}`);
  }

  function goToDayView(day: Date) {
    router.push(`/schedule?week=${getWeekStartStr(day)}&view=day&day=${toLocalDateStr(day)}`);
  }

  function switchToDayView() {
    const weekEnd = new Date(weekStartDate);
    weekEnd.setDate(weekStartDate.getDate() + 6);
    const target = today >= weekStartDate && today <= weekEnd ? today : new Date(weekStartDate);
    goToDayView(target);
  }

  function goToWeekView() {
    router.push(`/schedule?week=${weekStr}`);
  }

  const weekLabel = (() => {
    const end = new Date(weekStartDate);
    end.setDate(weekStartDate.getDate() + 6);
    const fmt = (d: Date) => d.toLocaleDateString("he-IL", { day: "numeric", month: "long" });
    return `${fmt(weekStartDate)} – ${fmt(end)}`;
  })();

  const dayLabel = selectedDayDate.toLocaleDateString("he-IL", {
    weekday: "long",
    day: "numeric",
    month: "long",
  });

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

  const daySessions = byDay[selectedDayIdx] ?? [];
  const isToday = (d: Date) => d.getTime() === today.getTime();

  return (
    <div className="space-y-3" dir="rtl">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <h1 className="text-2xl font-bold text-stone-800">לוח שיעורים</h1>
        <div className="flex items-center gap-2">
          {/* View toggle */}
          <div className="flex rounded-lg border border-stone-200 overflow-hidden bg-white shadow-sm">
            <button
              onClick={goToWeekView}
              className={`px-3 py-1.5 text-sm font-medium transition ${
                !isDayView ? "bg-amber-500 text-white" : "text-stone-600 hover:bg-stone-50"
              }`}
            >
              שבועי
            </button>
            <button
              onClick={isDayView ? undefined : switchToDayView}
              className={`px-3 py-1.5 text-sm font-medium transition ${
                isDayView ? "bg-amber-500 text-white" : "text-stone-600 hover:bg-stone-50"
              }`}
            >
              יומי
            </button>
          </div>

          {/* Today button */}
          <button
            onClick={goToToday}
            className="px-3 py-1.5 text-sm font-medium bg-white border border-stone-200 rounded-lg hover:bg-stone-50 shadow-sm transition text-stone-600"
          >
            היום
          </button>

          {/* Navigation */}
          {isDayView ? (
            <div className="flex items-center gap-2 bg-white border border-stone-200 rounded-xl px-2 py-1 shadow-sm">
              <button
                onClick={prevDay}
                className="w-8 h-8 flex items-center justify-center rounded-lg hover:bg-stone-100 transition text-stone-600 font-bold text-lg"
              >
                &lsaquo;
              </button>
              <span className="text-sm font-medium text-stone-700 min-w-[200px] text-center">
                {dayLabel}
              </span>
              <button
                onClick={nextDay}
                className="w-8 h-8 flex items-center justify-center rounded-lg hover:bg-stone-100 transition text-stone-600 font-bold text-lg"
              >
                &rsaquo;
              </button>
            </div>
          ) : (
            <div className="flex items-center gap-2 bg-white border border-stone-200 rounded-xl px-2 py-1 shadow-sm">
              <button
                onClick={prevWeek}
                className="w-8 h-8 flex items-center justify-center rounded-lg hover:bg-stone-100 transition text-stone-600 font-bold text-lg"
              >
                &lsaquo;
              </button>
              <span className="text-sm font-medium text-stone-700 min-w-[180px] text-center">
                {weekLabel}
              </span>
              <button
                onClick={nextWeek}
                className="w-8 h-8 flex items-center justify-center rounded-lg hover:bg-stone-100 transition text-stone-600 font-bold text-lg"
              >
                &rsaquo;
              </button>
            </div>
          )}
        </div>
      </div>

      {isDayView ? (
        /* Daily view */
        <div>
          {daySessions.length === 0 ? (
            <div className="text-center py-20 text-stone-400">
              <div className="text-5xl mb-4">📅</div>
              <div className="text-lg font-medium">אין שיעורים ביום זה</div>
            </div>
          ) : (
            <div className="space-y-3 max-w-2xl">
              {daySessions.map((s) => (
                <AdminDaySessionCard key={s.id} session={s} />
              ))}
            </div>
          )}
        </div>
      ) : (
        /* Weekly view */
        <div className="overflow-x-auto rounded-xl border border-stone-200 bg-white shadow-sm">
          <div className="min-w-[600px]">
            {/* Day headers — clickable to switch to daily view */}
            <div className="grid grid-cols-7 border-b border-stone-200">
              {days.map((day, idx) => {
                const isTodayDay = isToday(day);
                return (
                  <div
                    key={idx}
                    onClick={() => goToDayView(day)}
                    className={`px-2 py-3 text-center border-r border-stone-100 last:border-r-0 cursor-pointer hover:bg-amber-50/60 transition ${
                      isTodayDay ? "bg-amber-50" : ""
                    }`}
                  >
                    <div className={`text-xs font-semibold ${isTodayDay ? "text-amber-700" : "text-stone-500"}`}>
                      {DAYS_HE[idx]}
                    </div>
                    <div
                      className={`mt-1 text-sm font-bold inline-flex w-7 h-7 items-center justify-center rounded-full ${
                        isTodayDay ? "bg-amber-500 text-white" : "text-stone-700"
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
                const isTodayDay = isToday(day);
                return (
                  <div
                    key={idx}
                    className={`border-r border-stone-100 last:border-r-0 p-1.5 space-y-1.5 ${
                      isTodayDay ? "bg-amber-50/40" : ""
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
      )}
    </div>
  );
}
