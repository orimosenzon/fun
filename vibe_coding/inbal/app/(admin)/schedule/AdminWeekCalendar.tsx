"use client";

import { useRouter, usePathname } from "next/navigation";
import Link from "next/link";
import { useRef, useEffect, type RefObject } from "react";

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
  weekStart: string; // center week
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

function getWeekStartStr(date: Date): string {
  const d = new Date(date);
  d.setDate(d.getDate() - d.getDay());
  d.setHours(0, 0, 0, 0);
  return toLocalDateStr(d);
}

// Compact session card used in vertical weekly view
function AdminSessionCard({ session }: { session: AdminSessionData }) {
  const date = new Date(session.date);
  const timeStr = date.toLocaleTimeString("he-IL", { hour: "2-digit", minute: "2-digit" });
  const isCompleted = session.status === "COMPLETED";
  const isCancelled = session.status === "CANCELLED";

  const wheelFree = session.slots.wheelTotal - session.slots.wheel.length;
  const noWheelFree = session.slots.noWheelTotal - session.slots.noWheel.length;
  const totalTaken =
    session.slots.wheel.length + session.slots.noWheel.length + session.slots.unassigned.length;
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
      <div
        className={`px-2 py-1.5 border-b ${
          isCancelled
            ? "bg-red-100 border-red-200"
            : isCompleted
            ? "bg-stone-100 border-stone-200"
            : "bg-stone-50 border-stone-100"
        }`}
      >
        <div className="text-xs font-bold text-stone-800 leading-tight">{session.groupName}</div>
        <div className="flex items-center justify-between mt-0.5 gap-1">
          <span className="text-[11px] text-stone-500">{timeStr}</span>
          <div className="flex items-center gap-1">
            {isCompleted && (
              <span className="text-[10px] bg-stone-200 text-stone-600 px-1 rounded">הושלם</span>
            )}
            {isCancelled && (
              <span className="text-[10px] bg-red-200 text-red-700 px-1 rounded">בוטל</span>
            )}
            <span
              className={`text-[11px] font-semibold ${isFull ? "text-red-500" : "text-green-600"}`}
            >
              {totalTaken}/{totalSlots}
            </span>
          </div>
        </div>
      </div>

      <div className="p-2 space-y-2">
        <div>
          <div className="text-[10px] font-semibold text-stone-400 mb-1">
            🎡 אובן ({session.slots.wheelTotal})
          </div>
          <div className="space-y-0.5">
            {session.slots.wheel.map((name, i) => (
              <div
                key={i}
                className="text-xs bg-amber-100 text-amber-900 px-1.5 py-0.5 rounded font-medium"
              >
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
          <div className="text-[10px] font-semibold text-stone-400 mb-1">
            ✋ ללא אובן ({session.slots.noWheelTotal})
          </div>
          <div className="space-y-0.5">
            {session.slots.noWheel.map((name, i) => (
              <div
                key={i}
                className="text-xs bg-sky-100 text-sky-900 px-1.5 py-0.5 rounded font-medium"
              >
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
  const totalTaken =
    session.slots.wheel.length + session.slots.noWheel.length + session.slots.unassigned.length;
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
      <div
        className={`px-4 py-3 border-b ${
          isCancelled
            ? "bg-red-100 border-red-200"
            : isCompleted
            ? "bg-stone-100 border-stone-200"
            : "bg-stone-50 border-stone-100"
        }`}
      >
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
              {isCompleted && (
                <span className="text-xs bg-stone-200 text-stone-600 px-2 py-0.5 rounded">
                  הושלם
                </span>
              )}
              {isCancelled && (
                <span className="text-xs bg-red-200 text-red-700 px-2 py-0.5 rounded">בוטל</span>
              )}
              <span
                className={`text-sm font-semibold ${isFull ? "text-red-500" : "text-green-600"}`}
              >
                {totalTaken}/{totalSlots} מקומות
              </span>
            </div>
          </div>
        </div>
      </div>

      <div className="p-4 grid grid-cols-2 gap-4">
        <div>
          <div className="text-sm font-semibold text-stone-500 mb-2">
            🎡 אובן ({session.slots.wheelTotal})
          </div>
          <div className="space-y-1">
            {session.slots.wheel.map((name, i) => (
              <div
                key={i}
                className="text-sm bg-amber-100 text-amber-900 px-2 py-1 rounded font-medium"
              >
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
          <div className="text-sm font-semibold text-stone-500 mb-2">
            ✋ ללא אובן ({session.slots.noWheelTotal})
          </div>
          <div className="space-y-1">
            {session.slots.noWheel.map((name, i) => (
              <div
                key={i}
                className="text-sm bg-sky-100 text-sky-900 px-2 py-1 rounded font-medium"
              >
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

// A single day row within the vertical weekly view
function DaySection({
  day,
  sessions,
  isToday,
}: {
  day: Date;
  sessions: AdminSessionData[];
  isToday: boolean;
}) {
  if (sessions.length === 0) return null;

  const dayName = DAYS_HE[day.getDay()];
  const dateStr = day.toLocaleDateString("he-IL", {
    day: "numeric",
    month: "numeric",
    year: "2-digit",
  });

  // Group sessions in pairs of 2
  const pairs: AdminSessionData[][] = [];
  for (let i = 0; i < sessions.length; i += 2) {
    pairs.push(sessions.slice(i, i + 2));
  }

  return (
    <div className={`rounded-lg p-3 ${isToday ? "bg-amber-50/60 ring-1 ring-amber-200" : ""}`}>
      <div
        className={`text-sm font-semibold mb-2 ${isToday ? "text-amber-700" : "text-stone-500"}`}
      >
        יום {dayName}
        <span className="font-normal text-stone-400 mr-1">{dateStr}</span>
        {isToday && (
          <span className="mr-2 text-[11px] bg-amber-500 text-white px-1.5 py-0.5 rounded">
            היום
          </span>
        )}
      </div>
      <div className="space-y-2">
        {pairs.map((pair, pairIdx) => (
          <div
            key={pairIdx}
            className={`grid gap-2 ${pair.length === 2 ? "grid-cols-2" : "grid-cols-1 max-w-xs"}`}
          >
            {pair.map((s) => (
              <AdminSessionCard key={s.id} session={s} />
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}

// One week block in the vertical timeline
function WeekBlock({
  weekStart,
  sessions,
  isCurrentWeek,
  blockRef,
}: {
  weekStart: Date;
  sessions: AdminSessionData[];
  isCurrentWeek: boolean;
  blockRef?: RefObject<HTMLDivElement>;
}) {
  const today = new Date();
  today.setHours(0, 0, 0, 0);

  const weekEnd = new Date(weekStart);
  weekEnd.setDate(weekStart.getDate() + 6);
  const weekLabel = (() => {
    const fmt = (d: Date) =>
      d.toLocaleDateString("he-IL", { day: "numeric", month: "long" });
    return `${fmt(weekStart)} – ${fmt(weekEnd)}`;
  })();

  // Build 7 days for this week
  const days = Array.from({ length: 7 }, (_, i) => {
    const d = new Date(weekStart);
    d.setDate(weekStart.getDate() + i);
    return d;
  });

  // Sessions by day-of-week index
  const byDay: AdminSessionData[][] = Array.from({ length: 7 }, () => []);
  for (const s of sessions) {
    const d = new Date(s.date);
    // Map to this week's day
    const dayIdx = d.getDay();
    byDay[dayIdx].push(s);
  }
  // Sort each day's sessions by time
  for (const arr of byDay) {
    arr.sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
  }

  const hasAnySessions = byDay.some((arr) => arr.length > 0);

  return (
    <div
      ref={blockRef}
      className={`rounded-xl border ${
        isCurrentWeek
          ? "border-amber-300 shadow-md"
          : "border-stone-200 shadow-sm"
      } bg-white overflow-hidden`}
    >
      {/* Week header */}
      <div
        className={`px-4 py-2.5 border-b flex items-center gap-2 ${
          isCurrentWeek
            ? "bg-amber-500 border-amber-500"
            : "bg-stone-50 border-stone-200"
        }`}
      >
        <span
          className={`text-sm font-bold ${
            isCurrentWeek ? "text-white" : "text-stone-700"
          }`}
        >
          {weekLabel}
        </span>
        {isCurrentWeek && (
          <span className="text-[11px] bg-white/30 text-white px-1.5 py-0.5 rounded">
            השבוע
          </span>
        )}
      </div>

      {/* Days */}
      <div className="p-3 space-y-1">
        {hasAnySessions ? (
          days.map((day, idx) => (
            <DaySection
              key={idx}
              day={day}
              sessions={byDay[idx]}
              isToday={day.getTime() === today.getTime()}
            />
          ))
        ) : (
          <div className="text-center py-8 text-stone-300 text-sm">אין שיעורים בשבוע זה</div>
        )}
      </div>
    </div>
  );
}

export default function AdminWeekCalendar({
  sessions,
  weekStart,
  view: viewProp,
  day: dayProp,
}: Props) {
  const router = useRouter();
  const currentWeekRef = useRef<HTMLDivElement>(null);

  const centerWeekStart = new Date(weekStart);
  const today = new Date();
  today.setHours(0, 0, 0, 0);

  const isDayView = viewProp === "day";

  // Scroll current week into view on load
  useEffect(() => {
    if (!isDayView && currentWeekRef.current) {
      currentWeekRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [isDayView]);

  // Selected day for daily view
  const selectedDayDate = (() => {
    if (dayProp) {
      const d = new Date(dayProp + "T00:00:00");
      if (!isNaN(d.getTime())) return d;
    }
    const weekEnd = new Date(centerWeekStart);
    weekEnd.setDate(centerWeekStart.getDate() + 6);
    if (today >= centerWeekStart && today <= weekEnd) return new Date(today);
    return new Date(centerWeekStart);
  })();

  const selectedDayIdx = selectedDayDate.getDay();

  function goToToday() {
    if (isDayView) {
      router.push(
        `/schedule?week=${getWeekStartStr(today)}&view=day&day=${toLocalDateStr(today)}`
      );
    } else {
      router.push(`/schedule?week=${toLocalDateStr(today)}`);
    }
  }

  function prevWeek() {
    const d = new Date(centerWeekStart);
    d.setDate(d.getDate() - 7);
    router.push(`/schedule?week=${toLocalDateStr(d)}`);
  }

  function nextWeek() {
    const d = new Date(centerWeekStart);
    d.setDate(d.getDate() + 7);
    router.push(`/schedule?week=${toLocalDateStr(d)}`);
  }

  function prevDay() {
    const d = new Date(selectedDayDate);
    d.setDate(d.getDate() - 1);
    router.push(
      `/schedule?week=${getWeekStartStr(d)}&view=day&day=${toLocalDateStr(d)}`
    );
  }

  function nextDay() {
    const d = new Date(selectedDayDate);
    d.setDate(d.getDate() + 1);
    router.push(
      `/schedule?week=${getWeekStartStr(d)}&view=day&day=${toLocalDateStr(d)}`
    );
  }

  function switchToDayView() {
    const weekEnd = new Date(centerWeekStart);
    weekEnd.setDate(centerWeekStart.getDate() + 6);
    const target =
      today >= centerWeekStart && today <= weekEnd ? today : new Date(centerWeekStart);
    router.push(
      `/schedule?week=${getWeekStartStr(target)}&view=day&day=${toLocalDateStr(target)}`
    );
  }

  function goToWeekView() {
    router.push(`/schedule?week=${getWeekStartStr(centerWeekStart)}`);
  }

  const dayLabel = selectedDayDate.toLocaleDateString("he-IL", {
    weekday: "long",
    day: "numeric",
    month: "long",
  });

  // Split sessions into 3 week buckets: prev / center / next
  const prevWeekStart = new Date(centerWeekStart);
  prevWeekStart.setDate(prevWeekStart.getDate() - 7);
  const nextWeekStart = new Date(centerWeekStart);
  nextWeekStart.setDate(nextWeekStart.getDate() + 7);

  function getWeekEnd(ws: Date): Date {
    const d = new Date(ws);
    d.setDate(d.getDate() + 6);
    d.setHours(23, 59, 59, 999);
    return d;
  }

  const prevWeekEnd = getWeekEnd(prevWeekStart);
  const centerWeekEnd = getWeekEnd(centerWeekStart);
  const nextWeekEnd = getWeekEnd(nextWeekStart);

  const prevSessions = sessions.filter((s) => {
    const t = new Date(s.date).getTime();
    return t >= prevWeekStart.getTime() && t <= prevWeekEnd.getTime();
  });
  const centerSessions = sessions.filter((s) => {
    const t = new Date(s.date).getTime();
    return t >= centerWeekStart.getTime() && t <= centerWeekEnd.getTime();
  });
  const nextSessions = sessions.filter((s) => {
    const t = new Date(s.date).getTime();
    return t >= nextWeekStart.getTime() && t <= nextWeekEnd.getTime();
  });

  // Day sessions for daily view
  const daySessions = sessions
    .filter((s) => {
      const d = new Date(s.date);
      return d.getDay() === selectedDayIdx && toLocalDateStr(d) === toLocalDateStr(selectedDayDate);
    })
    .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());

  // Is center week the real current week?
  const isRealCurrentWeek =
    getWeekStartStr(today) === getWeekStartStr(centerWeekStart);

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
                &#8593;
              </button>
              <button
                onClick={nextWeek}
                className="w-8 h-8 flex items-center justify-center rounded-lg hover:bg-stone-100 transition text-stone-600 font-bold text-lg"
              >
                &#8595;
              </button>
            </div>
          )}
        </div>
      </div>

      {isDayView ? (
        /* Daily view — unchanged */
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
        /* Vertical weekly view — 3 weeks stacked */
        <div className="space-y-4">
          <WeekBlock
            weekStart={prevWeekStart}
            sessions={prevSessions}
            isCurrentWeek={false}
          />
          <WeekBlock
            weekStart={centerWeekStart}
            sessions={centerSessions}
            isCurrentWeek={isRealCurrentWeek}
            blockRef={currentWeekRef as RefObject<HTMLDivElement>}
          />
          <WeekBlock
            weekStart={nextWeekStart}
            sessions={nextSessions}
            isCurrentWeek={false}
          />
        </div>
      )}
    </div>
  );
}
