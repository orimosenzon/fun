"use client";

import { useRouter } from "next/navigation";
import Link from "next/link";
import { useRef, useEffect, type RefObject } from "react";

const DAYS_HE = ["ראשון", "שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת"];

type AdminSessionData = {
  id: string;
  date: string;
  status: string;
  groupName: string;
  groupLocation: string | null;
  groupDuration: number;
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

// Group sessions by time proximity (gap threshold in minutes)
function groupByProximity(
  sessions: AdminSessionData[],
  thresholdMinutes = 180
): AdminSessionData[][] {
  if (sessions.length === 0) return [];
  const groups: AdminSessionData[][] = [];
  let group = [sessions[0]];
  for (let i = 1; i < sessions.length; i++) {
    const diff =
      (new Date(sessions[i].date).getTime() - new Date(sessions[i - 1].date).getTime()) / 60000;
    if (diff <= thresholdMinutes) {
      group.push(sessions[i]);
    } else {
      groups.push(group);
      group = [sessions[i]];
    }
  }
  groups.push(group);
  return groups;
}

// A single slot square — filled with a name or empty (dashed)
function SlotSquare({ name, type }: { name?: string; type: "wheel" | "noWheel" }) {
  if (!name) {
    return (
      <div className="w-11 h-10 sm:w-[54px] sm:h-[48px] rounded border-2 border-dashed border-stone-200 shrink-0" />
    );
  }
  const display = name.split(" ")[0];
  return (
    <div
      className={`w-11 h-10 sm:w-[54px] sm:h-[48px] rounded border flex items-center justify-center text-center p-1 shrink-0 ${
        type === "wheel"
          ? "bg-amber-100 border-amber-200 text-amber-900"
          : "bg-sky-100 border-sky-200 text-sky-900"
      }`}
    >
      <span className="text-[11px] font-medium leading-tight">{display}</span>
    </div>
  );
}

// One session row inside a pair-block table
function SessionRow({
  session,
  hasBorderBottom,
}: {
  session: AdminSessionData;
  hasBorderBottom: boolean;
}) {
  const date = new Date(session.date);
  const startStr = date.toLocaleTimeString("he-IL", { hour: "2-digit", minute: "2-digit" });
  const endDate = new Date(date.getTime() + session.groupDuration * 60 * 1000);
  const endStr = endDate.toLocaleTimeString("he-IL", { hour: "2-digit", minute: "2-digit" });

  const isCompleted = session.status === "COMPLETED";
  const isCancelled = session.status === "CANCELLED";

  const wheelFree = session.slots.wheelTotal - session.slots.wheel.length;
  const noWheelFree = session.slots.noWheelTotal - session.slots.noWheel.length;
  const totalTaken =
    session.slots.wheel.length + session.slots.noWheel.length + session.slots.unassigned.length;
  const totalSlots = session.slots.wheelTotal + session.slots.noWheelTotal;
  const isFull = totalTaken >= totalSlots;

  const borderClass = hasBorderBottom ? "border-b border-stone-100" : "";
  const bgClass = isCancelled ? "bg-red-50/50" : isCompleted ? "bg-stone-50/60" : "";

  return (
    <div className={`px-3 py-2.5 ${borderClass} ${bgClass}`}>
      {/* Header: group name + time range + count */}
      <Link
        href={`/sessions/${session.id}`}
        className="flex items-center justify-between mb-2 hover:opacity-75 transition"
      >
        <span className="text-xs text-stone-400 truncate">{session.groupName}</span>
        <div className="flex items-center gap-2 shrink-0">
          <span
            className={`text-sm font-bold ${
              isCancelled ? "text-red-500" : isCompleted ? "text-stone-400" : "text-stone-700"
            }`}
          >
            {startStr} – {endStr}
          </span>
          {isCancelled && (
            <span className="text-[9px] bg-red-100 text-red-500 px-1 py-0.5 rounded">בוטל</span>
          )}
          {isCompleted && (
            <span className="text-[9px] bg-stone-100 text-stone-400 px-1 py-0.5 rounded">הושלם</span>
          )}
          <span className={`text-[10px] font-semibold ${isFull ? "text-red-400" : "text-green-500"}`}>
            {totalTaken}/{totalSlots}
          </span>
        </div>
      </Link>

      {/* Single slots row: wheel | noWheel */}
      <div className="overflow-x-auto">
      <div className="flex items-center gap-1">
        {session.slots.wheel.map((name, i) => (
          <SlotSquare key={i} name={name} type="wheel" />
        ))}
        {Array.from({ length: wheelFree }).map((_, i) => (
          <SlotSquare key={`we${i}`} type="wheel" />
        ))}
        <div className="w-px h-8 bg-stone-200 mx-0.5 shrink-0" />
        {session.slots.noWheel.map((name, i) => (
          <SlotSquare key={i} name={name} type="noWheel" />
        ))}
        {Array.from({ length: noWheelFree }).map((_, i) => (
          <SlotSquare key={`ne${i}`} type="noWheel" />
        ))}
        {session.slots.unassigned.length > 0 && (
          <>
            <div className="w-px h-8 bg-stone-200 mx-0.5 shrink-0" />
            {session.slots.unassigned.map((name, i) => (
              <div
                key={i}
                className="text-[11px] bg-stone-100 text-stone-600 px-1.5 py-0.5 rounded self-center"
              >
                {name}
              </div>
            ))}
          </>
        )}
      </div>
      </div>
    </div>
  );
}

// A block of sessions that are close in time — shown as a table
function SessionPairBlock({ sessions }: { sessions: AdminSessionData[] }) {
  return (
    <div className="rounded-lg border border-stone-200 bg-white">
      {sessions.map((session, idx) => (
        <SessionRow
          key={session.id}
          session={session}
          hasBorderBottom={idx < sessions.length - 1}
        />
      ))}
    </div>
  );
}

// A single day within a week block
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

  const groups = groupByProximity(sessions);

  return (
    <div className={`rounded-lg p-3 ${isToday ? "bg-amber-50/60 ring-1 ring-amber-200" : ""}`}>
      <div
        className={`text-sm font-semibold mb-3 flex items-center gap-2 ${
          isToday ? "text-amber-700" : "text-stone-500"
        }`}
      >
        <span>יום {dayName}</span>
        <span className="font-normal text-stone-400">{dateStr}</span>
        {isToday && (
          <span className="text-[11px] bg-amber-500 text-white px-1.5 py-0.5 rounded">היום</span>
        )}
      </div>
      <div className="space-y-4">
        {groups.map((group, gi) => (
          <SessionPairBlock key={gi} sessions={group} />
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

  const days = Array.from({ length: 7 }, (_, i) => {
    const d = new Date(weekStart);
    d.setDate(weekStart.getDate() + i);
    return d;
  });

  const byDay: AdminSessionData[][] = Array.from({ length: 7 }, () => []);
  for (const s of sessions) {
    const dayIdx = new Date(s.date).getDay();
    byDay[dayIdx].push(s);
  }
  for (const arr of byDay) {
    arr.sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
  }

  const hasAnySessions = byDay.some((arr) => arr.length > 0);

  return (
    <div
      ref={blockRef}
      className={`rounded-xl border ${
        isCurrentWeek ? "border-amber-300 shadow-md" : "border-stone-200 shadow-sm"
      } bg-white overflow-hidden`}
    >
      <div
        className={`px-4 py-2.5 border-b flex items-center gap-2 ${
          isCurrentWeek ? "bg-amber-500 border-amber-500" : "bg-stone-50 border-stone-200"
        }`}
      >
        <span className={`text-sm font-bold ${isCurrentWeek ? "text-white" : "text-stone-700"}`}>
          {weekLabel}
        </span>
        {isCurrentWeek && (
          <span className="text-[11px] bg-white/30 text-white px-1.5 py-0.5 rounded">השבוע</span>
        )}
      </div>

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

  useEffect(() => {
    if (!isDayView && currentWeekRef.current) {
      currentWeekRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [isDayView]);

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
      router.push(`/schedule?week=${getWeekStartStr(today)}&view=day&day=${toLocalDateStr(today)}`);
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
    router.push(`/schedule?week=${getWeekStartStr(d)}&view=day&day=${toLocalDateStr(d)}`);
  }

  function nextDay() {
    const d = new Date(selectedDayDate);
    d.setDate(d.getDate() + 1);
    router.push(`/schedule?week=${getWeekStartStr(d)}&view=day&day=${toLocalDateStr(d)}`);
  }

  function switchToDayView() {
    const weekEnd = new Date(centerWeekStart);
    weekEnd.setDate(centerWeekStart.getDate() + 6);
    const target =
      today >= centerWeekStart && today <= weekEnd ? today : new Date(centerWeekStart);
    router.push(`/schedule?week=${getWeekStartStr(target)}&view=day&day=${toLocalDateStr(target)}`);
  }

  function goToWeekView() {
    router.push(`/schedule?week=${getWeekStartStr(centerWeekStart)}`);
  }

  const dayLabel = selectedDayDate.toLocaleDateString("he-IL", {
    weekday: "long",
    day: "numeric",
    month: "long",
  });

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

  const daySessions = sessions
    .filter((s) => {
      const d = new Date(s.date);
      return (
        d.getDay() === selectedDayIdx &&
        toLocalDateStr(d) === toLocalDateStr(selectedDayDate)
      );
    })
    .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());

  const isRealCurrentWeek = getWeekStartStr(today) === getWeekStartStr(centerWeekStart);

  return (
    <div className="space-y-3" dir="rtl">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <h1 className="text-2xl font-bold text-stone-800">לוח שיעורים</h1>
        <div className="flex items-center gap-2">
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

          <button
            onClick={goToToday}
            className="px-3 py-1.5 text-sm font-medium bg-white border border-stone-200 rounded-lg hover:bg-stone-50 shadow-sm transition text-stone-600"
          >
            היום
          </button>

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
        <div>
          {daySessions.length === 0 ? (
            <div className="text-center py-20 text-stone-400">
              <div className="text-5xl mb-4">📅</div>
              <div className="text-lg font-medium">אין שיעורים ביום זה</div>
            </div>
          ) : (
            <div className="space-y-4 max-w-2xl">
              {groupByProximity(daySessions).map((group, gi) => (
                <SessionPairBlock key={gi} sessions={group} />
              ))}
            </div>
          )}
        </div>
      ) : (
        <div className="space-y-4">
          <WeekBlock weekStart={prevWeekStart} sessions={prevSessions} isCurrentWeek={false} />
          <WeekBlock
            weekStart={centerWeekStart}
            sessions={centerSessions}
            isCurrentWeek={isRealCurrentWeek}
            blockRef={currentWeekRef as RefObject<HTMLDivElement>}
          />
          <WeekBlock weekStart={nextWeekStart} sessions={nextSessions} isCurrentWeek={false} />
        </div>
      )}
    </div>
  );
}
