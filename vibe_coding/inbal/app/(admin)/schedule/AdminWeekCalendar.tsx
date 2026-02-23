"use client";

import { useRouter } from "next/navigation";
import Link from "next/link";

const DAYS_HE = ["ראשון", "שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת"];

type AdminSessionData = {
  id: string;
  date: string;
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

  function SlotList({ names, total, label }: { names: string[]; total: number; label: string }) {
    const free = total - names.length;
    return (
      <div>
        <div className="text-xs text-stone-500 mb-1">{label} ({total})</div>
        <div className="space-y-0.5">
          {names.map((name, i) => (
            <div
              key={i}
              className="text-xs bg-amber-50 border border-amber-200 text-amber-800 px-2 py-0.5 rounded font-medium"
            >
              {name}
            </div>
          ))}
          {Array.from({ length: free }).map((_, i) => (
            <div
              key={`free-${i}`}
              className="text-xs border border-dashed border-stone-200 text-stone-300 px-2 py-0.5 rounded"
            >
              פנוי
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <Link
      href={`/sessions/${session.id}`}
      className="block bg-white rounded-xl border border-stone-200 shadow-sm hover:border-amber-300 hover:shadow-md transition overflow-hidden"
    >
      <div className="bg-stone-50 px-3 py-2 border-b border-stone-100">
        <div className="font-semibold text-stone-800 text-sm">{session.groupName}</div>
        <div className="text-xs text-stone-500">{timeStr}</div>
        {session.groupLocation && (
          <div className="text-xs text-stone-400">📍 {session.groupLocation}</div>
        )}
      </div>

      <div className="px-3 py-3 space-y-3">
        <div className="grid grid-cols-2 gap-3">
          <SlotList
            names={session.slots.wheel}
            total={session.slots.wheelTotal}
            label="🎡 אובן"
          />
          <SlotList
            names={session.slots.noWheel}
            total={session.slots.noWheelTotal}
            label="✋ ללא אובן"
          />
        </div>
        {session.slots.unassigned.length > 0 && (
          <div>
            <div className="text-xs text-stone-400 mb-1">לא משוייך</div>
            <div className="flex flex-wrap gap-1">
              {session.slots.unassigned.map((name, i) => (
                <span
                  key={i}
                  className="text-xs bg-stone-100 text-stone-500 px-2 py-0.5 rounded"
                >
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
    const fmt = (d: Date) =>
      d.toLocaleDateString("he-IL", { day: "numeric", month: "short" });
    return `${fmt(weekStartDate)} – ${fmt(end)}`;
  })();

  const byDay: AdminSessionData[][] = Array.from({ length: 7 }, () => []);
  for (const s of sessions) {
    const d = new Date(s.date);
    byDay[d.getDay()].push(s);
  }

  const hasAnySessions = sessions.length > 0;

  return (
    <div className="space-y-4" dir="rtl">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-stone-800">לוח שיעורים</h1>
        <div className="flex items-center gap-3">
          <button
            onClick={prevWeek}
            className="p-2 rounded-lg hover:bg-stone-100 transition text-stone-600"
          >
            ←
          </button>
          <span className="text-sm font-medium text-stone-700 min-w-[140px] text-center">
            {weekLabel}
          </span>
          <button
            onClick={nextWeek}
            className="p-2 rounded-lg hover:bg-stone-100 transition text-stone-600"
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
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {byDay.map((daySessions, idx) => {
            if (daySessions.length === 0) return null;
            const dayDate = new Date(weekStartDate);
            dayDate.setDate(weekStartDate.getDate() + idx);

            return (
              <div key={idx} className="space-y-3">
                <div className="text-xs font-semibold text-stone-500 uppercase tracking-wide px-1">
                  {DAYS_HE[idx]}{" "}
                  <span className="text-stone-400 font-normal">
                    {dayDate.toLocaleDateString("he-IL", { day: "numeric", month: "numeric" })}
                  </span>
                </div>
                {daySessions.map((s) => (
                  <AdminSessionCard key={s.id} session={s} />
                ))}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
