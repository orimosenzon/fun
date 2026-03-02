"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";

type StandbyEntry = {
  id: string;
  position: number;
  name: string;
  phone: string | null;
  userId: string;
  slotType: string | null;
  notifiedAt: string | null;
  expiresAt: string | null;
  createdAt: string;
};

interface Props {
  sessionId: string;
  entries: StandbyEntry[];
}

const SLOT_LABEL: Record<string, string> = {
  WHEEL: "🎡 אובן",
  NO_WHEEL: "✋ ללא אובן",
};

export default function StandbyPanel({ sessionId, entries }: Props) {
  const router = useRouter();
  const [loading, setLoading] = useState<string | null>(null);

  async function handleRemove(entryUserId: string) {
    setLoading(entryUserId);
    await fetch(`/api/sessions/${sessionId}/standby?userId=${entryUserId}`, {
      method: "DELETE",
    });
    setLoading(null);
    router.refresh();
  }

  return (
    <div className="bg-white rounded-xl border border-sky-200 overflow-hidden">
      <div className="px-5 py-4 border-b border-sky-100 bg-sky-50">
        <h2 className="font-semibold text-sky-800">רשימת המתנה ({entries.length})</h2>
      </div>
      <div className="divide-y divide-stone-100">
        {entries.map((entry) => {
          const notified = entry.notifiedAt && entry.expiresAt;
          const expired = notified && new Date(entry.expiresAt!).getTime() < Date.now();
          return (
            <div key={entry.id} className="px-5 py-3 flex items-center justify-between gap-3">
              <div className="flex items-center gap-3">
                <span className="w-6 h-6 rounded-full bg-sky-100 text-sky-600 text-xs font-bold flex items-center justify-center shrink-0">
                  {entry.position}
                </span>
                <div>
                  <Link
                    href={`/students/${entry.userId}`}
                    className="font-medium text-stone-800 hover:text-amber-700"
                  >
                    {entry.name}
                  </Link>
                  {entry.phone && (
                    <div className="text-xs text-stone-400">{entry.phone}</div>
                  )}
                  <div className="text-xs text-stone-400 flex gap-2 mt-0.5">
                    {entry.slotType ? SLOT_LABEL[entry.slotType] : "כל סוג"}
                    {notified && !expired && (
                      <span className="text-green-600 font-medium">
                        הוודע — עד{" "}
                        {new Date(entry.expiresAt!).toLocaleTimeString("he-IL", {
                          hour: "2-digit",
                          minute: "2-digit",
                        })}
                      </span>
                    )}
                    {expired && (
                      <span className="text-red-400">פג תוקף</span>
                    )}
                  </div>
                </div>
              </div>
              <button
                onClick={() => handleRemove(entry.userId)}
                disabled={loading === entry.userId}
                className="text-xs text-stone-400 hover:text-red-500 border border-stone-200 hover:border-red-200 px-2 py-1 rounded transition disabled:opacity-50"
              >
                {loading === entry.userId ? "..." : "הסר"}
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}
