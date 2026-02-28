"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import type { SessionData, MyRegistration } from "./WeekCalendar";

interface Props {
  session: SessionData;
  myUpcomingRegistrations: MyRegistration[];
}

export default function StudentSessionCard({ session, myUpcomingRegistrations }: Props) {
  const router = useRouter();
  const [showCancelConfirm, setShowCancelConfirm] = useState(false);
  const [transferSlotType, setTransferSlotType] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const date = new Date(session.date);
  const timeStr = date.toLocaleTimeString("he-IL", { hour: "2-digit", minute: "2-digit" });

  const myReg = session.myRegistration;
  const isRegistered = myReg?.status === "REGISTERED";
  const isCancelled = myReg?.status === "CANCELLED";
  const isTransferred = myReg?.status === "TRANSFERRED";

  const now = Date.now();
  const hoursUntilSession = (new Date(session.date).getTime() - now) / (1000 * 60 * 60);
  const canCancel = hoursUntilSession >= 48;

  async function handleCancel() {
    if (!myReg) return;
    setLoading(true);
    const res = await fetch(`/api/registrations/${myReg.id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status: "CANCELLED" }),
    });
    setLoading(false);
    if (!res.ok) {
      const data = await res.json();
      alert(data.error || "שגיאה בביטול");
      return;
    }
    setShowCancelConfirm(false);
    router.refresh();
  }

  async function handleTransfer(fromRegistrationId: string) {
    setLoading(true);
    const res = await fetch("/api/registrations/transfer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ fromRegistrationId, toSessionId: session.id }),
    });
    setLoading(false);
    if (res.ok) {
      setTransferSlotType(null);
      router.refresh();
    } else {
      const data = await res.json();
      alert(data.error || "שגיאה בהעברה");
    }
  }

  // Slots for a given type
  function SlotDots({
    total,
    taken,
    type,
  }: {
    total: number;
    taken: number;
    type: "WHEEL" | "NO_WHEEL";
  }) {
    const mySlot = isRegistered && myReg?.slotType === type;
    // Adjust taken: if mySlot is included in taken, "me" dot replaces first taken dot
    const othersTaken = mySlot ? taken - 1 : taken;
    const free = total - taken;

    const canTransferHere =
      !isRegistered &&
      free > 0 &&
      myUpcomingRegistrations.some(
        (r) =>
          r.sessionId !== session.id &&
          (r.slotType === null || r.slotType === type) &&
          (new Date(r.sessionDate).getTime() - now) / (1000 * 60 * 60) >= 48
      );

    return (
      <div className="flex gap-0.5 flex-wrap">
        {mySlot && (
          <span
            title="המקום שלך"
            className="inline-block w-4 h-4 rounded-full bg-amber-500 ring-2 ring-amber-300"
          />
        )}
        {Array.from({ length: Math.max(0, othersTaken) }).map((_, i) => (
          <span key={`t${i}`} className="inline-block w-4 h-4 rounded-full bg-stone-300" />
        ))}
        {Array.from({ length: free }).map((_, i) => (
          <button
            key={`f${i}`}
            onClick={() => canTransferHere && setTransferSlotType(type)}
            title={canTransferHere ? "לחץ להעברה לכאן" : "פנוי"}
            className={`inline-block w-4 h-4 rounded-full border-2 transition ${
              canTransferHere
                ? "border-green-400 bg-green-50 hover:bg-green-200 cursor-pointer"
                : "border-stone-200 bg-white cursor-default"
            }`}
          />
        ))}
      </div>
    );
  }

  const cardBorder = isRegistered
    ? "border-amber-400 bg-amber-50"
    : isCancelled || isTransferred
    ? "border-stone-200 bg-stone-50 opacity-60"
    : "border-stone-200 bg-white hover:border-stone-300";

  return (
    <>
      <div className={`rounded-lg border p-1.5 text-xs space-y-1 transition ${cardBorder}`}>
        {/* Time + group */}
        <div className="font-semibold text-stone-800 leading-tight truncate">{session.groupName}</div>
        <div className="text-stone-400">{timeStr}</div>

        {/* Slot dots */}
        <div className="space-y-0.5 pt-0.5">
          <div className="flex items-center gap-1">
            <span className="text-stone-400" title="אובן">🎡</span>
            <SlotDots total={session.slots.wheelTotal} taken={session.slots.wheelTaken} type="WHEEL" />
          </div>
          <div className="flex items-center gap-1">
            <span className="text-stone-400" title="ללא אובן">✋</span>
            <SlotDots total={session.slots.noWheelTotal} taken={session.slots.noWheelTaken} type="NO_WHEEL" />
          </div>
        </div>

        {/* Status badge */}
        {(isCancelled || isTransferred) && (
          <div className="text-[10px] text-stone-400">
            {isCancelled ? "ביטלת" : "הועברת"}
          </div>
        )}

        {/* Cancel button */}
        {isRegistered && (
          <div className="pt-0.5">
            {canCancel ? (
              showCancelConfirm ? (
                <div className="flex gap-1">
                  <button
                    onClick={handleCancel}
                    disabled={loading}
                    className="flex-1 bg-red-600 text-white text-[10px] py-1 rounded disabled:opacity-50"
                  >
                    {loading ? "..." : "אשרי"}
                  </button>
                  <button
                    onClick={() => setShowCancelConfirm(false)}
                    className="flex-1 border border-stone-200 text-stone-500 text-[10px] py-1 rounded"
                  >
                    חזרה
                  </button>
                </div>
              ) : (
                <button
                  onClick={() => setShowCancelConfirm(true)}
                  className="w-full text-[10px] text-red-500 hover:text-red-700 border border-red-200 py-1 rounded hover:bg-red-50 transition"
                >
                  ביטול
                </button>
              )
            ) : (
              <div className="text-[10px] text-stone-400 text-center py-0.5">
                לא ניתן לבטל (פחות מ-48 שע׳)
              </div>
            )}
          </div>
        )}
      </div>

      {/* Transfer modal */}
      {transferSlotType && (
        <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50" dir="rtl">
          <div className="bg-white rounded-2xl p-5 w-full max-w-sm shadow-xl mx-4">
            <h3 className="font-bold text-stone-800 mb-1">העברה לשיעור</h3>
            <p className="text-sm text-stone-500 mb-4">
              {session.groupName} •{" "}
              {new Date(session.date).toLocaleDateString("he-IL", {
                weekday: "long", day: "numeric", month: "long",
              })}{" "}
              • {timeStr}
            </p>
            <p className="text-xs text-stone-500 mb-2">בחרי איזה שיעור לבטל:</p>
            <div className="space-y-2 mb-4 max-h-48 overflow-y-auto">
              {myUpcomingRegistrations
                .filter(
                  (r) =>
                    r.sessionId !== session.id &&
                    (r.slotType === null || r.slotType === transferSlotType) &&
                    (new Date(r.sessionDate).getTime() - now) / (1000 * 60 * 60) >= 48
                )
                .map((src) => (
                  <button
                    key={src.id}
                    onClick={() => handleTransfer(src.id)}
                    disabled={loading}
                    className="w-full text-right border border-stone-200 rounded-lg px-3 py-2 text-sm hover:bg-amber-50 hover:border-amber-300 transition disabled:opacity-50"
                  >
                    <div className="font-medium">{src.groupName}</div>
                    <div className="text-xs text-stone-400">
                      {new Date(src.sessionDate).toLocaleDateString("he-IL", {
                        weekday: "short", day: "numeric", month: "short",
                      })}
                    </div>
                  </button>
                ))}
            </div>
            <button
              onClick={() => setTransferSlotType(null)}
              className="w-full border border-stone-200 text-stone-600 py-2 rounded-lg text-sm hover:bg-stone-50"
            >
              ביטול
            </button>
          </div>
        </div>
      )}
    </>
  );
}
