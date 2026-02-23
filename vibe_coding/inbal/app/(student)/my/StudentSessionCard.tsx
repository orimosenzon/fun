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

  async function handleCancel() {
    if (!myReg) return;
    setLoading(true);
    await fetch(`/api/registrations/${myReg.id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status: "CANCELLED" }),
    });
    setLoading(false);
    setShowCancelConfirm(false);
    router.refresh();
  }

  // Registrations I can transfer FROM (same slot type, different session, in the future)
  const transferSources = myUpcomingRegistrations.filter(
    (r) =>
      r.sessionId !== session.id &&
      (r.slotType === null || r.slotType === transferSlotType)
  );

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

  function SlotDots({
    total,
    taken,
    type,
    mySlotType,
    isMe,
  }: {
    total: number;
    taken: number;
    type: "WHEEL" | "NO_WHEEL";
    mySlotType: string | null | undefined;
    isMe: boolean;
  }) {
    const dots = [];
    for (let i = 0; i < total; i++) {
      const isMeSlot = isMe && mySlotType === type && i === 0;
      const isFilled = isMeSlot || i < taken;
      const isFree = !isFilled;

      if (isMeSlot) {
        dots.push(
          <span
            key={i}
            title="המקום שלך"
            className="inline-flex items-center justify-center w-6 h-6 rounded-full bg-amber-500 text-white text-[10px] font-bold"
          >
            אני
          </span>
        );
      } else if (isFilled) {
        dots.push(
          <span
            key={i}
            className="inline-block w-6 h-6 rounded-full bg-stone-400"
          />
        );
      } else {
        // Free slot — clickable for transfer if student has a relevant registration
        const canTransfer =
          !isRegistered &&
          myUpcomingRegistrations.some(
            (r) => r.sessionId !== session.id && (r.slotType === null || r.slotType === type)
          );

        dots.push(
          <button
            key={i}
            title={canTransfer ? "לחץ להעברה לכאן" : "פנוי"}
            onClick={() => canTransfer && setTransferSlotType(type)}
            className={`inline-block w-6 h-6 rounded-full border-2 transition ${
              canTransfer
                ? "border-green-400 bg-green-50 hover:bg-green-100 cursor-pointer"
                : "border-stone-200 bg-white cursor-default"
            }`}
          />
        );
      }
    }
    return <div className="flex gap-1 flex-wrap">{dots}</div>;
  }

  return (
    <div
      className={`bg-white rounded-xl border shadow-sm overflow-hidden ${
        isRegistered ? "border-amber-300" : "border-stone-200"
      }`}
    >
      {/* Header */}
      <div
        className={`px-3 py-2 ${
          isRegistered ? "bg-amber-50" : "bg-stone-50"
        }`}
      >
        <div className="font-semibold text-stone-800 text-sm">{session.groupName}</div>
        <div className="text-xs text-stone-500">{timeStr}</div>
        {session.groupLocation && (
          <div className="text-xs text-stone-400">📍 {session.groupLocation}</div>
        )}
      </div>

      {/* Slot grid */}
      <div className="px-3 py-2 space-y-2">
        <div>
          <div className="text-xs text-stone-500 mb-1">🎡 אובן ({session.slots.wheelTotal})</div>
          <SlotDots
            total={session.slots.wheelTotal}
            taken={session.slots.wheelTaken}
            type="WHEEL"
            mySlotType={myReg?.slotType}
            isMe={isRegistered}
          />
        </div>
        <div>
          <div className="text-xs text-stone-500 mb-1">✋ ללא אובן ({session.slots.noWheelTotal})</div>
          <SlotDots
            total={session.slots.noWheelTotal}
            taken={session.slots.noWheelTaken}
            type="NO_WHEEL"
            mySlotType={myReg?.slotType}
            isMe={isRegistered}
          />
        </div>
      </div>

      {/* Actions for registered students */}
      {isRegistered && (
        <div className="px-3 pb-3">
          {showCancelConfirm ? (
            <div className="flex gap-2">
              <button
                onClick={handleCancel}
                disabled={loading}
                className="flex-1 bg-red-600 text-white text-xs py-1.5 rounded-lg disabled:opacity-50"
              >
                {loading ? "..." : "אשר ביטול"}
              </button>
              <button
                onClick={() => setShowCancelConfirm(false)}
                className="flex-1 border border-stone-200 text-stone-600 text-xs py-1.5 rounded-lg"
              >
                חזרה
              </button>
            </div>
          ) : (
            <button
              onClick={() => setShowCancelConfirm(true)}
              className="w-full text-xs text-red-600 hover:text-red-800 border border-red-200 py-1.5 rounded-lg hover:bg-red-50 transition"
            >
              ביטול שיעור
            </button>
          )}
        </div>
      )}

      {/* Transfer modal overlay */}
      {transferSlotType && (
        <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50">
          <div className="bg-white rounded-2xl p-5 w-full max-w-sm shadow-xl mx-4">
            <h3 className="font-bold text-stone-800 mb-1">העברה לשיעור</h3>
            <p className="text-sm text-stone-500 mb-4">
              {session.groupName} •{" "}
              {new Date(session.date).toLocaleDateString("he-IL", {
                weekday: "long",
                day: "numeric",
                month: "long",
              })}{" "}
              • {timeStr}
            </p>
            <p className="text-xs text-stone-500 mb-2">בחר/י איזה שיעור לבטל:</p>
            <div className="space-y-2 mb-4">
              {transferSources.length === 0 ? (
                <p className="text-sm text-stone-400">אין שיעורים מתאימים לביטול</p>
              ) : (
                transferSources.map((src) => (
                  <button
                    key={src.id}
                    onClick={() => handleTransfer(src.id)}
                    disabled={loading}
                    className="w-full text-right border border-stone-200 rounded-lg px-3 py-2 text-sm hover:bg-amber-50 hover:border-amber-300 transition disabled:opacity-50"
                  >
                    <div className="font-medium">{src.groupName}</div>
                    <div className="text-xs text-stone-400">
                      {new Date(src.sessionDate).toLocaleDateString("he-IL", {
                        weekday: "short",
                        day: "numeric",
                        month: "short",
                      })}
                    </div>
                  </button>
                ))
              )}
            </div>
            <button
              onClick={() => setTransferSlotType(null)}
              className="w-full border border-stone-200 text-stone-600 py-2 rounded-lg text-sm"
            >
              ביטול
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
