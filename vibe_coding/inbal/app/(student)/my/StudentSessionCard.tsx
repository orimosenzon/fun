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
  const [showRegisterModal, setShowRegisterModal] = useState(false);
  const [showStandbyModal, setShowStandbyModal] = useState(false);
  const [loading, setLoading] = useState(false);

  const date = new Date(session.date);
  const timeStr = date.toLocaleTimeString("he-IL", { hour: "2-digit", minute: "2-digit" });

  const myReg = session.myRegistration;
  const isRegistered = myReg?.status === "REGISTERED";
  const isCancelled = myReg?.status === "CANCELLED";
  const isTransferred = myReg?.status === "TRANSFERRED";
  const myStandby = session.myStandby;

  const now = Date.now();
  const hoursUntilSession = (new Date(session.date).getTime() - now) / (1000 * 60 * 60);
  const canCancel = hoursUntilSession >= 48;

  const wheelFree = session.slots.wheelTotal - session.slots.wheelTaken;
  const noWheelFree = session.slots.noWheelTotal - session.slots.noWheelTaken;
  const hasAnyFree = wheelFree > 0 || noWheelFree > 0;
  const isFull = !hasAnyFree;

  // תלמיד קיבל הודעת סטנד ביי ועדיין בתוך חלון הזמן
  const standbyNotified =
    myStandby?.notifiedAt &&
    myStandby.expiresAt &&
    new Date(myStandby.expiresAt).getTime() > now;

  // ------ actions ------

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

  async function handleDirectRegister(slotType: "WHEEL" | "NO_WHEEL") {
    setLoading(true);
    const res = await fetch(`/api/sessions/${session.id}/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ slotType }),
    });
    setLoading(false);
    if (res.ok) {
      setShowRegisterModal(false);
      router.refresh();
    } else {
      const data = await res.json();
      alert(data.error || "שגיאה ברישום");
    }
  }

  async function handleJoinStandby(slotType?: "WHEEL" | "NO_WHEEL") {
    setLoading(true);
    const res = await fetch(`/api/sessions/${session.id}/standby`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ slotType: slotType ?? null }),
    });
    setLoading(false);
    if (res.ok) {
      setShowStandbyModal(false);
      router.refresh();
    } else {
      const data = await res.json();
      alert(data.error || "שגיאה בהצטרפות לרשימת המתנה");
    }
  }

  async function handleLeaveStandby() {
    setLoading(true);
    const res = await fetch(`/api/sessions/${session.id}/standby`, { method: "DELETE" });
    setLoading(false);
    if (res.ok) router.refresh();
  }

  // ------ slot row ------

  function SlotRow({ type }: { type: "WHEEL" | "NO_WHEEL" }) {
    const label = type === "WHEEL" ? "🎡 אובניים" : "✋ ידני";
    const free = type === "WHEEL" ? wheelFree : noWheelFree;
    const isMySlot = isRegistered && myReg?.slotType === type;

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
      <div className="flex items-center justify-between gap-2">
        <span className="text-stone-500 text-[11px]">{label}</span>
        {isMySlot ? (
          <span className="text-[10px] bg-amber-100 text-amber-700 px-2 py-0.5 rounded-full font-medium">
            המקום שלך ✓
          </span>
        ) : isRegistered ? (
          <span className="text-[10px] text-stone-300">תפוס</span>
        ) : canTransferHere ? (
          <button
            onClick={() => setTransferSlotType(type)}
            disabled={loading}
            className="text-[10px] bg-amber-500 text-white px-2.5 py-1 rounded-full font-medium hover:bg-amber-600 transition disabled:opacity-50"
          >
            הרשם
          </button>
        ) : free > 0 ? (
          <button
            onClick={() => handleDirectRegister(type)}
            disabled={loading}
            className="text-[10px] bg-amber-500 text-white px-2.5 py-1 rounded-full font-medium hover:bg-amber-600 transition disabled:opacity-50"
          >
            {loading ? "..." : "הרשם"}
          </button>
        ) : (
          <span className="text-[10px] text-stone-400 bg-stone-100 px-2 py-0.5 rounded-full">
            תפוס
          </span>
        )}
      </div>
    );
  }

  const cardBorder = isRegistered
    ? "border-amber-400 bg-amber-50"
    : isCancelled || isTransferred
    ? "border-stone-200 bg-stone-50 opacity-60"
    : myStandby
    ? "border-sky-300 bg-sky-50"
    : "border-stone-200 bg-white hover:border-stone-300";

  return (
    <>
      <div className={`rounded-lg border p-1.5 text-xs space-y-1 transition ${cardBorder}`}>
        {/* Time + group */}
        <div className="font-semibold text-stone-800 leading-tight truncate">{session.groupName}</div>
        <div className="text-stone-400">{timeStr}</div>

        {/* Standby notification banner */}
        {standbyNotified && (
          <div className="bg-green-100 border border-green-300 rounded px-1.5 py-1 text-[10px] text-green-800 font-medium">
            🎉 יש מקום! הירשמי עד{" "}
            {new Date(myStandby!.expiresAt!).toLocaleTimeString("he-IL", {
              hour: "2-digit",
              minute: "2-digit",
            })}
          </div>
        )}

        {/* Slot availability rows */}
        <div className="space-y-1 pt-0.5">
          <SlotRow type="WHEEL" />
          <SlotRow type="NO_WHEEL" />
        </div>

        {/* Status / actions */}
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

        {/* Standby status + leave button */}
        {!isRegistered && myStandby && !standbyNotified && (
          <div className="pt-0.5 space-y-0.5">
            <div className="text-[10px] text-sky-600 font-medium">
              רשימת המתנה — מיקום {myStandby.position}
            </div>
            <button
              onClick={handleLeaveStandby}
              disabled={loading}
              className="w-full text-[10px] text-stone-400 border border-stone-200 py-0.5 rounded hover:bg-stone-50 transition"
            >
              {loading ? "..." : "הסר עצמי"}
            </button>
          </div>
        )}

        {/* Standby notification — register now button */}
        {!isRegistered && standbyNotified && (
          <button
            onClick={() => setShowRegisterModal(true)}
            disabled={loading}
            className="w-full text-[10px] bg-green-500 text-white py-1 rounded hover:bg-green-600 transition disabled:opacity-50"
          >
            הירשמי עכשיו
          </button>
        )}

        {/* Join standby button (full session, not registered, not on standby) */}
        {!isRegistered && !myStandby && isFull && (
          <button
            onClick={() => setShowStandbyModal(true)}
            className="w-full text-[10px] text-sky-600 border border-sky-200 py-0.5 rounded hover:bg-sky-50 transition"
          >
            הצטרפי לרשימת המתנה
          </button>
        )}

      </div>

      {/* Register modal */}
      {showRegisterModal && (
        <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50" dir="rtl">
          <div className="bg-white rounded-2xl p-5 w-full max-w-sm shadow-xl mx-4">
            <h3 className="font-bold text-stone-800 mb-1">רישום לשיעור</h3>
            <p className="text-sm text-stone-500 mb-4">
              {session.groupName} •{" "}
              {date.toLocaleDateString("he-IL", { weekday: "long", day: "numeric", month: "long" })}{" "}
              • {timeStr}
            </p>
            <p className="text-xs text-stone-500 mb-3">בחרי סוג סלוט:</p>
            <div className="space-y-2 mb-4">
              {wheelFree > 0 && (
                <button
                  onClick={() => handleDirectRegister("WHEEL")}
                  disabled={loading}
                  className="w-full border border-amber-200 rounded-lg px-3 py-2.5 text-sm text-right hover:bg-amber-50 hover:border-amber-400 transition disabled:opacity-50"
                >
                  <div className="font-medium">🎡 אובן</div>
                  <div className="text-xs text-stone-400">{wheelFree} מקומות פנויים</div>
                </button>
              )}
              {noWheelFree > 0 && (
                <button
                  onClick={() => handleDirectRegister("NO_WHEEL")}
                  disabled={loading}
                  className="w-full border border-sky-200 rounded-lg px-3 py-2.5 text-sm text-right hover:bg-sky-50 hover:border-sky-400 transition disabled:opacity-50"
                >
                  <div className="font-medium">✋ ללא אובן</div>
                  <div className="text-xs text-stone-400">{noWheelFree} מקומות פנויים</div>
                </button>
              )}
              {/* Transfer option if available */}
              {myUpcomingRegistrations.some(
                (r) =>
                  r.sessionId !== session.id &&
                  (new Date(r.sessionDate).getTime() - now) / (1000 * 60 * 60) >= 48
              ) && (
                <button
                  onClick={() => {
                    setShowRegisterModal(false);
                    setTransferSlotType("ANY");
                  }}
                  className="w-full border border-stone-200 rounded-lg px-3 py-2 text-sm text-right hover:bg-stone-50 transition text-stone-600"
                >
                  ↔ העברה משיעור אחר
                </button>
              )}
            </div>
            <button
              onClick={() => setShowRegisterModal(false)}
              className="w-full border border-stone-200 text-stone-600 py-2 rounded-lg text-sm hover:bg-stone-50"
            >
              ביטול
            </button>
          </div>
        </div>
      )}

      {/* Transfer modal */}
      {transferSlotType && (
        <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50" dir="rtl">
          <div className="bg-white rounded-2xl p-5 w-full max-w-sm shadow-xl mx-4">
            <h3 className="font-bold text-stone-800 mb-1">העברה לשיעור</h3>
            <p className="text-sm text-stone-500 mb-4">
              {session.groupName} •{" "}
              {date.toLocaleDateString("he-IL", { weekday: "long", day: "numeric", month: "long" })}{" "}
              • {timeStr}
            </p>
            <p className="text-xs text-stone-500 mb-2">בחרי איזה שיעור לבטל:</p>
            <div className="space-y-2 mb-4 max-h-48 overflow-y-auto">
              {myUpcomingRegistrations
                .filter(
                  (r) =>
                    r.sessionId !== session.id &&
                    (transferSlotType === "ANY" ||
                      r.slotType === null ||
                      r.slotType === transferSlotType) &&
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
                        weekday: "short",
                        day: "numeric",
                        month: "short",
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

      {/* Standby modal */}
      {showStandbyModal && (
        <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50" dir="rtl">
          <div className="bg-white rounded-2xl p-5 w-full max-w-sm shadow-xl mx-4">
            <h3 className="font-bold text-stone-800 mb-1">הצטרפות לרשימת המתנה</h3>
            <p className="text-sm text-stone-500 mb-4">
              {session.groupName} •{" "}
              {date.toLocaleDateString("he-IL", { weekday: "long", day: "numeric", month: "long" })}{" "}
              • {timeStr}
            </p>
            <p className="text-xs text-stone-500 mb-3">
              השיעור מלא. אם יתפנה מקום — תקבלי SMS ויהיה לך שעה להירשם.
            </p>
            <p className="text-xs text-stone-500 mb-3">בחרי העדפת סלוט:</p>
            <div className="space-y-2 mb-4">
              <button
                onClick={() => handleJoinStandby("WHEEL")}
                disabled={loading}
                className="w-full border border-amber-200 rounded-lg px-3 py-2.5 text-sm text-right hover:bg-amber-50 transition disabled:opacity-50"
              >
                🎡 אובן
              </button>
              <button
                onClick={() => handleJoinStandby("NO_WHEEL")}
                disabled={loading}
                className="w-full border border-sky-200 rounded-lg px-3 py-2.5 text-sm text-right hover:bg-sky-50 transition disabled:opacity-50"
              >
                ✋ ללא אובן
              </button>
              <button
                onClick={() => handleJoinStandby(undefined)}
                disabled={loading}
                className="w-full border border-stone-200 rounded-lg px-3 py-2 text-sm text-right hover:bg-stone-50 transition disabled:opacity-50 text-stone-500"
              >
                כל סוג
              </button>
            </div>
            <button
              onClick={() => setShowStandbyModal(false)}
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
