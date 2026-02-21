"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

export default function MySessionActions({
  registrationId,
  sessionId,
  userId,
}: {
  registrationId: string;
  sessionId: string;
  userId: string;
}) {
  const [loading, setLoading] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);
  const router = useRouter();

  async function handleCancel() {
    setLoading(true);
    await fetch(`/api/registrations/${registrationId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status: "CANCELLED" }),
    });
    setLoading(false);
    setShowConfirm(false);
    router.refresh();
  }

  if (showConfirm) {
    return (
      <div className="flex gap-2 text-xs">
        <button
          onClick={handleCancel}
          disabled={loading}
          className="bg-red-600 text-white px-3 py-1 rounded-lg disabled:opacity-50"
        >
          {loading ? "..." : "אשרי ביטול"}
        </button>
        <button
          onClick={() => setShowConfirm(false)}
          className="border border-stone-300 text-stone-600 px-3 py-1 rounded-lg"
        >
          חזרה
        </button>
      </div>
    );
  }

  return (
    <button
      onClick={() => setShowConfirm(true)}
      className="text-xs text-red-600 hover:text-red-800 border border-red-200 px-2 py-1 rounded-lg"
    >
      ביטול שיעור
    </button>
  );
}
