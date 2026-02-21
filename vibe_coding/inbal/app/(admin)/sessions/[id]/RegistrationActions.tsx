"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

export default function RegistrationActions({
  registrationId,
  currentStatus,
  userId,
  sessionId,
}: {
  registrationId: string;
  currentStatus: string;
  userId: string;
  sessionId: string;
}) {
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  async function updateStatus(status: string) {
    setLoading(true);
    await fetch(`/api/registrations/${registrationId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status }),
    });
    setLoading(false);
    router.refresh();
  }

  const buttons = [
    { status: "REGISTERED", label: "יבוא", cls: "text-green-700 hover:bg-green-50" },
    { status: "ABSENT", label: "נעדר", cls: "text-stone-500 hover:bg-stone-50" },
    { status: "CANCELLED", label: "ביטל", cls: "text-red-600 hover:bg-red-50" },
  ].filter((b) => b.status !== currentStatus);

  return (
    <div className="flex gap-1">
      {buttons.map((b) => (
        <button
          key={b.status}
          onClick={() => updateStatus(b.status)}
          disabled={loading}
          className={`text-xs px-2 py-1 rounded border border-stone-200 transition ${b.cls} disabled:opacity-50`}
        >
          {b.label}
        </button>
      ))}
    </div>
  );
}
