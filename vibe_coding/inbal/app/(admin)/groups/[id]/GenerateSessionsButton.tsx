"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

export default function GenerateSessionsButton({ groupId }: { groupId: string }) {
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  async function handleGenerate() {
    setLoading(true);
    const res = await fetch(`/api/groups/${groupId}/sessions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ weeks: 8 }),
    });
    const data = await res.json();
    alert(`נוצרו ${data.created} שיעורים חדשים`);
    setLoading(false);
    router.refresh();
  }

  return (
    <button
      onClick={handleGenerate}
      disabled={loading}
      className="border border-amber-300 text-amber-700 hover:bg-amber-50 px-4 py-2 rounded-lg text-sm font-medium transition disabled:opacity-50"
    >
      {loading ? "יוצר..." : "📅 הוסף שיעורים (8 שבועות)"}
    </button>
  );
}
