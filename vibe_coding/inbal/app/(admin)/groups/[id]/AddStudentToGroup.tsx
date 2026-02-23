"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";

interface User {
  id: string;
  name: string;
  email: string;
}

export default function AddStudentToGroup({ groupId }: { groupId: string }) {
  const [open, setOpen] = useState(false);
  const [students, setStudents] = useState<User[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [slotType, setSlotType] = useState("");
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  useEffect(() => {
    if (open) {
      fetch("/api/students")
        .then((r) => r.json())
        .then(setStudents);
    }
  }, [open]);

  async function handleAdd() {
    if (!selectedId) return;
    setLoading(true);

    const res = await fetch("/api/enrollments", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ userId: selectedId, groupId, slotType: slotType || null }),
    });

    if (res.ok) {
      setOpen(false);
      setSelectedId("");
      setSlotType("");
      router.refresh();
    } else {
      const data = await res.json();
      alert(data.error || "שגיאה");
    }
    setLoading(false);
  }

  if (!open) {
    return (
      <button
        onClick={() => setOpen(true)}
        className="text-sm text-amber-600 hover:text-amber-800 font-medium"
      >
        + הוסף תלמיד לקבוצה
      </button>
    );
  }

  return (
    <div className="space-y-2">
      <select
        value={selectedId}
        onChange={(e) => setSelectedId(e.target.value)}
        className="w-full border border-stone-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-amber-500"
      >
        <option value="">בחרי תלמיד...</option>
        {students.map((s) => (
          <option key={s.id} value={s.id}>
            {s.name} ({s.email})
          </option>
        ))}
      </select>

      <select
        value={slotType}
        onChange={(e) => setSlotType(e.target.value)}
        className="w-full border border-stone-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-amber-500"
      >
        <option value="">סוג עמדה (אופציונלי)</option>
        <option value="WHEEL">🎡 אובן (pottery wheel)</option>
        <option value="NO_WHEEL">✋ ללא אובן</option>
      </select>

      <div className="flex gap-2">
        <button
          onClick={handleAdd}
          disabled={!selectedId || loading}
          className="bg-amber-600 text-white text-sm px-4 py-1.5 rounded-lg disabled:opacity-50 hover:bg-amber-700 transition"
        >
          {loading ? "מוסיף..." : "הוסף"}
        </button>
        <button
          onClick={() => { setOpen(false); setSelectedId(""); setSlotType(""); }}
          className="text-stone-500 text-sm px-3 py-1.5 rounded-lg hover:bg-stone-100 transition"
        >
          ביטול
        </button>
      </div>
    </div>
  );
}
