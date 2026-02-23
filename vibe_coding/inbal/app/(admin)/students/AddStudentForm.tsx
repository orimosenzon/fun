"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";

interface Group {
  id: string;
  name: string;
}

export default function AddStudentForm() {
  const [open, setOpen] = useState(false);
  const [groups, setGroups] = useState<Group[]>([]);
  const [form, setForm] = useState({ name: "", email: "", phone: "", groupId: "", slotType: "" });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const router = useRouter();

  useEffect(() => {
    fetch("/api/groups").then((r) => r.json()).then(setGroups);
  }, []);

  function set(field: string, value: string) {
    setForm((f) => ({ ...f, [field]: value }));
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError("");

    const res = await fetch("/api/students", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ...form,
        slotType: form.slotType || null,
      }),
    });

    if (!res.ok) {
      const data = await res.json();
      setError(data.error || "שגיאה");
      setLoading(false);
      return;
    }

    setOpen(false);
    setForm({ name: "", email: "", phone: "", groupId: "", slotType: "" });
    router.refresh();
  }

  if (!open) {
    return (
      <button
        onClick={() => setOpen(true)}
        className="bg-amber-600 hover:bg-amber-700 text-white px-4 py-2 rounded-lg text-sm font-medium transition"
      >
        + תלמיד חדש
      </button>
    );
  }

  return (
    <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50">
      <div className="bg-white rounded-2xl p-6 w-full max-w-sm shadow-xl">
        <h2 className="text-lg font-bold text-stone-800 mb-4">הוספת תלמיד</h2>

        <form onSubmit={handleSubmit} className="space-y-3">
          <div>
            <label className="block text-sm font-medium text-stone-700 mb-1">שם מלא *</label>
            <input
              type="text"
              value={form.name}
              onChange={(e) => set("name", e.target.value)}
              className="w-full border border-stone-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-amber-500"
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-stone-700 mb-1">אימייל *</label>
            <input
              type="email"
              value={form.email}
              onChange={(e) => set("email", e.target.value)}
              className="w-full border border-stone-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-amber-500"
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-stone-700 mb-1">טלפון</label>
            <input
              type="tel"
              value={form.phone}
              onChange={(e) => set("phone", e.target.value)}
              className="w-full border border-stone-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-amber-500"
              placeholder="050-0000000"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-stone-700 mb-1">קבוצה (אופציונלי)</label>
            <select
              value={form.groupId}
              onChange={(e) => set("groupId", e.target.value)}
              className="w-full border border-stone-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-amber-500"
            >
              <option value="">ללא קבוצה</option>
              {groups.map((g) => (
                <option key={g.id} value={g.id}>{g.name}</option>
              ))}
            </select>
          </div>

          {form.groupId && (
            <div>
              <label className="block text-sm font-medium text-stone-700 mb-1">סוג עמדה</label>
              <select
                value={form.slotType}
                onChange={(e) => set("slotType", e.target.value)}
                className="w-full border border-stone-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-amber-500"
              >
                <option value="">לא משוייך</option>
                <option value="WHEEL">🎡 אובן (pottery wheel)</option>
                <option value="NO_WHEEL">✋ ללא אובן</option>
              </select>
            </div>
          )}

          <p className="text-xs text-stone-400">
            הסיסמה הראשונית תהיה: <strong>student123</strong>
          </p>

          {error && <p className="text-red-600 text-sm">{error}</p>}

          <div className="flex gap-2 pt-1">
            <button
              type="submit"
              disabled={loading}
              className="flex-1 bg-amber-600 hover:bg-amber-700 text-white py-2 rounded-lg text-sm font-medium transition disabled:opacity-50"
            >
              {loading ? "מוסיף..." : "הוסף"}
            </button>
            <button
              type="button"
              onClick={() => setOpen(false)}
              className="flex-1 border border-stone-300 text-stone-600 py-2 rounded-lg text-sm font-medium hover:bg-stone-50 transition"
            >
              ביטול
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
