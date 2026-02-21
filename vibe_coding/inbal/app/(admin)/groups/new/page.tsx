"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

const DAYS = ["ראשון", "שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת"];

export default function NewGroupPage() {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [form, setForm] = useState({
    name: "",
    description: "",
    dayOfWeek: 0,
    time: "10:00",
    duration: 90,
    maxStudents: 10,
    location: "סטודיו ראשי",
  });

  function set(field: string, value: string | number) {
    setForm((f) => ({ ...f, [field]: value }));
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError("");

    const res = await fetch("/api/groups", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ...form,
        dayOfWeek: Number(form.dayOfWeek),
        duration: Number(form.duration),
        maxStudents: Number(form.maxStudents),
      }),
    });

    if (!res.ok) {
      setError("שגיאה ביצירת הקבוצה");
      setLoading(false);
      return;
    }

    const group = await res.json();

    // Auto-generate 8 sessions
    await fetch(`/api/groups/${group.id}/sessions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ weeks: 8 }),
    });

    router.push(`/groups/${group.id}`);
  }

  return (
    <div className="max-w-xl">
      <h1 className="text-2xl font-bold text-stone-800 mb-6">קבוצה חדשה</h1>

      <form onSubmit={handleSubmit} className="bg-white rounded-xl border border-stone-200 p-6 space-y-4">
        <div>
          <label className="block text-sm font-medium text-stone-700 mb-1">שם הקבוצה *</label>
          <input
            type="text"
            value={form.name}
            onChange={(e) => set("name", e.target.value)}
            className="w-full border border-stone-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-amber-500"
            placeholder="לדוגמה: קבוצת בוקר ראשון"
            required
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-stone-700 mb-1">תיאור</label>
          <input
            type="text"
            value={form.description}
            onChange={(e) => set("description", e.target.value)}
            className="w-full border border-stone-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-amber-500"
            placeholder="מתחילים, מתקדמים..."
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-stone-700 mb-1">יום בשבוע *</label>
            <select
              value={form.dayOfWeek}
              onChange={(e) => set("dayOfWeek", Number(e.target.value))}
              className="w-full border border-stone-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-amber-500"
            >
              {DAYS.map((d, i) => (
                <option key={i} value={i}>יום {d}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-stone-700 mb-1">שעת התחלה *</label>
            <input
              type="time"
              value={form.time}
              onChange={(e) => set("time", e.target.value)}
              className="w-full border border-stone-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-amber-500"
              required
            />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-stone-700 mb-1">משך (דקות)</label>
            <input
              type="number"
              value={form.duration}
              onChange={(e) => set("duration", e.target.value)}
              className="w-full border border-stone-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-amber-500"
              min={30}
              max={360}
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-stone-700 mb-1">מקסימום תלמידים</label>
            <input
              type="number"
              value={form.maxStudents}
              onChange={(e) => set("maxStudents", e.target.value)}
              className="w-full border border-stone-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-amber-500"
              min={1}
              max={50}
            />
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-stone-700 mb-1">מיקום</label>
          <input
            type="text"
            value={form.location}
            onChange={(e) => set("location", e.target.value)}
            className="w-full border border-stone-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-amber-500"
            placeholder="סטודיו ראשי"
          />
        </div>

        {error && <p className="text-red-600 text-sm">{error}</p>}

        <div className="flex gap-3 pt-2">
          <button
            type="submit"
            disabled={loading}
            className="bg-amber-600 hover:bg-amber-700 text-white px-6 py-2.5 rounded-lg text-sm font-medium transition disabled:opacity-50"
          >
            {loading ? "יוצר..." : "צרי קבוצה (+ 8 שיעורים קדימה)"}
          </button>
          <button
            type="button"
            onClick={() => router.back()}
            className="border border-stone-300 text-stone-600 px-5 py-2.5 rounded-lg text-sm font-medium hover:bg-stone-50 transition"
          >
            ביטול
          </button>
        </div>
      </form>
    </div>
  );
}
