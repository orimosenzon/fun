"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

type Enrollment = {
  id: string;
  groupId: string;
  preferredSlotType: string | null;
  group: { name: string };
};

type Props = {
  studentId: string;
  name: string;
  phone: string | null;
  enrollments: Enrollment[];
};

const SLOT_LABELS: Record<string, string> = {
  WHEEL: "🎡 גלגל אובן",
  NO_WHEEL: "✋ ללא אובן",
};

export default function EditStudentForm({ studentId, name, phone, enrollments }: Props) {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [formName, setFormName] = useState(name);
  const [formPhone, setFormPhone] = useState(phone ?? "");
  const [slotTypes, setSlotTypes] = useState<Record<string, string>>(
    Object.fromEntries(enrollments.map((e) => [e.groupId, e.preferredSlotType ?? ""]))
  );

  async function handleSave() {
    setLoading(true);
    setError("");

    // Update name + phone
    const res = await fetch(`/api/students/${studentId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: formName, phone: formPhone }),
    });

    if (!res.ok) {
      setError("שגיאה בשמירה");
      setLoading(false);
      return;
    }

    // Update slotType per enrollment
    for (const enrollment of enrollments) {
      const newSlotType = slotTypes[enrollment.groupId];
      if (newSlotType !== (enrollment.preferredSlotType ?? "")) {
        await fetch(`/api/students/${studentId}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            name: formName,
            phone: formPhone,
            groupId: enrollment.groupId,
            preferredSlotType: newSlotType || null,
          }),
        });
      }
    }

    setLoading(false);
    setOpen(false);
    router.refresh();
  }

  if (!open) {
    return (
      <button
        onClick={() => setOpen(true)}
        className="text-xs text-amber-600 hover:text-amber-800 border border-amber-200 px-3 py-1 rounded-lg hover:bg-amber-50 transition"
      >
        עריכה
      </button>
    );
  }

  return (
    <div className="bg-white rounded-xl border border-amber-200 p-5 space-y-4">
      <h3 className="font-semibold text-stone-800">עריכת פרטי תלמיד</h3>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-xs font-medium text-stone-600 mb-1">שם</label>
          <input
            type="text"
            value={formName}
            onChange={(e) => setFormName(e.target.value)}
            className="w-full border border-stone-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-amber-500"
          />
        </div>
        <div>
          <label className="block text-xs font-medium text-stone-600 mb-1">טלפון</label>
          <input
            type="tel"
            value={formPhone}
            onChange={(e) => setFormPhone(e.target.value)}
            className="w-full border border-stone-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-amber-500"
            placeholder="052-0000000"
          />
        </div>
      </div>

      {enrollments.length > 0 && (
        <div>
          <label className="block text-xs font-medium text-stone-600 mb-2">סוג עמדה לפי קבוצה</label>
          <div className="space-y-2">
            {enrollments.map((e) => (
              <div key={e.groupId} className="flex items-center gap-3">
                <span className="text-sm text-stone-600 flex-1">{e.group.name}</span>
                <select
                  value={slotTypes[e.groupId] ?? ""}
                  onChange={(ev) => setSlotTypes((prev) => ({ ...prev, [e.groupId]: ev.target.value }))}
                  className="border border-stone-300 rounded-lg px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-amber-500"
                >
                  <option value="">ללא הגדרה</option>
                  {Object.entries(SLOT_LABELS).map(([val, label]) => (
                    <option key={val} value={val}>{label}</option>
                  ))}
                </select>
              </div>
            ))}
          </div>
        </div>
      )}

      {error && <p className="text-red-600 text-sm">{error}</p>}

      <div className="flex gap-3">
        <button
          onClick={handleSave}
          disabled={loading}
          className="bg-amber-600 hover:bg-amber-700 text-white px-5 py-2 rounded-lg text-sm font-medium transition disabled:opacity-50"
        >
          {loading ? "שומר..." : "שמור"}
        </button>
        <button
          onClick={() => setOpen(false)}
          className="border border-stone-300 text-stone-600 px-4 py-2 rounded-lg text-sm hover:bg-stone-50 transition"
        >
          ביטול
        </button>
      </div>
    </div>
  );
}
