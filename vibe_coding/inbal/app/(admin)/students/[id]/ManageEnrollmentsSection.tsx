"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

type Enrollment = {
  id: string;
  groupId: string;
  status: string;
  preferredSlotType: string | null;
  group: { id: string; name: string };
};

type Group = { id: string; name: string };

type Props = {
  studentId: string;
  enrollments: Enrollment[];
  allGroups: Group[];
};

const STATUS_LABEL: Record<string, string> = {
  ACTIVE: "פעיל",
  PAUSED: "מושהה",
  LEFT: "עזב",
};

const STATUS_CLASS: Record<string, string> = {
  ACTIVE: "bg-green-100 text-green-700",
  PAUSED: "bg-yellow-100 text-yellow-700",
  LEFT: "bg-stone-100 text-stone-500",
};

export default function ManageEnrollmentsSection({ studentId, enrollments, allGroups }: Props) {
  const router = useRouter();
  const [adding, setAdding] = useState(false);
  const [addGroupId, setAddGroupId] = useState("");
  const [addSlotType, setAddSlotType] = useState("");
  const [loadingKey, setLoadingKey] = useState<string | null>(null);
  const [error, setError] = useState("");

  const activeGroupIds = new Set(
    enrollments.filter((e) => e.status === "ACTIVE").map((e) => e.groupId)
  );
  const availableGroups = allGroups.filter((g) => !activeGroupIds.has(g.id));

  async function changeStatus(groupId: string, status: string) {
    setLoadingKey(groupId + status);
    await fetch("/api/enrollments", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ userId: studentId, groupId, status }),
    });
    setLoadingKey(null);
    router.refresh();
  }

  async function addToGroup() {
    if (!addGroupId) return;
    setLoadingKey("add");
    setError("");
    const res = await fetch("/api/enrollments", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ userId: studentId, groupId: addGroupId, slotType: addSlotType || null }),
    });
    if (!res.ok) {
      const data = await res.json();
      setError(data.error || "שגיאה");
      setLoadingKey(null);
      return;
    }
    setLoadingKey(null);
    setAdding(false);
    setAddGroupId("");
    setAddSlotType("");
    router.refresh();
  }

  return (
    <div className="bg-white rounded-xl border border-stone-200 overflow-hidden">
      <div className="px-5 py-4 border-b border-stone-100 flex items-center justify-between">
        <h2 className="font-semibold text-stone-800">קבוצות</h2>
        {availableGroups.length > 0 && (
          <button
            onClick={() => setAdding(true)}
            className="text-xs text-amber-600 hover:text-amber-800 border border-amber-200 px-3 py-1 rounded-lg hover:bg-amber-50 transition"
          >
            + הוסף קבוצה
          </button>
        )}
      </div>

      <div className="divide-y divide-stone-100">
        {enrollments.map((e) => (
          <div key={e.id} className="px-5 py-3 flex items-center justify-between">
            <div>
              <span className="font-medium text-stone-700">{e.group.name}</span>
              {e.preferredSlotType && (
                <span className="text-xs text-stone-400 mr-2">
                  {e.preferredSlotType === "WHEEL" ? "🎡 אובן" : "✋ ללא אובן"}
                </span>
              )}
            </div>
            <div className="flex items-center gap-2">
              <span className={`text-xs px-2 py-0.5 rounded-full ${STATUS_CLASS[e.status] ?? "bg-stone-100 text-stone-500"}`}>
                {STATUS_LABEL[e.status] ?? e.status}
              </span>

              {e.status === "ACTIVE" && (
                <>
                  <button
                    onClick={() => changeStatus(e.groupId, "PAUSED")}
                    disabled={!!loadingKey}
                    className="text-xs text-yellow-600 hover:text-yellow-800 px-2 py-0.5 rounded hover:bg-yellow-50 transition disabled:opacity-40"
                  >
                    השהה
                  </button>
                  <button
                    onClick={() => changeStatus(e.groupId, "LEFT")}
                    disabled={!!loadingKey}
                    className="text-xs text-red-400 hover:text-red-600 px-2 py-0.5 rounded hover:bg-red-50 transition disabled:opacity-40"
                  >
                    הסר
                  </button>
                </>
              )}

              {e.status === "PAUSED" && (
                <>
                  <button
                    onClick={() => changeStatus(e.groupId, "ACTIVE")}
                    disabled={!!loadingKey}
                    className="text-xs text-green-600 hover:text-green-800 px-2 py-0.5 rounded hover:bg-green-50 transition disabled:opacity-40"
                  >
                    הפעל
                  </button>
                  <button
                    onClick={() => changeStatus(e.groupId, "LEFT")}
                    disabled={!!loadingKey}
                    className="text-xs text-red-400 hover:text-red-600 px-2 py-0.5 rounded hover:bg-red-50 transition disabled:opacity-40"
                  >
                    הסר
                  </button>
                </>
              )}

              {e.status === "LEFT" && (
                <button
                  onClick={() => changeStatus(e.groupId, "ACTIVE")}
                  disabled={!!loadingKey}
                  className="text-xs text-green-600 hover:text-green-800 px-2 py-0.5 rounded hover:bg-green-50 transition disabled:opacity-40"
                >
                  הפעל מחדש
                </button>
              )}
            </div>
          </div>
        ))}

        {enrollments.length === 0 && !adding && (
          <div className="px-5 py-4 text-stone-400 text-sm">לא רשום לקבוצות</div>
        )}

        {/* Add to group form */}
        {adding && (
          <div className="px-5 py-4 space-y-3 bg-amber-50/40">
            <div className="flex gap-2">
              <select
                value={addGroupId}
                onChange={(e) => setAddGroupId(e.target.value)}
                className="flex-1 border border-stone-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-amber-500"
              >
                <option value="">בחר קבוצה...</option>
                {availableGroups.map((g) => (
                  <option key={g.id} value={g.id}>{g.name}</option>
                ))}
              </select>
              <select
                value={addSlotType}
                onChange={(e) => setAddSlotType(e.target.value)}
                className="border border-stone-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-amber-500"
              >
                <option value="">עמדה?</option>
                <option value="WHEEL">🎡 אובן</option>
                <option value="NO_WHEEL">✋ ללא אובן</option>
              </select>
            </div>
            {error && <p className="text-red-600 text-xs">{error}</p>}
            <div className="flex gap-2">
              <button
                onClick={addToGroup}
                disabled={!addGroupId || loadingKey === "add"}
                className="bg-amber-600 hover:bg-amber-700 text-white px-4 py-1.5 rounded-lg text-sm font-medium transition disabled:opacity-50"
              >
                {loadingKey === "add" ? "מוסיף..." : "הוסף"}
              </button>
              <button
                onClick={() => { setAdding(false); setError(""); setAddGroupId(""); setAddSlotType(""); }}
                className="border border-stone-300 text-stone-600 px-4 py-1.5 rounded-lg text-sm hover:bg-stone-50 transition"
              >
                ביטול
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
