import { useState } from "react";
import { Link } from "react-router-dom";
import { useReservations } from "@/hooks/useReservations";
import { RESERVATION_STATUS_LABELS_HE, ROOM_TYPE_LABELS_HE, type ReservationStatus } from "@hotel/shared";
import { Plus } from "lucide-react";
import { format } from "date-fns";
import { cn } from "@/lib/cn";

const statusColors: Record<ReservationStatus, string> = {
  PENDING: "bg-gray-100 text-gray-800",
  CONFIRMED: "bg-blue-100 text-blue-800",
  CHECKED_IN: "bg-green-100 text-green-800",
  CHECKED_OUT: "bg-slate-100 text-slate-800",
  CANCELLED: "bg-red-100 text-red-800",
  NO_SHOW: "bg-orange-100 text-orange-800",
};

export function Reservations() {
  const [dateRange, setDateRange] = useState({ from: "", to: "" });
  const { data: reservations, isLoading } = useReservations(dateRange.from, dateRange.to);

  if (isLoading) return <div className="text-center py-10 text-slate-500">טוען...</div>;

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-slate-800">הזמנות</h2>
        <Link to="/reservations/new" className="flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-lg text-sm hover:bg-blue-700">
          <Plus size={18} />
          הזמנה חדשה
        </Link>
      </div>

      <div className="flex gap-3 mb-6">
        <div>
          <label className="text-sm text-slate-600">מתאריך</label>
          <input type="date" value={dateRange.from} onChange={(e) => setDateRange((p) => ({ ...p, from: e.target.value }))} className="block border rounded-lg px-3 py-1.5 text-sm" dir="ltr" />
        </div>
        <div>
          <label className="text-sm text-slate-600">עד תאריך</label>
          <input type="date" value={dateRange.to} onChange={(e) => setDateRange((p) => ({ ...p, to: e.target.value }))} className="block border rounded-lg px-3 py-1.5 text-sm" dir="ltr" />
        </div>
      </div>

      <div className="bg-white rounded-lg shadow-sm border overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-slate-50">
            <tr>
              <th className="text-right px-4 py-3 font-medium text-slate-600">#</th>
              <th className="text-right px-4 py-3 font-medium text-slate-600">אורח</th>
              <th className="text-right px-4 py-3 font-medium text-slate-600">חדר</th>
              <th className="text-right px-4 py-3 font-medium text-slate-600">כניסה</th>
              <th className="text-right px-4 py-3 font-medium text-slate-600">יציאה</th>
              <th className="text-right px-4 py-3 font-medium text-slate-600">סטטוס</th>
              <th className="text-right px-4 py-3 font-medium text-slate-600">סכום</th>
            </tr>
          </thead>
          <tbody>
            {reservations?.map((r) => (
              <tr key={r.id} className="border-t hover:bg-slate-50">
                <td className="px-4 py-3">{r.id}</td>
                <td className="px-4 py-3 font-medium">{r.guest?.firstName} {r.guest?.lastName}</td>
                <td className="px-4 py-3">{r.room?.number} ({ROOM_TYPE_LABELS_HE[r.room?.type as keyof typeof ROOM_TYPE_LABELS_HE]})</td>
                <td className="px-4 py-3" dir="ltr">{format(new Date(r.checkIn), "dd/MM/yyyy")}</td>
                <td className="px-4 py-3" dir="ltr">{format(new Date(r.checkOut), "dd/MM/yyyy")}</td>
                <td className="px-4 py-3">
                  <span className={cn("px-2 py-1 rounded-full text-xs font-medium", statusColors[r.status as ReservationStatus])}>
                    {RESERVATION_STATUS_LABELS_HE[r.status as ReservationStatus]}
                  </span>
                </td>
                <td className="px-4 py-3">₪{Number(r.totalPrice).toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {reservations?.length === 0 && <p className="text-center text-slate-400 py-8">אין הזמנות</p>}
      </div>
    </div>
  );
}
