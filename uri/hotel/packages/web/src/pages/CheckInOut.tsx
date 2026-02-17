import { useTodayReservations, useCheckIn, useCheckOut } from "@/hooks/useReservations";
import { ROOM_TYPE_LABELS_HE } from "@hotel/shared";
import { LogIn, LogOut } from "lucide-react";

export function CheckInOut() {
  const { data, isLoading } = useTodayReservations();
  const checkIn = useCheckIn();
  const checkOut = useCheckOut();

  if (isLoading) return <div className="text-center py-10 text-slate-500">טוען...</div>;

  return (
    <div>
      <h2 className="text-2xl font-bold text-slate-800 mb-6">כניסה / יציאה</h2>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Arrivals */}
        <div>
          <h3 className="text-lg font-semibold text-slate-800 mb-4 flex items-center gap-2">
            <LogIn size={20} className="text-green-600" />
            הגעות היום ({data?.arrivals.length ?? 0})
          </h3>
          {data?.arrivals.length === 0 && (
            <div className="bg-white rounded-lg border p-8 text-center text-slate-400">אין הגעות צפויות היום</div>
          )}
          <div className="space-y-3">
            {data?.arrivals.map((r) => (
              <div key={r.id} className="bg-white rounded-lg border p-4 flex items-center justify-between">
                <div>
                  <p className="font-medium text-slate-800">{r.guest?.firstName} {r.guest?.lastName}</p>
                  <p className="text-sm text-slate-500">
                    חדר {r.room?.number} - {ROOM_TYPE_LABELS_HE[r.room?.type as keyof typeof ROOM_TYPE_LABELS_HE]}
                  </p>
                  <p className="text-sm text-slate-500">{r.numGuests} אורחים</p>
                </div>
                <button
                  onClick={() => checkIn.mutate(r.id)}
                  disabled={checkIn.isPending}
                  className="bg-green-600 text-white px-5 py-2 rounded-lg text-sm font-medium hover:bg-green-700 disabled:opacity-50"
                >
                  כניסה
                </button>
              </div>
            ))}
          </div>
        </div>

        {/* Departures */}
        <div>
          <h3 className="text-lg font-semibold text-slate-800 mb-4 flex items-center gap-2">
            <LogOut size={20} className="text-red-600" />
            עזיבות היום ({data?.departures.length ?? 0})
          </h3>
          {data?.departures.length === 0 && (
            <div className="bg-white rounded-lg border p-8 text-center text-slate-400">אין עזיבות צפויות היום</div>
          )}
          <div className="space-y-3">
            {data?.departures.map((r) => (
              <div key={r.id} className="bg-white rounded-lg border p-4 flex items-center justify-between">
                <div>
                  <p className="font-medium text-slate-800">{r.guest?.firstName} {r.guest?.lastName}</p>
                  <p className="text-sm text-slate-500">חדר {r.room?.number}</p>
                </div>
                <button
                  onClick={() => checkOut.mutate(r.id)}
                  disabled={checkOut.isPending}
                  className="bg-red-600 text-white px-5 py-2 rounded-lg text-sm font-medium hover:bg-red-700 disabled:opacity-50"
                >
                  יציאה
                </button>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
