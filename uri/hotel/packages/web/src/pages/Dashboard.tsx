import { useDashboard } from "@/hooks/useDashboard";
import { useTodayReservations } from "@/hooks/useReservations";
import { useRooms } from "@/hooks/useRooms";
import { BedDouble, UserCheck, UserX, BarChart3, ArrowDownToLine, ArrowUpFromLine } from "lucide-react";
import { ROOM_TYPE_LABELS_HE, ROOM_STATUS_LABELS_HE, type RoomStatus } from "@hotel/shared";
import { cn } from "@/lib/cn";

const statusColors: Record<string, { bg: string; border: string; text: string }> = {
  AVAILABLE: { bg: "bg-green-100", border: "border-green-400", text: "text-green-800" },
  OCCUPIED: { bg: "bg-blue-100", border: "border-blue-400", text: "text-blue-800" },
  CLEANING: { bg: "bg-yellow-100", border: "border-yellow-400", text: "text-yellow-800" },
  MAINTENANCE: { bg: "bg-red-100", border: "border-red-400", text: "text-red-800" },
};

function StatCard({ label, value, icon: Icon, color }: { label: string; value: string | number; icon: React.ElementType; color: string }) {
  return (
    <div className="bg-white rounded-lg shadow-sm border p-5 flex items-center gap-4">
      <div className={`p-3 rounded-lg ${color}`}>
        <Icon size={24} className="text-white" />
      </div>
      <div>
        <p className="text-sm text-slate-500">{label}</p>
        <p className="text-2xl font-bold text-slate-800">{value}</p>
      </div>
    </div>
  );
}

export function Dashboard() {
  const { data: summary, isLoading } = useDashboard();
  const { data: today } = useTodayReservations();
  const { data: rooms } = useRooms();

  if (isLoading) return <div className="text-center py-10 text-slate-500">טוען...</div>;
  if (!summary) return null;

  // Group rooms by floor
  const floors = rooms
    ? [...new Set(rooms.map((r) => r.floor))].sort((a, b) => a - b)
    : [];

  return (
    <div>
      <h2 className="text-2xl font-bold text-slate-800 mb-6">לוח בקרה</h2>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <StatCard label="חדרים תפוסים" value={`${summary.occupiedRooms} / ${summary.totalRooms}`} icon={BedDouble} color="bg-blue-500" />
        <StatCard label="אחוז תפוסה" value={`${summary.occupancyRate}%`} icon={BarChart3} color="bg-emerald-500" />
        <StatCard label="הגעות היום" value={summary.todayArrivals} icon={ArrowDownToLine} color="bg-amber-500" />
        <StatCard label="עזיבות היום" value={summary.todayDepartures} icon={ArrowUpFromLine} color="bg-purple-500" />
      </div>

      {/* Room Map */}
      <div className="bg-white rounded-lg shadow-sm border p-5 mb-6">
        <h3 className="text-lg font-semibold text-slate-800 mb-4 flex items-center gap-2">
          <BedDouble size={20} />
          מפת חדרים
        </h3>

        {/* Legend */}
        <div className="flex gap-4 mb-4 text-sm">
          {(["AVAILABLE", "OCCUPIED", "CLEANING", "MAINTENANCE"] as RoomStatus[]).map((s) => (
            <div key={s} className="flex items-center gap-1.5">
              <div className={cn("w-4 h-4 rounded border", statusColors[s].bg, statusColors[s].border)} />
              <span className="text-slate-600">{ROOM_STATUS_LABELS_HE[s]}</span>
            </div>
          ))}
        </div>

        {/* Floors */}
        {floors.map((floor) => {
          const floorRooms = rooms!.filter((r) => r.floor === floor).sort((a, b) => a.number.localeCompare(b.number));
          return (
            <div key={floor} className="mb-4">
              <p className="text-sm font-medium text-slate-500 mb-2">קומה {floor}</p>
              <div className="flex flex-wrap gap-2">
                {floorRooms.map((room) => {
                  const colors = statusColors[room.status] || statusColors.AVAILABLE;
                  return (
                    <div
                      key={room.id}
                      className={cn(
                        "w-20 h-16 rounded-lg border-2 flex flex-col items-center justify-center transition-all hover:shadow-md cursor-default",
                        colors.bg,
                        colors.border
                      )}
                      title={`${room.number} - ${ROOM_TYPE_LABELS_HE[room.type as keyof typeof ROOM_TYPE_LABELS_HE]} - ${ROOM_STATUS_LABELS_HE[room.status as RoomStatus]}`}
                    >
                      <span className={cn("text-sm font-bold", colors.text)}>{room.number}</span>
                      <span className={cn("text-[10px]", colors.text)}>
                        {ROOM_TYPE_LABELS_HE[room.type as keyof typeof ROOM_TYPE_LABELS_HE]}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>

      {/* Today's arrivals & departures */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg shadow-sm border p-5">
          <h3 className="text-lg font-semibold text-slate-800 mb-4 flex items-center gap-2">
            <UserCheck size={20} className="text-green-600" />
            הגעות היום
          </h3>
          {today?.arrivals.length === 0 && <p className="text-slate-400 text-sm">אין הגעות צפויות היום</p>}
          <div className="space-y-3">
            {today?.arrivals.map((r) => (
              <div key={r.id} className="flex items-center justify-between p-3 bg-green-50 rounded-lg">
                <div>
                  <span className="font-medium">{r.guest?.firstName} {r.guest?.lastName}</span>
                  <span className="text-sm text-slate-500 mr-3">
                    חדר {r.room?.number} ({ROOM_TYPE_LABELS_HE[r.room?.type as keyof typeof ROOM_TYPE_LABELS_HE]})
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-sm border p-5">
          <h3 className="text-lg font-semibold text-slate-800 mb-4 flex items-center gap-2">
            <UserX size={20} className="text-red-600" />
            עזיבות היום
          </h3>
          {today?.departures.length === 0 && <p className="text-slate-400 text-sm">אין עזיבות צפויות היום</p>}
          <div className="space-y-3">
            {today?.departures.map((r) => (
              <div key={r.id} className="flex items-center justify-between p-3 bg-red-50 rounded-lg">
                <div>
                  <span className="font-medium">{r.guest?.firstName} {r.guest?.lastName}</span>
                  <span className="text-sm text-slate-500 mr-3">חדר {r.room?.number}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
