import { useState } from "react";
import { useRooms, useCreateRoom, useUpdateRoomStatus } from "@/hooks/useRooms";
import { RoomCard } from "@/components/rooms/RoomCard";
import { RoomForm } from "@/components/rooms/RoomForm";
import { ROOM_STATUSES, ROOM_STATUS_LABELS_HE, type RoomStatus, type CreateRoomInput } from "@hotel/shared";
import { Plus, X } from "lucide-react";

export function Rooms() {
  const { data: rooms, isLoading } = useRooms();
  const createRoom = useCreateRoom();
  const updateStatus = useUpdateRoomStatus();
  const [showForm, setShowForm] = useState(false);
  const [selectedRoom, setSelectedRoom] = useState<number | null>(null);
  const [filterStatus, setFilterStatus] = useState<RoomStatus | "ALL">("ALL");

  const filteredRooms = rooms?.filter((r) => filterStatus === "ALL" || r.status === filterStatus);

  const handleCreateRoom = async (data: CreateRoomInput) => {
    await createRoom.mutateAsync(data);
    setShowForm(false);
  };

  const handleStatusChange = (roomId: number, status: RoomStatus) => {
    updateStatus.mutate({ id: roomId, status });
    setSelectedRoom(null);
  };

  if (isLoading) return <div className="text-center py-10 text-slate-500">טוען...</div>;

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-slate-800">חדרים</h2>
        <button onClick={() => setShowForm(true)} className="flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-lg text-sm hover:bg-blue-700">
          <Plus size={18} />
          חדר חדש
        </button>
      </div>

      {/* Filter bar */}
      <div className="flex gap-2 mb-6">
        <button
          onClick={() => setFilterStatus("ALL")}
          className={`px-3 py-1.5 rounded-lg text-sm ${filterStatus === "ALL" ? "bg-slate-800 text-white" : "bg-white border text-slate-600 hover:bg-slate-50"}`}
        >
          הכל ({rooms?.length})
        </button>
        {ROOM_STATUSES.map((s) => {
          const count = rooms?.filter((r) => r.status === s).length ?? 0;
          return (
            <button
              key={s}
              onClick={() => setFilterStatus(s)}
              className={`px-3 py-1.5 rounded-lg text-sm ${filterStatus === s ? "bg-slate-800 text-white" : "bg-white border text-slate-600 hover:bg-slate-50"}`}
            >
              {ROOM_STATUS_LABELS_HE[s]} ({count})
            </button>
          );
        })}
      </div>

      {/* Room Grid */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
        {filteredRooms?.map((room) => (
          <RoomCard
            key={room.id}
            number={room.number}
            floor={room.floor}
            type={room.type as any}
            status={room.status as any}
            basePrice={Number(room.basePrice)}
            onClick={() => setSelectedRoom(selectedRoom === room.id ? null : room.id)}
          />
        ))}
      </div>

      {/* Status change panel */}
      {selectedRoom && (
        <div className="fixed bottom-6 left-1/2 -translate-x-1/2 bg-white shadow-lg border rounded-xl p-4 flex items-center gap-3">
          <span className="text-sm font-medium text-slate-600">שנה סטטוס:</span>
          {ROOM_STATUSES.map((s) => (
            <button
              key={s}
              onClick={() => handleStatusChange(selectedRoom, s)}
              className="px-3 py-1.5 rounded-lg text-sm border hover:bg-slate-50"
            >
              {ROOM_STATUS_LABELS_HE[s]}
            </button>
          ))}
          <button onClick={() => setSelectedRoom(null)} className="text-slate-400 hover:text-slate-600 mr-2">
            <X size={18} />
          </button>
        </div>
      )}

      {/* New room modal */}
      {showForm && (
        <div className="fixed inset-0 bg-black/30 flex items-center justify-center z-50">
          <div className="bg-white rounded-xl shadow-xl p-6 w-full max-w-lg">
            <h3 className="text-lg font-bold text-slate-800 mb-4">חדר חדש</h3>
            <RoomForm onSubmit={handleCreateRoom} onCancel={() => setShowForm(false)} isLoading={createRoom.isPending} />
          </div>
        </div>
      )}
    </div>
  );
}
