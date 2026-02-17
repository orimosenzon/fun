import { ROOM_TYPE_LABELS_HE, type RoomStatus, type RoomType } from "@hotel/shared";
import { RoomStatusBadge } from "./RoomStatusBadge";
import { cn } from "@/lib/cn";

interface RoomCardProps {
  number: string;
  floor: number;
  type: RoomType;
  status: RoomStatus;
  basePrice: number;
  onClick?: () => void;
}

const statusBorders: Record<RoomStatus, string> = {
  AVAILABLE: "border-green-300",
  OCCUPIED: "border-blue-300",
  CLEANING: "border-yellow-300",
  MAINTENANCE: "border-red-300",
};

export function RoomCard({ number, floor, type, status, basePrice, onClick }: RoomCardProps) {
  return (
    <div
      onClick={onClick}
      className={cn(
        "bg-white rounded-lg border-2 p-4 cursor-pointer transition-all hover:shadow-md",
        statusBorders[status]
      )}
    >
      <div className="flex items-center justify-between mb-2">
        <span className="text-lg font-bold text-slate-800">{number}</span>
        <RoomStatusBadge status={status} />
      </div>
      <p className="text-sm text-slate-600">{ROOM_TYPE_LABELS_HE[type]}</p>
      <p className="text-sm text-slate-500">קומה {floor}</p>
      <p className="text-sm font-medium text-slate-700 mt-1">₪{basePrice} / לילה</p>
    </div>
  );
}
