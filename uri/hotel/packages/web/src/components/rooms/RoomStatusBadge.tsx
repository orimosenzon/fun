import { cn } from "@/lib/cn";
import { ROOM_STATUS_LABELS_HE, type RoomStatus } from "@hotel/shared";

const statusColors: Record<RoomStatus, string> = {
  AVAILABLE: "bg-green-100 text-green-800",
  OCCUPIED: "bg-blue-100 text-blue-800",
  CLEANING: "bg-yellow-100 text-yellow-800",
  MAINTENANCE: "bg-red-100 text-red-800",
};

export function RoomStatusBadge({ status }: { status: RoomStatus }) {
  return (
    <span className={cn("px-2 py-1 rounded-full text-xs font-medium", statusColors[status])}>
      {ROOM_STATUS_LABELS_HE[status]}
    </span>
  );
}
