import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";

interface DashboardSummary {
  totalRooms: number;
  occupiedRooms: number;
  availableRooms: number;
  cleaningRooms: number;
  maintenanceRooms: number;
  occupancyRate: number;
  todayArrivals: number;
  todayDepartures: number;
}

export function useDashboard() {
  return useQuery({
    queryKey: ["dashboard"],
    queryFn: () => api.get<DashboardSummary>("/dashboard/summary"),
    refetchInterval: 60_000,
  });
}
