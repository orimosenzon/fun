import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { Room, CreateRoomInput, UpdateRoomInput } from "@hotel/shared";

export function useRooms() {
  return useQuery({
    queryKey: ["rooms"],
    queryFn: () => api.get<Room[]>("/rooms"),
  });
}

export function useRoom(id: number) {
  return useQuery({
    queryKey: ["rooms", id],
    queryFn: () => api.get<Room>(`/rooms/${id}`),
    enabled: !!id,
  });
}

export function useAvailableRooms(checkIn: string, checkOut: string, type?: string) {
  return useQuery({
    queryKey: ["rooms", "availability", checkIn, checkOut, type],
    queryFn: () => {
      const params = new URLSearchParams({ checkIn, checkOut });
      if (type) params.set("type", type);
      return api.get<Room[]>(`/rooms/availability?${params}`);
    },
    enabled: !!checkIn && !!checkOut,
  });
}

export function useCreateRoom() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: CreateRoomInput) => api.post<Room>("/rooms", data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["rooms"] }),
  });
}

export function useUpdateRoom() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, data }: { id: number; data: UpdateRoomInput }) => api.patch<Room>(`/rooms/${id}`, data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["rooms"] }),
  });
}

export function useUpdateRoomStatus() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, status }: { id: number; status: string }) => api.patch<Room>(`/rooms/${id}/status`, { status }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["rooms"] }),
  });
}
