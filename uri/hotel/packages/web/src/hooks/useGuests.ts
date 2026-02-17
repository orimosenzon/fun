import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { Guest, CreateGuestInput } from "@hotel/shared";

export function useGuests() {
  return useQuery({
    queryKey: ["guests"],
    queryFn: () => api.get<Guest[]>("/guests"),
  });
}

export function useSearchGuests(query: string) {
  return useQuery({
    queryKey: ["guests", "search", query],
    queryFn: () => api.get<Guest[]>(`/guests/search?q=${encodeURIComponent(query)}`),
    enabled: query.length >= 2,
  });
}

export function useCreateGuest() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: CreateGuestInput) => api.post<Guest>("/guests", data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["guests"] }),
  });
}
