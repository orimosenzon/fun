import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { Reservation, CreateReservationInput } from "@hotel/shared";

export function useReservations(from?: string, to?: string) {
  return useQuery({
    queryKey: ["reservations", from, to],
    queryFn: () => {
      const params = new URLSearchParams();
      if (from) params.set("from", from);
      if (to) params.set("to", to);
      const qs = params.toString();
      return api.get<Reservation[]>(`/reservations${qs ? `?${qs}` : ""}`);
    },
  });
}

export function useReservation(id: number) {
  return useQuery({
    queryKey: ["reservations", id],
    queryFn: () => api.get<Reservation>(`/reservations/${id}`),
    enabled: !!id,
  });
}

export function useTodayReservations() {
  return useQuery({
    queryKey: ["reservations", "today"],
    queryFn: () => api.get<{ arrivals: Reservation[]; departures: Reservation[] }>("/reservations/today"),
    refetchInterval: 60_000,
  });
}

export function useCreateReservation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: CreateReservationInput) => api.post<Reservation>("/reservations", data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["reservations"] });
      qc.invalidateQueries({ queryKey: ["rooms"] });
      qc.invalidateQueries({ queryKey: ["dashboard"] });
    },
  });
}

export function useCheckIn() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: number) => api.post<Reservation>(`/reservations/${id}/check-in`, {}),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["reservations"] });
      qc.invalidateQueries({ queryKey: ["rooms"] });
      qc.invalidateQueries({ queryKey: ["dashboard"] });
    },
  });
}

export function useCheckOut() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: number) => api.post<Reservation>(`/reservations/${id}/check-out`, {}),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["reservations"] });
      qc.invalidateQueries({ queryKey: ["rooms"] });
      qc.invalidateQueries({ queryKey: ["dashboard"] });
    },
  });
}

export function useCancelReservation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: number) => api.post<Reservation>(`/reservations/${id}/cancel`, {}),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["reservations"] });
      qc.invalidateQueries({ queryKey: ["rooms"] });
      qc.invalidateQueries({ queryKey: ["dashboard"] });
    },
  });
}
