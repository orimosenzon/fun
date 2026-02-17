import type { ReservationStatus, ReservationSource } from "../constants/index.js";
import type { Room } from "./room.js";
import type { Guest } from "./guest.js";

export interface Reservation {
  id: number;
  roomId: number;
  guestId: number;
  checkIn: string;
  checkOut: string;
  status: ReservationStatus;
  numGuests: number;
  totalPrice: number;
  notes: string | null;
  source: string | null;
  createdAt: string;
  updatedAt: string;
  room?: Room;
  guest?: Guest;
}

export interface CreateReservationInput {
  roomId: number;
  guestId: number;
  checkIn: string;
  checkOut: string;
  numGuests?: number;
  totalPrice: number;
  notes?: string;
  source?: ReservationSource;
}

export interface UpdateReservationInput {
  roomId?: number;
  checkIn?: string;
  checkOut?: string;
  numGuests?: number;
  totalPrice?: number;
  notes?: string | null;
  source?: string | null;
}

export interface AvailabilityQuery {
  checkIn: string;
  checkOut: string;
  type?: string;
  minGuests?: number;
}
