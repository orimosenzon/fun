import type { RoomType, RoomStatus } from "../constants/index.js";

export interface Room {
  id: number;
  number: string;
  floor: number;
  type: RoomType;
  status: RoomStatus;
  basePrice: number;
  maxGuests: number;
  description: string | null;
  amenities: string[];
  notes: string | null;
  createdAt: string;
  updatedAt: string;
}

export interface CreateRoomInput {
  number: string;
  floor: number;
  type: RoomType;
  basePrice: number;
  maxGuests?: number;
  description?: string;
  amenities?: string[];
  notes?: string;
}

export interface UpdateRoomInput {
  number?: string;
  floor?: number;
  type?: RoomType;
  status?: RoomStatus;
  basePrice?: number;
  maxGuests?: number;
  description?: string | null;
  amenities?: string[];
  notes?: string | null;
}
