import { z } from "zod";
import { ROOM_TYPES, ROOM_STATUSES } from "../constants/index.js";

export const createRoomSchema = z.object({
  number: z.string().min(1, "מספר חדר נדרש"),
  floor: z.number().int().min(0),
  type: z.enum(ROOM_TYPES),
  basePrice: z.number().positive("מחיר חייב להיות חיובי"),
  maxGuests: z.number().int().min(1).max(10).default(2),
  description: z.string().optional(),
  amenities: z.array(z.string()).default([]),
  notes: z.string().optional(),
});

export const updateRoomSchema = createRoomSchema.partial().extend({
  status: z.enum(ROOM_STATUSES).optional(),
});
