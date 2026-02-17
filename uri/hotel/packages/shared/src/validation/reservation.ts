import { z } from "zod";
import { RESERVATION_SOURCES } from "../constants/index.js";

export const createReservationSchema = z
  .object({
    roomId: z.number().int().positive(),
    guestId: z.number().int().positive(),
    checkIn: z.string().date(),
    checkOut: z.string().date(),
    numGuests: z.number().int().min(1).default(1),
    totalPrice: z.number().positive("מחיר חייב להיות חיובי"),
    notes: z.string().optional(),
    source: z.enum(RESERVATION_SOURCES).optional(),
  })
  .refine((data) => new Date(data.checkOut) > new Date(data.checkIn), {
    message: "תאריך יציאה חייב להיות אחרי תאריך כניסה",
    path: ["checkOut"],
  });

export const updateReservationSchema = z.object({
  roomId: z.number().int().positive().optional(),
  checkIn: z.string().date().optional(),
  checkOut: z.string().date().optional(),
  numGuests: z.number().int().min(1).optional(),
  totalPrice: z.number().positive().optional(),
  notes: z.string().nullable().optional(),
  source: z.string().nullable().optional(),
});

export const availabilityQuerySchema = z
  .object({
    checkIn: z.string().date(),
    checkOut: z.string().date(),
    type: z.string().optional(),
    minGuests: z.coerce.number().int().min(1).optional(),
  })
  .refine((data) => new Date(data.checkOut) > new Date(data.checkIn), {
    message: "תאריך יציאה חייב להיות אחרי תאריך כניסה",
    path: ["checkOut"],
  });
