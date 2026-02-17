import { z } from "zod";

export const createGuestSchema = z.object({
  firstName: z.string().min(1, "שם פרטי נדרש"),
  lastName: z.string().min(1, "שם משפחה נדרש"),
  phone: z.string().min(9, "מספר טלפון לא תקין"),
  email: z.string().email("כתובת אימייל לא תקינה").optional(),
  idNumber: z.string().optional(),
  idType: z.enum(["tz", "passport"]).optional(),
  country: z.string().default("IL"),
  address: z.string().optional(),
  notes: z.string().optional(),
});

export const updateGuestSchema = createGuestSchema.partial();
