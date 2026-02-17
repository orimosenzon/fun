export const ROOM_TYPES = ["SINGLE", "DOUBLE", "TWIN", "SUITE", "FAMILY"] as const;
export type RoomType = (typeof ROOM_TYPES)[number];

export const ROOM_STATUSES = ["AVAILABLE", "OCCUPIED", "MAINTENANCE", "CLEANING"] as const;
export type RoomStatus = (typeof ROOM_STATUSES)[number];

export const RESERVATION_STATUSES = [
  "PENDING",
  "CONFIRMED",
  "CHECKED_IN",
  "CHECKED_OUT",
  "CANCELLED",
  "NO_SHOW",
] as const;
export type ReservationStatus = (typeof RESERVATION_STATUSES)[number];

export const ROOM_TYPE_LABELS_HE: Record<RoomType, string> = {
  SINGLE: "יחיד",
  DOUBLE: "זוגי",
  TWIN: "טווין",
  SUITE: "סוויטה",
  FAMILY: "משפחתי",
};

export const ROOM_STATUS_LABELS_HE: Record<RoomStatus, string> = {
  AVAILABLE: "פנוי",
  OCCUPIED: "תפוס",
  MAINTENANCE: "תחזוקה",
  CLEANING: "ניקיון",
};

export const RESERVATION_STATUS_LABELS_HE: Record<ReservationStatus, string> = {
  PENDING: "ממתין",
  CONFIRMED: "מאושר",
  CHECKED_IN: "נכנס",
  CHECKED_OUT: "יצא",
  CANCELLED: "בוטל",
  NO_SHOW: "לא הגיע",
};

export const RESERVATION_SOURCES = ["walk-in", "phone", "website", "booking.com", "expedia", "other"] as const;
export type ReservationSource = (typeof RESERVATION_SOURCES)[number];
