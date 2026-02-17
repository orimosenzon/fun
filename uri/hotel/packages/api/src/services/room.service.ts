import { prisma } from "../lib/prisma.js";

export async function getAllRooms(filters?: { status?: string; type?: string; floor?: number }) {
  return prisma.room.findMany({
    where: {
      status: filters?.status,
      type: filters?.type,
      floor: filters?.floor,
    },
    orderBy: [{ floor: "asc" }, { number: "asc" }],
  });
}

export async function getRoomById(id: number) {
  return prisma.room.findUnique({ where: { id } });
}

export async function createRoom(data: {
  number: string;
  floor: number;
  type: string;
  basePrice: number;
  maxGuests?: number;
  description?: string;
  amenities?: string;
  notes?: string;
}) {
  return prisma.room.create({ data });
}

export async function updateRoom(id: number, data: Record<string, unknown>) {
  return prisma.room.update({ where: { id }, data });
}

export async function updateRoomStatus(id: number, status: string) {
  return prisma.room.update({ where: { id }, data: { status } });
}

export async function findAvailableRooms(checkIn: Date, checkOut: Date, type?: string, minGuests?: number) {
  return prisma.room.findMany({
    where: {
      status: { not: "MAINTENANCE" },
      type: type || undefined,
      maxGuests: minGuests ? { gte: minGuests } : undefined,
      reservations: {
        none: {
          AND: [
            { checkIn: { lt: checkOut } },
            { checkOut: { gt: checkIn } },
            { status: { notIn: ["CANCELLED", "NO_SHOW", "CHECKED_OUT"] } },
          ],
        },
      },
    },
    orderBy: [{ floor: "asc" }, { number: "asc" }],
  });
}
