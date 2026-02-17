import { prisma } from "../lib/prisma.js";

export async function getAllGuests() {
  return prisma.guest.findMany({ orderBy: { lastName: "asc" } });
}

export async function getGuestById(id: number) {
  return prisma.guest.findUnique({
    where: { id },
    include: { reservations: { include: { room: true }, orderBy: { checkIn: "desc" } } },
  });
}

export async function createGuest(data: {
  firstName: string;
  lastName: string;
  phone: string;
  email?: string;
  idNumber?: string;
  idType?: string;
  country?: string;
  address?: string;
  notes?: string;
}) {
  return prisma.guest.create({ data });
}

export async function updateGuest(id: number, data: Record<string, unknown>) {
  return prisma.guest.update({ where: { id }, data });
}

export async function searchGuests(query: string) {
  return prisma.guest.findMany({
    where: {
      OR: [
        { firstName: { contains: query, mode: "insensitive" } },
        { lastName: { contains: query, mode: "insensitive" } },
        { phone: { contains: query } },
      ],
    },
    take: 20,
  });
}
