import { prisma } from "../lib/prisma.js";

export async function getAllReservations(filters?: {
  status?: string;
  from?: Date;
  to?: Date;
}) {
  return prisma.reservation.findMany({
    where: {
      status: filters?.status,
      checkIn: filters?.to ? { lte: filters.to } : undefined,
      checkOut: filters?.from ? { gte: filters.from } : undefined,
    },
    include: { room: true, guest: true },
    orderBy: { checkIn: "asc" },
  });
}

export async function getReservationById(id: number) {
  return prisma.reservation.findUnique({
    where: { id },
    include: { room: true, guest: true },
  });
}

export async function createReservation(data: {
  roomId: number;
  guestId: number;
  checkIn: Date;
  checkOut: Date;
  numGuests?: number;
  totalPrice: number;
  notes?: string;
  source?: string;
}) {
  // Check availability
  const conflicting = await prisma.reservation.findFirst({
    where: {
      roomId: data.roomId,
      checkIn: { lt: data.checkOut },
      checkOut: { gt: data.checkIn },
      status: { notIn: ["CANCELLED", "NO_SHOW", "CHECKED_OUT"] },
    },
  });

  if (conflicting) {
    throw Object.assign(new Error("החדר תפוס בתאריכים אלו"), { statusCode: 409 });
  }

  return prisma.reservation.create({
    data: { ...data, status: "CONFIRMED" },
    include: { room: true, guest: true },
  });
}

export async function updateReservation(id: number, data: Record<string, unknown>) {
  return prisma.reservation.update({
    where: { id },
    data,
    include: { room: true, guest: true },
  });
}

export async function checkIn(id: number) {
  return prisma.$transaction(async (tx) => {
    const reservation = await tx.reservation.update({
      where: { id },
      data: { status: "CHECKED_IN" },
      include: { room: true },
    });

    await tx.room.update({
      where: { id: reservation.roomId },
      data: { status: "OCCUPIED" },
    });

    return reservation;
  });
}

export async function checkOut(id: number) {
  return prisma.$transaction(async (tx) => {
    const reservation = await tx.reservation.update({
      where: { id },
      data: { status: "CHECKED_OUT" },
      include: { room: true },
    });

    await tx.room.update({
      where: { id: reservation.roomId },
      data: { status: "CLEANING" },
    });

    return reservation;
  });
}

export async function cancelReservation(id: number) {
  return prisma.reservation.update({
    where: { id },
    data: { status: "CANCELLED" },
    include: { room: true, guest: true },
  });
}

export async function getTodayReservations() {
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const tomorrow = new Date(today);
  tomorrow.setDate(tomorrow.getDate() + 1);

  const [arrivals, departures] = await Promise.all([
    prisma.reservation.findMany({
      where: { checkIn: { gte: today, lt: tomorrow }, status: "CONFIRMED" },
      include: { room: true, guest: true },
      orderBy: { room: { number: "asc" } },
    }),
    prisma.reservation.findMany({
      where: { checkOut: { gte: today, lt: tomorrow }, status: "CHECKED_IN" },
      include: { room: true, guest: true },
      orderBy: { room: { number: "asc" } },
    }),
  ]);

  return { arrivals, departures };
}
