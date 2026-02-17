import type { FastifyInstance } from "fastify";
import { prisma } from "../lib/prisma.js";

export async function dashboardRoutes(app: FastifyInstance) {
  app.get("/dashboard/summary", async () => {
    const totalRooms = await prisma.room.count();
    const occupiedRooms = await prisma.room.count({ where: { status: "OCCUPIED" } });
    const cleaningRooms = await prisma.room.count({ where: { status: "CLEANING" } });
    const maintenanceRooms = await prisma.room.count({ where: { status: "MAINTENANCE" } });
    const availableRooms = await prisma.room.count({ where: { status: "AVAILABLE" } });

    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const tomorrow = new Date(today);
    tomorrow.setDate(tomorrow.getDate() + 1);

    const todayArrivals = await prisma.reservation.count({
      where: { checkIn: { gte: today, lt: tomorrow }, status: "CONFIRMED" },
    });

    const todayDepartures = await prisma.reservation.count({
      where: { checkOut: { gte: today, lt: tomorrow }, status: "CHECKED_IN" },
    });

    const occupancyRate = totalRooms > 0 ? Math.round((occupiedRooms / totalRooms) * 100) : 0;

    return {
      totalRooms,
      occupiedRooms,
      availableRooms,
      cleaningRooms,
      maintenanceRooms,
      occupancyRate,
      todayArrivals,
      todayDepartures,
    };
  });
}
