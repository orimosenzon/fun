import type { FastifyInstance } from "fastify";
import { createReservationSchema, updateReservationSchema } from "@hotel/shared";
import * as reservationService from "../services/reservation.service.js";

export async function reservationRoutes(app: FastifyInstance) {
  app.get("/reservations", async (request) => {
    const { status, from, to } = request.query as {
      status?: string;
      from?: string;
      to?: string;
    };
    return reservationService.getAllReservations({
      status,
      from: from ? new Date(from) : undefined,
      to: to ? new Date(to) : undefined,
    });
  });

  app.get("/reservations/today", async () => {
    return reservationService.getTodayReservations();
  });

  app.get("/reservations/:id", async (request, reply) => {
    const { id } = request.params as { id: string };
    const reservation = await reservationService.getReservationById(Number(id));
    if (!reservation) return reply.status(404).send({ error: "הזמנה לא נמצאה" });
    return reservation;
  });

  app.post("/reservations", async (request, reply) => {
    const data = createReservationSchema.parse(request.body);
    const reservation = await reservationService.createReservation({
      ...data,
      checkIn: new Date(data.checkIn),
      checkOut: new Date(data.checkOut),
    });
    return reply.status(201).send(reservation);
  });

  app.patch("/reservations/:id", async (request) => {
    const { id } = request.params as { id: string };
    const data = updateReservationSchema.parse(request.body);
    return reservationService.updateReservation(Number(id), data);
  });

  app.post("/reservations/:id/check-in", async (request) => {
    const { id } = request.params as { id: string };
    return reservationService.checkIn(Number(id));
  });

  app.post("/reservations/:id/check-out", async (request) => {
    const { id } = request.params as { id: string };
    return reservationService.checkOut(Number(id));
  });

  app.post("/reservations/:id/cancel", async (request) => {
    const { id } = request.params as { id: string };
    return reservationService.cancelReservation(Number(id));
  });
}
