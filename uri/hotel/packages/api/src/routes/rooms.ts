import type { FastifyInstance } from "fastify";
import { createRoomSchema, updateRoomSchema, availabilityQuerySchema } from "@hotel/shared";
import * as roomService from "../services/room.service.js";

export async function roomRoutes(app: FastifyInstance) {
  app.get("/rooms", async (request) => {
    const { status, type, floor } = request.query as {
      status?: string;
      type?: string;
      floor?: string;
    };
    return roomService.getAllRooms({
      status,
      type,
      floor: floor ? Number(floor) : undefined,
    });
  });

  app.get("/rooms/availability", async (request) => {
    const query = availabilityQuerySchema.parse(request.query);
    return roomService.findAvailableRooms(
      new Date(query.checkIn),
      new Date(query.checkOut),
      query.type,
      query.minGuests
    );
  });

  app.get("/rooms/:id", async (request, reply) => {
    const { id } = request.params as { id: string };
    const room = await roomService.getRoomById(Number(id));
    if (!room) return reply.status(404).send({ error: "חדר לא נמצא" });
    return room;
  });

  app.post("/rooms", async (request, reply) => {
    const data = createRoomSchema.parse(request.body);
    const roomData = {
      ...data,
      amenities: JSON.stringify(data.amenities),
    };
    const room = await roomService.createRoom(roomData);
    return reply.status(201).send(room);
  });

  app.patch("/rooms/:id", async (request) => {
    const { id } = request.params as { id: string };
    const data = updateRoomSchema.parse(request.body);
    return roomService.updateRoom(Number(id), data);
  });

  app.patch("/rooms/:id/status", async (request) => {
    const { id } = request.params as { id: string };
    const { status } = request.body as { status: string };
    return roomService.updateRoomStatus(Number(id), status);
  });
}
