import type { FastifyInstance } from "fastify";
import { createGuestSchema, updateGuestSchema } from "@hotel/shared";
import * as guestService from "../services/guest.service.js";

export async function guestRoutes(app: FastifyInstance) {
  app.get("/guests", async () => {
    return guestService.getAllGuests();
  });

  app.get("/guests/search", async (request) => {
    const { q } = request.query as { q: string };
    if (!q || q.length < 2) return [];
    return guestService.searchGuests(q);
  });

  app.get("/guests/:id", async (request, reply) => {
    const { id } = request.params as { id: string };
    const guest = await guestService.getGuestById(Number(id));
    if (!guest) return reply.status(404).send({ error: "אורח לא נמצא" });
    return guest;
  });

  app.post("/guests", async (request, reply) => {
    const data = createGuestSchema.parse(request.body);
    const guest = await guestService.createGuest(data);
    return reply.status(201).send(guest);
  });

  app.patch("/guests/:id", async (request) => {
    const { id } = request.params as { id: string };
    const data = updateGuestSchema.parse(request.body);
    return guestService.updateGuest(Number(id), data);
  });
}
