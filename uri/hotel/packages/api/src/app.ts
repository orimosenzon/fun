import Fastify from "fastify";
import cors from "@fastify/cors";
import { errorHandler } from "./middleware/error-handler.js";
import { roomRoutes } from "./routes/rooms.js";
import { guestRoutes } from "./routes/guests.js";
import { reservationRoutes } from "./routes/reservations.js";
import { dashboardRoutes } from "./routes/dashboard.js";

export function buildApp() {
  const app = Fastify({ logger: true });

  app.register(cors, { origin: true });
  app.setErrorHandler(errorHandler);

  app.register(roomRoutes, { prefix: "/api/v1" });
  app.register(guestRoutes, { prefix: "/api/v1" });
  app.register(reservationRoutes, { prefix: "/api/v1" });
  app.register(dashboardRoutes, { prefix: "/api/v1" });

  app.get("/api/health", async () => ({ status: "ok" }));

  return app;
}
