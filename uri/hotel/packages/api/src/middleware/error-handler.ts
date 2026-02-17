import type { FastifyError, FastifyReply, FastifyRequest } from "fastify";
import { ZodError } from "zod";

export function errorHandler(error: FastifyError, _request: FastifyRequest, reply: FastifyReply) {
  if (error instanceof ZodError) {
    return reply.status(400).send({
      error: "Validation Error",
      details: error.errors,
    });
  }

  const statusCode = error.statusCode ?? 500;
  reply.status(statusCode).send({
    error: error.message || "Internal Server Error",
  });
}
