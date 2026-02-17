export const config = {
  port: Number(process.env.API_PORT) || 3001,
  host: process.env.API_HOST || "0.0.0.0",
};
