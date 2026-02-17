import { buildApp } from "./app.js";
import { config } from "./lib/config.js";

const app = buildApp();

app.listen({ port: config.port, host: config.host }, (err) => {
  if (err) {
    app.log.error(err);
    process.exit(1);
  }
});
