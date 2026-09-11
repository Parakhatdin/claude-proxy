#!/usr/bin/env node
import { config, log } from "./config.ts";
import { createProxyServer } from "./server.ts";

const server = createProxyServer();

server.on("error", (err: NodeJS.ErrnoException) => {
  if (err.code === "EADDRINUSE") {
    log("error", `${config.host}:${config.port} is already in use — claude-proxy may already be running.`);
    log("error", `Stop it with:  lsof -ti :${config.port} | xargs kill`);
    log("error", `Or pick another port:  CLAUDE_PROXY_PORT=8788 node dist/index.js`);
  } else if (err.code === "EACCES") {
    log("error", `Not permitted to bind ${config.host}:${config.port}. Use a port above 1024.`);
  } else {
    log("error", `Server error: ${err.message}`);
  }
  process.exit(1);
});

server.listen(config.port, config.host, () => {
  log("info", `claude-proxy listening on http://${config.host}:${config.port}`);
  log("info", `mode=${config.mode} model=${config.defaultModel} cwd=${config.cwd}`);
  if (!config.apiKey) {
    log(
      "warn",
      "CLAUDE_PROXY_API_KEY is unset: any process that can reach this port can spend your Claude quota.",
    );
  }
  if (config.host !== "127.0.0.1" && config.host !== "localhost" && !config.apiKey) {
    log("error", "Listening on a non-loopback address without an API key. Set CLAUDE_PROXY_API_KEY.");
  }
});

for (const signal of ["SIGINT", "SIGTERM"] as const) {
  process.on(signal, () => {
    log("info", `received ${signal}, shutting down`);
    server.close(() => process.exit(0));
    setTimeout(() => process.exit(0), 5_000).unref();
  });
}
