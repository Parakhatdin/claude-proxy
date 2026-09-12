#!/usr/bin/env node
/**
 * Config is loaded dynamically so that a bad environment — an unreadable token file, a
 * non-numeric port — exits with the one line that explains it, rather than with a
 * module-load stack trace an operator has to read through.
 */
let configModule: typeof import("./config.ts");
let serverModule: typeof import("./server.ts");
try {
  configModule = await import("./config.ts");
  serverModule = await import("./server.ts");
} catch (err) {
  console.error(`claude-proxy: ${err instanceof Error ? err.message : String(err)}`);
  process.exit(1);
}

const { config, log } = configModule;
const server = serverModule.createProxyServer();

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
  log("info", `CLI credential: ${config.authSource}`);
  if (config.oauthToken !== undefined && !config.oauthToken.startsWith("sk-ant-oat")) {
    log(
      "warn",
      "The configured token does not look like a `claude setup-token` credential (expected sk-ant-oat…).",
    );
  }
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
