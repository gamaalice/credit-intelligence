import { spawn } from "node:child_process";
import { setTimeout as delay } from "node:timers/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const packageDirectory = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const projectRoot = resolve(packageDirectory, "../..");
const pythonApi = spawn(
  "uv",
  [
    "run",
    "--project",
    projectRoot,
    "python",
    "-m",
    "uvicorn",
    "credit_intelligence.api:app",
    "--host",
    "127.0.0.1",
    "--port",
    "8000",
  ],
  {
    cwd: projectRoot,
    env: {
      ...process.env,
      PYTHONPATH: projectRoot,
      CREDIT_DATA_DIR:
        process.env.CREDIT_DATA_DIR ?? resolve(projectRoot, "data"),
    },
    stdio: "inherit",
  },
);

let nodeApi;
let shuttingDown = false;

function stopChildren(signal = "SIGTERM") {
  if (shuttingDown) return;
  shuttingDown = true;
  pythonApi.kill(signal);
  nodeApi?.kill(signal);
}

process.on("SIGINT", () => stopChildren("SIGINT"));
process.on("SIGTERM", () => stopChildren("SIGTERM"));
pythonApi.on("error", (error) => {
  console.error("Could not start the FastAPI service:", error);
  stopChildren();
  process.exitCode = 1;
});
pythonApi.on("exit", (code) => {
  if (!shuttingDown) {
    console.error(`FastAPI service exited unexpectedly (${code ?? "signal"}).`);
    stopChildren();
    process.exitCode = code ?? 1;
  }
});

let ready = false;
for (let attempt = 0; attempt < 60 && !shuttingDown; attempt += 1) {
  try {
    const response = await fetch("http://127.0.0.1:8000/openapi.json");
    if (response.ok) {
      ready = true;
      break;
    }
  } catch {
    // The service is still starting; retry for up to 30 seconds.
  }
  await delay(500);
}

if (!ready) {
  console.error("FastAPI did not become ready within 30 seconds.");
  stopChildren();
  process.exit(1);
}

nodeApi = spawn("node", ["--enable-source-maps", "./dist/index.mjs"], {
  cwd: packageDirectory,
  env: process.env,
  stdio: "inherit",
});
nodeApi.on("error", (error) => {
  console.error("Could not start the Express API gateway:", error);
  stopChildren();
  process.exitCode = 1;
});
nodeApi.on("exit", (code) => {
  if (!shuttingDown) {
    stopChildren();
    process.exitCode = code ?? 1;
  }
});
