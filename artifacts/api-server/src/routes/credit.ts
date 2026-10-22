import { Router, type IRouter } from "express";
import { logger } from "../lib/logger";

const router: IRouter = Router();

router.use("/credit", async (req, res) => {
  const path = req.originalUrl.replace(/^\/api/, "");
  const headers: Record<string, string> = {
    accept: req.header("accept") ?? "application/json",
  };

  if (req.body !== undefined && !["GET", "HEAD"].includes(req.method)) {
    headers["content-type"] =
      req.header("content-type") ?? "application/json";
  }

  try {
    const response = await fetch(`http://127.0.0.1:8000${path}`, {
      method: req.method,
      headers,
      body:
        req.body !== undefined && !["GET", "HEAD"].includes(req.method)
          ? JSON.stringify(req.body)
          : undefined,
    });
    const responseBody = await response.text();
    res.status(response.status);
    res.type(response.headers.get("content-type") ?? "application/json");
    res.send(responseBody);
  } catch (error) {
    logger.warn({ error, path }, "Credit Intelligence API is unavailable");
    res.status(503).json({
      detail:
        "The model API is not available. Start the API Server workflow and train the model.",
    });
  }
});

export default router;
