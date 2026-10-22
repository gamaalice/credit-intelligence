import { Router, type IRouter } from "express";
import healthRouter from "./health";
import creditRouter from "./credit";

const router: IRouter = Router();

router.use(healthRouter);
router.use(creditRouter);

export default router;
