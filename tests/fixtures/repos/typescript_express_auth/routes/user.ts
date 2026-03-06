import { Router, Request, Response } from "express";
import { adminOnly, editorOrAbove } from "../middleware";

const router = Router();

router.get("/me", (req: Request, res: Response) => {
  const userId = (req as any).userId;
  res.json({ userId, message: "Authenticated user profile" });
});

router.get("/users", adminOnly(), (_req: Request, res: Response) => {
  res.json({ users: [], message: "Admin-only user list" });
});

router.put("/users/:id", editorOrAbove(), (req: Request, res: Response) => {
  const { id } = req.params;
  res.json({ id, updated: true });
});

export default router;
