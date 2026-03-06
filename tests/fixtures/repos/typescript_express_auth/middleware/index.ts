import { Router } from "express";
import { verifyToken, requireRole } from "./auth";

export function applyAuthMiddleware(router: Router): void {
  router.use(verifyToken);
}

export function adminOnly() {
  return requireRole("admin");
}

export function editorOrAbove() {
  return requireRole("admin", "editor");
}

export { verifyToken, requireRole };
