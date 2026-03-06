import { Request, Response, NextFunction } from "express";
import jwt from "jsonwebtoken";

interface AuthenticatedRequest extends Request {
  userId?: string;
  roles?: string[];
}

const JWT_SECRET = process.env.JWT_SECRET ?? "";

export function verifyToken(
  req: AuthenticatedRequest,
  res: Response,
  next: NextFunction
): void {
  const header = req.headers.authorization;
  if (!header?.startsWith("Bearer ")) {
    res.status(401).json({ error: "Missing auth token" });
    return;
  }

  const token = header.slice(7);
  const payload = jwt.verify(token, JWT_SECRET) as {
    sub: string;
    roles: string[];
  };
  req.userId = payload.sub;
  req.roles = payload.roles;
  next();
}

export function requireRole(...allowed: string[]) {
  return (req: AuthenticatedRequest, res: Response, next: NextFunction) => {
    const roles = req.roles ?? [];
    if (!allowed.some((role) => roles.includes(role))) {
      res.status(403).json({ error: "Insufficient permissions" });
      return;
    }
    next();
  };
}
