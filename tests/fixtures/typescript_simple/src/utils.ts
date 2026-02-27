import { createHash } from "crypto";

const EMAIL_RE = /^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$/;

export function hashPassword(password: string): string {
  return createHash("sha256").update(password).digest("hex");
}

export function validateEmail(email: string): boolean {
  return EMAIL_RE.test(email);
}
