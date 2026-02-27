import { type AuthResult, type User } from "../types.js";
import { hashPassword, validateEmail } from "../utils.js";

const sessions = new Map<string, User>();

export function handleLogin(user: User, password: string): AuthResult {
  if (!validateEmail(user.email)) {
    return { success: false, error: "Invalid email address" };
  }
  const token = hashPassword(`${user.id}:${password}`);
  sessions.set(token, user);
  return { success: true, token, user };
}

export function handleLogout(token: string): void {
  sessions.delete(token);
}
