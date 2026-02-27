export enum Role {
  Admin = "admin",
  User = "user",
  Guest = "guest",
}

export interface User {
  id: number;
  name: string;
  email: string;
  role: Role;
}

export type AuthResult =
  | { success: true; token: string; user: User }
  | { success: false; error: string };
