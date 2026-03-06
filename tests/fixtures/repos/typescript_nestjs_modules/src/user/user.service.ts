import { Injectable } from "@nestjs/common";

interface User {
  id: string;
  email: string;
  name: string;
}

@Injectable()
export class UserService {
  private readonly users = new Map<string, User>();

  async findById(id: string): Promise<User | undefined> {
    return this.users.get(id);
  }

  async findByEmail(email: string): Promise<User | undefined> {
    for (const user of this.users.values()) {
      if (user.email === email) return user;
    }
    return undefined;
  }

  async create(email: string, name: string): Promise<User> {
    const id = crypto.randomUUID();
    const user: User = { id, email, name };
    this.users.set(id, user);
    return user;
  }

  async delete(id: string): Promise<boolean> {
    return this.users.delete(id);
  }
}
