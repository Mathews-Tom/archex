import { useState, useCallback, useEffect } from "react";
import { useLocalStorage } from "./useLocalStorage";

interface AuthState {
  token: string | null;
  userId: string | null;
  isAuthenticated: boolean;
}

export function useAuth() {
  const [storedToken, setStoredToken] = useLocalStorage<string | null>(
    "auth_token",
    null
  );
  const [authState, setAuthState] = useState<AuthState>({
    token: storedToken,
    userId: null,
    isAuthenticated: false,
  });

  useEffect(() => {
    if (storedToken) {
      const payload = JSON.parse(atob(storedToken.split(".")[1]));
      setAuthState({
        token: storedToken,
        userId: payload.sub,
        isAuthenticated: true,
      });
    }
  }, [storedToken]);

  const login = useCallback(
    async (email: string, password: string) => {
      const res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });
      const { token } = await res.json();
      setStoredToken(token);
    },
    [setStoredToken]
  );

  const logout = useCallback(() => {
    setStoredToken(null);
    setAuthState({ token: null, userId: null, isAuthenticated: false });
  }, [setStoredToken]);

  return { ...authState, login, logout };
}
