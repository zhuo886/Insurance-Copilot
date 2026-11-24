import type { AuthSession, AuthTokenPayload } from './types';

const STORAGE_KEY = 'insurance-mall-auth-v2';

export const DEFAULT_SESSION: AuthSession = {
  user: {
    id: 1,
    username: 'demo',
    email: 'demo@example.com',
    displayName: '演示用户',
  },
};

export function toAuthSession(payload: AuthTokenPayload): AuthSession {
  return {
    accessToken: payload.access_token,
    user: {
      id: payload.user.id,
      username: payload.user.username,
      email: payload.user.email,
      displayName: payload.user.display_name ?? payload.user.displayName ?? payload.user.username,
    },
  };
}

export function readSession(): AuthSession | null {
  try {
    const value = window.localStorage.getItem(STORAGE_KEY);
    return value ? (JSON.parse(value) as AuthSession) : null;
  } catch {
    window.localStorage.removeItem(STORAGE_KEY);
    return null;
  }
}

export function saveSession(session: AuthSession): void {
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(session));
}

export function clearSession(): void {
  window.localStorage.removeItem(STORAGE_KEY);
}
