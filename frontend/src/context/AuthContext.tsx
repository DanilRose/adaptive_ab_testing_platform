import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from 'react';
import type {
  AuthTokenResponse,
  AuthUser,
  LoginCredentials,
  PermissionKey,
  UserRole,
} from '../types';
import { authAPI } from '../utils/api';

interface AuthContextValue {
  user: AuthUser | null;
  token: string | null;
  loading: boolean;
  isAuthenticated: boolean;
  login: (credentials: LoginCredentials) => Promise<void>;
  logout: () => Promise<void>;
  refreshUser: () => Promise<void>;
  updateProfile: (payload: { full_name: string; email?: string | null; phone?: string | null; avatar_url?: string | null }) => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

const clearAuthStorage = (): void => {
  localStorage.removeItem('access_token');
  localStorage.removeItem('user_role');
  localStorage.removeItem('user_full_name');
  localStorage.removeItem('username');
};

interface MeResponse {
  id: number;
  username: string;
  role: UserRole;
  full_name: string;
  job_title?: 'developer' | 'analyst' | 'project_manager' | 'other';
  permissions?: PermissionKey[];
  email?: string | null;
  phone?: string | null;
  avatar_url?: string | null;
}

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [token, setToken] = useState<string | null>(localStorage.getItem('access_token'));
  const [loading, setLoading] = useState<boolean>(true);

  const resetSession = useCallback(() => {
    clearAuthStorage();
    setUser(null);
    setToken(null);
  }, []);

  const login = useCallback(async (credentials: LoginCredentials) => {
    const response = await authAPI.login(credentials);
    const data = response.data as AuthTokenResponse;

    localStorage.setItem('access_token', data.access_token);
    localStorage.setItem('user_role', data.role);
    localStorage.setItem('user_full_name', data.full_name);
    localStorage.setItem('username', credentials.username);

    setToken(data.access_token);

    const meResponse = await authAPI.me();
    const meData = meResponse.data as MeResponse;

    const currentUser: AuthUser = {
      id: meData.id,
      username: meData.username,
      role: meData.role,
      full_name: meData.full_name,
      job_title: meData.job_title,
      permissions: meData.permissions || [],
    };

    localStorage.setItem('user_role', currentUser.role);
    localStorage.setItem('user_full_name', currentUser.full_name);
    localStorage.setItem('username', currentUser.username);

    setUser(currentUser);
  }, []);

  const refreshUser = useCallback(async () => {
    const meResponse = await authAPI.me();
    const meData = meResponse.data as MeResponse;

    const restoredUser: AuthUser = {
      id: meData.id,
      username: meData.username,
      role: meData.role,
      full_name: meData.full_name,
      job_title: meData.job_title,
      permissions: meData.permissions || [],
      email: meData.email,
      phone: meData.phone,
      avatar_url: meData.avatar_url,
    };

    localStorage.setItem('user_role', restoredUser.role);
    localStorage.setItem('user_full_name', restoredUser.full_name);
    localStorage.setItem('username', restoredUser.username);

    setUser(restoredUser);
  }, []);

  const updateProfile = useCallback(async (payload: { full_name: string; email?: string | null; phone?: string | null; avatar_url?: string | null }) => {
    const response = await authAPI.updateProfile(payload);
    const updated = response.data as MeResponse;

    const currentUser: AuthUser = {
      id: updated.id,
      username: updated.username,
      role: updated.role,
      full_name: updated.full_name,
      job_title: updated.job_title,
      permissions: updated.permissions || [],
      email: updated.email,
      phone: updated.phone,
      avatar_url: updated.avatar_url,
    };

    localStorage.setItem('user_role', currentUser.role);
    localStorage.setItem('user_full_name', currentUser.full_name);
    localStorage.setItem('username', currentUser.username);

    setUser(currentUser);
  }, []);

  const logout = useCallback(async () => {
    try {
      await authAPI.logout();
    } catch {
      // Ошибка logout на backend не блокирует локальный выход
    }

    resetSession();
    if (window.location.pathname !== '/login') {
      window.location.href = '/login';
    }
  }, [resetSession]);

  useEffect(() => {
    const initializeAuth = async (): Promise<void> => {
      const storedToken = localStorage.getItem('access_token');

      if (!storedToken) {
        setLoading(false);
        return;
      }

      setToken(storedToken);

      try {
        await refreshUser();
      } catch {
        resetSession();
      } finally {
        setLoading(false);
      }
    };

    void initializeAuth();
  }, [resetSession, refreshUser]);

  const value = useMemo<AuthContextValue>(
    () => ({
      user,
      token,
      loading,
      isAuthenticated: Boolean(token && user),
      login,
      logout,
      refreshUser,
      updateProfile,
    }),
    [user, token, loading, login, logout, refreshUser, updateProfile]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export const useAuth = (): AuthContextValue => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
