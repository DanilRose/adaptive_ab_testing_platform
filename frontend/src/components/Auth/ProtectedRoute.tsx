import React from 'react';
import { Navigate } from 'react-router-dom';
import { Loader2 } from 'lucide-react';
import { useAuth } from '@/context/AuthContext';
import type { PermissionKey, UserRole } from '@/types';

interface ProtectedRouteProps {
  children: JSX.Element;
  allowedRoles?: UserRole[];
  allowedPermissions?: PermissionKey[];
}

export const ProtectedRoute: React.FC<ProtectedRouteProps> = ({ children, allowedRoles, allowedPermissions }) => {
  const { loading, isAuthenticated, user } = useAuth();

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background">
        <div className="flex flex-col items-center gap-4">
          <Loader2 className="h-12 w-12 animate-spin text-primary" />
          <p className="text-muted-foreground">Загрузка...</p>
        </div>
      </div>
    );
  }

  if (!isAuthenticated || !user) {
    return <Navigate to="/login" replace />;
  }

  if (allowedRoles && !allowedRoles.includes(user.role)) {
    return <Navigate to="/forbidden" replace />;
  }

  if (allowedPermissions && allowedPermissions.length > 0) {
    const userPermissions = user.permissions || [];
    const hasPermission = allowedPermissions.some((p) => userPermissions.includes(p));
    if (!hasPermission) {
      return <Navigate to="/forbidden" replace />;
    }
  }

  return children;
};
