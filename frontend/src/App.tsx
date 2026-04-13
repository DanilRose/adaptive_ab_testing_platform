import React from 'react';
import { BrowserRouter as Router, Navigate, Route, Routes } from 'react-router-dom';
import { AppLayout } from '@/components/Layout';
import { ABManager } from './components/ABManager/ABManager';
import { GANManager } from './components/GANManager/GANManager';
import { TemplatesPage } from './components/Templates/TemplatesPage';
import { ResultsPage } from './components/Results/ResultsPage';
import { LoginPage } from './components/Auth/LoginPage';
import { ProtectedRoute } from './components/Auth/ProtectedRoute';
import { useAuth } from './context/AuthContext';

const ForbiddenPage: React.FC = () => (
  <div className="flex min-h-screen items-center justify-center bg-background">
    <div className="text-center">
      <h1 className="text-4xl font-bold text-foreground">Доступ запрещён</h1>
      <p className="mt-2 text-muted-foreground">
        У вас недостаточно прав для просмотра этой страницы
      </p>
    </div>
  </div>
);

const AppRoutes: React.FC = () => {
  const { isAuthenticated, user } = useAuth();

  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route path="/forbidden" element={<ForbiddenPage />} />

      <Route
        path="/"
        element={<Navigate to="/ab-manager" replace />}
      />

      <Route
        path="/ab-manager"
        element={
          <ProtectedRoute allowedRoles={['developer', 'analyst']}>
            <AppLayout>
              <ABManager />
            </AppLayout>
          </ProtectedRoute>
        }
      />

      <Route
        path="/gan-manager"
        element={
          <ProtectedRoute allowedRoles={['developer', 'analyst']}>
            <AppLayout>
              <GANManager />
            </AppLayout>
          </ProtectedRoute>
        }
      />

      <Route
        path="/results"
        element={
          <ProtectedRoute allowedRoles={['developer', 'manager']}>
            <AppLayout>
              <ResultsPage />
            </AppLayout>
          </ProtectedRoute>
        }
      />

      <Route
        path="/templates"
        element={
          <ProtectedRoute allowedRoles={['developer', 'analyst']}>
            <AppLayout>
              <TemplatesPage />
            </AppLayout>
          </ProtectedRoute>
        }
      />

      <Route
        path="*"
        element={
          isAuthenticated && user ? (
            <Navigate to="/" replace />
          ) : (
            <Navigate to="/login" replace />
          )
        }
      />
    </Routes>
  );
};

const App: React.FC = () => {
  return (
    <Router>
      <AppRoutes />
    </Router>
  );
};

export default App;
