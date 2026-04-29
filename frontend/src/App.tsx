import React from 'react';
import { BrowserRouter as Router, Navigate, Route, Routes } from 'react-router-dom';
import { AppLayout } from '@/components/Layout';
import { ABManager } from './components/ABManager/ABManager';
import { GANManager } from './components/GANManager/GANManager';
import { TemplatesPage } from './components/Templates/TemplatesPage';
import { ResultsSectionPage } from './components/Results/ResultsSectionPage';
import { ResultsOverviewPage } from './components/Results/ResultsOverviewPage';
import { ResultsChartsPage } from './components/Results/ResultsChartsPage';
import { LoginPage } from './components/Auth/LoginPage';
import { ProtectedRoute } from './components/Auth/ProtectedRoute';
import { AdminPage } from './components/Admin/AdminPage';
import { useAuth } from './context/AuthContext';
import { ProfilePage } from './components/Profile/ProfilePage';

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

  const getHomeRoute = (): string => {
    if (!isAuthenticated || !user) return '/login';

    const permissions = user.permissions || [];

    if (
      permissions.includes('Просмотр_результатов_тестов') ||
      permissions.includes('Экспорт_результатов')
    ) {
      return '/results';
    }

    if (
      permissions.includes('AB_тесты_создание') ||
      permissions.includes('AB_тесты_управление')
    ) {
      return '/ab-manager';
    }

    if (
      permissions.includes('GAN_менеджер_обучение') ||
      permissions.includes('GAN_менеджер_генерация_данных')
    ) {
      return '/gan-manager';
    }

    if (
      permissions.includes('Шаблоны_просмотр') ||
      permissions.includes('Шаблоны_создание') ||
      permissions.includes('Шаблоны_редактирование') ||
      permissions.includes('Шаблоны_удаление')
    ) {
      return '/templates';
    }

    if (permissions.includes('Администрирование')) {
      return '/admin';
    }

    return '/profile';
  };

  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route path="/forbidden" element={<ForbiddenPage />} />

      <Route
        path="/"
        element={<Navigate to={getHomeRoute()} replace />}
      />

      <Route
        path="/ab-manager"
        element={
          <ProtectedRoute allowedPermissions={['AB_тесты_создание', 'AB_тесты_управление']}>
            <AppLayout>
              <ABManager />
            </AppLayout>
          </ProtectedRoute>
        }
      />

      <Route
        path="/gan-manager"
        element={
          <ProtectedRoute allowedPermissions={['GAN_менеджер_обучение', 'GAN_менеджер_генерация_данных']}>
            <AppLayout>
              <GANManager />
            </AppLayout>
          </ProtectedRoute>
        }
      />

      <Route
        path="/results"
        element={
          <ProtectedRoute allowedPermissions={['Просмотр_результатов_тестов', 'Экспорт_результатов']}>
            <AppLayout>
              <ResultsSectionPage />
            </AppLayout>
          </ProtectedRoute>
        }
      >
        <Route index element={<Navigate to="overview" replace />} />
        <Route path="overview" element={<ResultsOverviewPage />} />
        <Route path="charts" element={<ResultsChartsPage />} />
      </Route>

      <Route
        path="/templates"
        element={
          <ProtectedRoute allowedPermissions={['Шаблоны_просмотр', 'Шаблоны_создание', 'Шаблоны_редактирование', 'Шаблоны_удаление']}>
            <AppLayout>
              <TemplatesPage />
            </AppLayout>
          </ProtectedRoute>
        }
      />

      <Route
        path="/admin"
        element={
          <ProtectedRoute allowedPermissions={['Администрирование']}>
            <AppLayout>
              <AdminPage />
            </AppLayout>
          </ProtectedRoute>
        }
      />

      <Route
        path="/profile"
        element={
          <ProtectedRoute>
            <AppLayout>
              <ProfilePage />
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
