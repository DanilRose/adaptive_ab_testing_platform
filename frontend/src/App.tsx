// frontend/src/App.tsx
import React, { useMemo } from 'react';
import { BrowserRouter as Router, Link, Navigate, Route, Routes, useLocation } from 'react-router-dom';
import { Button, Layout, Menu, Space, Tag, Typography, theme } from 'antd';
import type { UserRole } from './types';
import { Dashboard } from './components/Dashboard/Dashboard';
import { TestCreator } from './components/TestCreator/TestCreator';
import { GANManager } from './components/GANManager/GANManager';
import { TemplatesPage } from './components/Templates/TemplatesPage';
import { LoginPage } from './components/Auth/LoginPage';
import { ProtectedRoute } from './components/Auth/ProtectedRoute';
import { useAuth } from './context/AuthContext';

const { Header, Content, Sider } = Layout;

const roleLabel: Record<UserRole, string> = {
  developer: 'developer',
  analyst: 'analyst',
  manager: 'manager',
};

interface MenuRouteItem {
  key: string;
  path: string;
  label: string;
  allowedRoles: UserRole[];
}

const menuRoutes: MenuRouteItem[] = [
  { key: '/', path: '/', label: 'Дашборд', allowedRoles: ['developer'] },
  { key: '/create-test', path: '/create-test', label: 'Создать тест', allowedRoles: ['developer', 'analyst'] },
  { key: '/gan-manager', path: '/gan-manager', label: 'GAN Менеджер', allowedRoles: ['developer', 'analyst'] },
  { key: '/results', path: '/results', label: 'Результаты', allowedRoles: ['developer', 'manager'] },
  { key: '/templates', path: '/templates', label: 'Шаблоны', allowedRoles: ['developer', 'analyst'] },
];

const ForbiddenPage: React.FC = () => (
  <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
    <Typography.Title level={3} style={{ margin: 0 }}>
      Недостаточно прав
    </Typography.Title>
  </div>
);

const AuthorizedLayout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { token: designToken } = theme.useToken();
  const location = useLocation();
  const { user, logout } = useAuth();

  const visibleMenuItems = useMemo(() => {
    if (!user) {
      return [];
    }

    return menuRoutes
      .filter((item) => item.allowedRoles.includes(user.role))
      .map((item) => ({
        key: item.key,
        label: <Link to={item.path}>{item.label}</Link>,
      }));
  }, [user]);

  const selectedKey = useMemo(() => {
    const directMatch = menuRoutes.find((item) => item.path === location.pathname);
    if (directMatch) {
      return [directMatch.key];
    }

    const partialMatch = menuRoutes.find((item) => location.pathname.startsWith(item.path) && item.path !== '/');
    return partialMatch ? [partialMatch.key] : [];
  }, [location.pathname]);

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider collapsible>
        <div
          style={{
            height: 32,
            margin: 16,
            background: 'rgba(255, 255, 255, 0.2)',
            borderRadius: 6,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'white',
            fontWeight: 'bold',
          }}
        >
          A/B Platform
        </div>

        <Menu theme="dark" mode="inline" items={visibleMenuItems} selectedKeys={selectedKey} />
      </Sider>

      <Layout>
        <Header
          style={{
            padding: '0 16px',
            background: designToken.colorBgContainer,
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}
        >
          <div />
          {user ? (
            <Space>
              <Typography.Text strong>{user.full_name}</Typography.Text>
              <Tag color="blue">{roleLabel[user.role]}</Tag>
              <Button onClick={() => void logout()}>Выйти</Button>
            </Space>
          ) : null}
        </Header>

        <Content style={{ margin: '16px' }}>{children}</Content>
      </Layout>
    </Layout>
  );
};

const AppRoutes: React.FC = () => {
  const { isAuthenticated, user } = useAuth();

  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route path="/forbidden" element={<ForbiddenPage />} />

      <Route
        path="/"
        element={
          <ProtectedRoute allowedRoles={['developer']}>
            <AuthorizedLayout>
              <Dashboard />
            </AuthorizedLayout>
          </ProtectedRoute>
        }
      />

      <Route
        path="/create-test"
        element={
          <ProtectedRoute allowedRoles={['developer', 'analyst']}>
            <AuthorizedLayout>
              <TestCreator />
            </AuthorizedLayout>
          </ProtectedRoute>
        }
      />

      <Route
        path="/gan-manager"
        element={
          <ProtectedRoute allowedRoles={['developer', 'analyst']}>
            <AuthorizedLayout>
              <GANManager />
            </AuthorizedLayout>
          </ProtectedRoute>
        }
      />

      <Route
        path="/results"
        element={
          <ProtectedRoute allowedRoles={['developer', 'manager']}>
            <AuthorizedLayout>
              <div>Results Page</div>
            </AuthorizedLayout>
          </ProtectedRoute>
        }
      />

      <Route
        path="/templates"
        element={
          <ProtectedRoute allowedRoles={['developer', 'analyst']}>
            <AuthorizedLayout>
              <TemplatesPage />
            </AuthorizedLayout>
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
