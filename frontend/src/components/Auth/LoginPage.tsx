import React, { useEffect, useState } from 'react';
import { Alert, Button, Card, Form, Input, Space, Typography } from 'antd';
import { Navigate, useNavigate } from 'react-router-dom';
import type { LoginCredentials, UserRole } from '../../types';
import { useAuth } from '../../context/AuthContext';

const roleHomeRoute: Record<UserRole, string> = {
  developer: '/',
  analyst: '/create-test',
  manager: '/results',
};

export const LoginPage: React.FC = () => {
  const navigate = useNavigate();
  const { login, isAuthenticated, loading, user } = useAuth();
  const [submitting, setSubmitting] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    if (isAuthenticated && user) {
      navigate(roleHomeRoute[user.role], { replace: true });
    }
  }, [isAuthenticated, user, navigate]);

  const handleFinish = async (values: LoginCredentials): Promise<void> => {
    setSubmitting(true);
    setErrorMessage(null);

    try {
      await login(values);
      const role = localStorage.getItem('user_role') as UserRole | null;
      if (role && roleHomeRoute[role]) {
        navigate(roleHomeRoute[role], { replace: true });
      } else {
        navigate('/', { replace: true });
      }
    } catch (error) {
      setErrorMessage('Неверный логин или пароль. Повторите попытку.');
    } finally {
      setSubmitting(false);
    }
  };

  if (!loading && isAuthenticated && user) {
    return <Navigate to={roleHomeRoute[user.role]} replace />;
  }

  return (
    <div
      style={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 24,
        background: '#f5f5f5',
      }}
    >
      <Card style={{ width: 420 }}>
        <Space direction="vertical" size={16} style={{ width: '100%' }}>
          <Typography.Title level={3} style={{ marginBottom: 0 }}>
            Вход в платформу
          </Typography.Title>

          {errorMessage && <Alert type="error" showIcon message={errorMessage} />}

          <Form<LoginCredentials> layout="vertical" onFinish={handleFinish} requiredMark={false}>
            <Form.Item
              label="Логин"
              name="username"
              rules={[{ required: true, message: 'Введите логин' }]}
            >
              <Input placeholder="Введите username" autoComplete="username" />
            </Form.Item>

            <Form.Item
              label="Пароль"
              name="password"
              rules={[{ required: true, message: 'Введите пароль' }]}
            >
              <Input.Password placeholder="Введите пароль" autoComplete="current-password" />
            </Form.Item>

            <Form.Item style={{ marginBottom: 0 }}>
              <Button type="primary" htmlType="submit" loading={submitting} block>
                Войти
              </Button>
            </Form.Item>
          </Form>

          <Card size="small" title="Тестовые учётные записи">
            <Typography.Paragraph style={{ marginBottom: 6 }}>
              developer / dev123
            </Typography.Paragraph>
            <Typography.Paragraph style={{ marginBottom: 6 }}>
              analyst / analyst123
            </Typography.Paragraph>
            <Typography.Paragraph style={{ marginBottom: 0 }}>
              manager / manager123
            </Typography.Paragraph>
          </Card>
        </Space>
      </Card>
    </div>
  );
};
