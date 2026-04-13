import React, { useEffect, useState } from 'react';
import { Navigate, useNavigate } from 'react-router-dom';
import { Moon, Sun, BrainCircuit, ArrowRight } from 'lucide-react';
import { useAuth } from '@/context/AuthContext';
import { useTheme } from '@/context/ThemeContext';
import type { LoginCredentials, UserRole } from '@/types';

const roleHomeRoute: Record<UserRole, string> = {
  developer: '/ab-manager',
  analyst: '/ab-manager',
  manager: '/results',
};

const testAccounts: { username: string; password: string; role: string; roleKey: UserRole }[] = [
  { username: 'developer', password: 'dev123',      role: 'Developer', roleKey: 'developer' },
  { username: 'analyst',   password: 'analyst123',  role: 'Analyst',   roleKey: 'analyst'   },
  { username: 'manager',   password: 'manager123',  role: 'Manager',   roleKey: 'manager'   },
];

const roleColors: Record<UserRole, { text: string; bg: string; textDark: string; bgDark: string }> = {
  developer: { text: '#92400e', bg: '#fef3c7', textDark: '#fcd34d', bgDark: 'rgba(252,211,77,0.15)' },
  analyst:   { text: '#065f46', bg: '#d1fae5', textDark: '#6ee7b7', bgDark: 'rgba(110,231,183,0.15)' },
  manager:   { text: '#78350f', bg: '#fde68a', textDark: '#fbbf24', bgDark: 'rgba(251,191,36,0.15)' },
};

export const LoginPage: React.FC = () => {
  const navigate = useNavigate();
  const { login, isAuthenticated, loading, user } = useAuth();
  const { theme, toggleTheme } = useTheme();
  const [submitting, setSubmitting] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [credentials, setCredentials] = useState<LoginCredentials>({ username: '', password: '' });
  const [focusedField, setFocusedField] = useState<string | null>(null);

  useEffect(() => {
    if (isAuthenticated && user) {
      navigate(roleHomeRoute[user.role], { replace: true });
    }
  }, [isAuthenticated, user, navigate]);

  const handleFinish = async (e: React.FormEvent): Promise<void> => {
    e.preventDefault();
    setSubmitting(true);
    setErrorMessage(null);
    try {
      await login(credentials);
      const role = localStorage.getItem('user_role') as UserRole | null;
      navigate(role && roleHomeRoute[role] ? roleHomeRoute[role] : '/', { replace: true });
    } catch {
      setErrorMessage('Неверный логин или пароль. Повторите попытку.');
    } finally {
      setSubmitting(false);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>): void => {
    const { name, value } = e.target;
    setCredentials((prev) => ({ ...prev, [name]: value }));
    if (errorMessage) setErrorMessage(null);
  };

  if (!loading && isAuthenticated && user) {
    return <Navigate to={roleHomeRoute[user.role]} replace />;
  }

  const isDark = theme === 'dark';

  const c = {
    pageBg:      isDark ? '#0f0d0b' : '#faf8f5',
    panelBg:     isDark ? '#1c1917' : '#ffffff',
    sideBg:      isDark ? '#171412' : '#f5f0e8',
    border:      isDark ? '#292524' : '#e7e5e4',
    textPrimary: isDark ? '#fafaf9' : '#1c1917',
    textMuted:   isDark ? '#78716c' : '#a8a29e',
    textSub:     isDark ? '#57534e' : '#c4b8a8',
    inputBg:     isDark ? '#292524' : '#fafaf9',
    inputBorder: isDark ? '#3c3330' : '#e7e5e4',
    inputFocus:  isDark ? '#d97706' : '#d97706',
    btnBg:       '#d97706',
    btnHov:      '#b45309',
    btnText:     '#ffffff',
    accentLight: isDark ? 'rgba(217,119,6,0.15)' : '#fef3c7',
    accentText:  isDark ? '#fcd34d' : '#92400e',
    // Claude signature warm cream tones
    cardShadow:  isDark ? '0 8px 32px rgba(0,0,0,0.4)' : '0 8px 32px rgba(28,25,23,0.08)',
  };

  const canSubmit = !submitting && credentials.username && credentials.password;

  return (
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      backgroundColor: c.pageBg,
      fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    }}>
      {/* Left decorative panel */}
      <div style={{
        display: 'none',
        width: '420px',
        flexShrink: 0,
        backgroundColor: c.sideBg,
        borderRight: `1px solid ${c.border}`,
        padding: '48px',
        flexDirection: 'column',
        justifyContent: 'space-between',
      }}
        className="login-side-panel"
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{
            width: '32px', height: '32px', borderRadius: '8px',
            backgroundColor: c.btnBg,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <BrainCircuit size={18} color="#fff" />
          </div>
          <span style={{ fontSize: '15px', fontWeight: '600', color: c.textPrimary, letterSpacing: '-0.3px' }}>
            A/B Platform
          </span>
        </div>

        <div>
          <p style={{ fontSize: '28px', fontWeight: '300', color: c.textPrimary, lineHeight: 1.4, letterSpacing: '-0.5px', margin: '0 0 16px' }}>
            Умная платформа<br />для A/B тестирования
          </p>
          <p style={{ fontSize: '14px', color: c.textMuted, lineHeight: 1.6, margin: 0 }}>
            Анализируйте данные, управляйте экспериментами и принимайте решения на основе статистики.
          </p>
        </div>

        <p style={{ fontSize: '12px', color: c.textSub, margin: 0 }}>SSTU © 2026</p>
      </div>

      {/* Main login area */}
      <div style={{
        flex: 1,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '24px',
        position: 'relative',
      }}>
        {/* Theme toggle */}
        <button
          onClick={toggleTheme}
          style={{
            position: 'absolute', top: '20px', right: '20px',
            width: '36px', height: '36px', borderRadius: '7px',
            border: `1px solid ${c.border}`,
            backgroundColor: c.panelBg,
            cursor: 'pointer',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            color: c.textMuted,
            transition: 'border-color 0.15s',
          }}
          title={isDark ? 'Светлая тема' : 'Тёмная тема'}
        >
          {isDark ? <Sun size={15} /> : <Moon size={15} />}
        </button>

        <div style={{ width: '100%', maxWidth: '380px' }}>
          {/* Logo mark */}
          <div style={{ marginBottom: '36px' }}>
            {/* <div style={{
              width: '40px', height: '40px', borderRadius: '10px',
              backgroundColor: c.btnBg,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              marginBottom: '20px',
            }}>
              <BrainCircuit size={22} color="#fff" />
            </div> */}
            <h1 style={{
              fontSize: '28px', fontWeight: '700',
              color: c.textPrimary, margin: '0 0 8px',
              letterSpacing: '-0.5px',
            }}>
              Добро пожаловать
            </h1>
            <p style={{ fontSize: '16px', color: c.textMuted, margin: 0 }}>
              Войдите чтобы продолжить
            </p>
          </div>

          {/* Error */}
          {errorMessage && (
            <div style={{
              padding: '11px 14px',
              borderRadius: '7px',
              backgroundColor: isDark ? 'rgba(239,68,68,0.1)' : '#fef2f2',
              border: `1px solid ${isDark ? 'rgba(239,68,68,0.25)' : '#fecaca'}`,
              color: isDark ? '#fca5a5' : '#dc2626',
              fontSize: '13px',
              marginBottom: '20px',
            }}>
              {errorMessage}
            </div>
          )}

          {/* Form */}
          <form onSubmit={handleFinish} style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
            <div>
              <label style={{
                display: 'block', fontSize: '14px', fontWeight: '600',
                color: focusedField === 'username' ? c.inputFocus : c.textMuted,
                marginBottom: '7px', letterSpacing: '0.1px',
                transition: 'color 0.15s',
              }}>
                Логин
              </label>
              <input
                name="username" type="text"
                placeholder="Введите username"
                autoComplete="username"
                value={credentials.username}
                onChange={handleInputChange}
                onFocus={() => setFocusedField('username')}
                onBlur={() => setFocusedField(null)}
                disabled={submitting}
                style={{
                  width: '100%', height: '46px', padding: '0 14px',
                  borderRadius: '8px',
                  border: `1.5px solid ${focusedField === 'username' ? c.inputFocus : c.inputBorder}`,
                  backgroundColor: c.inputBg,
                  color: c.textPrimary,
                  fontSize: '15px', outline: 'none',
                  transition: 'border-color 0.15s',
                  boxSizing: 'border-box',
                  boxShadow: focusedField === 'username' ? `0 0 0 3px rgba(217,119,6,0.12)` : 'none',
                }}
              />
            </div>

            <div>
              <label style={{
                display: 'block', fontSize: '14px', fontWeight: '600',
                color: focusedField === 'password' ? c.inputFocus : c.textMuted,
                marginBottom: '7px', letterSpacing: '0.1px',
                transition: 'color 0.15s',
              }}>
                Пароль
              </label>
              <input
                name="password" type="password"
                placeholder="Введите пароль"
                autoComplete="current-password"
                value={credentials.password}
                onChange={handleInputChange}
                onFocus={() => setFocusedField('password')}
                onBlur={() => setFocusedField(null)}
                disabled={submitting}
                style={{
                  width: '100%', height: '46px', padding: '0 14px',
                  borderRadius: '8px',
                  border: `1.5px solid ${focusedField === 'password' ? c.inputFocus : c.inputBorder}`,
                  backgroundColor: c.inputBg,
                  color: c.textPrimary,
                  fontSize: '15px', outline: 'none',
                  transition: 'border-color 0.15s',
                  boxSizing: 'border-box',
                  boxShadow: focusedField === 'password' ? `0 0 0 3px rgba(217,119,6,0.12)` : 'none',
                }}
              />
            </div>

            <button
              type="submit"
              disabled={!canSubmit}
              style={{
                marginTop: '6px',
                width: '100%', height: '46px',
                borderRadius: '8px', border: 'none',
                backgroundColor: canSubmit ? c.btnBg : (isDark ? '#292524' : '#e7e5e4'),
                color: canSubmit ? c.btnText : c.textMuted,
                fontSize: '16px', fontWeight: '600',
                cursor: canSubmit ? 'pointer' : 'not-allowed',
                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '7px',
                transition: 'background-color 0.15s',
                letterSpacing: '-0.1px',
              }}
              onMouseEnter={e => { if (canSubmit) e.currentTarget.style.backgroundColor = c.btnHov; }}
              onMouseLeave={e => { if (canSubmit) e.currentTarget.style.backgroundColor = c.btnBg; }}
            >
              {submitting ? (
                <>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" style={{ animation: 'spin 0.8s linear infinite' }}>
                    <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" strokeOpacity="0.3" />
                    <path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
                  </svg>
                  Вход...
                </>
              ) : (
                <>Войти <ArrowRight size={14} /></>
              )}
            </button>
          </form>

          {/* Test accounts */}
          <div style={{ marginTop: '28px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '12px' }}>
              <div style={{ flex: 1, height: '1px', backgroundColor: c.border }} />
              <span style={{ fontSize: '11px', color: c.textSub, whiteSpace: 'nowrap', letterSpacing: '0.3px' }}>
                ТЕСТОВЫЕ АККАУНТЫ
              </span>
              <div style={{ flex: 1, height: '1px', backgroundColor: c.border }} />
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              {testAccounts.map((account) => {
                const rc = isDark
                  ? roleColors[account.roleKey]
                  : roleColors[account.roleKey];
                const textC = isDark ? rc.textDark : rc.text;
                const bgC = isDark ? rc.bgDark : rc.bg;

                return (
                  <button
                    key={account.username}
                    onClick={() => setCredentials({ username: account.username, password: account.password })}
                    style={{
                      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                      padding: '9px 12px',
                      borderRadius: '7px',
                      border: `1px solid ${c.border}`,
                      backgroundColor: c.panelBg,
                      cursor: 'pointer',
                      transition: 'border-color 0.15s, background-color 0.15s',
                      textAlign: 'left', width: '100%',
                    }}
                    onMouseEnter={e => {
                      e.currentTarget.style.borderColor = isDark ? '#3c3330' : '#d6d0c8';
                      e.currentTarget.style.backgroundColor = isDark ? '#292524' : '#faf8f5';
                    }}
                    onMouseLeave={e => {
                      e.currentTarget.style.borderColor = c.border;
                      e.currentTarget.style.backgroundColor = c.panelBg;
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                      <div style={{
                        width: '26px', height: '26px', borderRadius: '6px',
                        backgroundColor: bgC,
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        fontSize: '11px', fontWeight: '600', color: textC,
                      }}>
                        {account.username[0].toUpperCase()}
                      </div>
                      <div>
                        <div style={{ fontSize: '15px', fontWeight: '600', color: c.textPrimary }}>
                          {account.username}
                        </div>
                        <div style={{ fontSize: '13px', color: c.textMuted, fontFamily: 'monospace' }}>
                          {account.password}
                        </div>
                      </div>
                    </div>
                    <span style={{
                      fontSize: '13px', fontWeight: '600',
                      color: textC, backgroundColor: bgC,
                      padding: '3px 10px', borderRadius: '4px',
                      letterSpacing: '0.1px',
                    }}>
                      {account.role}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      </div>

      <style>{`
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        @media (min-width: 900px) { .login-side-panel { display: flex !important; } }
      `}</style>
    </div>
  );
};
