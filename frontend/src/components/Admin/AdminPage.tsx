import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { AlertTriangle, Check, ExternalLink, RefreshCw, Search, ShieldCheck, UserPlus, X, Zap } from 'lucide-react';
import type { AdminCreateUserPayload, AdminUser, JobTitle, PermissionKey } from '@/types';
import { adminAPI } from '@/utils/api';
import { useTheme } from '@/context/ThemeContext';
import { useAuth } from '@/context/AuthContext';

const ALL_PERMISSIONS: PermissionKey[] = [
  'Администрирование',
  'GAN_менеджер_обучение',
  'GAN_менеджер_генерация_данных',
  'GAN_менеджер_редактирование',
  'AB_тесты_создание',
  'AB_тесты_управление',
  'AB_тесты_удаление_и_архивация',
  'Шаблоны_просмотр',
  'Шаблоны_создание',
  'Шаблоны_редактирование',
  'Шаблоны_удаление',
  'Просмотр_результатов_тестов',
  'Экспорт_результатов',
];

const GROUPS: Array<{ title: string; items: PermissionKey[] }> = [
  { title: 'Администрирование', items: ['Администрирование'] },
  {
    title: 'GAN Manager',
    items: ['GAN_менеджер_обучение', 'GAN_менеджер_генерация_данных', 'GAN_менеджер_редактирование'],
  },
  { title: 'A/B тесты', items: ['AB_тесты_создание', 'AB_тесты_управление', 'AB_тесты_удаление_и_архивация'] },
  { title: 'Шаблоны', items: ['Шаблоны_просмотр', 'Шаблоны_создание', 'Шаблоны_редактирование', 'Шаблоны_удаление'] },
  { title: 'Результаты', items: ['Просмотр_результатов_тестов', 'Экспорт_результатов'] },
];

interface ToastState {
  message: string;
  kind: 'success' | 'error';
  leaving?: boolean;
}

const defaultCreateForm: AdminCreateUserPayload = {
  username: '',
  password: '',
  full_name: '',
  role: 'user',
  job_title: 'other',
  permissions: [],
  email: '',
  phone: '',
};

export const AdminPage: React.FC = () => {
  const { theme } = useTheme();
  const isDark = theme === 'dark';
  const { user: currentUser, refreshUser } = useAuth();

  const c = useMemo(
    () => ({
      panelBg: isDark ? '#1c1917' : '#ffffff',
      panelSoft: isDark ? '#171412' : '#f5f0e8',
      border: isDark ? '#292524' : '#e7e5e4',
      textPrimary: isDark ? '#fafaf9' : '#1c1917',
      textMuted: isDark ? '#a8a29e' : '#78716c',
      textSub: isDark ? '#57534e' : '#a8a29e',
      inputBg: isDark ? '#292524' : '#fafaf9',
      inputBorder: isDark ? '#3c3330' : '#e7e5e4',
      accent: '#d97706',
      accentHov: '#b45309',
      accentSoft: isDark ? 'rgba(217,119,6,0.16)' : '#fef3c7',
      accentText: isDark ? '#fcd34d' : '#92400e',
      danger: isDark ? '#fca5a5' : '#dc2626',
      dangerSoft: isDark ? 'rgba(239,68,68,0.12)' : '#fef2f2',
      dangerBorder: isDark ? 'rgba(239,68,68,0.25)' : '#fecaca',
      success: isDark ? '#86efac' : '#166534',
      successSoft: isDark ? 'rgba(34,197,94,0.15)' : '#f0fdf4',
      shadow: isDark ? '0 10px 32px rgba(0,0,0,0.36)' : '0 8px 28px rgba(28,25,23,0.07)',
    }),
    [isDark],
  );

  const [users, setUsers] = useState<AdminUser[]>([]);
  const [loading, setLoading] = useState(true);
  const [query, setQuery] = useState('');
  const [selectedUserId, setSelectedUserId] = useState<number | null>(null);
  const [draftPermissions, setDraftPermissions] = useState<PermissionKey[]>([]);
  const [savingPermissions, setSavingPermissions] = useState(false);

  const [createForm, setCreateForm] = useState<AdminCreateUserPayload>(defaultCreateForm);
  const [createErrors, setCreateErrors] = useState<Record<string, string>>({});
  const [creatingUser, setCreatingUser] = useState(false);

  const [toast, setToast] = useState<ToastState | null>(null);
  const [selectedUserAvatarBlobUrl, setSelectedUserAvatarBlobUrl] = useState<string | null>(null);
  const [avatarLoading, setAvatarLoading] = useState(false);

  const showToast = (message: string, kind: 'success' | 'error') => {
    setToast({ message, kind, leaving: false });
    window.setTimeout(() => setToast((prev) => (prev ? { ...prev, leaving: true } : null)), 2600);
    window.setTimeout(() => setToast(null), 3000);
  };

  const loadUsers = useCallback(async () => {
    setLoading(true);
    try {
      const response = await adminAPI.listUsers();
      const items = (response.data.items || []).map((u) => ({ ...u, permissions: (u.permissions || []) as PermissionKey[] }));
      setUsers(items);

      if (!items.length) {
        setSelectedUserId(null);
        setDraftPermissions([]);
        return;
      }

      if (selectedUserId && items.some((u) => u.id === selectedUserId)) {
        const current = items.find((u) => u.id === selectedUserId);
        setDraftPermissions((current?.permissions || []) as PermissionKey[]);
      } else {
        setSelectedUserId(items[0].id);
        setDraftPermissions((items[0].permissions || []) as PermissionKey[]);
      }
    } catch {
      showToast('Не удалось загрузить пользователей', 'error');
    } finally {
      setLoading(false);
    }
  }, [selectedUserId]);

  useEffect(() => {
    void loadUsers();
  }, [loadUsers]);

  useEffect(() => {
    let revoked: string | null = null;

    const loadSelectedUserAvatar = async () => {
      if (!selectedUserId) {
        setSelectedUserAvatarBlobUrl(null);
        return;
      }

      setAvatarLoading(true);
      try {
        const response = await adminAPI.getUserAvatarBlob(selectedUserId);
        const url = URL.createObjectURL(response.data);
        revoked = url;
        setSelectedUserAvatarBlobUrl(url);
      } catch {
        const selected = users.find((u) => u.id === selectedUserId);
        if (selected?.avatar_url) {
          setSelectedUserAvatarBlobUrl(selected.avatar_url);
        } else {
          setSelectedUserAvatarBlobUrl(null);
        }
      } finally {
        setAvatarLoading(false);
      }
    };

    void loadSelectedUserAvatar();

    return () => {
      if (revoked) URL.revokeObjectURL(revoked);
    };
  }, [selectedUserId, users]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return users;
    return users.filter((u) => {
      return (
        u.username.toLowerCase().includes(q) ||
        (u.full_name || '').toLowerCase().includes(q) ||
        (u.email || '').toLowerCase().includes(q)
      );
    });
  }, [users, query]);

  const selectedUser = useMemo(() => users.find((u) => u.id === selectedUserId) || null, [users, selectedUserId]);

  const inputStyle: React.CSSProperties = {
    width: '100%',
    height: 38,
    borderRadius: 8,
    border: `1.5px solid ${c.inputBorder}`,
    backgroundColor: c.inputBg,
    color: c.textPrimary,
    fontSize: 14,
    padding: '0 11px',
    outline: 'none',
    boxSizing: 'border-box',
    fontFamily: 'inherit',
  };

  const savePermissions = async () => {
    if (!selectedUser) return;
    setSavingPermissions(true);
    try {
      const response = await adminAPI.updateUserPermissions(selectedUser.id, draftPermissions);
      const nextPermissions = (response.data.permissions || []) as PermissionKey[];

      setUsers((prev) => prev.map((u) => (u.id === selectedUser.id ? { ...u, permissions: nextPermissions } : u)));
      setDraftPermissions(nextPermissions);

      if (currentUser?.id === selectedUser.id) {
        await refreshUser();
      }

      showToast('Роли пользователя сохранены', 'success');
    } catch {
      showToast('Не удалось сохранить роли пользователя', 'error');
    } finally {
      setSavingPermissions(false);
    }
  };

  const togglePermission = (permission: PermissionKey, checked: boolean) => {
    setDraftPermissions((prev) => {
      if (checked) return Array.from(new Set([...prev, permission]));
      return prev.filter((p) => p !== permission);
    });
  };

  const toggleCreatePermission = (permission: PermissionKey, checked: boolean) => {
    setCreateForm((prev) => {
      const current = prev.permissions || [];
      if (checked) {
        return { ...prev, permissions: Array.from(new Set([...current, permission])) };
      }
      return { ...prev, permissions: current.filter((p) => p !== permission) };
    });
  };

  const validateCreateForm = (): boolean => {
    const errors: Record<string, string> = {};
    if (!createForm.username?.trim()) errors.username = 'Укажите логин';
    if (!createForm.password?.trim()) errors.password = 'Укажите пароль';
    if ((createForm.password || '').length > 0 && (createForm.password || '').length < 6) {
      errors.password = 'Минимум 6 символов';
    }
    if (!createForm.full_name?.trim()) errors.full_name = 'Укажите ФИО';

    if (createForm.email && !/^\S+@\S+\.\S+$/.test(createForm.email)) {
      errors.email = 'Некорректный email';
    }

    setCreateErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const createUser = async () => {
    if (!validateCreateForm()) return;

    setCreatingUser(true);
    try {
      await adminAPI.createUser({
        username: createForm.username.trim(),
        password: createForm.password,
        full_name: createForm.full_name.trim(),
        role: 'user',
        job_title: createForm.job_title as JobTitle,
        permissions: createForm.permissions || [],
        email: createForm.email?.trim() || null,
        phone: createForm.phone?.trim() || null,
      });

      setCreateForm(defaultCreateForm);
      setCreateErrors({});
      await loadUsers();
      showToast('Пользователь успешно создан', 'success');
    } catch (error: any) {
      const message = error?.response?.data?.detail || 'Не удалось создать пользователя';
      showToast(String(message), 'error');
    } finally {
      setCreatingUser(false);
    }
  };

  const profileRows = selectedUser
    ? [
        ['Логин', selectedUser.username || '—'],
        ['ФИО', selectedUser.full_name || '—'],
        ['Должность', selectedUser.job_title || 'other'],
        ['Email', selectedUser.email || '—'],
        ['Телефон', selectedUser.phone || '—'],
      ]
    : [];

  return (
    <div style={{ color: c.textPrimary, fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 22, gap: 12, flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div
            style={{
              width: 40,
              height: 40,
              borderRadius: 10,
              backgroundColor: c.accentSoft,
              border: `1px solid ${isDark ? 'rgba(217,119,6,0.25)' : '#fde68a'}`,
              color: c.accentText,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <Zap size={20} />
          </div>
          <div>
            <h1 style={{ margin: 0, fontSize: 22, fontWeight: 700, letterSpacing: '-0.4px' }}>Администрирование</h1>
            <p style={{ margin: 0, fontSize: 13, color: c.textMuted }}>Администрирование пользователей</p>
          </div>
        </div>

        <ActionButton
          onClick={() => void loadUsers()}
          disabled={loading}
          isDark={isDark}
          c={c}
          icon={<RefreshCw size={14} style={{ animation: loading ? 'spin 0.8s linear infinite' : 'none' }} />}
          label="Обновить"
          variant="secondary"
        />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '320px minmax(0, 1fr) 400px', gap: 14, alignItems: 'start' }}>
        <Panel c={c} title="Пользователи" icon={<Search size={14} />}>
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Поиск: логин / ФИО / email"
            style={{ ...inputStyle, marginBottom: 10 }}
          />

          <div style={{ maxHeight: 640, overflowY: 'auto', display: 'grid', gap: 8 }}>
            {loading ? (
              <EmptyState c={c} text="Загрузка пользователей..." />
            ) : filtered.length === 0 ? (
              <EmptyState c={c} text="Пользователи не найдены" />
            ) : (
              filtered.map((u) => {
                const active = selectedUserId === u.id;
                return (
                  <button
                    key={u.id}
                    onClick={() => {
                      setSelectedUserId(u.id);
                      setDraftPermissions((u.permissions || []) as PermissionKey[]);
                    }}
                    style={{
                      borderRadius: 10,
                      border: `1px solid ${active ? c.accent : c.border}`,
                      backgroundColor: active ? c.accentSoft : c.panelBg,
                      textAlign: 'left',
                      cursor: 'pointer',
                      padding: '10px 11px',
                    }}
                  >
                    <div style={{ fontSize: 14, fontWeight: 700, color: c.textPrimary }}>{u.username}</div>
                    <div style={{ fontSize: 12, color: c.textMuted, marginTop: 2 }}>{u.full_name || '—'}</div>
                    <div style={{ fontSize: 11, color: c.textSub, marginTop: 4 }}>{u.email || 'без email'}</div>
                  </button>
                );
              })
            )}
          </div>
        </Panel>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          <Panel c={c} title="Профиль пользователя" icon={<ShieldCheck size={14} />}>
            {!selectedUser ? (
              <EmptyState c={c} text="Выберите пользователя слева" />
            ) : (
              <>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 8, marginBottom: 12 }}>
                  {profileRows.map(([k, v]) => (
                    <div key={k} style={{ border: `1px solid ${c.border}`, borderRadius: 10, background: c.panelSoft, padding: '9px 11px' }}>
                      <div style={{ fontSize: 11, color: c.textSub, textTransform: 'uppercase', letterSpacing: '0.35px' }}>{k}</div>
                      <div style={{ marginTop: 4, fontSize: 13, color: c.textPrimary, fontWeight: 600, wordBreak: 'break-word' }}>{v}</div>
                    </div>
                  ))}
                </div>

                <div style={{ marginBottom: 10, display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
                  <div style={{ fontSize: 13, color: c.textMuted }}>
                    Назначено ролей: <span style={{ color: c.textPrimary, fontWeight: 700 }}>{draftPermissions.length}</span> / {ALL_PERMISSIONS.length}
                  </div>
                  {selectedUserAvatarBlobUrl ? (
                    <a
                      href={selectedUserAvatarBlobUrl}
                      target="_blank"
                      rel="noreferrer"
                      style={{
                        height: 32,
                        padding: '0 10px',
                        borderRadius: 8,
                        border: `1px solid ${c.border}`,
                        backgroundColor: c.panelBg,
                        color: c.textPrimary,
                        fontSize: 12,
                        fontWeight: 600,
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: 6,
                        textDecoration: 'none',
                      }}
                    >
                      <ExternalLink size={13} /> Открыть фото
                    </a>
                  ) : (
                    <span style={{ fontSize: 12, color: c.textSub }}>{avatarLoading ? 'Проверка фото...' : 'У выбранного пользователя фото недоступно'}</span>
                  )}
                </div>

                <div style={{ display: 'grid', gap: 10 }}>
                  {GROUPS.map((group) => (
                    <div key={group.title} style={{ border: `1px solid ${c.border}`, borderRadius: 10, backgroundColor: c.panelSoft, padding: 10 }}>
                      <div style={{ fontSize: 13, fontWeight: 700, color: c.textPrimary, marginBottom: 8 }}>{group.title}</div>
                      <div style={{ display: 'grid', gap: 6 }}>
                        {group.items.map((permission) => (
                          <PermissionCheckbox
                            key={permission}
                            label={permission}
                            checked={draftPermissions.includes(permission)}
                            onChange={(checked) => togglePermission(permission, checked)}
                            c={c}
                          />
                        ))}
                      </div>
                    </div>
                  ))}
                </div>

                <div style={{ marginTop: 12, display: 'flex', justifyContent: 'flex-end' }}>
                  <ActionButton
                    onClick={() => void savePermissions()}
                    disabled={savingPermissions}
                    isDark={isDark}
                    c={c}
                    icon={savingPermissions ? <RefreshCw size={14} style={{ animation: 'spin 0.8s linear infinite' }} /> : <Check size={14} />}
                    label={savingPermissions ? 'Сохранение...' : 'Сохранить роли'}
                    variant="primary"
                  />
                </div>
              </>
            )}
          </Panel>
        </div>

        <Panel c={c} title="Создание пользователя" icon={<UserPlus size={14} />}>
          <div style={{ display: 'grid', gap: 10 }}>
            <Field label="Логин *" c={c} error={createErrors.username}>
              <input
                value={createForm.username || ''}
                onChange={(e) => setCreateForm((p) => ({ ...p, username: e.target.value }))}
                style={inputStyle}
                placeholder="Например: ivan.petrov"
              />
            </Field>

            <Field label="Пароль *" c={c} error={createErrors.password}>
              <input
                type="password"
                value={createForm.password || ''}
                onChange={(e) => setCreateForm((p) => ({ ...p, password: e.target.value }))}
                style={inputStyle}
                placeholder="Минимум 6 символов"
              />
            </Field>

            <Field label="ФИО *" c={c} error={createErrors.full_name}>
              <input
                value={createForm.full_name || ''}
                onChange={(e) => setCreateForm((p) => ({ ...p, full_name: e.target.value }))}
                style={inputStyle}
                placeholder="Иван Петров"
              />
            </Field>

            <Field label="Должность" c={c}>
              <select
                value={createForm.job_title || 'other'}
                onChange={(e) => setCreateForm((p) => ({ ...p, job_title: e.target.value as JobTitle }))}
                style={inputStyle}
              >
                <option value="developer">developer</option>
                <option value="analyst">analyst</option>
                <option value="project_manager">project_manager</option>
                <option value="other">other</option>
              </select>
            </Field>

            <Field label="Email" c={c} error={createErrors.email}>
              <input
                value={createForm.email || ''}
                onChange={(e) => setCreateForm((p) => ({ ...p, email: e.target.value }))}
                style={inputStyle}
                placeholder="user@company.ru"
              />
            </Field>

            <Field label="Телефон" c={c}>
              <input
                value={createForm.phone || ''}
                onChange={(e) => setCreateForm((p) => ({ ...p, phone: e.target.value }))}
                style={inputStyle}
                placeholder="+7..."
              />
            </Field>

            <div style={{ marginTop: 2 }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: c.textMuted, marginBottom: 8 }}>Роли для нового пользователя</div>
              <div style={{ maxHeight: 220, overflowY: 'auto', border: `1px solid ${c.border}`, borderRadius: 10, background: c.panelSoft, padding: 10, display: 'grid', gap: 8 }}>
                {ALL_PERMISSIONS.map((permission) => (
                  <PermissionCheckbox
                    key={permission}
                    label={permission}
                    checked={(createForm.permissions || []).includes(permission)}
                    onChange={(checked) => toggleCreatePermission(permission, checked)}
                    c={c}
                  />
                ))}
              </div>
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, marginTop: 4 }}>
              <ActionButton
                onClick={() => {
                  setCreateForm(defaultCreateForm);
                  setCreateErrors({});
                }}
                isDark={isDark}
                c={c}
                label="Сбросить"
                variant="secondary"
              />
              <ActionButton
                onClick={() => void createUser()}
                disabled={creatingUser}
                isDark={isDark}
                c={c}
                icon={creatingUser ? <RefreshCw size={14} style={{ animation: 'spin 0.8s linear infinite' }} /> : <UserPlus size={14} />}
                label={creatingUser ? 'Создание...' : 'Создать пользователя'}
                variant="primary"
              />
            </div>
          </div>
        </Panel>
      </div>

      {toast && <Toast message={toast.message} kind={toast.kind} leaving={toast.leaving} onClose={() => setToast(null)} />}

      <style>{`
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes fadeOutDown { from { opacity: 1; transform: translateY(0); } to { opacity: 0; transform: translateY(16px); } }
      `}</style>
    </div>
  );
};

const Panel: React.FC<{ c: Record<string, string>; title: string; icon?: React.ReactNode; children: React.ReactNode }> = ({ c, title, icon, children }) => (
  <div style={{ borderRadius: 14, border: `1px solid ${c.border}`, backgroundColor: c.panelBg, boxShadow: c.shadow, overflow: 'hidden' }}>
    <div style={{ padding: '12px 14px', borderBottom: `1px solid ${c.border}`, backgroundColor: c.panelSoft, display: 'flex', alignItems: 'center', gap: 8 }}>
      <span style={{ color: c.accentText, display: 'inline-flex', alignItems: 'center' }}>{icon}</span>
      <span style={{ fontSize: 14, fontWeight: 700, color: c.textPrimary }}>{title}</span>
    </div>
    <div style={{ padding: 14 }}>{children}</div>
  </div>
);

const EmptyState: React.FC<{ c: Record<string, string>; text: string }> = ({ c, text }) => (
  <div style={{ padding: '26px 14px', border: `1px dashed ${c.border}`, borderRadius: 10, textAlign: 'center', color: c.textMuted, fontSize: 13 }}>
    {text}
  </div>
);

const PermissionCheckbox: React.FC<{
  checked: boolean;
  onChange: (checked: boolean) => void;
  label: string;
  c: Record<string, string>;
}> = ({ checked, onChange, label, c }) => (
  <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', userSelect: 'none' }}>
    <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
    <span style={{ fontSize: 13, color: checked ? c.textPrimary : c.textMuted }}>{label}</span>
  </label>
);

const Field: React.FC<{ label: string; c: Record<string, string>; error?: string; children: React.ReactNode }> = ({ label, c, error, children }) => (
  <div>
    <label style={{ display: 'block', marginBottom: 6, fontSize: 13, fontWeight: 600, color: c.textMuted }}>{label}</label>
    {children}
    {!!error && <div style={{ marginTop: 4, fontSize: 12, color: c.danger }}>{error}</div>}
  </div>
);

const Toast: React.FC<{ message: string; kind: 'success' | 'error'; leaving?: boolean; onClose: () => void }> = ({ message, kind, leaving, onClose }) => (
  <div
    style={{
      position: 'fixed',
      bottom: 28,
      right: 28,
      zIndex: 9999,
      display: 'flex',
      alignItems: 'center',
      gap: 10,
      padding: '12px 18px',
      borderRadius: 10,
      backgroundColor: kind === 'success' ? '#d97706' : '#dc2626',
      color: '#fff',
      fontSize: 14,
      fontWeight: 500,
      boxShadow: '0 8px 24px rgba(0,0,0,0.22)',
      animation: leaving ? 'fadeOutDown 0.35s ease forwards' : 'fadeIn 0.2s ease',
    }}
  >
    {kind === 'success' ? <Check size={15} /> : <AlertTriangle size={15} />}
    {message}
    <button onClick={onClose} style={{ marginLeft: 6, background: 'none', border: 'none', color: '#fff', cursor: 'pointer', padding: 0, display: 'flex' }}>
      <X size={14} />
    </button>
  </div>
);

interface ActionButtonProps {
  onClick: () => void;
  disabled?: boolean;
  isDark: boolean;
  c: Record<string, string>;
  icon?: React.ReactNode;
  label?: string;
  variant: 'primary' | 'secondary';
}

const ActionButton: React.FC<ActionButtonProps> = ({ onClick, disabled, isDark, c, icon, label, variant }) => {
  const [hov, setHov] = useState(false);

  const bg =
    variant === 'primary'
      ? disabled
        ? isDark
          ? '#292524'
          : '#e7e5e4'
        : hov
          ? c.accentHov
          : c.accent
      : hov
        ? isDark
          ? '#292524'
          : '#f0ede8'
        : c.panelBg;

  const border = variant === 'primary' ? (disabled ? (isDark ? '#292524' : '#e7e5e4') : hov ? c.accentHov : c.accent) : c.border;
  const color = variant === 'primary' ? (disabled ? c.textMuted : '#fff') : c.textMuted;

  return (
    <button
      onClick={disabled ? undefined : onClick}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        height: 38,
        padding: '0 14px',
        borderRadius: 8,
        border: `1px solid ${border}`,
        backgroundColor: bg,
        color,
        fontSize: 13,
        fontWeight: 600,
        cursor: disabled ? 'not-allowed' : 'pointer',
        display: 'flex',
        alignItems: 'center',
        gap: 6,
        transition: 'all 0.12s',
        whiteSpace: 'nowrap',
      }}
    >
      {icon}
      {label}
    </button>
  );
};
