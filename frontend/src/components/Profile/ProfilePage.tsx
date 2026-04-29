import React, { useEffect, useMemo, useState } from 'react';
import { Mail, Phone, Save, UserCircle2 } from 'lucide-react';
import { useAuth } from '@/context/AuthContext';
import { useTheme } from '@/context/ThemeContext';

const roleLabelMap: Record<string, string> = {
  developer: 'Разработчик',
  analyst: 'Аналитик',
  results_viewer: 'Просмотр_результатов_тестов',
  user: '',
};

const formatPhone = (input: string): string => {
  const digits = input.replace(/\D/g, '').replace(/^8/, '7').slice(0, 11);
  const normalized = digits.startsWith('7') ? digits : `7${digits}`;
  const d = normalized.slice(1);
  const p1 = d.slice(0, 3);
  const p2 = d.slice(3, 6);
  const p3 = d.slice(6, 8);
  const p4 = d.slice(8, 10);
  return `+7 (${p1.padEnd(3, '_')}) ${p2.padEnd(3, '_')} - ${p3.padEnd(2, '_')} - ${p4.padEnd(2, '_')}`;
};

export const ProfilePage: React.FC = () => {
  const { user, updateProfile } = useAuth();
  const { theme } = useTheme();
  const isDark = theme === 'dark';

  const [fullName, setFullName] = useState(user?.full_name || '');
  const [email, setEmail] = useState(user?.email || '');
  const [phone, setPhone] = useState(formatPhone(user?.phone || '+7'));
  const [avatarFile, setAvatarFile] = useState<File | null>(null);
  const [avatarPreviewUrl, setAvatarPreviewUrl] = useState<string | null>(null);
  const [avatarTick, setAvatarTick] = useState<number>(Date.now());
  const [avatarServerUrl, setAvatarServerUrl] = useState<string | null>(null);
  const [selectedAvatarFileName, setSelectedAvatarFileName] = useState<string>('');
  const [avatarUploaded, setAvatarUploaded] = useState<boolean>(false);
  const [saving, setSaving] = useState(false);
  const [okMessage, setOkMessage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const roleLabel = useMemo(() => {
    if (!user) return '';
    if (Object.prototype.hasOwnProperty.call(roleLabelMap, user.role)) {
      return roleLabelMap[user.role];
    }
    if (String(user.role).toLowerCase() === 'user') {
      return '';
    }
    return user.role;
  }, [user]);

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
      inputFocus: '#d97706',
      accent: '#d97706',
      accentHov: '#b45309',
      accentSoft: isDark ? 'rgba(217,119,6,0.16)' : '#fef3c7',
      accentText: isDark ? '#fcd34d' : '#92400e',
      success: isDark ? '#86efac' : '#166534',
      successSoft: isDark ? 'rgba(34,197,94,0.12)' : '#f0fdf4',
      danger: isDark ? '#fca5a5' : '#dc2626',
      dangerSoft: isDark ? 'rgba(239,68,68,0.12)' : '#fef2f2',
      shadow: isDark ? '0 10px 32px rgba(0,0,0,0.36)' : '0 8px 28px rgba(28,25,23,0.07)',
    }),
    [isDark],
  );

  if (!user) return null;

  useEffect(() => {
    let revokedUrl: string | null = null;

    const loadAvatar = async () => {
      try {
        const { authAPI } = await import('@/utils/api');
        const response = await authAPI.getAvatarBlob();
        const blobUrl = URL.createObjectURL(response.data);
        revokedUrl = blobUrl;
        setAvatarServerUrl(blobUrl);
        setAvatarUploaded(true);
      } catch {
        setAvatarServerUrl(null);
        setAvatarUploaded(false);
      }
    };

    void loadAvatar();

    return () => {
      if (revokedUrl) URL.revokeObjectURL(revokedUrl);
    };
  }, [avatarTick]);

  const onSave = async () => {
    setSaving(true);
    setOkMessage(null);
    setErrorMessage(null);
    try {
      await updateProfile({
        full_name: fullName,
        email: email || null,
        phone: phone.replace(/[_\s\-()]/g, ''),
      });
      if (avatarFile) {
        const { authAPI } = await import('@/utils/api');
        await authAPI.uploadAvatar(avatarFile);
        setAvatarTick(Date.now());
        setSelectedAvatarFileName('');
      }
      setOkMessage('Профиль успешно обновлён');
    } catch {
      setErrorMessage('Не удалось сохранить изменения профиля');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div style={{ color: c.textPrimary, fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 22 }}>
        <div style={{ width: 40, height: 40, borderRadius: 10, backgroundColor: c.accentSoft, border: `1px solid ${isDark ? 'rgba(217,119,6,0.25)' : '#fde68a'}`, display: 'flex', alignItems: 'center', justifyContent: 'center', color: c.accentText }}>
          <UserCircle2 size={20} />
        </div>
        <div>
          <h1 style={{ margin: 0, fontSize: 22, fontWeight: 700, letterSpacing: '-0.4px' }}>Профиль</h1>
          <p style={{ margin: 0, fontSize: 13, color: c.textMuted }}>Личные данные и настройки аккаунта</p>
        </div>
      </div>

      {(okMessage || errorMessage) && (
        <div style={{ marginBottom: 12, borderRadius: 10, border: `1px solid ${okMessage ? (isDark ? 'rgba(34,197,94,0.35)' : '#bbf7d0') : (isDark ? 'rgba(239,68,68,0.35)' : '#fecaca')}`, backgroundColor: okMessage ? c.successSoft : c.dangerSoft, color: okMessage ? c.success : c.danger, padding: '10px 12px', fontSize: 13, fontWeight: 600 }}>
          {okMessage || errorMessage}
        </div>
      )}

      <div style={{ borderRadius: 14, border: `1px solid ${c.border}`, backgroundColor: c.panelBg, boxShadow: c.shadow, overflow: 'hidden', marginBottom: 14 }}>
        <div style={{ padding: '12px 14px', borderBottom: `1px solid ${c.border}`, backgroundColor: c.panelSoft, display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8, flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            {avatarPreviewUrl || avatarServerUrl || user.avatar_url ? (
              <img src={avatarPreviewUrl || avatarServerUrl || user.avatar_url || ''} alt="avatar" style={{ width: 72, height: 72, borderRadius: 14, objectFit: 'cover', border: `1px solid ${c.border}` }} />
            ) : null}
            <div style={{ width: 72, height: 72, borderRadius: 14, border: `1px solid ${c.border}`, backgroundColor: c.panelBg, display: avatarPreviewUrl || avatarServerUrl || user.avatar_url ? 'none' : 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <UserCircle2 size={34} color={c.textSub} />
            </div>
            <div>
              <div style={{ fontSize: 15, fontWeight: 700 }}>{user.full_name}</div>
            </div>
          </div>
          {roleLabel ? (
            <span style={{ padding: '4px 10px', borderRadius: 999, border: `1px solid ${c.border}`, background: c.panelBg, color: c.textMuted, fontSize: 12, fontWeight: 600 }}>
              {roleLabel}
            </span>
          ) : null}
        </div>

        <div style={{ padding: 14 }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
            <Field label="ФИО" c={c}>
              <input value={fullName} onChange={(e) => setFullName(e.target.value)} placeholder="Введите ФИО" style={inputStyle(c)} />
            </Field>
            <Field label="E-mail" c={c} icon={<Mail size={14} />}>
              <input value={email} onChange={(e) => setEmail(e.target.value)} placeholder="name@example.com" style={inputStyle(c)} />
            </Field>
            <Field label="Телефон" c={c} icon={<Phone size={14} />}>
              <input value={phone} onChange={(e) => setPhone(formatPhone(e.target.value))} placeholder="+7 (___) ___ - __ - __" style={inputStyle(c)} />
            </Field>
          </div>

          <div style={{ marginTop: 12, display: 'grid', gridTemplateColumns: '1fr', gap: 12 }}>
            <Field label="Фото" c={c}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <input
                  id="avatar-upload-input"
                  type="file"
                  accept="image/png,image/jpeg,image/webp"
                  onChange={(e) => {
                    const file = e.target.files?.[0] || null;
                    setAvatarFile(file);
                    setSelectedAvatarFileName(file?.name || '');
                    if (file) {
                      const localUrl = URL.createObjectURL(file);
                      setAvatarPreviewUrl(localUrl);
                    } else {
                      setAvatarPreviewUrl(null);
                    }
                  }}
                  style={{ display: 'none' }}
                />
                <label
                  htmlFor="avatar-upload-input"
                  style={{
                    height: 36,
                    padding: '0 12px',
                    borderRadius: 8,
                    border: `1px solid ${c.border}`,
                    backgroundColor: c.panelSoft,
                    color: c.textPrimary,
                    fontSize: 13,
                    fontWeight: 600,
                    cursor: 'pointer',
                    display: 'inline-flex',
                    alignItems: 'center',
                    whiteSpace: 'nowrap',
                  }}
                >
                  Выбрать файл
                </label>
                <span style={{ fontSize: 13, color: c.textMuted, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {selectedAvatarFileName || (avatarUploaded ? 'Фото уже загружено' : 'Файл не выбран')}
                </span>
              </div>
            </Field>
          </div>

          <div style={{ marginTop: 16, display: 'flex', justifyContent: 'flex-end' }}>
            <ActionButton onClick={onSave} disabled={saving} c={c} icon={<Save size={14} />} label={saving ? 'Сохранение...' : 'Сохранить профиль'} />
          </div>
        </div>
      </div>
    </div>
  );
};

const Field: React.FC<{ label: string; c: Record<string, string>; children: React.ReactNode; icon?: React.ReactNode }> = ({ label, c, children, icon }) => (
  <div>
    <label style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6, fontSize: 13, fontWeight: 600, color: c.textMuted }}>
      {icon}
      <span>{label}</span>
    </label>
    {children}
  </div>
);

const inputStyle = (c: Record<string, string>, disabled = false): React.CSSProperties => ({
  width: '100%',
  height: 38,
  borderRadius: 8,
  border: `1.5px solid ${c.inputBorder}`,
  backgroundColor: disabled ? c.panelSoft : c.inputBg,
  color: disabled ? c.textSub : c.textPrimary,
  fontSize: 14,
  padding: '0 11px',
  outline: 'none',
  boxSizing: 'border-box',
  fontFamily: 'inherit',
});

const ActionButton: React.FC<{ onClick: () => void; disabled?: boolean; c: Record<string, string>; icon?: React.ReactNode; label: string }> = ({ onClick, disabled, c, icon, label }) => {
  const [hov, setHov] = useState(false);

  return (
    <button
      onClick={disabled ? undefined : onClick}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        height: 38,
        padding: '0 14px',
        borderRadius: 8,
        border: `1px solid ${disabled ? c.border : hov ? c.accentHov : c.accent}`,
        backgroundColor: disabled ? c.panelSoft : hov ? c.accentHov : c.accent,
        color: disabled ? c.textMuted : '#fff',
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
