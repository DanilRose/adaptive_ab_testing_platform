import React, { useEffect, useState } from 'react';
import { LogOut, Moon, Sun, User, ChevronDown } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { useTheme } from '@/context/ThemeContext';
import { useAuth } from '@/context/AuthContext';
import type { UserRole } from '@/types';

const roleLabels: Record<UserRole, { label: string; color: string; bg: string }> = {
  developer: { label: 'Разработчик', color: '#92400e', bg: '#fef3c7' },
  analyst: { label: 'Аналитик', color: '#065f46', bg: '#d1fae5' },
  results_viewer: { label: 'Просмотр результатов', color: '#1d4ed8', bg: '#dbeafe' },
  user: { label: '', color: '#6b7280', bg: '#f3f4f6' },
};

const roleLabels_dark: Record<UserRole, { label: string; color: string; bg: string }> = {
  developer: { label: 'Разработчик', color: '#fcd34d', bg: 'rgba(252,211,77,0.12)' },
  analyst: { label: 'Аналитик', color: '#6ee7b7', bg: 'rgba(110,231,183,0.12)' },
  results_viewer: { label: 'Просмотр результатов', color: '#93c5fd', bg: 'rgba(147,197,253,0.16)' },
  user: { label: '', color: '#d1d5db', bg: 'rgba(209,213,219,0.12)' },
};

interface HeaderProps {
  sidebarCollapsed: boolean;
}

export const Header: React.FC<HeaderProps> = () => {
  const { theme, toggleTheme } = useTheme();
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const isDark = theme === 'dark';
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [themeHover, setThemeHover] = useState(false);

  const [avatarUrl, setAvatarUrl] = useState<string | null>(null);

  useEffect(() => {
    let revokedUrl: string | null = null;

    const loadAvatar = async () => {
      try {
        const { authAPI } = await import('@/utils/api');
        const response = await authAPI.getAvatarBlob();
        const blobUrl = URL.createObjectURL(response.data);
        revokedUrl = blobUrl;
        setAvatarUrl(blobUrl);
      } catch {
        setAvatarUrl(null);
      }
    };

    void loadAvatar();

    return () => {
      if (revokedUrl) URL.revokeObjectURL(revokedUrl);
    };
  }, [user?.id, user?.full_name]);

  const getInitials = (name: string): string =>
    name.split(' ').map((n) => n[0]).join('').toUpperCase().slice(0, 2);

  const roleInfo = user?.role
    ? (isDark ? roleLabels_dark[user.role] : roleLabels[user.role])
    : null;

  const c = {
    bg:           isDark ? '#1c1917' : '#ffffff',
    border:       isDark ? '#292524' : '#e7e5e4',
    textPrimary:  isDark ? '#fafaf9' : '#1c1917',
    textMuted:    isDark ? '#78716c' : '#a8a29e',
    hoverBg:      isDark ? '#292524' : '#f5f5f4',
    dropdownBg:   isDark ? '#1c1917' : '#ffffff',
    dropdownBorder: isDark ? '#292524' : '#e7e5e4',
    dropdownHover:  isDark ? '#292524' : '#f5f5f4',
    avatarBg:     isDark ? '#292524' : '#f5f5f4',
    avatarText:   isDark ? '#fcd34d' : '#92400e',
  };

  return (
    <header style={{
      height: '61px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '0 20px',
      backgroundColor: c.bg,
      borderBottom: `1px solid ${c.border}`,
      flexShrink: 0,
      position: 'sticky',
      top: 0,
      zIndex: 10,
    }}>
      {/* Left — role badge */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        {roleInfo?.label && (
          <span style={{
            fontSize: '16px',
            fontWeight: '500',
            color: '#9ca3af',
            backgroundColor: 'transparent',
            padding: '6px 4px',
            borderRadius: '6px',
            letterSpacing: '0.2px',
          }}>
            {roleInfo.label}
          </span>
        )}
      </div>

      {/* Right */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '2px' }}>
        {/* Theme toggle */}
        <button
          onClick={toggleTheme}
          onMouseEnter={() => setThemeHover(true)}
          onMouseLeave={() => setThemeHover(false)}
          style={{
            width: '34px',
            height: '34px',
            borderRadius: '6px',
            border: 'none',
            backgroundColor: themeHover ? c.hoverBg : 'transparent',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: c.textMuted,
            transition: 'background-color 0.15s',
          }}
          title={isDark ? 'Светлая тема' : 'Тёмная тема'}
        >
          {isDark ? <Sun size={16} /> : <Moon size={16} />}
        </button>

        {/* Divider */}
        <div style={{ width: '1px', height: '20px', backgroundColor: c.border, margin: '0 6px' }} />

        {/* Profile */}
        {user && (
          <div style={{ position: 'relative' }}>
            <button
              onClick={() => setDropdownOpen((v) => !v)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                padding: '5px 8px 5px 5px',
                borderRadius: '7px',
                border: `1px solid ${dropdownOpen ? c.dropdownBorder : 'transparent'}`,
                backgroundColor: dropdownOpen ? c.hoverBg : 'transparent',
                cursor: 'pointer',
                transition: 'background-color 0.15s, border-color 0.15s',
              }}
              onMouseEnter={e => {
                if (!dropdownOpen) e.currentTarget.style.backgroundColor = c.hoverBg;
              }}
              onMouseLeave={e => {
                if (!dropdownOpen) e.currentTarget.style.backgroundColor = 'transparent';
              }}
            >
              {/* Avatar */}
              <div style={{
                width: '36px',
                height: '36px',
                borderRadius: '8px',
                backgroundColor: c.avatarBg,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '11px',
                fontWeight: '600',
                color: c.avatarText,
                flexShrink: 0,
                letterSpacing: '0.5px',
                overflow: 'hidden',
                border: `1px solid ${c.border}`,
              }}>
                {avatarUrl ? (
                  <img src={avatarUrl} alt="avatar" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                ) : (
                  <span style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', width: '100%', height: '100%' }}>
                    {getInitials(user.full_name)}
                  </span>
                )}
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start' }}>
                <span style={{ fontSize: '16px', fontWeight: '600', color: c.textPrimary, lineHeight: 1.3 }}>
                  {user.full_name}
                </span>
              </div>

              <ChevronDown
                size={13}
                color={c.textMuted}
                style={{
                  transform: dropdownOpen ? 'rotate(180deg)' : 'rotate(0deg)',
                  transition: 'transform 0.2s',
                }}
              />
            </button>

            {/* Dropdown */}
            {dropdownOpen && (
              <>
                <div
                  style={{ position: 'fixed', inset: 0, zIndex: 40 }}
                  onClick={() => setDropdownOpen(false)}
                />
                <div style={{
                  position: 'absolute',
                  top: 'calc(100% + 6px)',
                  right: 0,
                  width: '210px',
                  backgroundColor: c.dropdownBg,
                  border: `1px solid ${c.dropdownBorder}`,
                  borderRadius: '8px',
                  boxShadow: isDark
                    ? '0 8px 24px rgba(0,0,0,0.4)'
                    : '0 8px 24px rgba(28,25,23,0.1)',
                  zIndex: 50,
                  overflow: 'hidden',
                }}>
                  {/* User info */}
                  <div style={{
                    padding: '12px 14px',
                    borderBottom: `1px solid ${c.dropdownBorder}`,
                  }}>
                    <div style={{ fontSize: '16px', fontWeight: '600', color: c.textPrimary }}>
                      {user.full_name}
                    </div>
                  </div>

                  <div style={{ padding: '4px' }}>
                    <DropdownItem
                      icon={<User size={14} />}
                      label="Профиль"
                      onClick={() => { setDropdownOpen(false); navigate('/profile'); }}
                      hoverBg={c.dropdownHover}
                      textColor={c.textPrimary}
                    />

                    <div style={{ height: '1px', backgroundColor: c.dropdownBorder, margin: '4px 0' }} />

                    <DropdownItem
                      icon={<LogOut size={14} />}
                      label="Выйти"
                      onClick={() => { setDropdownOpen(false); void logout(); }}
                      hoverBg={isDark ? 'rgba(239,68,68,0.1)' : '#fef2f2'}
                      textColor="#ef4444"
                      danger
                    />
                  </div>
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </header>
  );
};

interface DropdownItemProps {
  icon: React.ReactNode;
  label: string;
  onClick: () => void;
  hoverBg: string;
  textColor: string;
  danger?: boolean;
}

const DropdownItem: React.FC<DropdownItemProps> = ({ icon, label, onClick, hoverBg, textColor }) => {
  const [hovered, setHovered] = useState(false);
  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        width: '100%',
        padding: '7px 10px',
        border: 'none',
        borderRadius: '5px',
        backgroundColor: hovered ? hoverBg : 'transparent',
        cursor: 'pointer',
        color: textColor,
        fontSize: '16px',
        textAlign: 'left',
        transition: 'background-color 0.12s',
      }}
    >
      <span style={{ opacity: 0.8 }}>{icon}</span>
      {label}
    </button>
  );
};
