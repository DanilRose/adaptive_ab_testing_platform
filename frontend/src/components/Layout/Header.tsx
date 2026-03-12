import React, { useState } from 'react';
import { LogOut, Moon, Sun, User, ChevronDown } from 'lucide-react';
import { useTheme } from '@/context/ThemeContext';
import { useAuth } from '@/context/AuthContext';
import type { UserRole } from '@/types';

const roleLabels: Record<UserRole, { label: string; color: string; bg: string }> = {
  developer: { label: 'Developer', color: '#92400e', bg: '#fef3c7' },
  analyst:   { label: 'Analyst',   color: '#065f46', bg: '#d1fae5' },
  manager:   { label: 'Manager',   color: '#78350f', bg: '#fde68a' },
};

const roleLabels_dark: Record<UserRole, { label: string; color: string; bg: string }> = {
  developer: { label: 'Developer', color: '#fcd34d', bg: 'rgba(252,211,77,0.12)' },
  analyst:   { label: 'Analyst',   color: '#6ee7b7', bg: 'rgba(110,231,183,0.12)' },
  manager:   { label: 'Manager',   color: '#fbbf24', bg: 'rgba(251,191,36,0.12)' },
};

interface HeaderProps {
  sidebarCollapsed: boolean;
}

export const Header: React.FC<HeaderProps> = () => {
  const { theme, toggleTheme } = useTheme();
  const { user, logout } = useAuth();
  const isDark = theme === 'dark';
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [themeHover, setThemeHover] = useState(false);

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
        {roleInfo && (
          <span style={{
            fontSize: '16px',
            fontWeight: '600',
            color: roleInfo.color,
            backgroundColor: roleInfo.bg,
            padding: '6px 14px',
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
                width: '28px',
                height: '28px',
                borderRadius: '6px',
                backgroundColor: c.avatarBg,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '11px',
                fontWeight: '600',
                color: c.avatarText,
                flexShrink: 0,
                letterSpacing: '0.5px',
              }}>
                {getInitials(user.full_name)}
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start' }}>
                <span style={{ fontSize: '16px', fontWeight: '600', color: c.textPrimary, lineHeight: 1.3 }}>
                  {user.full_name}
                </span>
                <span style={{ fontSize: '14px', color: c.textMuted, lineHeight: 1.3 }}>
                  @{user.username}
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
                    <div style={{ fontSize: '14px', color: c.textMuted, marginTop: '2px' }}>
                      @{user.username}
                    </div>
                  </div>

                  <div style={{ padding: '4px' }}>
                    <DropdownItem
                      icon={<User size={14} />}
                      label="Профиль"
                      onClick={() => setDropdownOpen(false)}
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
