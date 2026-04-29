import React, { useMemo, useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import {
  BarChart3,
  FileText,
  ChevronLeft,
  ChevronRight,
  BrainCircuit,
  Cpu,
  FlaskConical,
  Shield,
  type LucideProps,
} from 'lucide-react';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import type { PermissionKey, UserRole } from '@/types';
import { useTheme } from '@/context/ThemeContext';

type LucideIcon = React.ForwardRefExoticComponent<Omit<LucideProps, 'ref'> & React.RefAttributes<SVGSVGElement>>;

interface MenuRouteItem {
  key: string;
  path: string;
  label: string;
  icon: LucideIcon;
  allowedPermissions: PermissionKey[];
}

const menuRoutes: MenuRouteItem[] = [
  { key: '/ab-manager', path: '/ab-manager', label: 'A/B Менеджер', icon: FlaskConical, allowedPermissions: ['AB_тесты_создание', 'AB_тесты_управление'] },
  { key: '/gan-manager', path: '/gan-manager', label: 'GAN Менеджер', icon: Cpu, allowedPermissions: ['GAN_менеджер_обучение', 'GAN_менеджер_генерация_данных'] },
  { key: '/results', path: '/results', label: 'Результаты', icon: BarChart3, allowedPermissions: ['Просмотр_результатов_тестов', 'Экспорт_результатов'] },
  { key: '/templates', path: '/templates', label: 'Шаблоны', icon: FileText, allowedPermissions: ['Шаблоны_просмотр'] },
  { key: '/admin', path: '/admin', label: 'Администрирование', icon: Shield, allowedPermissions: ['Администрирование'] },
];

interface SidebarProps {
  user: { role: UserRole; permissions?: PermissionKey[] } | null;
  collapsed: boolean;
  onToggle: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ user, collapsed, onToggle }) => {
  const location = useLocation();
  const { theme } = useTheme();
  const isDark = theme === 'dark';
  const [hoveredKey, setHoveredKey] = useState<string | null>(null);

  const visibleMenuItems = useMemo(() => {
    if (!user) return [];
    const userPermissions = user.permissions || [];
    return menuRoutes.filter((item) => item.allowedPermissions.some((p) => userPermissions.includes(p)));
  }, [user]);

  const selectedKey = useMemo(() => {
    const directMatch = menuRoutes.find((item) => item.path === location.pathname);
    if (directMatch) return directMatch.key;
    const partialMatch = menuRoutes.find(
      (item) => location.pathname.startsWith(item.path) && item.path !== '/'
    );
    return partialMatch ? partialMatch.key : undefined;
  }, [location.pathname]);

  // Claude Theme — warm cream & amber
  const c = {
    bg:          isDark ? '#1c1917' : '#fafaf9',
    border:      isDark ? '#292524' : '#e7e5e4',
    logoBg:      isDark ? '#292524' : '#f5f5f4',
    logoText:    isDark ? '#fafaf9' : '#1c1917',
    navText:     isDark ? '#a8a29e' : '#78716c',
    navTextHov:  isDark ? '#fafaf9' : '#1c1917',
    navTextAct:  isDark ? '#fcd34d' : '#92400e',
    itemHov:     isDark ? '#292524' : '#f5f5f4',
    itemAct:     isDark ? '#292524' : '#fef3c7',
    dot:         isDark ? '#fcd34d' : '#d97706',
    muted:       isDark ? '#57534e' : '#a8a29e',
    toggleHov:   isDark ? '#292524' : '#f5f5f4',
  };

  return (
    <TooltipProvider delayDuration={0}>
      <aside
        style={{
          position: 'relative',
          display: 'flex',
          flexDirection: 'column',
          height: '100vh',
          backgroundColor: c.bg,
          borderRight: `1px solid ${c.border}`,
          width: collapsed ? '56px' : '232px',
          transition: 'width 0.2s ease',
          flexShrink: 0,
          overflow: 'hidden',
        }}
      >
        {/* Logo */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: collapsed ? 'center' : 'space-between',
          padding: collapsed ? '18px 0' : '18px 14px 18px 16px',
          height: '60px',
          flexShrink: 0,
        }}>
          {!collapsed && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', overflow: 'hidden' }}>
              <div style={{
                width: '28px',
                height: '28px',
                borderRadius: '6px',
                backgroundColor: c.logoBg,
                border: `1px solid ${c.border}`,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                flexShrink: 0,
              }}>
                <BrainCircuit size={16} color={c.navTextAct} />
              </div>
              <span style={{
                fontSize: '14px',
                fontWeight: '600',
                color: c.logoText,
                whiteSpace: 'nowrap',
                letterSpacing: '-0.3px',
              }}>
                A/B Platform
              </span>
            </div>
          )}

          {collapsed && (
            <div style={{
              width: '28px',
              height: '28px',
              borderRadius: '6px',
              backgroundColor: c.logoBg,
              border: `1px solid ${c.border}`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}>
              <BrainCircuit size={16} color={c.navTextAct} />
            </div>
          )}

          {/* {!collapsed && (
            <button
              onClick={onToggle}
              style={{
                width: '24px',
                height: '24px',
                borderRadius: '5px',
                border: 'none',
                backgroundColor: 'transparent',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: c.muted,
                transition: 'background-color 0.15s, color 0.15s',
                flexShrink: 0,
              }}
              onMouseEnter={e => {
                e.currentTarget.style.backgroundColor = c.toggleHov;
                e.currentTarget.style.color = c.navTextHov;
              }}
              onMouseLeave={e => {
                e.currentTarget.style.backgroundColor = 'transparent';
                e.currentTarget.style.color = c.muted;
              }}
            >
              <ChevronLeft size={14} />
            </button>
          )} */}
        </div>

        {/* Divider */}
        <div style={{ height: '1px', backgroundColor: c.border, flexShrink: 0 }} />

        {/* Nav */}
        <nav style={{ flex: 1, padding: '8px 6px', overflowY: 'auto', overflowX: 'hidden' }}>
          {visibleMenuItems.map((item) => {
            const Icon = item.icon;
            const isActive = selectedKey === item.key;
            const isHovered = hoveredKey === item.key;
            const iconColor = isActive
              ? c.navTextAct
              : isHovered ? c.navTextHov : c.navText;

            const itemContent = (
              <Link
                to={item.path}
                key={item.key}
                onMouseEnter={() => setHoveredKey(item.key)}
                onMouseLeave={() => setHoveredKey(null)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '9px',
                  padding: collapsed ? '8px 0' : '7px 10px',
                  margin: '1px 0',
                  borderRadius: '6px',
                  textDecoration: 'none',
                  backgroundColor: isActive
                    ? c.itemAct
                    : isHovered ? c.itemHov : 'transparent',
                  transition: 'background-color 0.12s',
                  justifyContent: collapsed ? 'center' : 'flex-start',
                }}
              >
                <Icon size={17} color={iconColor} strokeWidth={isActive ? 2.2 : 1.8} />

                {!collapsed && (
                  <span style={{
                    fontSize: '16px',
                    fontWeight: isActive ? '600' : '500',
                    color: isActive ? c.navTextAct : isHovered ? c.navTextHov : c.navText,
                    whiteSpace: 'nowrap',
                    letterSpacing: '-0.1px',
                  }}>
                    {item.label}
                  </span>
                )}

                {/* Active dot */}
                {isActive && !collapsed && (
                  <div style={{
                    marginLeft: 'auto',
                    width: '5px',
                    height: '5px',
                    borderRadius: '50%',
                    backgroundColor: c.dot,
                    flexShrink: 0,
                  }} />
                )}
              </Link>
            );

            if (collapsed) {
              return (
                <Tooltip key={item.key}>
                  <TooltipTrigger asChild>{itemContent}</TooltipTrigger>
                  <TooltipContent side="right" className="text-xs font-medium">
                    {item.label}
                  </TooltipContent>
                </Tooltip>
              );
            }

            return <React.Fragment key={item.key}>{itemContent}</React.Fragment>;
          })}
        </nav>

        {/* Footer */}
        <div style={{
          padding: collapsed ? '12px 0' : '12px 16px',
          borderTop: `1px solid ${c.border}`,
          flexShrink: 0,
          display: 'flex',
          alignItems: 'center',
          justifyContent: collapsed ? 'center' : 'space-between',
        }}>
          {!collapsed && (
            <span style={{ fontSize: '11px', color: c.muted, letterSpacing: '0.1px' }}>
              SSTU © 2026
            </span>
          )}
          <button
            onClick={onToggle}
            style={{
              width: '24px',
              height: '24px',
              borderRadius: '5px',
              border: 'none',
              backgroundColor: 'transparent',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: c.muted,
              transition: 'background-color 0.15s, color 0.15s',
            }}
            onMouseEnter={e => {
              e.currentTarget.style.backgroundColor = c.toggleHov;
              e.currentTarget.style.color = c.navTextHov;
            }}
            onMouseLeave={e => {
              e.currentTarget.style.backgroundColor = 'transparent';
              e.currentTarget.style.color = c.muted;
            }}
          >
            {collapsed ? <ChevronRight size={14} /> : <ChevronLeft size={14} />}
          </button>
        </div>
      </aside>
    </TooltipProvider>
  );
};
