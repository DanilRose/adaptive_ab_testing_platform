import React, { useEffect, useMemo, useState } from 'react';
import {
  FileText,
  Plus,
  RefreshCw,
  Eye,
  Edit2,
  Copy,
  Trash2,
  X,
  Settings,
  Database,
  FlaskConical,
  ChevronDown,
  Check,
  AlertTriangle,
} from 'lucide-react';
import { templatesAPI } from '../../utils/api';
import { useTheme } from '@/context/ThemeContext';
import { useAuth } from '@/context/AuthContext';

/* ────────────────────── types ────────────────────── */

type TemplateType = 'gan_config' | 'synthetic_data' | 'ab_test';

interface Template {
  id: number;
  name: string;
  description?: string;
  template_type: TemplateType;
  config_json: Record<string, any>;
  tags: string[];
  created_by?: string;
  created_at: string;
  updated_at: string;
}

/* ────────────────────── meta ────────────────────── */

const TYPE_META: Record<TemplateType, { label: string; icon: React.ReactNode; desc: string }> = {
  gan_config: {
    label: 'GAN конфигурация',
    icon: <Settings size={14} />,
    desc: 'Параметры обучения GAN-модели: эпохи, batch-size, learning rate и архитектура.',
  },
  synthetic_data: {
    label: 'Синтетические данные',
    icon: <Database size={14} />,
    desc: 'Настройки генерации синтетических данных: объём, распределения и фильтры.',
  },
  ab_test: {
    label: 'A/B тест',
    icon: <FlaskConical size={14} />,
    desc: 'Конфигурация A/B теста: метрики, варианты, доверие и распределение трафика.',
  },
};

/* ────────────────────── small shared UI ────────────────────── */

interface ToastProps {
  message: string;
  kind: 'success' | 'error';
  onClose: () => void;
}
const Toast: React.FC<ToastProps> = ({ message, kind, onClose }) => (
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
      animation: 'fadeIn 0.18s ease',
    }}
  >
    {kind === 'success' ? <Check size={15} /> : <AlertTriangle size={15} />}
    {message}
    <button
      onClick={onClose}
      style={{ marginLeft: 6, background: 'none', border: 'none', color: '#fff', cursor: 'pointer', padding: 0, display: 'flex' }}
    >
      <X size={14} />
    </button>
  </div>
);

/* ────────────────────── main component ────────────────────── */

export const TemplatesPage: React.FC = () => {
  const { theme } = useTheme();
  const { user } = useAuth();
  const isDark = theme === 'dark';

  const [templates, setTemplates] = useState<Template[]>([]);
  const [loading, setLoading] = useState(false);
  const [activeType, setActiveType] = useState<'all' | TemplateType>('all');

  /* modals */
  const [createOpen, setCreateOpen] = useState(false);
  const [editTarget, setEditTarget] = useState<Template | null>(null);
  const [viewTarget, setViewTarget] = useState<Template | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<Template | null>(null);

  /* form state */
  const emptyForm = { name: '', description: '', template_type: '' as TemplateType | '', tags: '', config_json: '' };
  const [form, setForm] = useState({ ...emptyForm });
  const [editForm, setEditForm] = useState({ ...emptyForm });
  const [formErrors, setFormErrors] = useState<Record<string, string>>({});
  const [editErrors, setEditErrors] = useState<Record<string, string>>({});
  const [saving, setSaving] = useState(false);

  /* toast */
  const [toast, setToast] = useState<{ message: string; kind: 'success' | 'error' } | null>(null);

  /* hovering rows */
  const [hoveredRow, setHoveredRow] = useState<number | null>(null);
  const [hoveredBtn, setHoveredBtn] = useState<string | null>(null);

  const permissions = user?.permissions || [];
  const canViewTemplates = permissions.includes('Шаблоны_просмотр');
  const canCreateTemplates = permissions.includes('Шаблоны_создание');
  const canEditTemplates = permissions.includes('Шаблоны_редактирование');
  const canDeleteTemplates = permissions.includes('Шаблоны_удаление');

  /* ── colours (same warm palette as Sidebar / LoginPage) ── */
  const c = useMemo(() => ({
    pageBg:      isDark ? '#0f0d0b' : '#faf8f5',
    panelBg:     isDark ? '#1c1917' : '#ffffff',
    panelSoft:   isDark ? '#171412' : '#f5f0e8',
    border:      isDark ? '#292524' : '#e7e5e4',
    textPrimary: isDark ? '#fafaf9' : '#1c1917',
    textMuted:   isDark ? '#a8a29e' : '#78716c',
    textSub:     isDark ? '#57534e' : '#a8a29e',
    inputBg:     isDark ? '#292524' : '#fafaf9',
    inputBorder: isDark ? '#3c3330' : '#e7e5e4',
    inputFocus:  '#d97706',
    accent:      '#d97706',
    accentHov:   '#b45309',
    accentSoft:  isDark ? 'rgba(217,119,6,0.16)' : '#fef3c7',
    accentText:  isDark ? '#fcd34d' : '#92400e',
    danger:      isDark ? '#fca5a5' : '#dc2626',
    dangerSoft:  isDark ? 'rgba(239,68,68,0.12)' : '#fef2f2',
    dangerBorder:isDark ? 'rgba(239,68,68,0.25)' : '#fecaca',
    shadow:      isDark ? '0 10px 32px rgba(0,0,0,0.36)' : '0 8px 28px rgba(28,25,23,0.07)',
    rowHov:      isDark ? '#211f1d' : '#faf8f5',
  }), [isDark]);

  const typeAccent = useMemo(() => ({
    gan_config: {
      border: isDark ? '#7c3aed' : '#7c3aed',
      bg: isDark ? 'rgba(124,58,237,0.14)' : '#f5f3ff',
      text: isDark ? '#c4b5fd' : '#5b21b6',
      dot: '#7c3aed',
    },
    synthetic_data: {
      border: isDark ? '#0284c7' : '#0284c7',
      bg: isDark ? 'rgba(2,132,199,0.14)' : '#f0f9ff',
      text: isDark ? '#7dd3fc' : '#0c4a6e',
      dot: '#0284c7',
    },
    ab_test: {
      border: isDark ? '#16a34a' : '#16a34a',
      bg: isDark ? 'rgba(22,163,74,0.14)' : '#f0fdf4',
      text: isDark ? '#86efac' : '#14532d',
      dot: '#16a34a',
    },
  }), [isDark]);

  /* ── helpers ── */

  const showToast = (message: string, kind: 'success' | 'error') => {
    setToast({ message, kind });
    setTimeout(() => setToast(null), 3200);
  };

  const fmtDate = (s: string) => s ? new Date(s).toLocaleString('ru-RU') : '—';

  const filteredTemplates = activeType === 'all'
    ? templates
    : templates.filter(t => t.template_type === activeType);

  const countOf = (type: TemplateType) => templates.filter(t => t.template_type === type).length;

  /* ── API ── */

  const loadTemplates = async () => {
    setLoading(true);
    try {
      const res = await templatesAPI.listTemplates();
      setTemplates(res.data.items || []);
    } catch (err: any) {
      showToast(`Ошибка загрузки: ${err.response?.data?.detail || err.message}`, 'error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { loadTemplates(); }, []);

  /* ── validation ── */

  const validateForm = (f: typeof emptyForm, isEdit = false) => {
    const errs: Record<string, string> = {};
    if (!f.name.trim()) errs.name = 'Укажите название';
    if (!isEdit && !f.template_type) errs.template_type = 'Выберите тип';
    if (!f.config_json.trim()) {
      errs.config_json = 'Укажите конфигурацию';
    } else {
      try { JSON.parse(f.config_json); } catch { errs.config_json = 'Некорректный JSON'; }
    }
    return errs;
  };

  /* ── handlers ── */

  const handleCreate = async () => {
    const errs = validateForm(form);
    if (Object.keys(errs).length) { setFormErrors(errs); return; }
    setSaving(true);
    try {
      await templatesAPI.createTemplate({
        name: form.name,
        description: form.description || null,
        template_type: form.template_type as TemplateType,
        config_json: JSON.parse(form.config_json),
        tags: form.tags.split(',').map(t => t.trim()).filter(Boolean),
      });
      showToast('Шаблон создан', 'success');
      setCreateOpen(false);
      setForm({ ...emptyForm });
      setFormErrors({});
      loadTemplates();
    } catch (err: any) {
      showToast(`Ошибка: ${err.response?.data?.detail || err.message}`, 'error');
    } finally {
      setSaving(false);
    }
  };

  const handleEdit = async () => {
    const errs = validateForm(editForm, true);
    if (Object.keys(errs).length) { setEditErrors(errs); return; }
    if (!editTarget) return;
    setSaving(true);
    try {
      await templatesAPI.updateTemplate(editTarget.id, {
        name: editForm.name,
        description: editForm.description,
        config_json: JSON.parse(editForm.config_json),
        tags: editForm.tags.split(',').map(t => t.trim()).filter(Boolean),
      });
      showToast('Шаблон обновлён', 'success');
      setEditTarget(null);
      setEditErrors({});
      loadTemplates();
    } catch (err: any) {
      showToast(`Ошибка: ${err.response?.data?.detail || err.message}`, 'error');
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async () => {
    if (!deleteTarget) return;
    try {
      await templatesAPI.deleteTemplate(deleteTarget.id);
      showToast('Шаблон удалён', 'success');
      setDeleteTarget(null);
      loadTemplates();
    } catch (err: any) {
      showToast(`Ошибка: ${err.response?.data?.detail || err.message}`, 'error');
    }
  };

  const handleDuplicate = async (t: Template) => {
    try {
      await templatesAPI.createTemplate({
        name: `${t.name} (копия)`,
        description: t.description,
        template_type: t.template_type,
        config_json: t.config_json,
        tags: t.tags,
      });
      showToast('Шаблон продублирован', 'success');
      loadTemplates();
    } catch (err: any) {
      showToast(`Ошибка: ${err.response?.data?.detail || err.message}`, 'error');
    }
  };

  const openEdit = (t: Template) => {
    setEditForm({
      name: t.name,
      description: t.description || '',
      template_type: t.template_type,
      tags: (t.tags || []).join(', '),
      config_json: JSON.stringify(t.config_json, null, 2),
    });
    setEditErrors({});
    setEditTarget(t);
  };

  /* ── inline style helpers ── */

  const inputStyle = (focused: boolean, error?: string): React.CSSProperties => ({
    width: '100%',
    padding: '9px 12px',
    borderRadius: 8,
    border: `1.5px solid ${error ? c.danger : focused ? c.inputFocus : c.inputBorder}`,
    backgroundColor: c.inputBg,
    color: c.textPrimary,
    fontSize: 14,
    outline: 'none',
    boxSizing: 'border-box',
    transition: 'border-color 0.15s',
    boxShadow: focused ? `0 0 0 3px rgba(217,119,6,0.12)` : 'none',
    fontFamily: 'inherit',
  });

  const labelStyle: React.CSSProperties = {
    display: 'block',
    fontSize: 13,
    fontWeight: 600,
    color: c.textMuted,
    marginBottom: 5,
    letterSpacing: '0.1px',
  };

  const errorStyle: React.CSSProperties = {
    fontSize: 12,
    color: c.danger,
    marginTop: 4,
  };

  /* ─────────────── render ─────────────── */

  return (
    <div style={{ color: c.textPrimary, fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" }}>

      {/* ── HEADER ── */}
      <div style={{
        display: 'flex',
        alignItems: 'flex-start',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: 12,
        marginBottom: 24,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{
            width: 40,
            height: 40,
            borderRadius: 10,
            backgroundColor: c.accentSoft,
            border: `1px solid ${isDark ? 'rgba(217,119,6,0.25)' : '#fde68a'}`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: c.accentText,
            flexShrink: 0,
          }}>
            <FileText size={20} />
          </div>
          <div>
            <h1 style={{ margin: 0, fontSize: 22, fontWeight: 700, letterSpacing: '-0.4px', color: c.textPrimary }}>
              Шаблоны
            </h1>
            <p style={{ margin: 0, fontSize: 13, color: c.textMuted }}>
              Сохранённые конфигурации для GAN, синтетических данных и A/B тестов
            </p>
          </div>
        </div>

        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <ActionButton
            onClick={loadTemplates}
            disabled={loading}
            isDark={isDark}
            c={c}
            icon={<RefreshCw size={14} style={{ animation: loading ? 'spin 0.8s linear infinite' : 'none' }} />}
            label="Обновить"
            variant="secondary"
          />
          {canCreateTemplates && (
            <ActionButton
              onClick={() => { setCreateOpen(true); setForm({ ...emptyForm }); setFormErrors({}); }}
              isDark={isDark}
              c={c}
              icon={<Plus size={14} />}
              label="Создать шаблон"
              variant="primary"
            />
          )}
        </div>
      </div>

      {/* ── TYPE CARDS ── */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 14, marginBottom: 20 }}>
        {(Object.keys(TYPE_META) as TemplateType[]).map(type => {
          const meta = TYPE_META[type];
          const accent = typeAccent[type];
          const isActive = activeType === type;
          return (
            <button
              key={type}
              onClick={() => setActiveType(isActive ? 'all' : type)}
              style={{
                padding: '14px 16px',
                borderRadius: 12,
                border: `1.5px solid ${isActive ? accent.border : c.border}`,
                backgroundColor: isActive ? accent.bg : c.panelBg,
                cursor: 'pointer',
                textAlign: 'left',
                transition: 'border-color 0.15s, background-color 0.15s',
                position: 'relative',
              }}
              onMouseEnter={e => {
                if (!isActive) {
                  e.currentTarget.style.borderColor = accent.border;
                  e.currentTarget.style.backgroundColor = isDark ? '#211f1d' : '#faf8f5';
                }
              }}
              onMouseLeave={e => {
                if (!isActive) {
                  e.currentTarget.style.borderColor = c.border;
                  e.currentTarget.style.backgroundColor = c.panelBg;
                }
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 8 }}>
                <div>
                  <div style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: 6,
                    padding: '3px 10px',
                    borderRadius: 999,
                    border: `1px solid ${accent.border}`,
                    color: accent.text,
                    fontSize: 12,
                    fontWeight: 600,
                    marginBottom: 8,
                  }}>
                    {meta.icon}
                    {meta.label}
                  </div>
                  <p style={{ margin: 0, fontSize: 12, color: c.textMuted, lineHeight: 1.5 }}>
                    {meta.desc}
                  </p>
                </div>
                <span style={{
                  flexShrink: 0,
                  minWidth: 24,
                  height: 24,
                  borderRadius: 999,
                  backgroundColor: isActive ? accent.dot : c.textSub,
                  color: '#fff',
                  fontSize: 12,
                  fontWeight: 700,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  padding: '0 6px',
                }}>
                  {countOf(type)}
                </span>
              </div>
            </button>
          );
        })}
      </div>

      {/* ── FILTER PILLS ── */}
      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 16 }}>
        {([['all', `Все (${templates.length})`], ...
          (Object.keys(TYPE_META) as TemplateType[]).map(t => [t, `${TYPE_META[t].label} (${countOf(t)})`])
        ] as [string, string][]).map(([key, label]) => {
          const isActive = activeType === key;
          const acc = key !== 'all' ? typeAccent[key as TemplateType] : null;
          return (
            <button
              key={key}
              onClick={() => setActiveType(key as 'all' | TemplateType)}
              style={{
                padding: '4px 14px',
                borderRadius: 999,
                border: `1px solid ${isActive ? (acc?.border ?? c.accent) : c.border}`,
                backgroundColor: isActive ? (acc?.bg ?? c.accentSoft) : c.panelBg,
                color: isActive ? (acc?.text ?? c.accentText) : c.textMuted,
                fontSize: 13,
                fontWeight: isActive ? 600 : 400,
                cursor: 'pointer',
                transition: 'all 0.12s',
              }}
            >
              {label}
            </button>
          );
        })}
      </div>

      {/* ── TABLE PANEL ── */}
      <div style={{
        borderRadius: 14,
        border: `1px solid ${c.border}`,
        backgroundColor: c.panelBg,
        boxShadow: c.shadow,
        overflow: 'hidden',
      }}>
        {loading ? (
          <div style={{ padding: '64px 24px', textAlign: 'center', color: c.textMuted }}>
            <RefreshCw size={28} style={{ animation: 'spin 0.8s linear infinite', marginBottom: 12 }} />
            <p style={{ margin: 0, fontSize: 14 }}>Загрузка шаблонов…</p>
          </div>
        ) : filteredTemplates.length === 0 ? (
          <div style={{ padding: '64px 24px', textAlign: 'center' }}>
            <FileText size={36} color={c.textSub} style={{ marginBottom: 12 }} />
            <p style={{ margin: '0 0 16px', color: c.textMuted, fontSize: 15 }}>Шаблоны не найдены</p>
            {canCreateTemplates && (
              <ActionButton
                onClick={() => { setCreateOpen(true); setForm({ ...emptyForm }); setFormErrors({}); }}
                isDark={isDark}
                c={c}
                icon={<Plus size={14} />}
                label="Создать первый шаблон"
                variant="primary"
              />
            )}
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 14 }}>
              <thead>
                <tr style={{ backgroundColor: c.panelSoft, borderBottom: `1px solid ${c.border}` }}>
                  {['Название', 'Тип', 'Теги', 'Создал', 'Обновлено', 'Действия'].map(h => (
                    <th
                      key={h}
                      style={{
                        padding: '11px 16px',
                        textAlign: 'left',
                        fontSize: 11,
                        fontWeight: 600,
                        color: c.textSub,
                        letterSpacing: '0.4px',
                        textTransform: 'uppercase',
                        whiteSpace: 'nowrap',
                      }}
                    >
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {filteredTemplates.map((tpl, idx) => {
                  const acc = typeAccent[tpl.template_type];
                  const isHov = hoveredRow === tpl.id;
                  return (
                    <tr
                      key={tpl.id}
                      onMouseEnter={() => setHoveredRow(tpl.id)}
                      onMouseLeave={() => setHoveredRow(null)}
                      style={{
                        backgroundColor: isHov ? c.rowHov : 'transparent',
                        borderBottom: idx < filteredTemplates.length - 1 ? `1px solid ${c.border}` : 'none',
                        transition: 'background-color 0.1s',
                      }}
                    >
                      {/* name */}
                      <td style={{ padding: '12px 16px', minWidth: 180 }}>
                        <div style={{ fontWeight: 600, color: c.textPrimary }}>{tpl.name}</div>
                        {tpl.description && (
                          <div style={{ fontSize: 12, color: c.textMuted, marginTop: 2 }}>
                            {tpl.description.length > 80 ? `${tpl.description.slice(0, 80)}…` : tpl.description}
                          </div>
                        )}
                      </td>

                      {/* type */}
                      <td style={{ padding: '12px 16px', whiteSpace: 'nowrap' }}>
                        <span style={{
                          display: 'inline-flex', alignItems: 'center', gap: 5,
                          padding: '3px 10px', borderRadius: 999,
                          border: `1px solid ${acc.border}`,
                          backgroundColor: acc.bg, color: acc.text,
                          fontSize: 12, fontWeight: 600,
                        }}>
                          {TYPE_META[tpl.template_type].icon}
                          {TYPE_META[tpl.template_type].label}
                        </span>
                      </td>

                      {/* tags */}
                      <td style={{ padding: '12px 16px', maxWidth: 200 }}>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                          {(tpl.tags || []).map(tag => (
                            <span key={tag} style={{
                              padding: '2px 8px', borderRadius: 999,
                              border: `1px solid ${c.border}`,
                              backgroundColor: c.panelSoft,
                              color: c.textMuted, fontSize: 11,
                            }}>
                              {tag}
                            </span>
                          ))}
                          {(!tpl.tags || tpl.tags.length === 0) && (
                            <span style={{ color: c.textSub, fontSize: 12 }}>—</span>
                          )}
                        </div>
                      </td>

                      {/* created_by */}
                      <td style={{ padding: '12px 16px', color: c.textMuted, whiteSpace: 'nowrap' }}>
                        {tpl.created_by || '—'}
                      </td>

                      {/* updated_at */}
                      <td style={{ padding: '12px 16px', color: c.textMuted, whiteSpace: 'nowrap', fontSize: 13 }}>
                        {fmtDate(tpl.updated_at)}
                      </td>

                      {/* actions */}
                      <td style={{ padding: '12px 16px', whiteSpace: 'nowrap' }}>
                        <div style={{ display: 'flex', gap: 4 }}>
                          {[
                            ...(canViewTemplates ? [{ id: `view-${tpl.id}`, icon: <Eye size={14} />, title: 'Просмотреть', onClick: () => setViewTarget(tpl) }] : []),
                            ...(canEditTemplates ? [
                              { id: `edit-${tpl.id}`, icon: <Edit2 size={14} />, title: 'Редактировать', onClick: () => openEdit(tpl) },
                              { id: `dup-${tpl.id}`, icon: <Copy size={14} />, title: 'Дублировать', onClick: () => handleDuplicate(tpl) },
                            ] : []),
                          ].map(btn => (
                            <button
                              key={btn.id}
                              title={btn.title}
                              onClick={btn.onClick}
                              onMouseEnter={() => setHoveredBtn(btn.id)}
                              onMouseLeave={() => setHoveredBtn(null)}
                              style={{
                                width: 30, height: 30,
                                borderRadius: 7,
                                border: `1px solid ${hoveredBtn === btn.id ? c.border : 'transparent'}`,
                                backgroundColor: hoveredBtn === btn.id ? c.panelSoft : 'transparent',
                                color: c.textMuted,
                                cursor: 'pointer',
                                display: 'flex', alignItems: 'center', justifyContent: 'center',
                                transition: 'all 0.12s',
                              }}
                            >
                              {btn.icon}
                            </button>
                          ))}
                          {canDeleteTemplates && (
                            <button
                              title="Удалить"
                              onClick={() => setDeleteTarget(tpl)}
                              onMouseEnter={() => setHoveredBtn(`del-${tpl.id}`)}
                              onMouseLeave={() => setHoveredBtn(null)}
                              style={{
                                width: 30, height: 30,
                                borderRadius: 7,
                                border: `1px solid ${hoveredBtn === `del-${tpl.id}` ? c.dangerBorder : 'transparent'}`,
                                backgroundColor: hoveredBtn === `del-${tpl.id}` ? c.dangerSoft : 'transparent',
                                color: hoveredBtn === `del-${tpl.id}` ? c.danger : c.textMuted,
                                cursor: 'pointer',
                                display: 'flex', alignItems: 'center', justifyContent: 'center',
                                transition: 'all 0.12s',
                              }}
                            >
                              <Trash2 size={14} />
                            </button>
                          )}
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* ── CREATE MODAL ── */}
      {createOpen && (
        <TemplateModal
          title="Создать новый шаблон"
          icon={<Plus size={16} />}
          isDark={isDark}
          c={c}
          saving={saving}
          onClose={() => { setCreateOpen(false); setFormErrors({}); }}
          onConfirm={handleCreate}
          confirmLabel="Создать"
        >
          <FormBody
            f={form}
            onChange={v => setForm(prev => ({ ...prev, ...v }))}
            errors={formErrors}
            isDark={isDark}
            c={c}
            inputStyle={inputStyle}
            labelStyle={labelStyle}
            errorStyle={errorStyle}
            showTypeField
          />
        </TemplateModal>
      )}

      {/* ── EDIT MODAL ── */}
      {editTarget && (
        <TemplateModal
          title={`Редактировать: ${editTarget.name}`}
          icon={<Edit2 size={16} />}
          isDark={isDark}
          c={c}
          saving={saving}
          onClose={() => { setEditTarget(null); setEditErrors({}); }}
          onConfirm={handleEdit}
          confirmLabel="Сохранить"
        >
          <FormBody
            f={editForm}
            onChange={v => setEditForm(prev => ({ ...prev, ...v }))}
            errors={editErrors}
            isDark={isDark}
            c={c}
            inputStyle={inputStyle}
            labelStyle={labelStyle}
            errorStyle={errorStyle}
          />
        </TemplateModal>
      )}

      {/* ── VIEW MODAL ── */}
      {viewTarget && (
        <Overlay onClose={() => setViewTarget(null)} c={c}>
          <ModalPanel width={740} c={c} isDark={isDark}>
            <ModalHeader
              icon={<Eye size={16} />}
              title={viewTarget.name}
              onClose={() => setViewTarget(null)}
              c={c}
            />
            <div style={{ padding: '20px 24px', overflowY: 'auto', maxHeight: 'calc(90vh - 130px)' }}>
              {/* meta grid */}
              <div style={{
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: 1,
                borderRadius: 10,
                border: `1px solid ${c.border}`,
                overflow: 'hidden',
                marginBottom: 20,
              }}>
                {[
                  ['Тип', (
                    <span style={{
                      display: 'inline-flex', alignItems: 'center', gap: 5,
                      padding: '3px 10px', borderRadius: 999,
                      border: `1px solid ${typeAccent[viewTarget.template_type].border}`,
                      backgroundColor: typeAccent[viewTarget.template_type].bg,
                      color: typeAccent[viewTarget.template_type].text,
                      fontSize: 12, fontWeight: 600,
                    }}>
                      {TYPE_META[viewTarget.template_type].icon}
                      {TYPE_META[viewTarget.template_type].label}
                    </span>
                  )],
                  ['Создал', viewTarget.created_by || '—'],
                  ['Создан', fmtDate(viewTarget.created_at)],
                  ['Обновлён', fmtDate(viewTarget.updated_at)],
                  ...(viewTarget.description ? [['Описание', viewTarget.description]] : []),
                  ...(viewTarget.tags?.length ? [['Теги', (
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                      {viewTarget.tags.map(tag => (
                        <span key={tag} style={{
                          padding: '2px 8px', borderRadius: 999,
                          border: `1px solid ${c.border}`,
                          backgroundColor: c.panelSoft,
                          color: c.textMuted, fontSize: 12,
                        }}>
                          {tag}
                        </span>
                      ))}
                    </div>
                  )]] : []),
                ].map(([label, value], i) => (
                  <div key={i as number} style={{
                    backgroundColor: i % 2 === 0 ? c.panelSoft : c.panelBg,
                    padding: '10px 16px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 8,
                    borderBottom: `1px solid ${c.border}`,
                  }}>
                    <span style={{ fontSize: 12, color: c.textSub, minWidth: 80 }}>{label as string}</span>
                    <span style={{ fontSize: 13, color: c.textPrimary }}>{value as React.ReactNode}</span>
                  </div>
                ))}
              </div>

              <div style={{ fontWeight: 600, fontSize: 13, color: c.textPrimary, marginBottom: 8 }}>
                Конфигурация (JSON):
              </div>
              <pre style={{
                margin: 0,
                backgroundColor: c.panelSoft,
                color: c.textPrimary,
                padding: 16,
                borderRadius: 10,
                fontSize: 12,
                maxHeight: 380,
                overflowY: 'auto',
                border: `1px solid ${c.border}`,
                fontFamily: 'monospace',
                lineHeight: 1.6,
              }}>
                {JSON.stringify(viewTarget.config_json, null, 2)}
              </pre>
            </div>

            {/* footer */}
            <div style={{
              padding: '14px 24px',
              borderTop: `1px solid ${c.border}`,
              display: 'flex',
              justifyContent: 'flex-end',
              gap: 8,
            }}>
              {canEditTemplates && (
                <ActionButton
                  onClick={() => { setViewTarget(null); if (viewTarget) handleDuplicate(viewTarget); }}
                  isDark={isDark}
                  c={c}
                  icon={<Copy size={14} />}
                  label="Дублировать"
                  variant="secondary"
                />
              )}
              {canEditTemplates && (
                <ActionButton
                  onClick={() => { setViewTarget(null); if (viewTarget) openEdit(viewTarget); }}
                  isDark={isDark}
                  c={c}
                  icon={<Edit2 size={14} />}
                  label="Редактировать"
                  variant="primary"
                />
              )}
              <ActionButton
                onClick={() => setViewTarget(null)}
                isDark={isDark}
                c={c}
                label="Закрыть"
                variant="ghost"
              />
            </div>
          </ModalPanel>
        </Overlay>
      )}

      {/* ── DELETE CONFIRM ── */}
      {deleteTarget && (
        <Overlay onClose={() => setDeleteTarget(null)} c={c}>
          <ModalPanel width={400} c={c} isDark={isDark}>
            <ModalHeader
              icon={<Trash2 size={16} />}
              title="Удалить шаблон?"
              onClose={() => setDeleteTarget(null)}
              c={c}
              danger
            />
            <div style={{ padding: '20px 24px' }}>
              <p style={{ margin: 0, fontSize: 14, color: c.textMuted, lineHeight: 1.6 }}>
                Шаблон <span style={{ fontWeight: 600, color: c.textPrimary }}>«{deleteTarget.name}»</span> будет
                удалён безвозвратно. Это действие нельзя отменить.
              </p>
            </div>
            <div style={{
              padding: '14px 24px',
              borderTop: `1px solid ${c.border}`,
              display: 'flex',
              justifyContent: 'flex-end',
              gap: 8,
            }}>
              <ActionButton
                onClick={() => setDeleteTarget(null)}
                isDark={isDark}
                c={c}
                label="Отмена"
                variant="ghost"
              />
              <button
                onClick={handleDelete}
                style={{
                  height: 38,
                  padding: '0 18px',
                  borderRadius: 8,
                  border: `1px solid ${c.dangerBorder}`,
                  backgroundColor: isDark ? 'rgba(239,68,68,0.15)' : '#fef2f2',
                  color: c.danger,
                  fontSize: 14,
                  fontWeight: 600,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  transition: 'background-color 0.15s',
                }}
                onMouseEnter={e => {
                  e.currentTarget.style.backgroundColor = isDark ? 'rgba(239,68,68,0.25)' : '#fee2e2';
                }}
                onMouseLeave={e => {
                  e.currentTarget.style.backgroundColor = isDark ? 'rgba(239,68,68,0.15)' : '#fef2f2';
                }}
              >
                <Trash2 size={14} />
                Удалить
              </button>
            </div>
          </ModalPanel>
        </Overlay>
      )}

      {/* Toast */}
      {toast && <Toast message={toast.message} kind={toast.kind} onClose={() => setToast(null)} />}

      <style>{`
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes modalIn { from { opacity: 0; transform: scale(0.96) translateY(8px); } to { opacity: 1; transform: scale(1) translateY(0); } }
      `}</style>
    </div>
  );
};

/* ══════════════════════════════════════════════════════
   Sub-components
══════════════════════════════════════════════════════ */

/* ── Overlay ── */
interface OverlayProps {
  onClose: () => void;
  c?: Record<string, string>;
  children: React.ReactNode;
}
const Overlay: React.FC<OverlayProps> = ({ onClose, children }) => (
  <div
    onClick={e => { if (e.target === e.currentTarget) onClose(); }}
    style={{
      position: 'fixed', inset: 0, zIndex: 1000,
      backgroundColor: 'rgba(0,0,0,0.46)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      padding: 24,
    }}
  >
    {children}
  </div>
);

/* ── ModalPanel ── */
interface ModalPanelProps {
  width: number;
  c: Record<string, string>;
  isDark: boolean;
  children: React.ReactNode;
}
const ModalPanel: React.FC<ModalPanelProps> = ({ width, c, children, isDark: _isDark }) => (
  <div style={{
    width: '100%',
    maxWidth: width,
    maxHeight: '90vh',
    backgroundColor: c.panelBg,
    borderRadius: 14,
    border: `1px solid ${c.border}`,
    boxShadow: '0 20px 60px rgba(0,0,0,0.3)',
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
    animation: 'modalIn 0.18s ease',
  }}>
    {children}
  </div>
);

/* ── ModalHeader ── */
interface ModalHeaderProps {
  icon?: React.ReactNode;
  title: string;
  onClose: () => void;
  c: Record<string, string>;
  danger?: boolean;
}
const ModalHeader: React.FC<ModalHeaderProps> = ({ icon, title, onClose, c, danger }) => (
  <div style={{
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    padding: '16px 24px',
    borderBottom: `1px solid ${c.border}`,
    backgroundColor: c.panelSoft,
    flexShrink: 0,
  }}>
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      {icon && (
        <span style={{
          color: danger ? c.danger : c.accentText,
          display: 'flex', alignItems: 'center',
        }}>
          {icon}
        </span>
      )}
      <span style={{ fontSize: 16, fontWeight: 700, color: c.textPrimary, letterSpacing: '-0.2px' }}>
        {title}
      </span>
    </div>
    <button
      onClick={onClose}
      style={{
        width: 30, height: 30, borderRadius: 7,
        border: 'none', backgroundColor: 'transparent',
        cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
        color: c.textMuted, transition: 'background-color 0.12s',
      }}
      onMouseEnter={e => e.currentTarget.style.backgroundColor = c.border}
      onMouseLeave={e => e.currentTarget.style.backgroundColor = 'transparent'}
    >
      <X size={16} />
    </button>
  </div>
);

/* ── TemplateModal (create / edit wrapper) ── */
interface TemplateModalProps {
  title: string;
  icon?: React.ReactNode;
  isDark: boolean;
  c: Record<string, string>;
  saving: boolean;
  onClose: () => void;
  onConfirm: () => void;
  confirmLabel: string;
  children: React.ReactNode;
}
const TemplateModal: React.FC<TemplateModalProps> = ({
  title, icon, isDark, c, saving, onClose, onConfirm, confirmLabel, children
}) => (
  <Overlay onClose={onClose} c={c}>
    <ModalPanel width={720} c={c} isDark={isDark}>
      <ModalHeader icon={icon} title={title} onClose={onClose} c={c} />
      <div style={{ padding: '20px 24px', overflowY: 'auto', flex: 1 }}>
        {children}
      </div>
      <div style={{
        padding: '14px 24px',
        borderTop: `1px solid ${c.border}`,
        display: 'flex', justifyContent: 'flex-end', gap: 8,
        flexShrink: 0,
      }}>
        <ActionButton
          onClick={onClose}
          isDark={isDark}
          c={c}
          label="Отмена"
          variant="ghost"
        />
        <ActionButton
          onClick={onConfirm}
          disabled={saving}
          isDark={isDark}
          c={c}
          icon={saving
            ? <RefreshCw size={14} style={{ animation: 'spin 0.8s linear infinite' }} />
            : <Check size={14} />}
          label={saving ? 'Сохранение…' : confirmLabel}
          variant="primary"
        />
      </div>
    </ModalPanel>
  </Overlay>
);

/* ── FormBody ── */
type FormState = { name: string; description: string; template_type: TemplateType | ''; tags: string; config_json: string };
interface FormBodyProps {
  f: FormState;
  onChange: (v: Partial<FormState>) => void;
  errors: Record<string, string>;
  isDark: boolean;
  c: Record<string, string>;
  inputStyle: (focused: boolean, error?: string) => React.CSSProperties;
  labelStyle: React.CSSProperties;
  errorStyle: React.CSSProperties;
  showTypeField?: boolean;
}

const FormBody: React.FC<FormBodyProps> = ({ f, onChange, errors, c, inputStyle, labelStyle, errorStyle, showTypeField }) => {
  const [focused, setFocused] = useState<string | null>(null);
  const [typeOpen, setTypeOpen] = useState(false);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* row: name + type */}
      <div style={{ display: 'grid', gridTemplateColumns: showTypeField ? '1fr 200px' : '1fr', gap: 14 }}>
        <div>
          <label style={labelStyle}>Название шаблона *</label>
          <input
            value={f.name}
            onChange={e => onChange({ name: e.target.value })}
            onFocus={() => setFocused('name')}
            onBlur={() => setFocused(null)}
            placeholder="Например: GAN — Быстрое обучение"
            style={inputStyle(focused === 'name', errors.name)}
          />
          {errors.name && <p style={errorStyle}>{errors.name}</p>}
        </div>

        {showTypeField && (
          <div style={{ position: 'relative' }}>
            <label style={labelStyle}>Тип шаблона *</label>
            <button
              type="button"
              onClick={() => setTypeOpen(v => !v)}
              style={{
                ...inputStyle(focused === 'type', errors.template_type),
                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                cursor: 'pointer', height: 40,
              }}
            >
              <span style={{ color: f.template_type ? c.textPrimary : c.textSub }}>
                {f.template_type ? TYPE_META[f.template_type as TemplateType].label : 'Выберите тип'}
              </span>
              <ChevronDown size={14} color={c.textMuted} />
            </button>
            {errors.template_type && <p style={errorStyle}>{errors.template_type}</p>}
            {typeOpen && (
              <div style={{
                position: 'absolute', top: 'calc(100% + 4px)', left: 0, right: 0,
                backgroundColor: c.panelBg,
                border: `1px solid ${c.border}`,
                borderRadius: 8,
                boxShadow: '0 8px 24px rgba(0,0,0,0.15)',
                zIndex: 100,
                overflow: 'hidden',
              }}>
                {(Object.keys(TYPE_META) as TemplateType[]).map(type => (
                  <button
                    key={type}
                    type="button"
                    onClick={() => { onChange({ template_type: type }); setTypeOpen(false); }}
                    style={{
                      width: '100%', padding: '10px 14px',
                      display: 'flex', alignItems: 'center', gap: 8,
                      backgroundColor: f.template_type === type ? c.accentSoft : 'transparent',
                      border: 'none',
                      color: f.template_type === type ? c.accentText : c.textPrimary,
                      fontSize: 14, cursor: 'pointer', textAlign: 'left',
                    }}
                    onMouseEnter={e => { if (f.template_type !== type) e.currentTarget.style.backgroundColor = c.panelSoft; }}
                    onMouseLeave={e => { if (f.template_type !== type) e.currentTarget.style.backgroundColor = 'transparent'; }}
                  >
                    {TYPE_META[type].icon}
                    {TYPE_META[type].label}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* description */}
      <div>
        <label style={labelStyle}>Описание</label>
        <textarea
          value={f.description}
          onChange={e => onChange({ description: e.target.value })}
          onFocus={() => setFocused('description')}
          onBlur={() => setFocused(null)}
          rows={2}
          placeholder="Краткое описание назначения шаблона"
          style={{
            ...inputStyle(focused === 'description'),
            resize: 'vertical',
            minHeight: 70,
            lineHeight: 1.5,
          }}
        />
      </div>

      {/* tags */}
      <div>
        <label style={labelStyle}>Теги <span style={{ fontWeight: 400, color: c.textSub }}>(через запятую)</span></label>
        <input
          value={f.tags}
          onChange={e => onChange({ tags: e.target.value })}
          onFocus={() => setFocused('tags')}
          onBlur={() => setFocused(null)}
          placeholder="быстрый, прототип, мобильные"
          style={inputStyle(focused === 'tags')}
        />
      </div>

      {/* config_json */}
      <div>
        <label style={labelStyle}>
          Конфигурация JSON *
          <span style={{ fontWeight: 400, color: c.textSub, marginLeft: 6 }}>
            — например: {'{'}«epochs»: 50{'}'}
          </span>
        </label>
        <textarea
          value={f.config_json}
          onChange={e => onChange({ config_json: e.target.value })}
          onFocus={() => setFocused('config_json')}
          onBlur={() => setFocused(null)}
          rows={10}
          placeholder={'{\n  "epochs": 50,\n  "batch_size": 256\n}'}
          style={{
            ...inputStyle(focused === 'config_json', errors.config_json),
            fontFamily: "'JetBrains Mono', 'Fira Mono', monospace",
            fontSize: 12,
            resize: 'vertical',
            minHeight: 200,
            lineHeight: 1.6,
          }}
        />
        {errors.config_json && <p style={errorStyle}>{errors.config_json}</p>}
      </div>
    </div>
  );
};

/* ── ActionButton ── */
interface ActionButtonProps {
  onClick: () => void;
  disabled?: boolean;
  isDark: boolean;
  c: Record<string, string>;
  icon?: React.ReactNode;
  label?: string;
  variant: 'primary' | 'secondary' | 'ghost';
}
const ActionButton: React.FC<ActionButtonProps> = ({ onClick, disabled, isDark, c, icon, label, variant }) => {
  const [hov, setHov] = useState(false);

  const bg =
    variant === 'primary'
      ? disabled ? (isDark ? '#292524' : '#e7e5e4') : hov ? c.accentHov : c.accent
      : variant === 'secondary'
      ? hov ? (isDark ? '#292524' : '#f0ede8') : c.panelBg
      : hov ? (isDark ? '#292524' : '#f0ede8') : 'transparent';

  const border =
    variant === 'primary'
      ? disabled ? (isDark ? '#292524' : '#e7e5e4') : hov ? c.accentHov : c.accent
      : variant === 'secondary'
      ? c.border
      : 'transparent';

  const color =
    variant === 'primary'
      ? disabled ? c.textMuted : '#fff'
      : c.textMuted;

  return (
    <button
      onClick={disabled ? undefined : onClick}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        height: 38,
        padding: '0 16px',
        borderRadius: 8,
        border: `1px solid ${border}`,
        backgroundColor: bg,
        color,
        fontSize: 14,
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
