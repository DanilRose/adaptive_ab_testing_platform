import React, { useState, useEffect, useRef, useMemo } from 'react';
import {
  Plus,
  RefreshCw,
  Trash2,
  X,
  PlayCircle,
  PauseCircle,
  Folder,
  FileText,
  AlertTriangle,
  Check,
  Zap,
} from 'lucide-react';
import { abTestAPI, dataAPI, templatesAPI } from '../../utils/api';
import { useTheme } from '@/context/ThemeContext';
import { useAuth } from '@/context/AuthContext';

/* ────────────────────── types ────────────────────── */

interface Test {
  test_id: string;
  test_name: string;
  description?: string;
  status: string;
  simulation_status?: string;
  variants: string[];
  primary_metric: string;
  metric_type: string;
  sample_size?: number;
  total_users: number;
  completion_percentage: number;
  created_at: string;
  archive_reason?: string;
}

interface TestsByStatus {
  prepared_tests: Test[];
  active_tests: Test[];
  paused_tests: Test[];
  completed_tests: Test[];
  archived_tests: Test[];
  counts: {
    prepared: number;
    active: number;
    paused: number;
    completed: number;
    archived: number;
  };
}

interface Template {
  id: number;
  name: string;
  description?: string;
  template_type: string;
  config_json: Record<string, any>;
  tags: string[];
  created_by?: string;
  created_at: string;
  updated_at: string;
}

/* ────────────────────── Toast ────────────────────── */

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

/* ────────────────────── Main Component ────────────────────── */

export const ABManager: React.FC = () => {
  const { theme } = useTheme();
  const { user } = useAuth();
  const isDark = theme === 'dark';

  // State
  const [activeTab, setActiveTab] = useState<'dashboard' | 'create'>('dashboard');
  const [testsData, setTestsData] = useState<TestsByStatus | null>(null);
  const [datasets, setDatasets] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [simulationLoading, setSimulationLoading] = useState<string | null>(null);
  const pollingRef = useRef<number | null>(null);
  const [toast, setToast] = useState<{ message: string; kind: 'success' | 'error' } | null>(null);

  // Create test form state
  const [formData, setFormData] = useState<Record<string, any>>({
    testName: '',
    variants: 'A, B',
    primaryMetric: '',
    metricType: 'continuous',
    description: '',
    confidenceLevel: 0.95,
    power: 0.8,
    minEffectSize: 0.1,
    trafficSplitType: 'fixed',
    simulationDurationMinutes: 20,
    analysisMode: 'fixed_experiment',
    earlyStoppingEnabled: false,
  });
  const [formErrors, setFormErrors] = useState<Record<string, string>>({});
  const [saving, setSaving] = useState(false);
  const [appliedTemplate, setAppliedTemplate] = useState<string | null>(null);
  const [appliedVariantEffects, setAppliedVariantEffects] = useState<Record<string, any> | null>(null);

  // Template modal
  const [templateModal, setTemplateModal] = useState(false);
  const [templates, setTemplates] = useState<Template[]>([]);
  const [loadingTemplates, setLoadingTemplates] = useState(false);

  // Colors
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
    tabActive:   isDark ? '#292524' : '#fef3c7',
    tabHover:    isDark ? '#211f1d' : '#f5f0e8',
  }), [isDark]);

  const userPermissions = user?.permissions || [];
  const canCreateAB = userPermissions.includes('AB_тесты_создание');
  const canManageAB = userPermissions.includes('AB_тесты_управление');
  const canArchiveDeleteAB = userPermissions.includes('AB_тесты_удаление_и_архивация');

  useEffect(() => {
    if (activeTab === 'create' && !canCreateAB && canManageAB) {
      setActiveTab('dashboard');
    }
    if (activeTab === 'dashboard' && !canManageAB && canCreateAB) {
      setActiveTab('create');
    }
  }, [activeTab, canCreateAB, canManageAB]);

  const statusColors = useMemo(() => ({
    prepared: { bg: isDark ? 'rgba(59,130,246,0.14)' : '#eff6ff', text: isDark ? '#93c5fd' : '#1e40af', border: '#3b82f6' },
    active: { bg: isDark ? 'rgba(34,197,94,0.14)' : '#f0fdf4', text: isDark ? '#86efac' : '#166534', border: '#22c55e' },
    paused: { bg: isDark ? 'rgba(251,146,60,0.14)' : '#fff7ed', text: isDark ? '#fdba74' : '#9a3412', border: '#fb923c' },
    completed: { bg: isDark ? 'rgba(168,85,247,0.14)' : '#faf5ff', text: isDark ? '#d8b4fe' : '#6b21a8', border: '#a855f7' },
    archived: { bg: isDark ? 'rgba(107,114,128,0.14)' : '#f9fafb', text: isDark ? '#9ca3af' : '#4b5563', border: '#6b7280' },
  }), [isDark]);

  const statTones = useMemo(() => ({
    prepared: { bg: isDark ? 'rgba(59,130,246,0.14)' : '#eff6ff', text: isDark ? '#93c5fd' : '#1d4ed8', border: isDark ? 'rgba(59,130,246,0.35)' : '#bfdbfe' },
    active: { bg: isDark ? 'rgba(34,197,94,0.14)' : '#f0fdf4', text: isDark ? '#86efac' : '#166534', border: isDark ? 'rgba(34,197,94,0.35)' : '#bbf7d0' },
    completed: { bg: isDark ? 'rgba(168,85,247,0.14)' : '#faf5ff', text: isDark ? '#d8b4fe' : '#6b21a8', border: isDark ? 'rgba(168,85,247,0.35)' : '#e9d5ff' },
    archived: { bg: isDark ? 'rgba(107,114,128,0.14)' : '#f9fafb', text: isDark ? '#d1d5db' : '#4b5563', border: isDark ? 'rgba(107,114,128,0.35)' : '#d1d5db' },
  }), [isDark]);

  // Load data
  useEffect(() => {
    loadDashboardData();
    loadDatasets();
    
    pollingRef.current = window.setInterval(() => {
      loadDashboardData();
    }, 3000);
    
    return () => {
      if (pollingRef.current !== null) {
        window.clearInterval(pollingRef.current);
      }
    };
  }, []);

  const showToast = (message: string, kind: 'success' | 'error') => {
    setToast({ message, kind });
    setTimeout(() => setToast(null), 3200);
  };

  const loadDashboardData = async () => {
    try {
      const response = await abTestAPI.getAllTests();
      setTestsData(response.data);
    } catch (error) {
      console.error('Ошибка загрузки данных дашборда:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadDatasets = async () => {
    try {
      const response = await dataAPI.listGeneratedHistory(100);
      const syntheticDatasets = response.data.items.filter((d: any) => d.data_type === 'synthetic');
      setDatasets(syntheticDatasets);
    } catch (error) {
      console.error('Ошибка загрузки датасетов:', error);
    }
  };

  const loadTemplates = async () => {
    setLoadingTemplates(true);
    try {
      const response = await templatesAPI.listTemplates('ab_test');
      setTemplates(response.data.items || []);
    } catch (e) {
      showToast('Ошибка загрузки шаблонов', 'error');
    } finally {
      setLoadingTemplates(false);
    }
  };

  const handleRunSimulation = async (testId: string) => {
    if (datasets.length === 0) {
      showToast('Для запуска симуляции необходимо сначала сгенерировать синтетические данные в GAN Менеджер', 'error');
      return;
    }

    setSimulationLoading(testId);
    try {
      await abTestAPI.startSimulation(testId, {});
      showToast('Симуляция запущена', 'success');
      setTimeout(() => {
        loadDashboardData();
      }, 3000);
    } catch (error: any) {
      const errorMsg = error.response?.data?.detail || error.message || 'Неизвестная ошибка';
      showToast(`Ошибка запуска симуляции: ${errorMsg}`, 'error');
    } finally {
      setSimulationLoading(null);
    }
  };

  const handlePauseTest = async (testId: string) => {
    try {
      await abTestAPI.pauseTest(testId);
      showToast('Тест поставлен на паузу', 'success');
      loadDashboardData();
    } catch (error: any) {
      showToast(`Ошибка паузы теста: ${error.response?.data?.detail || error.message}`, 'error');
    }
  };

  const handleResumeTest = async (testId: string) => {
    try {
      await abTestAPI.resumeTest(testId);
      showToast('Тест продолжен', 'success');
      loadDashboardData();
    } catch (error: any) {
      showToast(`Ошибка продолжения теста: ${error.response?.data?.detail || error.message}`, 'error');
    }
  };

  const handleArchiveTest = async (testId: string) => {
    try {
      await abTestAPI.archiveTest(testId);
      showToast('Тест перемещен в архив', 'success');
      loadDashboardData();
    } catch (error: any) {
      showToast(`Ошибка архивирования: ${error.response?.data?.detail || error.message}`, 'error');
    }
  };

  const handlePermanentlyDeleteTest = async (testId: string) => {
    try {
      await abTestAPI.permanentlyDeleteTest(testId);
      showToast('Тест полностью удален', 'success');
      loadDashboardData();
    } catch (error: any) {
      showToast(`Ошибка удаления: ${error.response?.data?.detail || error.message}`, 'error');
    }
  };

  const applyTemplate = (template: Template) => {
    const cfg = template.config_json || {};

    const pick = <T,>(...keys: string[]): T | undefined => {
      for (const key of keys) {
        if (cfg[key] !== undefined && cfg[key] !== null) {
          return cfg[key] as T;
        }
      }
      return undefined;
    };

    const rawVariants = pick<any>('variants', 'variant_list');
    const normalizedVariants = Array.isArray(rawVariants)
      ? rawVariants.join(', ')
      : (rawVariants || 'A, B');

    const trafficSplitType = pick<string>('trafficSplitType', 'traffic_split_type') || 'fixed';
    const resolvedAnalysisMode =
      pick<'fixed_experiment' | 'adaptive_bandit'>('analysisMode', 'analysis_mode') ||
      (trafficSplitType === 'adaptive' ? 'adaptive_bandit' : 'fixed_experiment');

    const guardrailsConfig = pick<Record<string, any>>('guardrailsConfig', 'guardrails_config');

    setFormData({
      testName: pick<string>('testName', 'test_name') || '',
      variants: normalizedVariants,
      primaryMetric: pick<string>('primaryMetric', 'primary_metric') || '',
      metricType: pick<string>('metricType', 'metric_type') || 'continuous',
      description: pick<string>('description') || '',
      confidenceLevel: pick<number>('confidenceLevel', 'confidence_level') ?? 0.95,
      power: pick<number>('power') ?? 0.8,
      minEffectSize: pick<number>('minEffectSize', 'min_effect_size') ?? 0.1,
      trafficSplitType,
      simulationDurationMinutes: pick<number>('simulationDurationMinutes', 'simulation_duration_minutes') ?? 20,
      sampleSize: pick<number>('sampleSize', 'sample_size') ?? undefined,
      analysisMode: resolvedAnalysisMode,
      guardrailsConfigJson: guardrailsConfig ? JSON.stringify(guardrailsConfig, null, 2) : undefined,
      earlyStoppingEnabled: pick<boolean>('earlyStoppingEnabled', 'early_stopping_enabled') ?? false,
      datasetId: pick<number>('datasetId', 'dataset_id'),
    });

    setAppliedVariantEffects(pick<Record<string, any>>('variantEffects', 'variant_effects') || null);
    setAppliedTemplate(template.name);
    setTemplateModal(false);
    showToast(`Шаблон "${template.name}" применён`, 'success');
  };

  const handleCreateTest = async () => {
    // Validation
    const errors: Record<string, string> = {};
    if (!formData.testName?.trim()) errors.testName = 'Укажите название теста';
    if (!formData.primaryMetric?.trim()) errors.primaryMetric = 'Укажите основную метрику';
    if (!formData.datasetId) errors.datasetId = 'Выберите синтетический датасет';

    if (Object.keys(errors).length) {
      setFormErrors(errors);
      return;
    }

    setSaving(true);
    try {
      let variantsArray: string[];
      if (typeof formData.variants === 'string') {
        variantsArray = formData.variants
          .split(',')
          .map((v: string) => v.trim())
          .filter((v: string) => v.length > 0);
      } else if (Array.isArray(formData.variants)) {
        variantsArray = formData.variants
          .map((v: string) => String(v).trim())
          .filter((v: string) => v.length > 0);
      } else {
        throw new Error('Варианты должны быть строкой или массивом');
      }

      if (variantsArray.length < 2) {
        showToast('Необходимо минимум 2 варианта для A/B теста', 'error');
        setSaving(false);
        return;
      }

      let guardrailsConfig: Record<string, any> | null = null;
      if (formData.guardrailsConfigJson && String(formData.guardrailsConfigJson).trim().length > 0) {
        try {
          guardrailsConfig = JSON.parse(formData.guardrailsConfigJson);
        } catch {
          throw new Error('Некорректный JSON в guardrails-конфиге');
        }
      }

      const resolvedAnalysisMode = formData.analysisMode || (formData.trafficSplitType === 'adaptive' ? 'adaptive_bandit' : 'fixed_experiment');

      const testData = {
        test_name: formData.testName,
        variants: variantsArray,
        primary_metric: formData.primaryMetric,
        metric_type: formData.metricType,
        description: formData.description || '',
        sample_size: formData.sampleSize || null,
        confidence_level: formData.confidenceLevel || 0.95,
        power: formData.power || 0.8,
        min_effect_size: formData.minEffectSize || 0.1,
        dataset_id: formData.datasetId || null,
        simulation_duration_minutes: formData.simulationDurationMinutes || 20,
        traffic_split_type: formData.trafficSplitType || 'fixed',
        variant_effects: appliedVariantEffects,
        analysis_mode: resolvedAnalysisMode,
        guardrails_config: guardrailsConfig,
        early_stopping_enabled: Boolean(formData.earlyStoppingEnabled),
      };

      const response = await abTestAPI.createTest(testData);
      showToast(`Тест успешно создан! ID: ${response.data.test_id}`, 'success');
      
      // Reset form
      setFormData({
        testName: '',
        variants: 'A, B',
        primaryMetric: '',
        metricType: 'continuous',
        description: '',
        confidenceLevel: 0.95,
        power: 0.8,
        minEffectSize: 0.1,
        trafficSplitType: 'fixed',
        simulationDurationMinutes: 20,
        analysisMode: 'fixed_experiment',
        earlyStoppingEnabled: false,
      });
      setAppliedTemplate(null);
      setAppliedVariantEffects(null);
      setFormErrors({});
      
      // Switch to dashboard
      setActiveTab('dashboard');
      loadDashboardData();

    } catch (error: any) {
      const errorMsg = error.response?.data?.detail || error.message || 'Неизвестная ошибка';
      showToast(`Ошибка при создании теста: ${errorMsg}`, 'error');
    } finally {
      setSaving(false);
    }
  };

  const getStatusLabel = (status: string): string => {
    const labels: Record<string, string> = {
      prepared: 'Подготовлен',
      active: 'Активен',
      paused: 'На паузе',
      completed: 'Завершен',
      archived: 'Архив',
    };
    return labels[status] || status;
  };

  const fmtDate = (s: string) => s ? new Date(s).toLocaleString('ru-RU') : '—';
  const fmtPercent = (value: number) => {
    const safeValue = Number.isFinite(value) ? value : 0;
    return new Intl.NumberFormat('ru-RU', { maximumFractionDigits: 2 }).format(safeValue);
  };

  // Render helpers
  const renderStatCard = (title: string, value: number, tone?: { bg: string; text: string; border: string }) => (
    <StatCard title={title} value={String(value)} c={c} tone={tone} />
  );

  const renderTestCard = (test: Test) => {
    const isSimulating = simulationLoading === test.test_id;
    const isPaused = test.status === 'paused';
    const statusStyle = statusColors[test.status as keyof typeof statusColors] || statusColors.prepared;
    const progress = Math.max(0, Math.min(100, Number(test.completion_percentage || 0)));

    return (
      <div
        key={test.test_id}
        style={{
          borderRadius: 12,
          border: `1px solid ${c.border}`,
          backgroundColor: isPaused ? (isDark ? '#292524' : '#fff7ed') : c.panelBg,
          padding: 14,
          marginBottom: 10,
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10, flexWrap: 'wrap' }}>
          <div style={{ flex: 1, minWidth: 280 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', marginBottom: 6 }}>
              <div style={{ fontSize: 15, fontWeight: 700, color: c.textPrimary, letterSpacing: '-0.2px' }}>{test.test_name}</div>
              <span style={{
                padding: '3px 10px',
                borderRadius: 999,
                border: `1px solid ${statusStyle.border}`,
                backgroundColor: statusStyle.bg,
                color: statusStyle.text,
                fontSize: 12,
                fontWeight: 600,
              }}>
                {getStatusLabel(test.status)}
              </span>
              {test.simulation_status === 'running' && test.status !== 'archived' && (
                <span style={{
                  padding: '3px 10px',
                  borderRadius: 999,
                  border: '1px solid #dc2626',
                  backgroundColor: isDark ? 'rgba(239,68,68,0.14)' : '#fee2e2',
                  color: '#dc2626',
                  fontSize: 12,
                  fontWeight: 600,
                }}>
                  Симуляция запущена
                </span>
              )}
            </div>

            {test.description && (
              <div style={{ fontSize: 13, color: c.textMuted, marginBottom: 10, lineHeight: 1.5 }}>
                {test.description}
              </div>
            )}

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 8, marginBottom: 10 }}>
              {[
                ['Варианты', test.variants?.join(', ') || '—'],
                ['Метрика', test.primary_metric || '—'],
                ['Пользователи', String(test.total_users || 0)],
                ['Создан', fmtDate(test.created_at)],
              ].map(([k, v]) => (
                <div key={k} style={{ padding: '8px 10px', borderRadius: 8, border: `1px solid ${c.border}`, backgroundColor: c.panelSoft }}>
                  <div style={{ fontSize: 11, color: c.textSub, textTransform: 'uppercase', letterSpacing: '0.35px' }}>{k}</div>
                  <div style={{ marginTop: 4, fontSize: 13, color: c.textPrimary, fontWeight: 600 }}>{v}</div>
                </div>
              ))}
            </div>

            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                <span style={{ fontSize: 12, color: c.textMuted }}>Прогресс эксперимента</span>
                <span style={{ fontSize: 12, color: c.textPrimary, fontWeight: 700 }}>{fmtPercent(progress)}%</span>
              </div>
              <div style={{ height: 8, borderRadius: 999, background: isDark ? '#211f1d' : '#ece7df', overflow: 'hidden' }}>
                <div style={{ width: `${progress}%`, height: '100%', background: 'linear-gradient(90deg, #d97706, #22c55e)' }} />
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', gap: 6, flexShrink: 0, alignSelf: 'flex-start', flexWrap: 'wrap' }}>
            {test.status === 'prepared' && (
              <>
                <ActionButton
                  onClick={() => handleRunSimulation(test.test_id)}
                  disabled={datasets.length === 0 || isSimulating}
                  loading={isSimulating}
                  icon={<PlayCircle size={14} />}
                  label="Запустить"
                  variant="primary"
                  c={c}
                  isDark={isDark}
                />
                {canArchiveDeleteAB && (
                  <ActionButton
                    onClick={() => handleArchiveTest(test.test_id)}
                    icon={<Folder size={14} />}
                    label="В архив"
                    variant="secondary"
                    c={c}
                    isDark={isDark}
                  />
                )}
              </>
            )}
            {test.status === 'active' && canManageAB && (
              <ActionButton
                onClick={() => handlePauseTest(test.test_id)}
                icon={<PauseCircle size={14} />}
                label="Пауза"
                variant="secondary"
                c={c}
                isDark={isDark}
              />
            )}
            {test.status === 'paused' && canManageAB && (
              <ActionButton
                onClick={() => handleResumeTest(test.test_id)}
                icon={<PlayCircle size={14} />}
                label="Продолжить"
                variant="primary"
                c={c}
                isDark={isDark}
              />
            )}
            {test.status === 'completed' && canArchiveDeleteAB && (
              <ActionButton
                onClick={() => handleArchiveTest(test.test_id)}
                icon={<Folder size={14} />}
                label="В архив"
                variant="secondary"
                c={c}
                isDark={isDark}
              />
            )}
            {test.status === 'archived' && canArchiveDeleteAB && (
              <ActionButton
                onClick={() => handlePermanentlyDeleteTest(test.test_id)}
                icon={<Trash2 size={14} />}
                label="Удалить"
                variant="danger"
                c={c}
                isDark={isDark}
              />
            )}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div style={{ color: c.textPrimary, fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" }}>
      
      {/* Header */}
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
            <Zap size={20} />
          </div>
          <div>
            <h1 style={{ margin: 0, fontSize: 22, fontWeight: 700, letterSpacing: '-0.4px', color: c.textPrimary }}>
              A/B Менеджер
            </h1>
            <p style={{ margin: 0, fontSize: 13, color: c.textMuted }}>
              Управление A/B тестами и создание экспериментов
            </p>
          </div>
        </div>

        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <ActionButton
            onClick={loadDashboardData}
            disabled={loading}
            isDark={isDark}
            c={c}
            icon={<RefreshCw size={14} style={{ animation: loading ? 'spin 0.8s linear infinite' : 'none' }} />}
            label="Обновить"
            variant="secondary"
          />
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 6, marginBottom: 16, flexWrap: 'wrap' }}>
        {canManageAB && (
          <PillTab active={activeTab === 'dashboard'} onClick={() => setActiveTab('dashboard')} label="Дашборд тестов" c={c} />
        )}
        {canCreateAB && (
          <PillTab active={activeTab === 'create'} onClick={() => setActiveTab('create')} label="Создать A/B тест" c={c} />
        )}
      </div>

      {/* Tab content */}
      {activeTab === 'dashboard' && canManageAB && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 12, marginBottom: 12 }}>
            {renderStatCard('Подготовленные', testsData?.counts.prepared || 0, statTones.prepared)}
            {renderStatCard('Активные', (testsData?.counts.active || 0) + (testsData?.counts.paused || 0), statTones.active)}
            {renderStatCard('Завершенные', testsData?.counts.completed || 0, statTones.completed)}
            {renderStatCard('Архив', testsData?.counts.archived || 0, statTones.archived)}
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) 280px', gap: 14, alignItems: 'start' }}>
            <div>
              {loading ? (
                <div style={{ padding: '64px 24px', textAlign: 'center', color: c.textMuted }}>
                  <RefreshCw size={28} style={{ animation: 'spin 0.8s linear infinite', marginBottom: 12 }} />
                  <p style={{ margin: 0, fontSize: 14 }}>Загрузка тестов…</p>
                </div>
              ) : (
                <>
                  <Panel c={c} title="Активные тесты" icon={<PlayCircle size={14} />}>
                    <SectionHeader title="Активные и на паузе" count={[...(testsData?.active_tests || []), ...(testsData?.paused_tests || [])].length} c={c} />
                    {[...(testsData?.active_tests || []), ...(testsData?.paused_tests || [])].map(renderTestCard)}
                    {[...(testsData?.active_tests || []), ...(testsData?.paused_tests || [])].length === 0 && (
                      <EmptyState message="Нет активных тестов" c={c} />
                    )}
                  </Panel>

                  <Panel c={c} title="Подготовленные тесты" icon={<FileText size={14} />}>
                    <SectionHeader title="Готовы к запуску" count={(testsData?.prepared_tests || []).length} c={c} />
                    {(testsData?.prepared_tests || []).map(renderTestCard)}
                    {(testsData?.prepared_tests || []).length === 0 && (
                      <EmptyState message="Нет подготовленных тестов" c={c} />
                    )}
                  </Panel>

                  <Panel c={c} title="Завершенные тесты" icon={<Check size={14} />}>
                    <SectionHeader title="Результаты готовы" count={(testsData?.completed_tests || []).length} c={c} />
                    {(testsData?.completed_tests || []).map(renderTestCard)}
                    {(testsData?.completed_tests || []).length === 0 && (
                      <EmptyState message="Нет завершенных тестов" c={c} />
                    )}
                  </Panel>

                  <Panel c={c} title="Архив" icon={<Folder size={14} />}>
                    <SectionHeader title="Архивные тесты" count={(testsData?.archived_tests || []).length} c={c} />
                    {(testsData?.archived_tests || []).map(renderTestCard)}
                    {(testsData?.archived_tests || []).length === 0 && (
                      <EmptyState message="Нет архивных тестов" c={c} />
                    )}
                  </Panel>
                </>
              )}
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {datasets.length === 0 && (
                <div style={{
                  padding: 14,
                  borderRadius: 12,
                  border: `1px solid ${isDark ? 'rgba(251,146,60,0.25)' : '#fed7aa'}`,
                  backgroundColor: isDark ? 'rgba(251,146,60,0.1)' : '#fff7ed',
                  color: isDark ? '#fdba74' : '#9a3412',
                  fontSize: 13,
                  lineHeight: 1.5,
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontWeight: 700, marginBottom: 6 }}>
                    <AlertTriangle size={15} />
                    Нет данных для симуляции
                  </div>
                  Сначала создайте синтетические данные во вкладке GAN Менеджер, затем возвращайтесь к запуску A/B тестов.
                </div>
              )}

              <div style={{
                borderRadius: 14,
                border: `1px solid ${c.border}`,
                backgroundColor: c.panelBg,
                boxShadow: c.shadow,
                padding: 14,
              }}>
                <div style={{ fontSize: 12, textTransform: 'uppercase', letterSpacing: '0.35px', color: c.textSub, marginBottom: 10 }}>Как читать статусы</div>
                <div style={{ display: 'grid', gap: 6 }}>
                  {[
                    ['Подготовлен', 'Тест создан и готов к запуску симуляции'],
                    ['Активен', 'Тест собирает данные в режиме симуляции'],
                    ['На паузе', 'Тест остановлен и может быть продолжен'],
                    ['Завершен', 'Сбор данных окончен, результаты доступны'],
                  ].map(([title, desc]) => (
                    <div key={title} style={{ fontSize: 12, color: c.textMuted }}>
                      <span style={{ color: c.textPrimary, fontWeight: 700 }}>{title}:</span> {desc}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'create' && canCreateAB && (
        <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) 280px', gap: 14, alignItems: 'start' }}>
          <div style={{
            borderRadius: 14,
            border: `1px solid ${c.border}`,
            backgroundColor: c.panelBg,
            boxShadow: c.shadow,
            overflow: 'hidden',
          }}>
            <div style={{
              padding: '14px 16px',
              borderBottom: `1px solid ${c.border}`,
              backgroundColor: c.panelSoft,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: 8,
              flexWrap: 'wrap',
            }}>
              <div>
                <h2 style={{ margin: 0, fontSize: 18, fontWeight: 700, color: c.textPrimary }}>Создать новый A/B тест</h2>
                <p style={{ margin: '2px 0 0', fontSize: 12, color: c.textMuted }}>
                  Заполните параметры эксперимента и выберите синтетический датасет
                </p>
              </div>
              <ActionButton
                onClick={() => { setTemplateModal(true); loadTemplates(); }}
                icon={<FileText size={14} />}
                label="Из шаблонов"
                variant="secondary"
                c={c}
                isDark={isDark}
              />
            </div>

            <div style={{ padding: 16 }}>
              {appliedTemplate && (
                <div style={{
                  padding: 12,
                  borderRadius: 10,
                  border: `1px solid ${isDark ? 'rgba(34,197,94,0.25)' : '#bbf7d0'}`,
                  backgroundColor: isDark ? 'rgba(34,197,94,0.1)' : '#f0fdf4',
                  color: isDark ? '#86efac' : '#166534',
                  fontSize: 13,
                  marginBottom: 14,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <Check size={14} />
                    <span>Применён шаблон: <strong>"{appliedTemplate}"</strong></span>
                  </div>
                  <button
                    onClick={() => setAppliedTemplate(null)}
                    style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'inherit', display: 'flex' }}
                  >
                    <X size={14} />
                  </button>
                </div>
              )}

              <CreateTestForm
                formData={formData}
                setFormData={setFormData}
                formErrors={formErrors}
                datasets={datasets}
                c={c}
                isDark={isDark}
              />

              <div style={{ marginTop: 16, display: 'flex', gap: 8, justifyContent: 'flex-end', flexWrap: 'wrap' }}>
                <ActionButton
                  onClick={() => {
                    setFormData({
                      testName: '',
                      variants: 'A, B',
                      primaryMetric: '',
                      metricType: 'continuous',
                      description: '',
                      confidenceLevel: 0.95,
                      power: 0.8,
                      minEffectSize: 0.1,
                      trafficSplitType: 'fixed',
                      simulationDurationMinutes: 20,
                      analysisMode: 'fixed_experiment',
                      earlyStoppingEnabled: false,
                    });
                    setFormErrors({});
                    setAppliedTemplate(null);
                  }}
                  label="Сбросить"
                  variant="secondary"
                  c={c}
                  isDark={isDark}
                />
                <ActionButton
                  onClick={handleCreateTest}
                  disabled={saving}
                  loading={saving}
                  icon={saving ? <RefreshCw size={14} style={{ animation: 'spin 0.8s linear infinite' }} /> : <Plus size={14} />}
                  label={saving ? 'Создание…' : 'Создать A/B тест'}
                  variant="primary"
                  c={c}
                  isDark={isDark}
                />
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <div style={{
              borderRadius: 14,
              border: `1px solid ${c.border}`,
              backgroundColor: c.panelBg,
              boxShadow: c.shadow,
              padding: 14,
            }}>
              <div style={{ fontSize: 12, textTransform: 'uppercase', letterSpacing: '0.35px', color: c.textSub, marginBottom: 8 }}>Превью конфигурации</div>
              <div style={{ display: 'grid', gap: 8 }}>
                <div style={{ fontSize: 13, color: c.textMuted }}>Название: <span style={{ color: c.textPrimary, fontWeight: 600 }}>{formData.testName?.trim() || '—'}</span></div>
                <div style={{ fontSize: 13, color: c.textMuted }}>Варианты: <span style={{ color: c.textPrimary, fontWeight: 600 }}>{String(formData.variants || '').split(',').map((x: string) => x.trim()).filter(Boolean).length || 0}</span></div>
                <div style={{ fontSize: 13, color: c.textMuted }}>Метрика: <span style={{ color: c.textPrimary, fontWeight: 600 }}>{formData.primaryMetric?.trim() || '—'}</span></div>
                <div style={{ fontSize: 13, color: c.textMuted }}>Режим: <span style={{ color: c.textPrimary, fontWeight: 600 }}>{formData.analysisMode || 'fixed_experiment'}</span></div>
              </div>
            </div>

            <div style={{
              borderRadius: 14,
              border: `1px solid ${c.border}`,
              backgroundColor: c.panelBg,
              boxShadow: c.shadow,
              padding: 14,
            }}>
              <div style={{ fontSize: 12, textTransform: 'uppercase', letterSpacing: '0.35px', color: c.textSub, marginBottom: 10 }}>Рекомендации</div>
              <ul style={{ margin: 0, paddingLeft: 18, fontSize: 13, color: c.textMuted, lineHeight: 1.6 }}>
                <li>Первый вариант (A) используйте как контрольный</li>
                <li>Для валидных выводов выбирайте <strong>Фиксированный эксперимент</strong></li>
                <li>Перед созданием теста проверьте наличие <strong>верного</strong> GAN-датасета</li>
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* Template Modal */}
      {templateModal && (
        <Overlay onClose={() => setTemplateModal(false)}>
          <ModalPanel width={800} c={c} isDark={isDark}>
            <ModalHeader
              icon={<FileText size={16} />}
              title="Выберите шаблон A/B теста"
              onClose={() => setTemplateModal(false)}
              c={c}
            />
            <div style={{ padding: '20px 24px', overflowY: 'auto', maxHeight: 'calc(90vh - 130px)' }}>
              <p style={{ margin: '0 0 16px', fontSize: 13, color: c.textMuted }}>
                Выберите готовый шаблон для быстрого заполнения формы. Все поля можно изменить после применения.
              </p>
              {loadingTemplates ? (
                <div style={{ padding: '40px 24px', textAlign: 'center', color: c.textMuted }}>
                  <RefreshCw size={28} style={{ animation: 'spin 0.8s linear infinite', marginBottom: 12 }} />
                  <p style={{ margin: 0, fontSize: 14 }}>Загрузка шаблонов…</p>
                </div>
              ) : templates.length === 0 ? (
                <EmptyState message="Нет доступных шаблонов A/B тестов" c={c} />
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                  {templates.map(template => (
                    <div
                      key={template.id}
                      style={{
                        padding: 16,
                        borderRadius: 10,
                        border: `1px solid ${c.border}`,
                        backgroundColor: c.panelBg,
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'flex-start',
                      }}
                    >
                      <div style={{ flex: 1 }}>
                        <div style={{ fontSize: 15, fontWeight: 600, color: c.textPrimary, marginBottom: 4 }}>
                          {template.name}
                        </div>
                        {template.description && (
                          <div style={{ fontSize: 12, color: c.textMuted, marginBottom: 8 }}>
                            {template.description.length > 100 ? `${template.description.slice(0, 100)}…` : template.description}
                          </div>
                        )}
                        {template.tags && template.tags.length > 0 && (
                          <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                            {template.tags.map(tag => (
                              <span
                                key={tag}
                                style={{
                                  padding: '2px 8px',
                                  borderRadius: 999,
                                  border: `1px solid ${c.border}`,
                                  backgroundColor: c.panelSoft,
                                  color: c.textMuted,
                                  fontSize: 11,
                                }}
                              >
                                {tag}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                      <ActionButton
                        onClick={() => applyTemplate(template)}
                        icon={<Zap size={14} />}
                        label="Применить"
                        variant="primary"
                        c={c}
                        isDark={isDark}
                      />
                    </div>
                  ))}
                </div>
              )}
            </div>
          </ModalPanel>
        </Overlay>
      )}

      {toast && <Toast message={toast.message} kind={toast.kind} onClose={() => setToast(null)} />}

      <style>{`
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
      `}</style>
    </div>
  );
};

/* ══════════════════════════════════════════════════════
   Sub-components
══════════════════════════════════════════════════════ */

const Panel: React.FC<{ c: Record<string, string>; title: string; icon?: React.ReactNode; children: React.ReactNode }> = ({ c, title, icon, children }) => (
  <div style={{ borderRadius: 14, border: `1px solid ${c.border}`, backgroundColor: c.panelBg, boxShadow: c.shadow, overflow: 'hidden', marginBottom: 14 }}>
    <div style={{ padding: '12px 14px', borderBottom: `1px solid ${c.border}`, backgroundColor: c.panelSoft, display: 'flex', alignItems: 'center', gap: 8 }}>
      <span style={{ color: c.accentText, display: 'inline-flex', alignItems: 'center' }}>{icon}</span>
      <span style={{ fontSize: 14, fontWeight: 700, color: c.textPrimary }}>{title}</span>
    </div>
    <div style={{ padding: 14 }}>{children}</div>
  </div>
);

const PillTab: React.FC<{ active: boolean; onClick: () => void; label: string; c: Record<string, string> }> = ({ active, onClick, label, c }) => (
  <button
    onClick={onClick}
    style={{
      padding: '7px 14px',
      borderRadius: 999,
      border: `1px solid ${active ? c.accent : c.border}`,
      backgroundColor: active ? c.accentSoft : c.panelBg,
      color: active ? c.accentText : c.textMuted,
      fontSize: 13,
      fontWeight: active ? 700 : 600,
      cursor: 'pointer',
    }}
  >
    {label}
  </button>
);

const StatCard: React.FC<{
  title: string;
  value: string;
  c: Record<string, string>;
  tone?: { bg: string; text: string; border: string };
}> = ({ title, value, c, tone }) => (
  <div style={{ borderRadius: 12, border: `1px solid ${tone?.border ?? c.border}`, backgroundColor: tone?.bg ?? c.panelBg, boxShadow: c.shadow, padding: '14px 16px' }}>
    <div style={{ fontSize: 11, color: c.textSub, textTransform: 'uppercase', letterSpacing: '0.35px' }}>{title}</div>
    <div style={{ marginTop: 8, color: tone?.text ?? c.textPrimary, fontSize: 18, fontWeight: 700, lineHeight: 1.35 }}>{value}</div>
  </div>
);

interface SectionHeaderProps {
  title: string;
  count: number;
  c: Record<string, string>;
}

const SectionHeader: React.FC<SectionHeaderProps> = ({ title, count, c }) => (
  <div style={{
    display: 'flex',
    alignItems: 'center',
    gap: 10,
    marginBottom: 12,
    paddingBottom: 8,
    borderBottom: `1px solid ${c.border}`,
  }}>
    <h3 style={{ margin: 0, fontSize: 16, fontWeight: 600, color: c.textPrimary }}>
      {title}
    </h3>
    <span style={{
      padding: '2px 8px',
      borderRadius: 999,
      backgroundColor: c.accentSoft,
      color: c.accentText,
      fontSize: 12,
      fontWeight: 600,
    }}>
      {count}
    </span>
  </div>
);

interface EmptyStateProps {
  message: string;
  c: Record<string, string>;
}

const EmptyState: React.FC<EmptyStateProps> = ({ message, c }) => (
  <div style={{
    padding: '40px 24px',
    textAlign: 'center',
    borderRadius: 10,
    border: `1px solid ${c.border}`,
    backgroundColor: c.panelSoft,
  }}>
    <FileText size={32} color={c.textSub} style={{ marginBottom: 12 }} />
    <p style={{ margin: 0, color: c.textMuted, fontSize: 14 }}>{message}</p>
  </div>
);

interface ActionButtonProps {
  onClick: () => void;
  disabled?: boolean;
  loading?: boolean;
  isDark: boolean;
  c: Record<string, string>;
  icon?: React.ReactNode;
  label?: string;
  variant: 'primary' | 'secondary' | 'danger';
}

const ActionButton: React.FC<ActionButtonProps> = ({ onClick, disabled, loading, isDark, c, icon, label, variant }) => {
  const [hov, setHov] = React.useState(false);

  const bg =
    variant === 'primary'
      ? disabled ? (isDark ? '#292524' : '#e7e5e4') : hov ? c.accentHov : c.accent
      : variant === 'danger'
      ? hov ? (isDark ? 'rgba(239,68,68,0.25)' : '#fee2e2') : (isDark ? 'rgba(239,68,68,0.15)' : '#fef2f2')
      : hov ? (isDark ? '#292524' : '#f0ede8') : c.panelBg;

  const border =
    variant === 'primary'
      ? disabled ? (isDark ? '#292524' : '#e7e5e4') : hov ? c.accentHov : c.accent
      : variant === 'danger'
      ? c.dangerBorder
      : c.border;

  const color =
    variant === 'primary'
      ? disabled ? c.textMuted : '#fff'
      : variant === 'danger'
      ? c.danger
      : c.textMuted;

  return (
    <button
      onClick={disabled || loading ? undefined : onClick}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        height: 38,
        padding: label ? '0 16px' : '0 12px',
        borderRadius: 8,
        border: `1px solid ${border}`,
        backgroundColor: bg,
        color,
        fontSize: 14,
        fontWeight: 600,
        cursor: disabled || loading ? 'not-allowed' : 'pointer',
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

interface OverlayProps {
  onClose: () => void;
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

interface ModalPanelProps {
  width: number;
  c: Record<string, string>;
  isDark: boolean;
  children: React.ReactNode;
}

const ModalPanel: React.FC<ModalPanelProps> = ({ width, c, children }) => (
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
    animation: 'fadeIn 0.18s ease',
  }}>
    {children}
  </div>
);

interface ModalHeaderProps {
  icon?: React.ReactNode;
  title: string;
  onClose: () => void;
  c: Record<string, string>;
}

const ModalHeader: React.FC<ModalHeaderProps> = ({ icon, title, onClose, c }) => (
  <div style={{
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    padding: '16px 24px',
    borderBottom: `1px solid ${c.border}`,
    backgroundColor: c.panelSoft,
    flexShrink: 0,
  }}>
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      {icon && (
        <span style={{ color: c.accentText, display: 'flex', alignItems: 'center' }}>
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

interface CreateTestFormProps {
  formData: Record<string, any>;
  setFormData: React.Dispatch<React.SetStateAction<Record<string, any>>>;
  formErrors: Record<string, string>;
  datasets: any[];
  c: Record<string, string>;
  isDark: boolean;
}

const CreateTestForm: React.FC<CreateTestFormProps> = ({ formData, setFormData, formErrors, datasets, c, isDark }) => {
  const [focused, setFocused] = React.useState<string | null>(null);

  const inputStyle = (focused: boolean, error?: string): React.CSSProperties => ({
    width: '100%',
    height: 40,
    padding: '0 12px',
    borderRadius: 10,
    border: `1.5px solid ${error ? c.danger : focused ? c.inputFocus : c.inputBorder}`,
    backgroundColor: c.inputBg,
    color: c.textPrimary,
    fontSize: 14,
    outline: 'none',
    boxSizing: 'border-box',
    transition: 'border-color 0.15s, box-shadow 0.15s',
    boxShadow: focused ? `0 0 0 3px rgba(217,119,6,0.12)` : 'none',
    fontFamily: 'inherit',
  });

  const labelStyle: React.CSSProperties = {
    display: 'block',
    fontSize: 13,
    fontWeight: 600,
    color: c.textMuted,
    marginBottom: 6,
    letterSpacing: '0.1px',
  };

  const hintStyle: React.CSSProperties = {
    marginTop: 4,
    fontSize: 12,
    color: c.textSub,
    lineHeight: 1.45,
  };

  const errorStyle: React.CSSProperties = {
    fontSize: 12,
    color: c.danger,
    marginTop: 4,
  };

  const sectionStyle: React.CSSProperties = {
    border: `1px solid ${c.border}`,
    borderRadius: 12,
    backgroundColor: isDark ? '#191613' : '#fcfbf9',
    padding: 14,
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      <div style={sectionStyle}>
        <div style={{ marginBottom: 12 }}>
          <div style={{ fontSize: 12, textTransform: 'uppercase', letterSpacing: '0.35px', color: c.textSub }}>Базовая конфигурация</div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          <div>
            <label style={labelStyle}>Название теста *</label>
            <input
              value={formData.testName || ''}
              onChange={e => setFormData(prev => ({ ...prev, testName: e.target.value }))}
              onFocus={() => setFocused('testName')}
              onBlur={() => setFocused(null)}
              placeholder="Например: Тест цвета кнопки CTA"
              style={inputStyle(focused === 'testName', formErrors.testName)}
            />
            {formErrors.testName ? <p style={errorStyle}>{formErrors.testName}</p> : <p style={hintStyle}>Короткое бизнес-описание гипотезы</p>}
          </div>

          <div>
            <label style={labelStyle}>Основная метрика *</label>
            <input
              value={formData.primaryMetric || ''}
              onChange={e => setFormData(prev => ({ ...prev, primaryMetric: e.target.value }))}
              onFocus={() => setFocused('primaryMetric')}
              onBlur={() => setFocused(null)}
              placeholder="conversion, revenue, ctr"
              style={inputStyle(focused === 'primaryMetric', formErrors.primaryMetric)}
            />
            {formErrors.primaryMetric ? <p style={errorStyle}>{formErrors.primaryMetric}</p> : <p style={hintStyle}>Поле должно совпадать с метрикой в данных</p>}
          </div>
        </div>

        <div style={{ marginTop: 12 }}>
          <label style={labelStyle}>Описание теста</label>
          <textarea
            value={formData.description || ''}
            onChange={e => setFormData(prev => ({ ...prev, description: e.target.value }))}
            onFocus={() => setFocused('description')}
            onBlur={() => setFocused(null)}
            rows={3}
            placeholder="Что именно проверяем и почему это важно"
            style={{
              ...inputStyle(focused === 'description'),
              height: 'auto',
              padding: '10px 12px',
              minHeight: 84,
              resize: 'vertical',
              lineHeight: 1.5,
            }}
          />
        </div>

        <div style={{ marginTop: 12, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          <div>
            <label style={labelStyle}>Варианты (через запятую) *</label>
            <input
              value={formData.variants || ''}
              onChange={e => setFormData(prev => ({ ...prev, variants: e.target.value }))}
              onFocus={() => setFocused('variants')}
              onBlur={() => setFocused(null)}
              placeholder="A, B, C"
              style={inputStyle(focused === 'variants')}
            />
            <p style={hintStyle}>Минимум 2 варианта. Первый — контрольный</p>
          </div>

          <div>
            <label style={labelStyle}>Тип метрики *</label>
            <select
              value={formData.metricType || 'continuous'}
              onChange={e => setFormData(prev => ({ ...prev, metricType: e.target.value }))}
              onFocus={() => setFocused('metricType')}
              onBlur={() => setFocused(null)}
              style={inputStyle(focused === 'metricType')}
            >
              <option value="binary">Бинарная (0/1)</option>
              <option value="continuous">Непрерывная (числа)</option>
              <option value="ratio">Отношение (проценты)</option>
            </select>
          </div>
        </div>
      </div>

      <div style={sectionStyle}>
        <div style={{ marginBottom: 12 }}>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: 12 }}>
          <div>
            <label style={labelStyle}>Уровень доверия (1 − α)</label>
            <input
              type="number"
              value={formData.confidenceLevel ?? 0.95}
              onChange={e => setFormData(prev => ({ ...prev, confidenceLevel: parseFloat(e.target.value) }))}
              onFocus={() => setFocused('confidenceLevel')}
              onBlur={() => setFocused(null)}
              min={0.8}
              max={0.99}
              step={0.01}
              style={inputStyle(focused === 'confidenceLevel')}
            />
          </div>
          <div>
            <label style={labelStyle}>Мощность теста (1 − β)</label>
            <input
              type="number"
              value={formData.power ?? 0.8}
              onChange={e => setFormData(prev => ({ ...prev, power: parseFloat(e.target.value) }))}
              onFocus={() => setFocused('power')}
              onBlur={() => setFocused(null)}
              min={0.5}
              max={0.95}
              step={0.05}
              style={inputStyle(focused === 'power')}
            />
          </div>
          <div>
            <label style={labelStyle}>MDE</label>
            <input
              type="number"
              value={formData.minEffectSize ?? 0.1}
              onChange={e => setFormData(prev => ({ ...prev, minEffectSize: parseFloat(e.target.value) }))}
              onFocus={() => setFocused('minEffectSize')}
              onBlur={() => setFocused(null)}
              min={0.01}
              max={1.0}
              step={0.01}
              style={inputStyle(focused === 'minEffectSize')}
            />
          </div>
        </div>

        <div style={{ marginTop: 12, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          <div>
            <label style={labelStyle}>Стратегия разделения трафика</label>
            <select
              value={formData.trafficSplitType || 'fixed'}
              onChange={e => setFormData(prev => ({ ...prev, trafficSplitType: e.target.value }))}
              onFocus={() => setFocused('trafficSplitType')}
              onBlur={() => setFocused(null)}
              style={inputStyle(focused === 'trafficSplitType')}
            >
              <option value="fixed">Равномерное разделение (fixed)</option>
              <option value="adaptive">Адаптивное разделение (adaptive)</option>
            </select>
          </div>

          <div>
            <label style={labelStyle}>Режим анализа</label>
            <select
              value={formData.analysisMode || 'fixed_experiment'}
              onChange={e => setFormData(prev => ({ ...prev, analysisMode: e.target.value }))}
              onFocus={() => setFocused('analysisMode')}
              onBlur={() => setFocused(null)}
              style={inputStyle(focused === 'analysisMode')}
            >
              <option value="fixed_experiment">Фиксированный эксперимент (валидные выводы)</option>
              <option value="adaptive_bandit">Адаптивный бандит (исследовательский режим)</option>
            </select>
          </div>
        </div>
      </div>

      <div style={sectionStyle}>
        <div style={{ marginBottom: 12 }}>
          <div style={{ fontSize: 12, textTransform: 'uppercase', letterSpacing: '0.35px', color: c.textSub }}>Данные и время симуляции</div>
          <div style={{ fontSize: 13, color: c.textMuted, marginTop: 2 }}>Выберите источник данных и длительность эксперимента</div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 220px', gap: 12 }}>
          <div>
            <label style={labelStyle}>Синтетический датасет *</label>
            <select
              value={formData.datasetId || ''}
              onChange={e => setFormData(prev => ({ ...prev, datasetId: parseInt(e.target.value) }))}
              onFocus={() => setFocused('datasetId')}
              onBlur={() => setFocused(null)}
              style={inputStyle(focused === 'datasetId', formErrors.datasetId)}
            >
              <option value="">Выберите датасет</option>
              {datasets.map((ds: any) => (
                <option key={ds.id} value={ds.id}>
                  ID: {ds.id} | {ds.dataset_name || 'Без имени'} | Записей: {ds.sample_count?.toLocaleString('ru-RU')} | {new Date(ds.created_at).toLocaleDateString('ru-RU')}
                </option>
              ))}
            </select>
            {formErrors.datasetId ? <p style={errorStyle}>{formErrors.datasetId}</p> : <p style={hintStyle}></p>}
          </div>

          <div>
            <label style={labelStyle}>Длительность (мин)</label>
            <input
              type="number"
              value={formData.simulationDurationMinutes ?? 20}
              onChange={e => setFormData(prev => ({ ...prev, simulationDurationMinutes: parseInt(e.target.value) }))}
              onFocus={() => setFocused('simulationDurationMinutes')}
              onBlur={() => setFocused(null)}
              min={1}
              max={180}
              style={inputStyle(focused === 'simulationDurationMinutes')}
            />
          </div>
        </div>
      </div>
    </div>
  );
};
