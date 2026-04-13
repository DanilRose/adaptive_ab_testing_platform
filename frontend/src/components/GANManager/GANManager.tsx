import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertTriangle,
  Check,
  CircleHelp,
  Cpu,
  Download,
  FileText,
  PauseCircle,
  Play,
  RefreshCw,
  Settings,
  Sparkles,
  Trash2,
  X,
} from 'lucide-react';
import { dataAPI, templatesAPI } from '../../utils/api';
import type { GANTrainingPayload, GeneratedHistoryItem, SyntheticGenerationPayload } from '../../types';
import { useTheme } from '@/context/ThemeContext';

type TemplateType = 'gan_config' | 'synthetic_data';

interface TemplateItem {
  id: number;
  name: string;
  description?: string;
  template_type: TemplateType;
  config_json: Record<string, unknown>;
  tags?: string[];
}

interface GANStatus {
  status?: string;
  is_trained?: boolean;
  available_checkpoints?: number;
  loaded_checkpoint_name?: string;
  current_epoch?: number;
  total_epochs?: number;
  training_progress?: number;
  config?: Record<string, unknown>;
  loss_history?: {
    total_epochs?: number;
    latest_g_loss?: number;
    latest_d_loss?: number;
    latest_wasserstein?: number;
  };
}

interface CheckpointItem {
  id: number;
  name?: string;
  filename?: string;
  created_at?: string;
  metrics?: {
    size?: number;
  };
}

interface NumericRangeDef {
  min: number;
  max: number;
}

interface FilterOptions {
  cities?: string[];
  devices?: string[];
  os?: string[];
  browsers?: string[];
  user_types?: string[];
  traffic_sources?: string[];
  genders?: string[];
  numeric_ranges?: Record<string, NumericRangeDef>;
  tech_tree?: Record<string, Record<string, string[]>>;
  compatibility?: {
    os_to_devices?: Record<string, string[]>;
    browser_to_os?: Record<string, string[]>;
  };
}

interface TrainFormState {
  epochs: number;
  real_data_samples: number;
  save_checkpoint: boolean;
  checkpoint_name: string;
  LATENT_DIM?: number;
  BATCH_SIZE?: number;
  LEARNING_RATE?: number;
  DROPOUT_RATE?: number;
  LAMBDA_GP?: number;
  N_CRITIC?: number;
  GENERATOR_LAYERS?: string;
  DISCRIMINATOR_LAYERS?: string;
  USE_WGAN_GP?: boolean;
}

interface SyntheticFormState {
  num_users: number;
  evaluation_metrics: boolean;
  dataset_name: string;
}

const GAN_CONFIG_FIELDS: Array<{
  name: keyof TrainFormState;
  label: string;
  tooltip: string;
  type: 'number' | 'text' | 'boolean';
  min?: number;
  max?: number;
  step?: number;
}> = [
  {
    name: 'LATENT_DIM',
    label: 'Размер латентного вектора',
    tooltip: 'Размерность случайного шума (Z), который подаётся на вход генератору. Чем больше значение, тем богаче вариативность синтетических данных, но тем сложнее стабилизировать обучение. Обычно используют 64–256.',
    type: 'number',
    min: 32,
    max: 512,
  },
  {
    name: 'BATCH_SIZE',
    label: 'Размер батча',
    tooltip: 'Сколько строк данных модель обрабатывает за один шаг. Большой батч делает обучение стабильнее, но требует больше памяти. Типичный диапазон: 128–512.',
    type: 'number',
    min: 64,
    max: 4096,
    step: 64,
  },
  {
    name: 'LEARNING_RATE',
    label: 'Скорость обучения',
    tooltip: 'Шаг обновления весов нейросети. Слишком высокий шаг может «раскачать» обучение, слишком низкий — сильно замедлить. Для GAN обычно 0.0001–0.0003.',
    type: 'number',
    min: 0.00001,
    max: 0.01,
    step: 0.00001,
  },
  {
    name: 'DROPOUT_RATE',
    label: 'Дропаут',
    tooltip: 'Доля нейронов, временно отключаемых на каждом шаге обучения для борьбы с переобучением. Часто ставят 0.2–0.5.',
    type: 'number',
    min: 0,
    max: 0.8,
    step: 0.05,
  },
  {
    name: 'LAMBDA_GP',
    label: 'Lambda GP',
    tooltip: 'Вес gradient penalty в WGAN-GP. Этот параметр контролирует «штраф» за нестабильные градиенты дискриминатора и напрямую влияет на устойчивость обучения. Базовое рабочее значение — 10.',
    type: 'number',
    min: 1,
    max: 50,
    step: 1,
  },
  {
    name: 'N_CRITIC',
    label: 'N Critic',
    tooltip: 'Сколько обновлений дискриминатора выполняется на одно обновление генератора. Обычно 3–5, чтобы дискриминатор успевал давать качественный сигнал генератору.',
    type: 'number',
    min: 1,
    max: 10,
    step: 1,
  },
  {
    name: 'GENERATOR_LAYERS',
    label: 'Слои генератора',
    tooltip: 'Архитектура генератора через запятую. Пример: 256,512,256 — три скрытых слоя с указанным числом нейронов.',
    type: 'text',
  },
  {
    name: 'DISCRIMINATOR_LAYERS',
    label: 'Слои дискриминатора',
    tooltip: 'Архитектура дискриминатора через запятую. Пример: 512,256. Обычно выбирают сопоставимую или немного более «сильную» сеть, чем у генератора.',
    type: 'text',
  },
  {
    name: 'USE_WGAN_GP',
    label: 'WGAN-GP режим',
    tooltip: 'Включает стабильный режим Wasserstein GAN + Gradient Penalty. Как правило, для табличных данных его рекомендуется оставлять включённым.',
    type: 'boolean',
  },
];

const MULTI_FILTER_FIELDS: Array<{ key: keyof FilterOptions; label: string }> = [
  { key: 'cities', label: 'Города' },
  { key: 'devices', label: 'Устройства' },
  { key: 'os', label: 'ОС' },
  { key: 'browsers', label: 'Браузеры' },
  { key: 'user_types', label: 'Типы пользователей' },
  { key: 'traffic_sources', label: 'Источники трафика' },
  { key: 'genders', label: 'Пол' },
];

const FILTER_VALUE_LABELS: Partial<Record<keyof FilterOptions, Record<string, string>>> = {
  genders: {
    male: 'Мужской',
    female: 'Женский',
  },
  user_types: {
    browser: 'Просматривающий',
    shopper: 'Покупатель',
    researcher: 'Исследователь',
    returning: 'Вернувшийся',
  },
  traffic_sources: {
    social: 'Соцсети',
    direct: 'Прямой',
    email: 'Email',
    organic: 'Органический',
  },
  browsers: {
    'samsung internet': 'Samsung Internet Browser',
  },
};

const NUMERIC_RANGE_LABELS: Record<string, string> = {
  age: 'Возраст',
  income: 'Доход',
  sessions: 'Количество сессий',
  pages_viewed: 'Просмотрено страниц',
  time_on_site: 'Время на сайте',
  purchase_count: 'Количество покупок',
  avg_order_value: 'Средний чек',
  previous_purchases: 'Предыдущие покупки',
  total_spent: 'Всего потрачено',
  visits_per_week: 'Посещений в неделю',
  session_duration: 'Длительность сессии',
  pages_per_session: 'Страниц за сессию',
};

interface ToastState {
  message: string;
  kind: 'success' | 'error';
  leaving?: boolean;
}

const formatDate = (value?: string | null): string => (value ? new Date(value).toLocaleString('ru-RU') : '—');

const parseLayers = (value?: string): number[] | undefined => {
  if (!value) return undefined;
  const parsed = value
    .split(',')
    .map((v) => Number(v.trim()))
    .filter((v) => !Number.isNaN(v) && v > 0);
  return parsed.length ? parsed : undefined;
};

const asNumber = (v: unknown): number | undefined => {
  if (v === '' || v === null || v === undefined) return undefined;
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
};

const getDatasetRecords = (dataset: Record<string, unknown> | undefined): Array<Record<string, unknown>> => {
  if (!dataset) return [];
  const records = dataset.records ?? dataset.synthetic_preview ?? dataset.preview_json;
  if (Array.isArray(records)) {
    return records.filter((r): r is Record<string, unknown> => typeof r === 'object' && r !== null);
  }
  return [];
};

const translateFilterValue = (fieldKey: keyof FilterOptions, value: string): string => {
  const dict = FILTER_VALUE_LABELS[fieldKey];
  return dict?.[value.toLowerCase()] || value;
};

const translateRangeLabel = (key: string): string => NUMERIC_RANGE_LABELS[key] || key;

const FILTER_KEY_LABELS: Record<string, string> = {
  cities: 'Города',
  devices: 'Устройства',
  os: 'ОС',
  browsers: 'Браузеры',
  user_types: 'Тип пользователей',
  traffic_sources: 'Источник трафика',
  genders: 'Пол',
  email_subscribed: 'Подписка на email',
  push_enabled: 'Push уведомления',
  is_weekend: 'Выходные',
};

const buildAppliedFiltersSummary = (draft: Record<string, unknown>): string[] => {
  const parts: string[] = [];

  Object.entries(draft).forEach(([key, value]) => {
    if (key === 'numeric_ranges' && value && typeof value === 'object') {
      Object.entries(value as Record<string, { min?: number; max?: number }>).forEach(([rangeKey, range]) => {
        const hasMin = typeof range.min === 'number';
        const hasMax = typeof range.max === 'number';
        if (!hasMin && !hasMax) return;
        parts.push(`${translateRangeLabel(rangeKey)}: ${hasMin ? range.min : 'мин'}–${hasMax ? range.max : 'макс'}`);
      });
      return;
    }

    if (Array.isArray(value)) {
      if (!value.length) return;
      const translatedValues = value.map((v) => {
        if (typeof v !== 'string') return String(v);
        if ((['cities', 'devices', 'os', 'browsers', 'user_types', 'traffic_sources', 'genders'] as string[]).includes(key)) {
          return translateFilterValue(key as keyof FilterOptions, v);
        }
        return v;
      });
      parts.push(`${FILTER_KEY_LABELS[key] || key}: ${translatedValues.join(', ')}`);
      return;
    }

    if (typeof value === 'boolean') {
      parts.push(`${FILTER_KEY_LABELS[key] || key}: ${value ? 'Да' : 'Нет'}`);
      return;
    }

    if (value !== null && value !== undefined && value !== '') {
      parts.push(`${FILTER_KEY_LABELS[key] || key}: ${String(value)}`);
    }
  });

  return parts;
};

export const GANManager: React.FC = () => {
  const { theme } = useTheme();
  const isDark = theme === 'dark';

  const c = useMemo(
    () => ({
      pageBg: isDark ? '#0f0d0b' : '#faf8f5',
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
      danger: isDark ? '#fca5a5' : '#dc2626',
      dangerSoft: isDark ? 'rgba(239,68,68,0.12)' : '#fef2f2',
      dangerBorder: isDark ? 'rgba(239,68,68,0.25)' : '#fecaca',
      success: isDark ? '#86efac' : '#166534',
      successSoft: isDark ? 'rgba(34,197,94,0.15)' : '#f0fdf4',
      info: isDark ? '#93c5fd' : '#1d4ed8',
      infoSoft: isDark ? 'rgba(59,130,246,0.15)' : '#eff6ff',
      shadow: isDark ? '0 10px 32px rgba(0,0,0,0.36)' : '0 8px 28px rgba(28,25,23,0.07)',
      rowHov: isDark ? '#211f1d' : '#faf8f5',
    }),
    [isDark],
  );

  const [ganStatus, setGanStatus] = useState<GANStatus>({});
  const [checkpoints, setCheckpoints] = useState<CheckpointItem[]>([]);
  const [generatedHistory, setGeneratedHistory] = useState<GeneratedHistoryItem[]>([]);
  const [filterOptions, setFilterOptions] = useState<FilterOptions | null>(null);

  const [activeTab, setActiveTab] = useState<'train' | 'synthetic'>('train');
  const [loadingStatus, setLoadingStatus] = useState(false);
  const [training, setTraining] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [loadingTemplates, setLoadingTemplates] = useState(false);

  const [toast, setToast] = useState<ToastState | null>(null);

  const [trainForm, setTrainForm] = useState<TrainFormState>({
    epochs: 50,
    real_data_samples: 50000,
    save_checkpoint: true,
    checkpoint_name: 'best_wgan_config',
    USE_WGAN_GP: true,
  });

  const [syntheticForm, setSyntheticForm] = useState<SyntheticFormState>({
    num_users: 10000,
    evaluation_metrics: true,
    dataset_name: '',
  });

  const [filterDraft, setFilterDraft] = useState<Record<string, unknown>>({});

  const [ganTemplates, setGanTemplates] = useState<TemplateItem[]>([]);
  const [syntheticTemplates, setSyntheticTemplates] = useState<TemplateItem[]>([]);
  const [ganTemplateModal, setGanTemplateModal] = useState(false);
  const [syntheticTemplateModal, setSyntheticTemplateModal] = useState(false);

  const [dataPreviewModal, setDataPreviewModal] = useState<{ visible: boolean; dataset?: Record<string, unknown> }>({ visible: false });

  const [confirmDeleteCheckpoint, setConfirmDeleteCheckpoint] = useState<CheckpointItem | null>(null);
  const [confirmDeleteHistoryItem, setConfirmDeleteHistoryItem] = useState<GeneratedHistoryItem | null>(null);
  const [lastLoadedCheckpointName, setLastLoadedCheckpointName] = useState<string | null>(() => localStorage.getItem('gan_last_loaded_checkpoint_name'));
  const [activeFilterGroup, setActiveFilterGroup] = useState<'geo' | 'profile' | 'tech' | 'activity' | 'ranges'>('geo');

  const showToast = (message: string, kind: 'success' | 'error') => {
    setToast({ message, kind, leaving: false });
    window.setTimeout(() => {
      setToast((prev) => (prev ? { ...prev, leaving: true } : null));
    }, 2600);
    window.setTimeout(() => setToast(null), 3000);
  };

  const loadGANStatus = async (): Promise<void> => {
    try {
      const response = await dataAPI.getGANStatus();
      setGanStatus((response.data || {}) as GANStatus);
    } catch {
      setGanStatus({ status: 'error', is_trained: false });
    }
  };

  const loadCheckpoints = async (): Promise<void> => {
    try {
      const response = await dataAPI.getGANCheckpoints();
      setCheckpoints(((response.data?.checkpoints || []) as CheckpointItem[]) ?? []);
    } catch {
      setCheckpoints([]);
    }
  };

  const loadFilterOptions = async (): Promise<void> => {
    try {
      const response = await dataAPI.getDatasetStats();
      setFilterOptions((response.data || null) as FilterOptions | null);
    } catch {
      setFilterOptions(null);
    }
  };

  const loadGeneratedHistory = async (withSpinner = true): Promise<void> => {
    try {
      if (withSpinner) setHistoryLoading(true);
      const response = await dataAPI.listGeneratedHistory(25);
      setGeneratedHistory(((response.data?.items || []) as GeneratedHistoryItem[]) ?? []);
    } catch {
      setGeneratedHistory([]);
    } finally {
      if (withSpinner) setHistoryLoading(false);
    }
  };

  useEffect(() => {
    const init = async () => {
      setLoadingStatus(true);
      await Promise.all([loadGANStatus(), loadCheckpoints(), loadFilterOptions(), loadGeneratedHistory()]);
      setLoadingStatus(false);
    };
    init();

    const interval = window.setInterval(() => {
      loadGANStatus();
      loadGeneratedHistory(false);
    }, 5000);

    return () => window.clearInterval(interval);
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      if (confirmDeleteHistoryItem) {
        setConfirmDeleteHistoryItem(null);
        return;
      }
      if (confirmDeleteCheckpoint) {
        setConfirmDeleteCheckpoint(null);
        return;
      }
      if (dataPreviewModal.visible) {
        setDataPreviewModal({ visible: false });
        return;
      }
      if (syntheticTemplateModal) {
        setSyntheticTemplateModal(false);
        return;
      }
      if (ganTemplateModal) {
        setGanTemplateModal(false);
      }
    };

    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [
    ganTemplateModal,
    syntheticTemplateModal,
    dataPreviewModal.visible,
    confirmDeleteCheckpoint,
    confirmDeleteHistoryItem,
  ]);

  const isTraining = ganStatus.status === 'training';
  const isStopped = ganStatus.status === 'training_paused';
  const isTrainingOrStopped = isTraining || isStopped;
  const canGenerateSynthetic = Boolean(ganStatus.is_trained) || ganStatus.status === 'checkpoint_loaded';

  const getFriendlyStatus = (): string => {
    if (ganStatus.status === 'checkpoint_not_loaded' || !ganStatus.status) return 'Чекпоинт не загружен';
    if (ganStatus.status === 'checkpoint_loaded') {
      const backendLoadedName =
        typeof ganStatus.loaded_checkpoint_name === 'string'
          ? ganStatus.loaded_checkpoint_name.trim()
          : '';

      const looksLikeTempName = /^tmp[\w-]+$/i.test(backendLoadedName);
      const loadedName = backendLoadedName && !looksLikeTempName ? backendLoadedName : lastLoadedCheckpointName;

      return loadedName ? `Загружен чекпоинт: ${loadedName}` : 'Загружен чекпоинт';
    }
    if (ganStatus.status === 'training') {
      return `Обучение: ${ganStatus.current_epoch || 0}/${ganStatus.total_epochs || 0} эпох`;
    }
    if (ganStatus.status === 'training_paused') {
      return `Пауза обучения: ${ganStatus.current_epoch || 0}/${ganStatus.total_epochs || 0} эпох`;
    }
    if ((ganStatus.status || '').includes('error')) {
      return `Ошибка: ${(ganStatus.status || '').replace('error: ', '')}`;
    }
    return ganStatus.status || '—';
  };

  const getStatusTone = ():
    | { bg: string; text: string; border: string }
    | { bg: string; text: string; border: string } => {
    if ((ganStatus.status || '').includes('error')) {
      return { bg: c.dangerSoft, text: c.danger, border: c.dangerBorder };
    }
    if (isTraining || isStopped) {
      return { bg: c.infoSoft, text: c.info, border: isDark ? 'rgba(59,130,246,0.35)' : '#bfdbfe' };
    }
    if (ganStatus.is_trained || ganStatus.status === 'checkpoint_loaded') {
      return { bg: c.successSoft, text: c.success, border: isDark ? 'rgba(34,197,94,0.35)' : '#bbf7d0' };
    }
    return { bg: c.panelSoft, text: c.textMuted, border: c.border };
  };

  const handleTrainGAN = async () => {
    if (!trainForm.epochs || !trainForm.real_data_samples || !trainForm.checkpoint_name.trim()) {
      showToast('Заполните обязательные поля обучения', 'error');
      return;
    }

    setTraining(true);
    try {
      const configOverrides: Record<string, string | number | boolean | number[]> = {};

      if (asNumber(trainForm.LATENT_DIM) !== undefined) configOverrides.LATENT_DIM = asNumber(trainForm.LATENT_DIM) as number;
      if (asNumber(trainForm.BATCH_SIZE) !== undefined) configOverrides.BATCH_SIZE = asNumber(trainForm.BATCH_SIZE) as number;
      if (asNumber(trainForm.LEARNING_RATE) !== undefined) configOverrides.LEARNING_RATE = asNumber(trainForm.LEARNING_RATE) as number;
      if (asNumber(trainForm.DROPOUT_RATE) !== undefined) configOverrides.DROPOUT_RATE = asNumber(trainForm.DROPOUT_RATE) as number;
      if (asNumber(trainForm.LAMBDA_GP) !== undefined) configOverrides.LAMBDA_GP = asNumber(trainForm.LAMBDA_GP) as number;
      if (asNumber(trainForm.N_CRITIC) !== undefined) configOverrides.N_CRITIC = asNumber(trainForm.N_CRITIC) as number;

      const generatorLayers = parseLayers(trainForm.GENERATOR_LAYERS);
      if (generatorLayers?.length) configOverrides.GENERATOR_LAYERS = generatorLayers;

      const discriminatorLayers = parseLayers(trainForm.DISCRIMINATOR_LAYERS);
      if (discriminatorLayers?.length) configOverrides.DISCRIMINATOR_LAYERS = discriminatorLayers;

      if (trainForm.USE_WGAN_GP !== undefined) configOverrides.USE_WGAN_GP = trainForm.USE_WGAN_GP;

      const payload: GANTrainingPayload = {
        epochs: Number(trainForm.epochs),
        real_data_samples: Number(trainForm.real_data_samples),
        save_checkpoint: trainForm.save_checkpoint,
        checkpoint_name: trainForm.checkpoint_name.trim(),
        gan_config: Object.keys(configOverrides).length ? configOverrides : undefined,
      };

      await dataAPI.trainGAN(payload);
      showToast('Обучение GAN запущено', 'success');
      loadGANStatus();
    } catch (error: any) {
      showToast(`Ошибка запуска обучения: ${error.response?.data?.detail || error.message}`, 'error');
    } finally {
      setTraining(false);
    }
  };

  const handleStopTraining = async () => {
    try {
      await dataAPI.stopGANTraining();
      showToast('Запрос на остановку отправлен', 'success');
      loadGANStatus();
    } catch (error: any) {
      showToast(`Ошибка остановки: ${error.response?.data?.detail || error.message}`, 'error');
    }
  };

  const handleResumeTraining = async () => {
    try {
      await dataAPI.resumeGANTraining();
      showToast('Обучение возобновлено', 'success');
      loadGANStatus();
    } catch (error: any) {
      showToast(`Ошибка возобновления: ${error.response?.data?.detail || error.message}`, 'error');
    }
  };

  const handleResetTraining = async () => {
    try {
      await dataAPI.resetGANTraining();
      showToast('Обучение GAN сброшено', 'success');
      loadGANStatus();
    } catch (error: any) {
      showToast(`Ошибка сброса: ${error.response?.data?.detail || error.message}`, 'error');
    }
  };

  const updateFilterValue = (key: string, value: unknown) => {
    setFilterDraft((prev) => {
      const next = { ...prev };
      if (value === undefined || value === null || (Array.isArray(value) && value.length === 0) || value === '') {
        delete next[key];
      } else {
        next[key] = value;
      }
      return next;
    });
  };

  const toggleMultiFilterOption = (key: string, value: string) => {
    const current = Array.isArray(filterDraft[key]) ? (filterDraft[key] as string[]) : [];
    const exists = current.includes(value);
    const next = exists ? current.filter((x) => x !== value) : [...current, value];
    updateFilterValue(key, next);
  };

  const updateNumericRange = (field: string, bound: 'min' | 'max', value?: number) => {
    setFilterDraft((prev) => {
      const ranges = { ...((prev.numeric_ranges as Record<string, { min?: number; max?: number }>) || {}) };
      const existing = { ...(ranges[field] || {}) };

      if (value === undefined || Number.isNaN(value)) {
        delete existing[bound];
      } else {
        existing[bound] = value;
      }

      if (Object.keys(existing).length) {
        ranges[field] = existing;
      } else {
        delete ranges[field];
      }

      const next = { ...prev };
      if (Object.keys(ranges).length) {
        next.numeric_ranges = ranges;
      } else {
        delete next.numeric_ranges;
      }
      return next;
    });
  };

  const handleGenerateData = async () => {
    if (!syntheticForm.num_users || syntheticForm.num_users < 100) {
      showToast('Укажите корректное количество пользователей (от 100)', 'error');
      return;
    }

    setGenerating(true);
    try {
      const payload: SyntheticGenerationPayload = {
        num_users: Number(syntheticForm.num_users),
        evaluation_metrics: syntheticForm.evaluation_metrics,
        dataset_name: syntheticForm.dataset_name.trim() || undefined,
        filters: Object.keys(filterDraft).length ? (filterDraft as Record<string, string | number | boolean | null>) : undefined,
      };

      const response = await dataAPI.generateSynthetic(payload);
      showToast(`Сгенерировано ${response.data.synthetic_samples} пользователей`, 'success');
      setDataPreviewModal({ visible: true, dataset: response.data as Record<string, unknown> });
      loadGeneratedHistory();
    } catch (error: any) {
      showToast(`Ошибка генерации: ${error.response?.data?.detail || error.message}`, 'error');
    } finally {
      setGenerating(false);
    }
  };

  const handleLoadCheckpoint = async (checkpointName: string) => {
    try {
      await dataAPI.loadGANCheckpoint(checkpointName);
      setLastLoadedCheckpointName(checkpointName);
      localStorage.setItem('gan_last_loaded_checkpoint_name', checkpointName);
      showToast(`Модель загружена: ${checkpointName}`, 'success');
      loadGANStatus();
    } catch (error: any) {
      showToast(`Ошибка загрузки модели: ${error.response?.data?.detail || error.message}`, 'error');
    }
  };

  const handleDeleteCheckpoint = async () => {
    if (!confirmDeleteCheckpoint) return;
    try {
      await dataAPI.deleteGANCheckpoint(confirmDeleteCheckpoint.id);
      showToast('Чекпоинт удалён', 'success');
      setConfirmDeleteCheckpoint(null);
      await loadCheckpoints();
      await loadGANStatus();
    } catch (error: any) {
      showToast(`Ошибка удаления чекпоинта: ${error.response?.data?.detail || error.message}`, 'error');
    }
  };

  const handleDeleteDatasetHistoryItem = async () => {
    if (!confirmDeleteHistoryItem) return;
    try {
      await dataAPI.deleteGeneratedHistoryItem(confirmDeleteHistoryItem.id);
      showToast('Запись о CSV удалена', 'success');
      if (dataPreviewModal.dataset?.id === confirmDeleteHistoryItem.id) {
        setDataPreviewModal({ visible: false });
      }
      setConfirmDeleteHistoryItem(null);
      await loadGeneratedHistory();
    } catch (error: any) {
      showToast(`Ошибка удаления записи: ${error.response?.data?.detail || error.message}`, 'error');
    }
  };

  const downloadDatasetAsCSV = (dataset: Record<string, unknown>) => {
    const records = getDatasetRecords(dataset);
    if (!records.length) {
      showToast('Нет данных для скачивания', 'error');
      return;
    }

    const headers = Object.keys(records[0]);
    const csvContent = [
      headers.join(','),
      ...records.map((row) =>
        headers
          .map((header) => {
            const value = row[header];
            const text = value === null || value === undefined ? '' : String(value);
            if (text.includes(',') || text.includes('"')) {
              return `"${text.replace(/"/g, '""')}"`;
            }
            return text;
          })
          .join(','),
      ),
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    const fallbackName = `synthetic_data_${new Date().toISOString().slice(0, 10)}.csv`;
    const datasetName = typeof dataset.dataset_name === 'string' ? dataset.dataset_name : '';
    const filename = datasetName ? `${datasetName}.csv` : fallbackName;

    link.setAttribute('href', url);
    link.setAttribute('download', filename);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    showToast(`Файл ${filename} скачан`, 'success');
  };

  const handleDownloadCSV = async () => {
    const id = asNumber(dataPreviewModal.dataset?.id);
    if (!id) return;
    try {
      const response = await dataAPI.getFullDataset(id);
      downloadDatasetAsCSV(response.data as Record<string, unknown>);
    } catch (error: any) {
      showToast(`Ошибка загрузки данных: ${error.response?.data?.detail || error.message}`, 'error');
    }
  };

  const loadTemplates = async (templateType: TemplateType) => {
    setLoadingTemplates(true);
    try {
      const response = await templatesAPI.listTemplates(templateType);
      const items = (response.data?.items || []) as TemplateItem[];
      if (templateType === 'gan_config') {
        setGanTemplates(items);
      } else {
        setSyntheticTemplates(items);
      }
    } catch {
      showToast('Ошибка загрузки шаблонов', 'error');
    } finally {
      setLoadingTemplates(false);
    }
  };

  const applyGANTemplate = (template: TemplateItem) => {
    const cfg = template.config_json as Record<string, unknown>;

    setTrainForm((prev) => ({
      ...prev,
      epochs: asNumber(cfg.epochs) ?? prev.epochs,
      real_data_samples: asNumber(cfg.real_data_samples) ?? prev.real_data_samples,
      save_checkpoint: typeof cfg.save_checkpoint === 'boolean' ? cfg.save_checkpoint : prev.save_checkpoint,
      checkpoint_name: typeof cfg.checkpoint_name === 'string' ? cfg.checkpoint_name : prev.checkpoint_name,
      LATENT_DIM: asNumber(cfg.LATENT_DIM),
      BATCH_SIZE: asNumber(cfg.BATCH_SIZE),
      LEARNING_RATE: asNumber(cfg.LEARNING_RATE),
      DROPOUT_RATE: asNumber(cfg.DROPOUT_RATE),
      LAMBDA_GP: asNumber(cfg.LAMBDA_GP),
      N_CRITIC: asNumber(cfg.N_CRITIC),
      GENERATOR_LAYERS: Array.isArray(cfg.GENERATOR_LAYERS)
        ? (cfg.GENERATOR_LAYERS as number[]).join(',')
        : (typeof cfg.GENERATOR_LAYERS === 'string' ? cfg.GENERATOR_LAYERS : undefined),
      DISCRIMINATOR_LAYERS: Array.isArray(cfg.DISCRIMINATOR_LAYERS)
        ? (cfg.DISCRIMINATOR_LAYERS as number[]).join(',')
        : (typeof cfg.DISCRIMINATOR_LAYERS === 'string' ? cfg.DISCRIMINATOR_LAYERS : undefined),
      USE_WGAN_GP: typeof cfg.USE_WGAN_GP === 'boolean' ? cfg.USE_WGAN_GP : prev.USE_WGAN_GP,
    }));

    setGanTemplateModal(false);
    showToast(`Шаблон GAN «${template.name}» применён`, 'success');
  };

  const applySyntheticTemplate = (template: TemplateItem) => {
    const cfg = template.config_json as Record<string, unknown>;

    setSyntheticForm((prev) => ({
      ...prev,
      num_users: asNumber(cfg.num_users) ?? prev.num_users,
      evaluation_metrics: typeof cfg.evaluation_metrics === 'boolean' ? cfg.evaluation_metrics : prev.evaluation_metrics,
      dataset_name: typeof cfg.dataset_name === 'string' ? cfg.dataset_name : prev.dataset_name,
    }));

    if (cfg.filters && typeof cfg.filters === 'object') {
      setFilterDraft(cfg.filters as Record<string, unknown>);
    }

    setSyntheticTemplateModal(false);
    showToast(`Шаблон «${template.name}» применён`, 'success');
  };

  const statusTone = getStatusTone();
  const checkpointCount = Math.max(Number(ganStatus.available_checkpoints || 0), checkpoints.length);
  const appliedFiltersSummary = buildAppliedFiltersSummary(filterDraft);
  const appliedFiltersPreview =
    appliedFiltersSummary.length > 2
      ? `${appliedFiltersSummary.slice(0, 2).join(' · ')} +${appliedFiltersSummary.length - 2}`
      : appliedFiltersSummary.join(' · ');


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
            <Cpu size={20} />
          </div>
          <div>
            <h1 style={{ margin: 0, fontSize: 22, fontWeight: 700, letterSpacing: '-0.4px' }}>GAN менеджер</h1>
            <p style={{ margin: 0, fontSize: 13, color: c.textMuted }}>Обучение GAN, управление чекпоинтами и генерация синтетических данных</p>
          </div>
        </div>

        <ActionButton
          onClick={() => {
            setLoadingStatus(true);
            Promise.all([loadGANStatus(), loadCheckpoints(), loadGeneratedHistory(), loadFilterOptions()]).finally(() => setLoadingStatus(false));
          }}
          disabled={loadingStatus}
          isDark={isDark}
          c={c}
          icon={<RefreshCw size={14} style={{ animation: loadingStatus ? 'spin 0.8s linear infinite' : 'none' }} />}
          label="Обновить"
          variant="secondary"
        />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 12, marginBottom: 16 }}>
        <StatCard title="Статус GAN" value={getFriendlyStatus()} c={c} tone={statusTone} />
        <StatCard title="Чекпоинты" value={String(checkpointCount)} c={c} />
        <StatCard
          title="Эпох обучения"
          value={ganStatus.is_trained && ganStatus.loss_history?.total_epochs ? String(ganStatus.loss_history.total_epochs) : 'Загрузите модель'}
          c={c}
        />
      </div>

      {!!ganStatus.config && Object.keys(ganStatus.config).length > 0 && (
        <Panel c={c} title={!ganStatus.is_trained && !isTraining ? 'Последняя конфигурация обучаемой модели' : 'Текущая конфигурация модели'} icon={<Settings size={14} />}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 10 }}>
            {[
              ['Эпохи', ganStatus.config.EPOCHS],
              ['Размер батча', ganStatus.config.BATCH_SIZE],
              ['Скорость обучения', ganStatus.config.LEARNING_RATE],
              ['WGAN-GP', ganStatus.config.USE_WGAN_GP ? 'Да' : 'Нет'],
              ['Lambda GP', ganStatus.config.LAMBDA_GP],
              ['N Critic', ganStatus.config.N_CRITIC],
              ['LATENT_DIM', ganStatus.config.LATENT_DIM],
              ['Дропаут', ganStatus.config.DROPOUT_RATE],
              ['Устройство', ganStatus.config.DEVICE],
              ['G Loss', ganStatus.loss_history?.latest_g_loss?.toFixed(4)],
              ['D Loss', ganStatus.loss_history?.latest_d_loss?.toFixed(4)],
              ['Wasserstein', ganStatus.loss_history?.latest_wasserstein?.toFixed(4)],
            ].map(([label, value]) => (
              <div key={String(label)} style={{ border: `1px solid ${c.border}`, borderRadius: 10, padding: '10px 12px', background: c.panelSoft }}>
                <div style={{ fontSize: 11, color: c.textSub, textTransform: 'uppercase', letterSpacing: '0.35px' }}>{String(label)}</div>
                <div style={{ marginTop: 5, fontSize: 14, color: c.textPrimary, fontWeight: 600 }}>{value ? String(value) : 'Н/Д'}</div>
              </div>
            ))}
          </div>
        </Panel>
      )}

      <div style={{ display: 'flex', gap: 6, marginBottom: 12, flexWrap: 'wrap' }}>
        <PillTab active={activeTab === 'train'} onClick={() => setActiveTab('train')} label="Конфигурация и обучение GAN" c={c} />
        <PillTab active={activeTab === 'synthetic'} onClick={() => setActiveTab('synthetic')} label="Генерация синтетических данных" c={c} />
      </div>

      {activeTab === 'train' ? (
        <Panel c={c} title="Обучение и чекпоинты" icon={<Cpu size={14} />}>
          <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10, flexWrap: 'wrap', marginBottom: 14 }}>
            <ActionButton
              onClick={() => {
                setGanTemplateModal(true);
                loadTemplates('gan_config');
              }}
              isDark={isDark}
              c={c}
              icon={<FileText size={14} />}
              label="Выбрать из шаблонов GAN"
              variant="template"
              disabled={isTrainingOrStopped}
            />

            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              {!isTrainingOrStopped && (
                <ActionButton
                  onClick={handleTrainGAN}
                  disabled={training}
                  isDark={isDark}
                  c={c}
                  icon={training ? <RefreshCw size={14} style={{ animation: 'spin 0.8s linear infinite' }} /> : <Play size={14} />}
                  label={training ? 'Запуск...' : 'Обучить GAN с нуля'}
                  variant="primary"
                />
              )}
              {isTraining && (
                <ActionButton onClick={handleStopTraining} isDark={isDark} c={c} icon={<PauseCircle size={14} />} label="Остановить" variant="secondary" />
              )}
              {isStopped && (
                <>
                  <ActionButton onClick={handleResumeTraining} isDark={isDark} c={c} icon={<Play size={14} />} label="Возобновить" variant="primary" />
                  <ActionButton onClick={handleResetTraining} isDark={isDark} c={c} icon={<Trash2 size={14} />} label="Сбросить обучение" variant="ghost" />
                </>
              )}
              {!isTrainingOrStopped && (
                <ActionButton
                  onClick={() =>
                    setTrainForm({
                      epochs: 50,
                      real_data_samples: 50000,
                      save_checkpoint: true,
                      checkpoint_name: 'best_wgan_config',
                      USE_WGAN_GP: true,
                    })
                  }
                  isDark={isDark}
                  c={c}
                  icon={<RefreshCw size={14} />}
                  label="Сбросить форму"
                  variant="ghost"
                />
              )}
            </div>
          </div>

          {isTrainingOrStopped && (
            <div style={{ marginBottom: 16, padding: 14, borderRadius: 10, border: `1px solid ${statusTone.border}`, backgroundColor: statusTone.bg }}>
              <div style={{ color: statusTone.text, fontSize: 13, fontWeight: 600, marginBottom: 8 }}>{getFriendlyStatus()}</div>
              <div style={{ height: 10, borderRadius: 999, background: isDark ? '#292524' : '#e7e5e4', overflow: 'hidden' }}>
                <div style={{ width: `${Math.min(100, ganStatus.training_progress || 0)}%`, height: '100%', background: 'linear-gradient(90deg, #3b82f6, #22c55e)' }} />
              </div>
              <div style={{ marginTop: 8, color: c.textMuted, fontSize: 13 }}>
                Эпоха: {ganStatus.current_epoch || 0}/{ganStatus.total_epochs || 0}
              </div>
            </div>
          )}

          {!isTrainingOrStopped && (
            <>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 12 }}>
                <Field label="Эпохи *" c={c}>
                  <input
                    type="number"
                    min={10}
                    max={500}
                    value={trainForm.epochs}
                    onChange={(e) => setTrainForm((p) => ({ ...p, epochs: Number(e.target.value) }))}
                    style={inputStyle(c)}
                  />
                </Field>
                <Field label="Количество записей для обучения *" c={c}>
                  <input
                    type="number"
                    min={1000}
                    max={100000}
                    step={1000}
                    value={trainForm.real_data_samples}
                    onChange={(e) => setTrainForm((p) => ({ ...p, real_data_samples: Number(e.target.value) }))}
                    style={inputStyle(c)}
                  />
                </Field>
                <Field label="Сохранять чекпоинт" c={c}>
                  <Toggle checked={trainForm.save_checkpoint} onChange={(v) => setTrainForm((p) => ({ ...p, save_checkpoint: v }))} c={c} />
                </Field>
              </div>

              <div style={{ marginTop: 12 }}>
                <Field label="Имя чекпоинта *" c={c}>
                  <input
                    type="text"
                    value={trainForm.checkpoint_name}
                    onChange={(e) => setTrainForm((p) => ({ ...p, checkpoint_name: e.target.value }))}
                    placeholder="Например: best_wgan_config"
                    style={inputStyle(c)}
                  />
                </Field>
              </div>

              <SectionTitle c={c}>Переопределение конфигурации</SectionTitle>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: 12 }}>
                {GAN_CONFIG_FIELDS.map((field) => (
                  <Field
                    key={String(field.name)}
                    label={
                      <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                        {field.label}
                        <TooltipHint c={c} text={field.tooltip} placement={field.name === 'LATENT_DIM' || field.name === 'LAMBDA_GP' ? 'right' : 'top'} />
                      </span>
                    }
                    c={c}
                    compact
                  >
                    {field.type === 'number' ? (
                      <input
                        type="number"
                        min={field.min}
                        max={field.max}
                        step={field.step}
                        value={(trainForm[field.name] as number | undefined) ?? ''}
                        onChange={(e) => {
                          const value = e.target.value === '' ? undefined : Number(e.target.value);
                          setTrainForm((prev) => ({ ...prev, [field.name]: value }));
                        }}
                        placeholder="По умолчанию"
                        style={inputStyle(c)}
                      />
                    ) : field.type === 'text' ? (
                      <input
                        type="text"
                        value={(trainForm[field.name] as string | undefined) ?? ''}
                        onChange={(e) => setTrainForm((prev) => ({ ...prev, [field.name]: e.target.value || undefined }))}
                        placeholder="Например: 512,256,128"
                        style={inputStyle(c)}
                      />
                    ) : (
                      <Toggle checked={Boolean(trainForm[field.name])} onChange={(v) => setTrainForm((prev) => ({ ...prev, [field.name]: v }))} c={c} />
                    )}
                  </Field>
                ))}
              </div>
            </>
          )}

          <SectionTitle c={c}>Доступные чекпоинты</SectionTitle>
          <div style={{ border: `1px solid ${c.border}`, borderRadius: 12, overflow: 'hidden' }}>
            {checkpoints.length === 0 ? (
              <EmptyState c={c} text="Нет доступных чекпоинтов" />
            ) : (
              checkpoints.map((checkpoint, idx) => (
                <div
                  key={checkpoint.id}
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    gap: 10,
                    padding: '12px 14px',
                    borderBottom: idx < checkpoints.length - 1 ? `1px solid ${c.border}` : 'none',
                    background: c.panelBg,
                    flexWrap: 'wrap',
                  }}
                >
                  <div>
                    <div style={{ fontSize: 14, fontWeight: 600 }}>{checkpoint.name || checkpoint.filename || 'Без имени'}</div>
                    <div style={{ fontSize: 12, color: c.textMuted, marginTop: 2 }}>
                      Размер: {checkpoint.metrics?.size ? `${(checkpoint.metrics.size / 1024 / 1024).toFixed(2)} МБ` : 'Н/Д'} · Изменён: {formatDate(checkpoint.created_at)}
                    </div>
                  </div>
                  <div style={{ display: 'flex', gap: 8 }}>
                    <RowActionButton
                      onClick={() => handleLoadCheckpoint(checkpoint.name || checkpoint.filename || '')}
                      disabled={isTrainingOrStopped || !(checkpoint.name || checkpoint.filename)}
                      c={c}
                      label="Загрузить"
                      tone="load"
                    />
                    <RowActionButton onClick={() => setConfirmDeleteCheckpoint(checkpoint)} c={c} icon={<Trash2 size={14} />} label="Удалить" tone="danger" />
                  </div>
                </div>
              ))
            )}
          </div>
        </Panel>
      ) : (
        <Panel c={c} title="Генерация синтетических данных" icon={<Sparkles size={14} />}>
          <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10, flexWrap: 'wrap', marginBottom: 14 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
              <ActionButton
                onClick={() => {
                  setSyntheticTemplateModal(true);
                  loadTemplates('synthetic_data');
                }}
                isDark={isDark}
                c={c}
                icon={<FileText size={14} />}
                label="Выбрать из шаблонов генерации"
                variant="template"
              />
              {appliedFiltersSummary.length > 0 && (
                <span
                  title={appliedFiltersSummary.join(' · ')}
                  style={{
                    padding: '4px 10px',
                    borderRadius: 999,
                    border: `1px solid ${isDark ? 'rgba(34,197,94,0.35)' : '#bbf7d0'}`,
                    background: c.successSoft,
                    color: c.success,
                    fontSize: 12,
                    fontWeight: 600,
                    maxWidth: 520,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                    display: 'inline-block',
                  }}
                >
                  {appliedFiltersPreview}
                </span>
              )}
            </div>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              <ActionButton
                onClick={handleGenerateData}
                disabled={generating || !canGenerateSynthetic || isTrainingOrStopped}
                isDark={isDark}
                c={c}
                icon={generating ? <RefreshCw size={14} style={{ animation: 'spin 0.8s linear infinite' }} /> : <Sparkles size={14} />}
                label={generating ? 'Генерация...' : 'Сгенерировать данные'}
                variant="primary"
              />
              <ActionButton
                onClick={() => {
                  setSyntheticForm({ num_users: 10000, evaluation_metrics: true, dataset_name: '' });
                  setFilterDraft({});
                }}
                isDark={isDark}
                c={c}
                icon={<RefreshCw size={14} />}
                label="Сбросить форму"
                variant="ghost"
              />
            </div>
          </div>

          {!canGenerateSynthetic && (
            <div style={{ marginBottom: 14, border: `1px solid ${statusTone.border}`, backgroundColor: statusTone.bg, color: statusTone.text, borderRadius: 10, padding: '10px 12px', fontSize: 13 }}>
              Для генерации нужно обучить модель или загрузить чекпоинт.
            </div>
          )}

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 12 }}>
            <Field label="Кол-во пользователей *" c={c}>
              <input
                type="number"
                min={100}
                max={100000}
                step={100}
                value={syntheticForm.num_users}
                onChange={(e) => setSyntheticForm((p) => ({ ...p, num_users: Number(e.target.value) }))}
                style={inputStyle(c)}
              />
            </Field>
            <Field
              label={
                <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                  Рассчитывать метрики
                  <TooltipHint
                    c={c}
                    text="Если включено, после генерации дополнительно считаются проверочные метрики качества синтетических данных (сравнение распределений и базовые статистики). Это помогает быстро понять, насколько данные похожи на реальные."
                    placement="top"
                  />
                </span>
              }
              c={c}
            >
              <Toggle checked={syntheticForm.evaluation_metrics} onChange={(v) => setSyntheticForm((p) => ({ ...p, evaluation_metrics: v }))} c={c} />
            </Field>
            <Field label="Название набора" c={c}>
              <input
                type="text"
                value={syntheticForm.dataset_name}
                onChange={(e) => setSyntheticForm((p) => ({ ...p, dataset_name: e.target.value }))}
                placeholder="Например: iphone_spb_samara"
                style={inputStyle(c)}
              />
            </Field>
          </div>

          <SectionTitle c={c}>Фильтры генерации</SectionTitle>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 12 }}>
            {[
              { key: 'geo', label: 'География' },
              { key: 'profile', label: 'Профиль пользователей' },
              { key: 'tech', label: 'Устройства и браузеры' },
              { key: 'activity', label: 'Активность и каналы' },
              { key: 'ranges', label: 'Числовые параметры' },
            ].map((group) => {
              const active = activeFilterGroup === group.key;
              return (
                <button
                  key={group.key}
                  type="button"
                  onClick={() => setActiveFilterGroup(group.key as 'geo' | 'profile' | 'tech' | 'activity' | 'ranges')}
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
                  {group.label}
                </button>
              );
            })}
          </div>

          {activeFilterGroup !== 'ranges' && (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 12 }}>
              {MULTI_FILTER_FIELDS.filter((field) => {
                if (activeFilterGroup === 'geo') return ['cities'].includes(String(field.key));
                if (activeFilterGroup === 'profile') return ['user_types', 'genders'].includes(String(field.key));
                if (activeFilterGroup === 'tech') return ['devices', 'os', 'browsers'].includes(String(field.key));
                if (activeFilterGroup === 'activity') return ['traffic_sources'].includes(String(field.key));
                return false;
              }).map((field) => {
                const options = (filterOptions?.[field.key] || []) as string[];
                return (
                  <div key={String(field.key)} style={{ border: `1px solid ${c.border}`, borderRadius: 12, background: c.panelSoft, padding: '12px 14px' }}>
                    <div style={{ fontSize: 13, fontWeight: 700, color: c.textMuted, marginBottom: 10 }}>{field.label}</div>
                    {!options.length ? (
                      <div style={{ fontSize: 13, color: c.textSub }}>Нет значений</div>
                    ) : (
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 7, maxHeight: 124, overflowY: 'auto' }}>
                        {options.map((opt) => {
                          const selected = Array.isArray(filterDraft[field.key as string]) && (filterDraft[field.key as string] as string[]).includes(opt);
                          return (
                            <button
                              key={opt}
                              type="button"
                              onClick={() => toggleMultiFilterOption(field.key as string, opt)}
                              style={{
                                borderRadius: 999,
                                border: `1px solid ${selected ? c.accent : c.border}`,
                                background: selected ? c.accentSoft : c.panelBg,
                                color: selected ? c.accentText : c.textMuted,
                                padding: '5px 11px',
                                fontSize: 12,
                                fontWeight: selected ? 600 : 500,
                                cursor: 'pointer',
                              }}
                            >
                              {translateFilterValue(field.key, opt)}
                            </button>
                          );
                        })}
                      </div>
                    )}
                  </div>
                );
              })}

              {activeFilterGroup === 'activity' && (
                <>
                  <Field label="Подписка на email" c={c}>
                    <ChoiceSegment
                      c={c}
                      value={
                        filterDraft.email_subscribed === true
                          ? 'true'
                          : filterDraft.email_subscribed === false
                            ? 'false'
                            : ''
                      }
                      onChange={(next) => updateFilterValue('email_subscribed', next === '' ? undefined : next === 'true')}
                    />
                  </Field>
                  <Field label="Push уведомления" c={c}>
                    <ChoiceSegment
                      c={c}
                      value={
                        filterDraft.push_enabled === true
                          ? 'true'
                          : filterDraft.push_enabled === false
                            ? 'false'
                            : ''
                      }
                      onChange={(next) => updateFilterValue('push_enabled', next === '' ? undefined : next === 'true')}
                    />
                  </Field>
                  <Field label="Выходные" c={c}>
                    <ChoiceSegment
                      c={c}
                      value={
                        filterDraft.is_weekend === true
                          ? 'true'
                          : filterDraft.is_weekend === false
                            ? 'false'
                            : ''
                      }
                      onChange={(next) => updateFilterValue('is_weekend', next === '' ? undefined : next === 'true')}
                    />
                  </Field>
                </>
              )}
            </div>
          )}

          {activeFilterGroup === 'ranges' && (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 12 }}>
              {!!filterOptions?.numeric_ranges && Object.keys(filterOptions.numeric_ranges).length > 0 ? (
                Object.entries(filterOptions.numeric_ranges).map(([key, range]) => (
                  <div key={key} style={{ border: `1px solid ${c.border}`, borderRadius: 12, background: c.panelSoft, padding: '12px 14px' }}>
                    <div style={{ fontSize: 13, fontWeight: 700, color: c.textMuted, marginBottom: 10 }}>
                      {translateRangeLabel(key)}
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                      <input
                        type="number"
                        placeholder={`Мин (${range.min})`}
                        value={asNumber((filterDraft.numeric_ranges as Record<string, { min?: number }> | undefined)?.[key]?.min) ?? ''}
                        onChange={(e) => updateNumericRange(key, 'min', e.target.value === '' ? undefined : Number(e.target.value))}
                        style={inputStyle(c)}
                      />
                      <input
                        type="number"
                        placeholder={`Макс (${range.max})`}
                        value={asNumber((filterDraft.numeric_ranges as Record<string, { max?: number }> | undefined)?.[key]?.max) ?? ''}
                        onChange={(e) => updateNumericRange(key, 'max', e.target.value === '' ? undefined : Number(e.target.value))}
                        style={inputStyle(c)}
                      />
                    </div>
                  </div>
                ))
              ) : (
                <EmptyState c={c} text="Нет числовых диапазонов" />
              )}
            </div>
          )}

          <SectionTitle c={c}>История генераций</SectionTitle>
          <div style={{ border: `1px solid ${c.border}`, borderRadius: 12, overflow: 'hidden' }}>
            {historyLoading ? (
              <EmptyState c={c} text="Загрузка истории..." />
            ) : generatedHistory.filter((item) => item.data_type === 'synthetic').length === 0 ? (
              <EmptyState c={c} text="Пока нет синтетических датасетов" />
            ) : (
              generatedHistory
                .filter((item) => item.data_type === 'synthetic')
                .map((item, idx, arr) => (
                  <div
                    key={item.id}
                    style={{
                      padding: '12px 14px',
                      borderBottom: idx < arr.length - 1 ? `1px solid ${c.border}` : 'none',
                      display: 'flex',
                      gap: 12,
                      justifyContent: 'space-between',
                      flexWrap: 'wrap',
                      background: c.panelBg,
                    }}
                  >
                    <div>
                      <div style={{ fontWeight: 600, fontSize: 14 }}>{item.dataset_name || 'Без имени'}</div>
                      <div style={{ marginTop: 2, fontSize: 12, color: c.textMuted }}>
                        Кол-во: {item.sample_count} · Создано: {formatDate(item.created_at)}
                      </div>
                    </div>
                    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                      <RowActionButton
                        onClick={() => setDataPreviewModal({ visible: true, dataset: item as unknown as Record<string, unknown> })}
                        c={c}
                        label="Просмотр"
                        tone="preview"
                      />
                      <RowActionButton
                        onClick={() => downloadDatasetAsCSV(item as unknown as Record<string, unknown>)}
                        c={c}
                        icon={<Download size={14} />}
                        label="Скачать CSV"
                        tone="download"
                      />
                      <RowActionButton onClick={() => setConfirmDeleteHistoryItem(item)} c={c} icon={<Trash2 size={14} />} label="Удалить" tone="danger" />
                    </div>
                  </div>
                ))
            )}
          </div>
        </Panel>
      )}

      {dataPreviewModal.visible && (
        <Overlay onClose={() => setDataPreviewModal({ visible: false })}>
          <ModalPanel width={1000} c={c}>
            <ModalHeader
              icon={<Sparkles size={16} />}
              title={(dataPreviewModal.dataset?.dataset_name as string) || (dataPreviewModal.dataset?.extra_metadata as { dataset_name?: string } | undefined)?.dataset_name || 'Превью синтетических данных'}
              onClose={() => setDataPreviewModal({ visible: false })}
              c={c}
              showClose={false}
            />
            <div style={{ padding: '16px 20px', overflowY: 'auto', maxHeight: 'calc(90vh - 140px)' }}>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 10 }}>
                <Badge c={c} color="blue">
                  Всего записей: {Number(dataPreviewModal.dataset?.synthetic_samples || getDatasetRecords(dataPreviewModal.dataset).length || dataPreviewModal.dataset?.sample_count || 0)}
                </Badge>
                <Badge c={c} color="green">
                  Предпросмотр: первые 10 записей
                </Badge>
              </div>

              <SimpleDataTable rows={getDatasetRecords(dataPreviewModal.dataset).slice(0, 10)} c={c} />
            </div>
            <div style={{ borderTop: `1px solid ${c.border}`, padding: '12px 20px', display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
              <ActionButton onClick={handleDownloadCSV} isDark={isDark} c={c} icon={<Download size={14} />} label="Скачать CSV" variant="primary" />
              <ActionButton onClick={() => setDataPreviewModal({ visible: false })} isDark={isDark} c={c} label="Закрыть" variant="ghost" />
            </div>
          </ModalPanel>
        </Overlay>
      )}

      {ganTemplateModal && (
        <Overlay onClose={() => setGanTemplateModal(false)}>
          <ModalPanel width={860} c={c}>
            <ModalHeader icon={<FileText size={16} />} title="Выберите шаблон GAN конфигурации" onClose={() => setGanTemplateModal(false)} c={c} />
            <div style={{ padding: '16px 20px', overflowY: 'auto' }}>
              <p style={{ margin: '0 0 12px', fontSize: 13, color: c.textMuted }}>
                Шаблон заполнит форму обучения GAN. После применения все поля можно изменить.
              </p>
              <TemplateList c={c} items={ganTemplates} loading={loadingTemplates} onApply={applyGANTemplate} />
            </div>
          </ModalPanel>
        </Overlay>
      )}

      {syntheticTemplateModal && (
        <Overlay onClose={() => setSyntheticTemplateModal(false)}>
          <ModalPanel width={860} c={c}>
            <ModalHeader icon={<FileText size={16} />} title="Выберите шаблон генерации данных" onClose={() => setSyntheticTemplateModal(false)} c={c} />
            <div style={{ padding: '16px 20px', overflowY: 'auto' }}>
              <p style={{ margin: '0 0 12px', fontSize: 13, color: c.textMuted }}>
                Шаблон заполнит форму генерации синтетических данных вместе с фильтрами.
              </p>
              <TemplateList c={c} items={syntheticTemplates} loading={loadingTemplates} onApply={applySyntheticTemplate} />
            </div>
          </ModalPanel>
        </Overlay>
      )}

      {confirmDeleteCheckpoint && (
        <Overlay onClose={() => setConfirmDeleteCheckpoint(null)}>
          <ModalPanel width={420} c={c}>
            <ModalHeader icon={<Trash2 size={16} />} title="Удалить чекпоинт?" onClose={() => setConfirmDeleteCheckpoint(null)} c={c} danger showClose={false} />
            <div style={{ padding: '16px 20px', color: c.textMuted, fontSize: 14 }}>
              Чекпоинт «<span style={{ color: c.textPrimary, fontWeight: 600 }}>{confirmDeleteCheckpoint.name || confirmDeleteCheckpoint.filename || 'Без имени'}</span>» будет удалён безвозвратно.
            </div>
            <div style={{ borderTop: `1px solid ${c.border}`, padding: '12px 20px', display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
              <ActionButton onClick={() => setConfirmDeleteCheckpoint(null)} isDark={isDark} c={c} label="Отмена" variant="ghost" />
              <DangerActionButton onClick={handleDeleteCheckpoint} c={c} icon={<Trash2 size={14} />} label="Удалить" />
            </div>
          </ModalPanel>
        </Overlay>
      )}

      {confirmDeleteHistoryItem && (
        <Overlay onClose={() => setConfirmDeleteHistoryItem(null)}>
          <ModalPanel width={420} c={c}>
            <ModalHeader icon={<Trash2 size={16} />} title="Удалить запись о CSV?" onClose={() => setConfirmDeleteHistoryItem(null)} c={c} danger showClose={false} />
            <div style={{ padding: '16px 20px', color: c.textMuted, fontSize: 14 }}>
              Запись «<span style={{ color: c.textPrimary, fontWeight: 600 }}>{confirmDeleteHistoryItem.dataset_name || `#${confirmDeleteHistoryItem.id}`}</span>» будет удалена.
            </div>
            <div style={{ borderTop: `1px solid ${c.border}`, padding: '12px 20px', display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
              <ActionButton onClick={() => setConfirmDeleteHistoryItem(null)} isDark={isDark} c={c} label="Отмена" variant="ghost" />
              <DangerActionButton onClick={handleDeleteDatasetHistoryItem} c={c} icon={<Trash2 size={14} />} label="Удалить" />
            </div>
          </ModalPanel>
        </Overlay>
      )}

      {toast && <Toast message={toast.message} kind={toast.kind} leaving={toast.leaving} onClose={() => setToast(null)} />}

      <style>{`
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes fadeOutDown { from { opacity: 1; transform: translateY(0); } to { opacity: 0; transform: translateY(16px); } }
        @keyframes modalIn { from { opacity: 0; transform: scale(0.96) translateY(8px); } to { opacity: 1; transform: scale(1) translateY(0); } }
      `}</style>
    </div>
  );
};

const inputStyle = (c: Record<string, string>): React.CSSProperties => ({
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
});

const EmptyState: React.FC<{ c: Record<string, string>; text: string }> = ({ c, text }) => (
  <div style={{ padding: '34px 16px', textAlign: 'center', color: c.textMuted, fontSize: 13 }}>{text}</div>
);

const SectionTitle: React.FC<{ c: Record<string, string>; children: React.ReactNode }> = ({ c, children }) => (
  <div style={{ margin: '16px 0 10px', fontSize: 13, fontWeight: 700, color: c.textMuted, letterSpacing: '0.3px', textTransform: 'uppercase' }}>{children}</div>
);

const Field: React.FC<{ label: React.ReactNode; c: Record<string, string>; compact?: boolean; children: React.ReactNode }> = ({ label, c, compact, children }) => (
  <div>
    <label style={{ display: 'block', marginBottom: 6, fontSize: compact ? 12 : 13, fontWeight: 600, color: c.textMuted }}>{label}</label>
    {children}
  </div>
);

const Toggle: React.FC<{ checked: boolean; onChange: (v: boolean) => void; c: Record<string, string> }> = ({ checked, onChange, c }) => (
  <button
    type="button"
    onClick={() => onChange(!checked)}
    style={{
      width: 56,
      height: 30,
      borderRadius: 999,
      border: `1px solid ${checked ? c.accent : c.border}`,
      backgroundColor: checked ? c.accentSoft : c.panelBg,
      padding: 2,
      cursor: 'pointer',
      position: 'relative',
      boxSizing: 'border-box',
      transition: 'all 0.25s ease',
    }}
  >
    <span
      style={{
        width: 24,
        height: 24,
        borderRadius: '50%',
        backgroundColor: checked ? c.accent : c.textSub,
        position: 'absolute',
        top: 2,
        left: checked ? 30 : 2,
        transition: 'left 0.25s ease, background-color 0.25s ease',
      }}
    />
  </button>
);

const ChoiceSegment: React.FC<{
  c: Record<string, string>;
  value: '' | 'true' | 'false';
  onChange: (value: '' | 'true' | 'false') => void;
}> = ({ c, value, onChange }) => {
  const options: Array<{ value: '' | 'true' | 'false'; label: string }> = [
    { value: '', label: 'Любое' },
    { value: 'true', label: 'Да' },
    { value: 'false', label: 'Нет' },
  ];

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(3, minmax(0, 1fr))',
        gap: 6,
        padding: 4,
        borderRadius: 10,
        border: `1px solid ${c.inputBorder}`,
        backgroundColor: c.inputBg,
      }}
    >
      {options.map((opt) => {
        const active = value === opt.value;
        return (
          <button
            key={opt.label}
            type="button"
            onClick={() => onChange(opt.value)}
            style={{
              height: 32,
              borderRadius: 8,
              border: `1px solid ${active ? c.accent : 'transparent'}`,
              backgroundColor: active ? c.accentSoft : 'transparent',
              color: active ? c.accentText : c.textMuted,
              fontSize: 12,
              fontWeight: active ? 700 : 600,
              cursor: 'pointer',
              transition: 'all 0.12s',
            }}
          >
            {opt.label}
          </button>
        );
      })}
    </div>
  );
};

const StatCard: React.FC<{
  title: string;
  value: string;
  c: Record<string, string>;
  tone?: { bg: string; text: string; border: string };
}> = ({ title, value, c, tone }) => (
  <div style={{ borderRadius: 12, border: `1px solid ${tone?.border ?? c.border}`, backgroundColor: tone?.bg ?? c.panelBg, boxShadow: c.shadow, padding: '14px 16px' }}>
    <div style={{ fontSize: 11, color: c.textSub, textTransform: 'uppercase', letterSpacing: '0.35px' }}>{title}</div>
    <div style={{ marginTop: 8, color: tone?.text ?? c.textPrimary, fontSize: 15, fontWeight: 700, lineHeight: 1.4 }}>{value}</div>
  </div>
);

const Badge: React.FC<{ c: Record<string, string>; color: 'green' | 'blue'; children: React.ReactNode }> = ({ c, color, children }) => (
  <span
    style={{
      padding: '4px 10px',
      borderRadius: 999,
      border: `1px solid ${color === 'green' ? (c.successSoft.includes('rgba') ? 'rgba(34,197,94,0.35)' : '#bbf7d0') : (c.infoSoft.includes('rgba') ? 'rgba(59,130,246,0.35)' : '#bfdbfe')}`,
      backgroundColor: color === 'green' ? c.successSoft : c.infoSoft,
      color: color === 'green' ? c.success : c.info,
      fontSize: 12,
      fontWeight: 600,
    }}
  >
    {children}
  </span>
);

const TooltipHint: React.FC<{ c: Record<string, string>; text: string; placement?: 'top' | 'right' }> = ({ c, text, placement = 'top' }) => {
  const [open, setOpen] = useState(false);

  return (
    <span
      style={{ position: 'relative', display: 'inline-flex', alignItems: 'center' }}
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
    >
      <span style={{ display: 'inline-flex', alignItems: 'center', color: c.textSub, cursor: 'help' }}>
        <CircleHelp size={13} />
      </span>
      {open && (
        <span
          style={{
            position: 'absolute',
            ...(placement === 'right'
              ? { left: 'calc(100% + 8px)', top: '50%', transform: 'translateY(-50%)' }
              : { bottom: 'calc(100% + 8px)', right: 0 }),
            zIndex: 20,
            width: 'min(360px, calc(100vw - 44px))',
            maxWidth: 'calc(100vw - 44px)',
            padding: '8px 10px',
            borderRadius: 8,
            border: `1px solid ${c.border}`,
            backgroundColor: c.panelBg,
            color: c.textPrimary,
            fontSize: 12,
            lineHeight: 1.45,
            boxShadow: '0 8px 24px rgba(0,0,0,0.2)',
            whiteSpace: 'normal',
          }}
        >
          {text}
        </span>
      )}
    </span>
  );
};

const SimpleDataTable: React.FC<{ rows: Array<Record<string, unknown>>; c: Record<string, string> }> = ({ rows, c }) => {
  if (!rows.length) {
    return <EmptyState c={c} text="Нет данных" />;
  }

  const headers = Object.keys(rows[0]);

  return (
    <div style={{ overflowX: 'auto', border: `1px solid ${c.border}`, borderRadius: 10 }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
        <thead>
          <tr style={{ backgroundColor: c.panelSoft, borderBottom: `1px solid ${c.border}` }}>
            {headers.map((h) => (
              <th key={h} style={{ padding: '9px 12px', textAlign: 'left', whiteSpace: 'nowrap', fontSize: 11, color: c.textSub, textTransform: 'uppercase', letterSpacing: '0.35px' }}>
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, idx) => (
            <tr key={idx} style={{ borderBottom: idx < rows.length - 1 ? `1px solid ${c.border}` : 'none' }}>
              {headers.map((h) => (
                <td key={`${idx}-${h}`} style={{ padding: '9px 12px', color: c.textPrimary, whiteSpace: 'nowrap' }}>
                  {row[h] === null || row[h] === undefined ? '—' : String(row[h])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

const TemplateList: React.FC<{
  c: Record<string, string>;
  items: TemplateItem[];
  loading: boolean;
  onApply: (item: TemplateItem) => void;
}> = ({ c, items, loading, onApply }) => {
  if (loading) {
    return <EmptyState c={c} text="Загрузка шаблонов..." />;
  }
  if (!items.length) {
    return <EmptyState c={c} text="Нет доступных шаблонов" />;
  }

  return (
    <div style={{ border: `1px solid ${c.border}`, borderRadius: 12, overflow: 'hidden' }}>
      {items.map((item, idx) => (
        <div
          key={item.id}
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            gap: 12,
            padding: '12px 14px',
            borderBottom: idx < items.length - 1 ? `1px solid ${c.border}` : 'none',
            flexWrap: 'wrap',
            backgroundColor: c.panelBg,
          }}
        >
          <div style={{ flex: '1 1 280px' }}>
            <div style={{ fontWeight: 700, fontSize: 14 }}>{item.name}</div>
            {item.description && (
              <div style={{ marginTop: 3, color: c.textMuted, fontSize: 12 }}>{item.description.length > 120 ? `${item.description.slice(0, 120)}...` : item.description}</div>
            )}
            {!!item.tags?.length && (
              <div style={{ marginTop: 6, display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                {item.tags.map((tag) => (
                  <span key={tag} style={{ border: `1px solid ${c.border}`, borderRadius: 999, padding: '2px 8px', fontSize: 11, color: c.textMuted, background: c.panelSoft }}>
                    {tag}
                  </span>
                ))}
              </div>
            )}
          </div>
          <div>
            <button
              onClick={() => onApply(item)}
              style={{
                height: 36,
                padding: '0 14px',
                borderRadius: 8,
                border: '1px solid #d97706',
                background: '#d97706',
                color: '#fff',
                fontSize: 13,
                fontWeight: 600,
                cursor: 'pointer',
              }}
            >
              Применить
            </button>
          </div>
        </div>
      ))}
    </div>
  );
};

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
      padding: '6px 14px',
      borderRadius: 999,
      border: `1px solid ${active ? c.accent : c.border}`,
      backgroundColor: active ? c.accentSoft : c.panelBg,
      color: active ? c.accentText : c.textMuted,
      fontSize: 13,
      fontWeight: active ? 600 : 500,
      cursor: 'pointer',
    }}
  >
    {label}
  </button>
);

interface ToastProps {
  message: string;
  kind: 'success' | 'error';
  onClose: () => void;
}
const Toast: React.FC<ToastProps & { leaving?: boolean }> = ({ message, kind, onClose, leaving }) => (
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

const Overlay: React.FC<{ onClose: () => void; children: React.ReactNode }> = ({ onClose, children }) => (
  <div
    onClick={(e) => {
      if (e.target === e.currentTarget) onClose();
    }}
    style={{
      position: 'fixed',
      inset: 0,
      zIndex: 1000,
      backgroundColor: 'rgba(0,0,0,0.46)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: 24,
    }}
  >
    {children}
  </div>
);

const ModalPanel: React.FC<{ width: number; c: Record<string, string>; children: React.ReactNode }> = ({ width, c, children }) => (
  <div
    style={{
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
    }}
  >
    {children}
  </div>
);

const ModalHeader: React.FC<{
  icon?: React.ReactNode;
  title: string;
  onClose: () => void;
  c: Record<string, string>;
  danger?: boolean;
  showClose?: boolean;
}> = ({ icon, title, onClose, c, danger, showClose = true }) => (
  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '14px 20px', borderBottom: `1px solid ${c.border}`, backgroundColor: c.panelSoft }}>
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <span style={{ color: danger ? c.danger : c.accentText, display: 'inline-flex', alignItems: 'center' }}>{icon}</span>
      <span style={{ fontSize: 16, fontWeight: 700, color: c.textPrimary }}>{title}</span>
    </div>
    {showClose && (
      <button onClick={onClose} style={{ width: 30, height: 30, borderRadius: 7, border: 'none', background: 'transparent', cursor: 'pointer', color: c.textMuted }}>
        <X size={16} />
      </button>
    )}
  </div>
);

const DangerActionButton: React.FC<{ onClick: () => void; c: Record<string, string>; icon?: React.ReactNode; label: string }> = ({ onClick, c, icon, label }) => {
  const [hov, setHov] = useState(false);

  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        height: 38,
        padding: '0 18px',
        borderRadius: 8,
        border: `1px solid ${c.dangerBorder}`,
        backgroundColor: hov ? (c.dangerSoft.includes('rgba') ? 'rgba(239,68,68,0.25)' : '#fee2e2') : c.dangerSoft,
        color: c.danger,
        fontSize: 14,
        fontWeight: 600,
        cursor: 'pointer',
        display: 'flex',
        alignItems: 'center',
        gap: 6,
        transition: 'background-color 0.15s',
      }}
    >
      {icon}
      {label}
    </button>
  );
};

interface RowActionButtonProps {
  onClick: () => void;
  disabled?: boolean;
  c: Record<string, string>;
  icon?: React.ReactNode;
  label: string;
  tone: 'danger' | 'load' | 'preview' | 'download';
}

const RowActionButton: React.FC<RowActionButtonProps> = ({ onClick, disabled, c, icon, label, tone }) => {
  const [hov, setHov] = useState(false);

  const palette = {
    danger: {
      baseBg: c.dangerSoft,
      baseBorder: c.dangerBorder,
      baseColor: c.danger,
      hovBg: c.dangerSoft.includes('rgba') ? 'rgba(239,68,68,0.22)' : '#fee2e2',
      hovBorder: c.danger,
      hovColor: c.danger,
    },
    load: {
      baseBg: 'transparent',
      baseBorder: c.border,
      baseColor: c.textMuted,
      hovBg: c.successSoft,
      hovBorder: c.success,
      hovColor: c.success,
    },
    preview: {
      baseBg: 'transparent',
      baseBorder: c.border,
      baseColor: c.textMuted,
      hovBg: c.infoSoft,
      hovBorder: c.info,
      hovColor: c.info,
    },
    download: {
      baseBg: 'transparent',
      baseBorder: c.border,
      baseColor: c.textMuted,
      hovBg: 'rgba(139,92,246,0.14)',
      hovBorder: '#8b5cf6',
      hovColor: '#8b5cf6',
    },
  }[tone];

  return (
    <button
      onClick={disabled ? undefined : onClick}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        height: 34,
        padding: '0 12px',
        borderRadius: 8,
        border: `1px solid ${hov ? palette.hovBorder : palette.baseBorder}`,
        backgroundColor: hov ? palette.hovBg : palette.baseBg,
        color: hov ? palette.hovColor : palette.baseColor,
        fontSize: 13,
        fontWeight: 600,
        cursor: disabled ? 'not-allowed' : 'pointer',
        display: 'flex',
        alignItems: 'center',
        gap: 6,
        transition: 'all 0.12s',
        whiteSpace: 'nowrap',
        opacity: disabled ? 0.6 : 1,
      }}
    >
      {icon}
      {label}
    </button>
  );
};

interface ActionButtonProps {
  onClick: () => void;
  disabled?: boolean;
  isDark: boolean;
  c: Record<string, string>;
  icon?: React.ReactNode;
  label?: string;
  variant: 'primary' | 'secondary' | 'ghost' | 'template';
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
      : variant === 'secondary'
        ? hov
          ? isDark
            ? '#292524'
            : '#f0ede8'
          : c.panelBg
        : variant === 'template'
          ? hov
            ? (isDark ? 'rgba(180,83,9,0.16)' : '#f8f1e7')
            : c.panelBg
          : hov
            ? isDark
              ? '#292524'
              : '#f0ede8'
            : 'transparent';

  const border =
    variant === 'primary'
      ? disabled
        ? isDark
          ? '#292524'
          : '#e7e5e4'
        : hov
          ? c.accentHov
          : c.accent
      : variant === 'secondary'
        ? c.border
        : variant === 'template'
          ? (hov ? (isDark ? 'rgba(180,83,9,0.42)' : '#e9d5bd') : c.border)
          : 'transparent';

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
