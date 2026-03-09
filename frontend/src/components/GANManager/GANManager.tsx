import React, { useState, useEffect, useMemo } from 'react';
import {
  Card,
  Button,
  Progress,
  Statistic,
  Row,
  Col,
  Descriptions,
  message,
  List,
  Tag,
  Modal,
  Form,
  InputNumber,
  Input,
  Select,
  Divider,
  Tooltip,
  Space,
  Tabs,
  Table,
  Switch,
  Popconfirm,
} from 'antd';
import { PlusOutlined, ReloadOutlined, DownloadOutlined, StopOutlined, QuestionCircleOutlined, DeleteOutlined, PlayCircleOutlined } from '@ant-design/icons';
import { dataAPI } from '../../utils/api';
import type { GANTrainingPayload, SyntheticGenerationPayload } from '../../types';

interface GANTuningForm extends GANTrainingPayload {
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
interface SyntheticFormValues extends SyntheticGenerationPayload {}

const GAN_CONFIG_FIELDS: Array<{ name: keyof GANTuningForm; label: string; tooltip?: string; type: 'number' | 'text' | 'boolean'; min?: number; max?: number; step?: number }> = [
  {
    name: 'LATENT_DIM',
    label: 'Размер латентного вектора',
    tooltip: 'Размерность случайного шума (Z), из которого генератор создает данные. Больше = больше вариативности, но медленнее обучение. Рекомендуется: 64-256',
    type: 'number',
    min: 32,
    max: 512
  },
  {
    name: 'BATCH_SIZE',
    label: 'Размер батча',
    tooltip: 'Количество примеров, обрабатываемых за один шаг обучения. Больше = стабильнее градиенты, но больше памяти. Рекомендуется: 128-512',
    type: 'number',
    min: 64,
    max: 4096,
    step: 64
  },
  {
    name: 'LEARNING_RATE',
    label: 'Learning rate',
    tooltip: 'Скорость обучения модели. Меньше = медленнее, но стабильнее. Больше = быстрее, но может не сойтись. Рекомендуется: 0.0001-0.0003',
    type: 'number',
    min: 0.00001,
    max: 0.01,
    step: 0.00001
  },
  {
    name: 'DROPOUT_RATE',
    label: 'Dropout',
    tooltip: 'Процент нейронов, отключаемых при обучении для предотвращения переобучения. Рекомендуется: 0.2-0.5',
    type: 'number',
    min: 0,
    max: 0.8,
    step: 0.05
  },
  {
    name: 'LAMBDA_GP',
    label: 'Lambda GP',
    tooltip: 'Вес штрафа за нарушение условия Липшица (gradient penalty) в WGAN-GP. Контролирует стабильность обучения. Рекомендуется: 10',
    type: 'number',
    min: 1,
    max: 50,
    step: 1
  },
  {
    name: 'N_CRITIC',
    label: 'N Critic',
    tooltip: 'Сколько раз обновляется дискриминатор на одно обновление генератора. Больше = дискриминатор сильнее. Рекомендуется: 3-5',
    type: 'number',
    min: 1,
    max: 10,
    step: 1
  },
  {
    name: 'GENERATOR_LAYERS',
    label: 'Слои генератора (через запятую)',
    tooltip: 'Архитектура нейросети генератора. Например: 256,512,256 означает 3 слоя с указанным количеством нейронов',
    type: 'text'
  },
  {
    name: 'DISCRIMINATOR_LAYERS',
    label: 'Слои дискриминатора (через запятую)',
    tooltip: 'Архитектура нейросети дискриминатора. Обычно зеркальна генератору',
    type: 'text'
  },
  {
    name: 'USE_WGAN_GP',
    label: 'WGAN-GP режим',
    tooltip: 'Wasserstein GAN с Gradient Penalty - более стабильная версия GAN. Рекомендуется включить',
    type: 'boolean'
  },
];

export const GANManager: React.FC = () => {
  const [ganStatus, setGanStatus] = useState<any>({});
  const [checkpoints, setCheckpoints] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [training, setTraining] = useState(false);
  const [configForm] = Form.useForm<GANTuningForm>();
  const [syntheticForm] = Form.useForm<SyntheticFormValues>();
  const [filterOptions, setFilterOptions] = useState<any>(null);
  const [generatedHistory, setGeneratedHistory] = useState<any[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [dataPreviewModal, setDataPreviewModal] = useState<{ visible: boolean; dataset?: any }>({ visible: false });
  const [filterDraft, setFilterDraft] = useState<Record<string, any>>({});

  useEffect(() => {
    configForm.setFieldsValue({
      epochs: 50,
      real_data_samples: 50000,
      save_checkpoint: true,
    });
    syntheticForm.setFieldsValue({
      num_users: 10000,
      evaluation_metrics: true,
    });
  }, [configForm, syntheticForm]);

  useEffect(() => {
    loadGANStatus();
    loadCheckpoints();
    loadFilterOptions();
    loadGeneratedHistory();
    const interval = setInterval(() => {
      loadGANStatus();
      loadGeneratedHistory(false);
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  const loadGANStatus = async () => {
    try {
      const response = await dataAPI.getGANStatus();
      setGanStatus(response.data || {});
    } catch (error) {
      console.error('Error loading GAN status:', error);
      setGanStatus({ status: 'error', is_trained: false });
    }
  };

  const loadCheckpoints = async () => {
    try {
      const response = await dataAPI.getGANCheckpoints();
      setCheckpoints(response.data?.checkpoints || []);
    } catch (error) {
      console.error('Error loading checkpoints:', error);
      setCheckpoints([]);
    }
  };

  const loadFilterOptions = async () => {
    try {
      const response = await dataAPI.getDatasetStats();
      setFilterOptions(response.data || {});
    } catch (error) {
      console.error('Error loading dataset stats:', error);
    }
  };

  const loadGeneratedHistory = async (withSpinner: boolean = true) => {
    try {
      if (withSpinner) setHistoryLoading(true);
      const response = await dataAPI.listGeneratedHistory(25);
      setGeneratedHistory(response.data?.items || []);
    } catch (error) {
      console.error('Error loading history:', error);
    } finally {
      if (withSpinner) setHistoryLoading(false);
    }
  };

  const parseLayers = (value?: string): number[] | undefined => {
    if (!value) return undefined;
    return value
      .split(',')
      .map((v) => Number(v.trim()))
      .filter((v) => !Number.isNaN(v) && v > 0);
  };

  const handleTrainGAN = async () => {
    try {
      const values = await configForm.validateFields();
      setTraining(true);

      const {
        epochs,
        real_data_samples,
        save_checkpoint,
        checkpoint_name,
        LATENT_DIM,
        BATCH_SIZE,
        LEARNING_RATE,
        DROPOUT_RATE,
        LAMBDA_GP,
        N_CRITIC,
        GENERATOR_LAYERS,
        DISCRIMINATOR_LAYERS,
        USE_WGAN_GP,
      } = values;

      const configOverrides: Record<string, string | number | boolean | number[]> = {};
      if (LATENT_DIM) configOverrides.LATENT_DIM = LATENT_DIM;
      if (BATCH_SIZE) configOverrides.BATCH_SIZE = BATCH_SIZE;
      if (LEARNING_RATE) configOverrides.LEARNING_RATE = LEARNING_RATE;
      if (DROPOUT_RATE !== undefined) configOverrides.DROPOUT_RATE = DROPOUT_RATE;
      if (LAMBDA_GP) configOverrides.LAMBDA_GP = LAMBDA_GP;
      if (N_CRITIC) configOverrides.N_CRITIC = N_CRITIC;
      const generatorLayers = parseLayers(GENERATOR_LAYERS);
      if (generatorLayers?.length) configOverrides.GENERATOR_LAYERS = generatorLayers;
      const discriminatorLayers = parseLayers(DISCRIMINATOR_LAYERS);
      if (discriminatorLayers?.length) configOverrides.DISCRIMINATOR_LAYERS = discriminatorLayers;
      if (USE_WGAN_GP !== undefined) configOverrides.USE_WGAN_GP = USE_WGAN_GP;

      const payload: GANTrainingPayload = {
        epochs,
        real_data_samples,
        save_checkpoint,
        checkpoint_name,
        gan_config: Object.keys(configOverrides).length ? configOverrides : undefined,
      };

      await dataAPI.trainGAN(payload);
      message.success('Обучение GAN запущено!');
      loadGANStatus();
    } catch (error: any) {
      if (error?.errorFields) {
        return;
      }
      message.error('Ошибка запуска обучения GAN: ' + (error.response?.data?.detail || error.message));
    } finally {
      setTraining(false);
    }
  };

  const handleStopTraining = async () => {
    try {
      await dataAPI.stopGANTraining();
      message.success('Запрос на остановку обучения отправлен');
      loadGANStatus();
    } catch (error: any) {
      message.error('Ошибка остановки обучения: ' + (error.response?.data?.detail || error.message));
    }
  };

  const handleResumeTraining = async () => {
    try {
      await dataAPI.resumeGANTraining();
      message.success('Обучение возобновлено');
      loadGANStatus();
    } catch (error: any) {
      message.error('Ошибка возобновления обучения: ' + (error.response?.data?.detail || error.message));
    }
  };

  const handleResetTraining = async () => {
    try {
      await dataAPI.resetGANTraining();
      message.success('Обучение GAN сброшено');
      loadGANStatus();
    } catch (error: any) {
      message.error('Ошибка сброса обучения: ' + (error.response?.data?.detail || error.message));
    }
  };

  const handleGenerateData = async () => {
    try {
      const values = await syntheticForm.validateFields();
      setLoading(true);
      const payload: SyntheticGenerationPayload = {
        ...values,
        filters: Object.keys(filterDraft).length ? filterDraft : undefined,
      };
      const response = await dataAPI.generateSynthetic(payload);
      message.success(`Сгенерировано ${response.data.synthetic_samples} синтетических пользователей!`);
      setDataPreviewModal({ visible: true, dataset: response.data });
      loadGeneratedHistory();
    } catch (error: any) {
      if (error?.errorFields) {
        return;
      }
      message.error('Ошибка генерации данных: ' + (error.response?.data?.detail || error.message));
    } finally {
      setLoading(false);
    }
  };

  const handleLoadCheckpoint = async (checkpointName: string) => {
    try {
      await dataAPI.loadGANCheckpoint(checkpointName);
      message.success(`Модель загружена из ${checkpointName}`);
      loadGANStatus();
    } catch (error: any) {
      console.error(' Load checkpoint error:', error);
      message.error('Ошибка загрузки модели: ' + (error.response?.data?.detail || error.message));
    }
  };

  const updateFilterValue = (key: string, value: any) => {
    setFilterDraft((prev) => {
      const next = { ...prev };
      if (value === undefined || (Array.isArray(value) && value.length === 0)) {
        delete next[key];
      } else {
        next[key] = value;
      }
      return next;
    });
  };

  const updateNumericRange = (field: string, bound: 'min' | 'max', value?: number) => {
    setFilterDraft((prev) => {
      const ranges = { ...(prev.numeric_ranges || {}) } as Record<string, Record<'min' | 'max', number>>;
      const existing = { ...(ranges[field] || {}) } as Record<'min' | 'max', number>;
      if (value === undefined || value === null) {
        delete existing[bound];
      } else {
        existing[bound] = value;
      }
      if (Object.keys(existing).length) {
        ranges[field] = existing;
      } else {
        delete ranges[field];
      }
      const cleanedEntries = Object.entries(ranges).filter(([, v]) => Object.keys(v).length);
      const cleaned = Object.fromEntries(cleanedEntries);
      return { ...prev, numeric_ranges: cleaned };
    });
  };

  const filterControls = useMemo(() => {
    if (!filterOptions) return null;
    return (
      <Space direction="vertical" style={{ width: '100%' }} size="middle">
        <Select
          mode="multiple"
          allowClear
          placeholder="Города"
          value={filterDraft.cities}
          onChange={(value) => updateFilterValue('cities', value)}
          options={(filterOptions.cities || []).map((city: string) => ({ label: city, value: city }))}
          style={{ width: '100%' }}
        />
        <Select
          mode="multiple"
          allowClear
          placeholder="Устройства"
          value={filterDraft.devices}
          onChange={(value) => updateFilterValue('devices', value)}
          options={(filterOptions.devices || []).map((device: string) => ({ label: device, value: device }))}
          style={{ width: '100%' }}
        />
        <Select
          mode="multiple"
          allowClear
          placeholder="ОС"
          value={filterDraft.os}
          onChange={(value) => updateFilterValue('os', value)}
          options={(filterOptions.os || []).map((os: string) => ({ label: os, value: os }))}
          style={{ width: '100%' }}
        />
        <Select
          mode="multiple"
          allowClear
          placeholder="Браузеры"
          value={filterDraft.browsers}
          onChange={(value) => updateFilterValue('browsers', value)}
          options={(filterOptions.browsers || []).map((browser: string) => ({ label: browser, value: browser }))}
          style={{ width: '100%' }}
        />
        <Select
          mode="multiple"
          allowClear
          placeholder="Типы пользователей"
          value={filterDraft.user_types}
          onChange={(value) => updateFilterValue('user_types', value)}
          options={(filterOptions.user_types || []).map((type: string) => ({ label: type, value: type }))}
          style={{ width: '100%' }}
        />
        <Select
          mode="multiple"
          allowClear
          placeholder="Источники трафика"
          value={filterDraft.traffic_sources}
          onChange={(value) => updateFilterValue('traffic_sources', value)}
          options={(filterOptions.traffic_sources || []).map((src: string) => ({ label: src, value: src }))}
          style={{ width: '100%' }}
        />
        <Select
          mode="multiple"
          allowClear
          placeholder="Пол"
          value={filterDraft.genders}
          onChange={(value) => updateFilterValue('genders', value)}
          options={(filterOptions.genders || []).map((gender: string) => ({ label: gender, value: gender }))}
          style={{ width: '100%' }}
        />
        <Select
          placeholder="Подписка на email"
          allowClear
          value={filterDraft.email_subscribed}
          onChange={(value) => updateFilterValue('email_subscribed', value)}
          options={[{ label: 'Да', value: true }, { label: 'Нет', value: false }]}
          style={{ width: '100%' }}
        />
        <Select
          placeholder="Push уведомления"
          allowClear
          value={filterDraft.push_enabled}
          onChange={(value) => updateFilterValue('push_enabled', value)}
          options={[{ label: 'Да', value: true }, { label: 'Нет', value: false }]}
          style={{ width: '100%' }}
        />
        <Select
          placeholder="Выходные"
          allowClear
          value={filterDraft.is_weekend}
          onChange={(value) => updateFilterValue('is_weekend', value)}
          options={[{ label: 'Да', value: true }, { label: 'Нет', value: false }]}
          style={{ width: '100%' }}
        />
        <Divider>Числовые диапазоны</Divider>
        {(filterOptions.numeric_ranges ? Object.entries(filterOptions.numeric_ranges) : []).map(([key, range]: any) => (
          <div key={key} style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <span style={{ width: 160, fontWeight: 500 }}>{key}</span>
            <InputNumber
              placeholder={`Мин (${range.min})`}
              style={{ flex: 1 }}
              value={filterDraft.numeric_ranges?.[key]?.min}
              onChange={(value) => updateNumericRange(key, 'min', value ?? undefined)}
            />
            <InputNumber
              placeholder={`Макс (${range.max})`}
              style={{ flex: 1 }}
              value={filterDraft.numeric_ranges?.[key]?.max}
              onChange={(value) => updateNumericRange(key, 'max', value ?? undefined)}
            />
          </div>
        ))}
      </Space>
    );
  }, [filterOptions, filterDraft]);

  const handleClearFilters = () => setFilterDraft({});

  const getDatasetRecords = (dataset: any) => {
    return dataset?.records || dataset?.synthetic_preview || dataset?.preview_json || [];
  };

  const downloadDatasetAsCSV = (dataset: any) => {
    if (!dataset) return;

    const records = getDatasetRecords(dataset);
    if (records.length === 0) {
      message.warning('Нет данных для скачивания');
      return;
    }

    const headers = Object.keys(records[0]);
    const csvContent = [
      headers.join(','),
      ...records.map((row: any) =>
        headers
          .map((header) => {
            const value = row[header];
            if (typeof value === 'string' && (value.includes(',') || value.includes('"'))) {
              return `"${value.replace(/"/g, '""')}"`;
            }
            return value;
          })
          .join(','),
      ),
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    const filename = dataset.dataset_name ? `${dataset.dataset_name}.csv` : `synthetic_data_${new Date().toISOString().slice(0, 10)}.csv`;

    link.setAttribute('href', url);
    link.setAttribute('download', filename);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    message.success(`Файл ${filename} скачан`);
  };

  const handleDownloadCSV = async () => {
    if (!dataPreviewModal.dataset?.id) return;
    
    try {
      // Загружаем полный датасет только при скачивании
      const response = await dataAPI.getFullDataset(dataPreviewModal.dataset.id);
      downloadDatasetAsCSV(response.data);
    } catch (error: any) {
      message.error('Ошибка загрузки данных: ' + (error.response?.data?.detail || error.message));
    }
  };

  const handleDeleteDatasetHistoryItem = async (id: number) => {
    try {
      await dataAPI.deleteGeneratedHistoryItem(id);
      message.success('Запись о CSV удалена');
      await loadGeneratedHistory();
      if (dataPreviewModal.dataset?.id === id) {
        setDataPreviewModal({ visible: false });
      }
    } catch (error: any) {
      message.error('Ошибка удаления записи: ' + (error.response?.data?.detail || error.message));
    }
  };

  const handleDeleteCheckpoint = async (id: number) => {
    try {
      await dataAPI.deleteGANCheckpoint(id);
      message.success('Чекпоинт удален');
      await loadCheckpoints();
      await loadGANStatus();
    } catch (error: any) {
      message.error('Ошибка удаления чекпоинта: ' + (error.response?.data?.detail || error.message));
    }
  };

  const historyColumns = (onPreview?: (record: any) => void) => [
    {
      title: 'Название набора',
      dataIndex: 'dataset_name',
      render: (value: string | undefined) => value || 'Без имени',
    },
    {
      title: 'Кол-во',
      dataIndex: 'sample_count',
    },
    {
      title: 'Создано',
      dataIndex: 'created_at',
      render: (value: string) => (value ? new Date(value).toLocaleString() : '—'),
    },
    {
      title: 'Действия',
      key: 'actions',
      render: (_: any, record: any) => (
        <Space>
          <Button type="link" onClick={() => onPreview?.(record)}>
            Просмотр
          </Button>
          <Button type="link" icon={<DownloadOutlined />} onClick={() => downloadDatasetAsCSV(record)}>
            Скачать CSV
          </Button>
          <Popconfirm title="Удалить запись о CSV?" onConfirm={() => handleDeleteDatasetHistoryItem(record.id)}>
            <Button type="link" danger icon={<DeleteOutlined />}>
              Удалить
            </Button>
          </Popconfirm>
        </Space>
      ),
    },
  ];

  const isTraining = ganStatus.status === 'training';
  const isStopped = ganStatus.status === 'training_paused';
  const isTrainingOrStopped = isTraining || isStopped;

  const getFriendlyStatus = () => {
    // Статус 1.1: Чекпоинт не загружен
    if (ganStatus.status === 'checkpoint_not_loaded' || !ganStatus.status) {
      return 'Чекпоинт не загружен';
    }
    
    // Статус 1.2: Загружен чекпоинт: Имя чекпоинта
    if (ganStatus.status === 'checkpoint_loaded' && ganStatus.loaded_checkpoint_name) {
      return `Загружен чекпоинт: ${ganStatus.loaded_checkpoint_name}`;
    }
    
    // Статус 1.3: Обучение: 0/N эпох
    if (ganStatus.status === 'training') {
      return `Обучение: ${ganStatus.current_epoch}/${ganStatus.total_epochs} эпох`;
    }
    
    // Статус 1.4: Пауза обучения: 0/N эпох
    if (ganStatus.status === 'training_paused') {
      return `Пауза обучения: ${ganStatus.current_epoch}/${ganStatus.total_epochs} эпох`;
    }
    
    // Обработка ошибок
    if (ganStatus.status?.includes('error')) {
      return `Ошибка: ${ganStatus.status.replace('error: ', '')}`;
    }
    
    // Fallback на случай неизвестного статуса
    return ganStatus.status;
  };

  return (
    <div style={{ padding: '20px' }}>
      <Row gutter={[16, 16]}>
        <Col span={8}>
          <Card>
            <Statistic
              title="Статус GAN"
              value={getFriendlyStatus()}
              valueStyle={{
                color: ganStatus.status?.includes('error')
                  ? 'red'
                  : (ganStatus.status === 'training' || ganStatus.status === 'training_paused')
                    ? '#1890ff'
                    : ganStatus.is_trained
                      ? 'green'
                      : 'gray'
              }}
            />
          </Card>
        </Col>
        <Col span={8}>
          <Card>
            <Statistic title="Чекпоинты" value={ganStatus.available_checkpoints || 0} />
          </Card>
        </Col>
        <Col span={8}>
          <Card>
            <Statistic
              title="Эпох обучения"
              value={
                ganStatus.is_trained && ganStatus.loss_history?.total_epochs
                  ? ganStatus.loss_history.total_epochs
                  : 'Загрузите модель для отображения'
              }
            />
          </Card>
        </Col>
      </Row>
      {ganStatus.config && Object.keys(ganStatus.config).length > 0 && (
        <Card title={!ganStatus.is_trained && !isTraining ? "Последняя конфигурация обучаемой модели" : "Текущая конфигурация модели"} style={{ marginTop: 20 }}>
          <Descriptions bordered column={3}>
            <Descriptions.Item label="Эпохи">{ganStatus.config.EPOCHS || 'N/A'}</Descriptions.Item>
            <Descriptions.Item label="Размер батча">{ganStatus.config.BATCH_SIZE || 'N/A'}</Descriptions.Item>
            <Descriptions.Item label="Learning Rate">{ganStatus.config.LEARNING_RATE || 'N/A'}</Descriptions.Item>
            <Descriptions.Item label="WGAN-GP режим">{ganStatus.config.USE_WGAN_GP ? 'Да' : 'Нет'}</Descriptions.Item>
            <Descriptions.Item label="Lambda GP">{ganStatus.config.LAMBDA_GP || 'N/A'}</Descriptions.Item>
            <Descriptions.Item label="N Critic">{ganStatus.config.N_CRITIC || 'N/A'}</Descriptions.Item>
            <Descriptions.Item label="Latent Dim">{ganStatus.config.LATENT_DIM || 'N/A'}</Descriptions.Item>
            <Descriptions.Item label="Dropout">{ganStatus.config.DROPOUT_RATE || 'N/A'}</Descriptions.Item>
            <Descriptions.Item label="Устройство">{ganStatus.config.DEVICE || 'N/A'}</Descriptions.Item>
            <Descriptions.Item label="G Loss" span={1}>
              {ganStatus.loss_history?.latest_g_loss?.toFixed(4) || 'N/A'}
            </Descriptions.Item>
            <Descriptions.Item label="D Loss" span={1}>
              {ganStatus.loss_history?.latest_d_loss?.toFixed(4) || 'N/A'}
            </Descriptions.Item>
            <Descriptions.Item label="Wasserstein" span={1}>
              {ganStatus.loss_history?.latest_wasserstein?.toFixed(4) || 'N/A'}
            </Descriptions.Item>
          </Descriptions>
        </Card>
      )}

      <Card style={{ marginTop: 20 }}>
        <Tabs
          defaultActiveKey="train"
          items={[
            {
              key: 'train',
              label: 'Конфигурация и обучение GAN',
              children: (
                <Form form={configForm} layout="vertical">
                  {isTrainingOrStopped && (
                    <div style={{ marginBottom: 24, padding: 16, background: '#f0f2f5', borderRadius: 8 }}>
                      <div style={{ marginBottom: 12 }}>
                        <Tag color={isStopped ? "warning" : "processing"} style={{ fontSize: 14, padding: '4px 12px' }}>
                          {getFriendlyStatus()}
                        </Tag>
                      </div>
                      <Progress
                        percent={Math.min(100, ganStatus.training_progress || 0)}
                        status={isStopped ? "exception" : "active"}
                        strokeColor={{ '0%': '#108ee9', '100%': '#87d068' }}
                      />
                      <div style={{ marginTop: 8, color: '#666' }}>
                        Эпоха: {ganStatus.current_epoch}/{ganStatus.total_epochs}
                      </div>
                    </div>
                  )}
                  
                  <Space style={{ marginBottom: 16 }} wrap>
                    {!isTrainingOrStopped && (
                      <Button type="primary" onClick={handleTrainGAN} loading={training} icon={<PlusOutlined />}>
                        Обучить GAN с нуля
                      </Button>
                    )}
                    {isTraining && (
                      <Button danger onClick={handleStopTraining} icon={<StopOutlined />}>
                        Остановить обучение
                      </Button>
                    )}
                    {isStopped && (
                      <>
                        <Button
                          type="primary"
                          onClick={handleResumeTraining}
                          icon={<PlayCircleOutlined />}
                        >
                          Возобновить обучение
                        </Button>
                        <Popconfirm
                          title="Сбросить обучение GAN?"
                          description="Это полностью отменит текущее обучение. Чекпоинт не сохранится."
                          onConfirm={handleResetTraining}
                          okText="Да, сбросить"
                          cancelText="Отмена"
                        >
                          <Button danger icon={<DeleteOutlined />}>
                            Сбросить обучение GAN
                          </Button>
                        </Popconfirm>
                      </>
                    )}
                    {isTraining && (
                      <Button disabled icon={<PlayCircleOutlined />} style={{ color: '#d9d9d9' }}>
                        Возобновить обучение
                      </Button>
                    )}
                    {!isTrainingOrStopped && (
                      <Button onClick={() => configForm.resetFields()} icon={<ReloadOutlined />}>Сбросить форму</Button>
                    )}
                  </Space>

                  {!isTrainingOrStopped && (
                    <>
                      <Row gutter={16}>
                        <Col span={8}>
                          <Form.Item name="epochs" label="Эпох" rules={[{ required: true, message: 'Укажите количество эпох' }]}>
                            <InputNumber min={10} max={500} style={{ width: '100%' }} />
                          </Form.Item>
                        </Col>
                        <Col span={8}>
                          <Form.Item
                            name="real_data_samples"
                            label="Samples для обучения"
                            rules={[{ required: true, message: 'Укажите количество примеров' }]}
                          >
                            <InputNumber min={1000} max={100000} step={1000} style={{ width: '100%' }} />
                          </Form.Item>
                        </Col>
                        <Col span={8}>
                          <Form.Item name="save_checkpoint" label="Сохранять чекпоинт" valuePropName="checked">
                            <Switch defaultChecked />
                          </Form.Item>
                        </Col>
                      </Row>
                      <Row gutter={16}>
                        <Col span={12}>
                          <Form.Item name="checkpoint_name" label="Имя чекпоинта" rules={[{ required: true, message: 'Укажите имя чекпоинта' }]}>
                            <Input placeholder="Например, best_wgan_config" />
                          </Form.Item>
                        </Col>
                      </Row>

                      <Divider orientation="left">Переопределение конфигурации</Divider>
                      <Row gutter={16}>
                        {GAN_CONFIG_FIELDS.map((field) => (
                          <Col span={12} key={field.name} style={{ marginBottom: 16 }}>
                            <Form.Item
                              name={field.name}
                              label={
                                <span>
                                  {field.label}
                                  {field.tooltip && (
                                    <Tooltip title={field.tooltip}>
                                      <QuestionCircleOutlined style={{ marginLeft: 8, color: '#1890ff' }} />
                                    </Tooltip>
                                  )}
                                </span>
                              }
                              valuePropName={field.type === 'boolean' ? 'checked' : undefined}
                            >
                              {field.type === 'number' ? (
                                <InputNumber
                                  min={field.min}
                                  max={field.max}
                                  step={field.step}
                                  style={{ width: '100%' }}
                                  placeholder="По умолчанию"
                                />
                              ) : field.type === 'text' ? (
                                <Input placeholder="Например: 512,256,128" />
                              ) : (
                                <Switch />
                              )}
                            </Form.Item>
                          </Col>
                        ))}
                      </Row>
                    </>
                  )}

                  <Divider orientation="left">Доступные чекпоинты</Divider>
                  <List
                    dataSource={checkpoints}
                    locale={{ emptyText: 'Нет доступных чекпоинтов' }}
                    renderItem={(checkpoint: any) => (
                      <List.Item
                        actions={[
                          <Button type="link" onClick={() => handleLoadCheckpoint(checkpoint.name || checkpoint.filename)} disabled={isTrainingOrStopped}>
                            Загрузить
                          </Button>,
                          <Popconfirm title="Удалить чекпоинт?" onConfirm={() => handleDeleteCheckpoint(checkpoint.id)}>
                            <Button type="link" danger icon={<DeleteOutlined />}>
                              Удалить
                            </Button>
                          </Popconfirm>,
                        ]}
                      >
                        <List.Item.Meta
                          title={checkpoint.name || checkpoint.filename}
                          description={
                            <div>
                              <div>Размер: {checkpoint.metrics?.size ? `${(checkpoint.metrics.size / 1024 / 1024).toFixed(2)} MB` : 'N/A'}</div>
                              <div>Изменен: {checkpoint.created_at ? new Date(checkpoint.created_at).toLocaleString() : 'N/A'}</div>
                            </div>
                          }
                        />
                      </List.Item>
                    )}
                  />
                </Form>
              ),
            },
            {
              key: 'synthetic',
              label: 'Генерация синтетических данных',
              children: (
                <Form form={syntheticForm} layout="vertical">
                  <Row gutter={16}>
                    <Col span={6}>
                      <Form.Item
                        name="num_users"
                        label={
                          <span>
                            Кол-во пользователей
                            <Tooltip title="Функция генерирует синтетических пользователей на основе обученной GAN-модели и выбранных фильтров.">
                              <QuestionCircleOutlined style={{ marginLeft: 8, color: '#1890ff' }} />
                            </Tooltip>
                          </span>
                        }
                        rules={[{ required: true }]}
                      >
                        <InputNumber min={100} max={100000} step={100} style={{ width: '100%' }} />
                      </Form.Item>
                    </Col>
                    <Col span={6}>
                      <Form.Item name="evaluation_metrics" label="Рассчитывать метрики" valuePropName="checked">
                        <Switch defaultChecked />
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item name="dataset_name" label="Название набора">
                        <Input placeholder="Например: iphone_spb_samara" />
                      </Form.Item>
                    </Col>
                  </Row>
                  <Divider orientation="left">Фильтры генерации</Divider>
                  {filterControls}
                  {Object.keys(filterDraft).length > 0 && (
                    <div style={{ marginTop: 12, marginBottom: 16 }}>
                      <Tag color="green">Используются фильтры ({Object.keys(filterDraft).length})</Tag>
                      <Button type="link" onClick={handleClearFilters}>
                        Очистить фильтры
                      </Button>
                    </div>
                  )}
                  <Space>
                    <Button type="primary" onClick={handleGenerateData} loading={loading} disabled={!ganStatus.is_trained || isTrainingOrStopped} icon={<PlusOutlined />}>
                      Сгенерировать данные
                    </Button>
                    <Button onClick={() => syntheticForm.resetFields()} icon={<ReloadOutlined />}>Сбросить форму</Button>
                  </Space>

                  <Divider orientation="left">История генераций</Divider>
                  <Table
                    rowKey="id"
                    dataSource={generatedHistory.filter((item) => item.data_type === 'synthetic')}
                    columns={historyColumns((record) => setDataPreviewModal({ visible: true, dataset: record }))}
                    loading={historyLoading}
                    pagination={{ pageSize: 5 }}
                    locale={{ emptyText: 'Пока нет синтетических датасетов' }}
                  />
                </Form>
              ),
            },
          ]}
        />
      </Card>


      <Modal
        title={dataPreviewModal.dataset?.dataset_name || dataPreviewModal.dataset?.extra_metadata?.dataset_name || 'Превью синтетических данных'}
        open={dataPreviewModal.visible}
        onCancel={() => setDataPreviewModal({ visible: false })}
        footer={[
          <Button key="download" type="primary" icon={<DownloadOutlined />} onClick={handleDownloadCSV}>
            Скачать CSV
          </Button>,
          <Button key="close" onClick={() => setDataPreviewModal({ visible: false })}>
            Закрыть
          </Button>
        ]}
        width={900}
      >
        {dataPreviewModal.dataset ? (
          <>
            <div style={{ marginBottom: 16 }}>
              <Tag color="blue">Всего записей: {dataPreviewModal.dataset.synthetic_samples || getDatasetRecords(dataPreviewModal.dataset).length || dataPreviewModal.dataset.sample_count || 0}</Tag>
              <Tag color="green">Предпросмотр: первые 10 записей</Tag>
            </div>
            <Table
              dataSource={getDatasetRecords(dataPreviewModal.dataset).slice(0, 10)}
              columns={Object.keys(getDatasetRecords(dataPreviewModal.dataset)[0] || {}).map((feature: string) => ({
                title: feature,
                dataIndex: feature,
                ellipsis: true,
              }))}
              rowKey={(record: any) => record.user_id ?? record.id ?? Math.random()}
              pagination={{ pageSize: 10 }}
              size="small"
              scroll={{ x: 'max-content' }}
            />
          </>
        ) : (
          'Нет данных'
        )}
      </Modal>

    </div>
  );
};
