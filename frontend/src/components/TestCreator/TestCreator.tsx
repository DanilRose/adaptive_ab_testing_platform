import React, { useState, useEffect } from 'react';
import {
  Button,
  Form,
  Input,
  Select,
  InputNumber,
  Card,
  message,
  Modal,
  Table,
  Tag,
  Space,
  Typography,
  Tooltip,
  Alert,
} from 'antd';
import { FileTextOutlined, ThunderboltOutlined } from '@ant-design/icons';
import { abTestAPI, dataAPI, templatesAPI } from '../../utils/api';

const { Option } = Select;
const { Text } = Typography;

interface Template {
  id: number;
  name: string;
  description?: string;
  template_type: string;
  config_json: Record<string, any>;
  tags: string[];
}

export const TestCreator: React.FC = () => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [datasets, setDatasets] = useState<any[]>([]);
  const [templates, setTemplates] = useState<Template[]>([]);
  const [templateModal, setTemplateModal] = useState(false);
  const [loadingTemplates, setLoadingTemplates] = useState(false);
  const [appliedTemplate, setAppliedTemplate] = useState<string | null>(null);
  // Хранит variantEffects из применённого шаблона
  const [appliedVariantEffects, setAppliedVariantEffects] = useState<Record<string, any> | null>(null);

  useEffect(() => {
    const loadDatasets = async () => {
      try {
        const response = await dataAPI.listGeneratedHistory(100);
        const syntheticDatasets = response.data.items.filter((d: any) => d.data_type === 'synthetic');
        setDatasets(syntheticDatasets);
      } catch (e) {
        console.error('Ошибка загрузки датасетов:', e);
      }
    };
    loadDatasets();
  }, []);

  const loadTemplates = async () => {
    setLoadingTemplates(true);
    try {
      const response = await templatesAPI.listTemplates('ab_test');
      setTemplates(response.data.items || []);
    } catch (e) {
      message.error('Ошибка загрузки шаблонов');
    } finally {
      setLoadingTemplates(false);
    }
  };

  const openTemplateModal = () => {
    setTemplateModal(true);
    loadTemplates();
  };

  const applyTemplate = (template: Template) => {
    const cfg = template.config_json;
    form.setFieldsValue({
      testName: cfg.testName || '',
      variants: cfg.variants || 'A, B',
      primaryMetric: cfg.primaryMetric || '',
      metricType: cfg.metricType || 'continuous',
      description: cfg.description || '',
      confidenceLevel: cfg.confidenceLevel ?? 0.95,
      power: cfg.power ?? 0.8,
      minEffectSize: cfg.minEffectSize ?? 0.1,
      trafficSplitType: cfg.trafficSplitType || 'fixed',
      simulationDurationMinutes: cfg.simulationDurationMinutes ?? 20,
      sampleSize: cfg.sampleSize ?? undefined,
    });
    setAppliedVariantEffects(cfg.variantEffects || null);
    setAppliedTemplate(template.name);
    setTemplateModal(false);
    message.success(`Шаблон "${template.name}" применён`);
  };

  const templateColumns = [
    {
      title: 'Название',
      dataIndex: 'name',
      render: (name: string, record: Template) => (
        <div>
          <Text strong>{name}</Text>
          {record.description && (
            <div>
              <Text type="secondary" style={{ fontSize: 12 }}>
                {record.description.length > 100 ? record.description.slice(0, 100) + '...' : record.description}
              </Text>
            </div>
          )}
        </div>
      ),
    },
    {
      title: 'Теги',
      dataIndex: 'tags',
      width: 200,
      render: (tags: string[]) => (
        <Space size={4} wrap>
          {(tags || []).map((tag) => (
            <Tag key={tag} style={{ fontSize: 11 }}>{tag}</Tag>
          ))}
        </Space>
      ),
    },
    {
      title: '',
      key: 'action',
      width: 120,
      render: (_: any, record: Template) => (
        <Button type="primary" size="small" icon={<ThunderboltOutlined />} onClick={() => applyTemplate(record)}>
          Применить
        </Button>
      ),
    },
  ];

  const onFinish = async (values: any) => {
    setLoading(true);
    try {
      let variantsArray: string[];
      if (typeof values.variants === 'string') {
        variantsArray = values.variants.split(',').map((v: string) => v.trim());
      } else if (Array.isArray(values.variants)) {
        variantsArray = values.variants;
      } else {
        throw new Error('Варианты должны быть строкой или массивом');
      }

      if (variantsArray.length < 2) {
        message.error('Необходимо минимум 2 варианта для A/B теста');
        setLoading(false);
        return;
      }

      const testData = {
        test_name: values.testName,
        variants: variantsArray,
        primary_metric: values.primaryMetric,
        metric_type: values.metricType,
        description: values.description || '',
        sample_size: values.sampleSize || null,
        confidence_level: values.confidenceLevel || 0.95,
        power: values.power || 0.8,
        min_effect_size: values.minEffectSize || 0.1,
        dataset_id: values.datasetId || null,
        simulation_duration_minutes: values.simulationDurationMinutes || 20,
        traffic_split_type: values.trafficSplitType || 'fixed',
        variant_effects: appliedVariantEffects,
      };

      const response = await abTestAPI.createTest(testData);
      message.success(`✅ Тест успешно создан! ID: ${response.data.test_id}`);
      form.resetFields();
      setAppliedTemplate(null);
      setAppliedVariantEffects(null);

    } catch (error: any) {
      const errorMsg = error.response?.data?.detail || error.message || 'Неизвестная ошибка';
      message.error(`Ошибка при создании теста: ${errorMsg}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card
      title="Создать новый A/B тест"
      style={{ maxWidth: 800, margin: '20px auto' }}
      extra={
        <Button
          icon={<FileTextOutlined />}
          onClick={openTemplateModal}
          type="dashed"
        >
          Выбрать из шаблонов
        </Button>
      }
    >
      {appliedTemplate && (
        <Alert
          type="success"
          showIcon
          icon={<FileTextOutlined />}
          message={`Применён шаблон: "${appliedTemplate}"`}
          description="Поля заполнены из шаблона. Вы можете изменить любое значение перед созданием."
          style={{ marginBottom: 16 }}
          closable
          onClose={() => setAppliedTemplate(null)}
        />
      )}

      <Form
        form={form}
        layout="vertical"
        onFinish={onFinish}
        initialValues={{
          confidenceLevel: 0.95,
          power: 0.8,
          minEffectSize: 0.1,
          metricType: 'continuous',
          variants: 'A, B',
          trafficSplitType: 'fixed',
          simulationDurationMinutes: 20,
        }}
      >
        <Form.Item
          name="testName"
          label="Название теста"
          rules={[{ required: true, message: 'Введите название теста' }]}
          tooltip="Понятное имя для идентификации теста"
        >
          <Input placeholder="Например: Тест цвета кнопки CTA" />
        </Form.Item>

        <Form.Item
          name="variants"
          label="Варианты (через запятую)"
          rules={[{ required: true, message: 'Введите варианты теста' }]}
          tooltip="Укажите названия вариантов через запятую. Первый вариант — контрольный (A)"
        >
          <Input placeholder="A, B, C" />
        </Form.Item>

        <Form.Item
          name="primaryMetric"
          label="Основная метрика"
          rules={[{ required: true, message: 'Введите основную метрику' }]}
          tooltip="Метрика для оценки успеха теста"
        >
          <Input placeholder="Например: conversion, revenue, ctr, click_rate" />
        </Form.Item>

        <Form.Item
          name="metricType"
          label="Тип метрики"
          rules={[{ required: true }]}
          tooltip="Тип данных метрики для правильного статистического анализа"
        >
          <Select>
            <Option value="binary">Бинарная (конверсия, клик: 0 или 1)</Option>
            <Option value="continuous">Непрерывная (доход, время: числа)</Option>
            <Option value="ratio">Отношение (CTR, CR: проценты)</Option>
          </Select>
        </Form.Item>

        <Form.Item
          name="description"
          label="Описание теста"
          tooltip="Опишите цель и гипотезу теста"
        >
          <Input.TextArea
            rows={3}
            placeholder="Опишите цель и гипотезу теста. Например: Проверяем влияние цвета кнопки на конверсию"
          />
        </Form.Item>

        <Form.Item
          name="sampleSize"
          label="Размер выборки (опционально)"
          tooltip="Оставьте пустым для автоматического расчёта на основе MDE и мощности"
        >
          <InputNumber
            min={100}
            max={1000000}
            style={{ width: '100%' }}
            placeholder="Автоматический расчёт"
          />
        </Form.Item>

        <Form.Item
          name="confidenceLevel"
          label="Уровень доверия (1 − α)"
          tooltip="Вероятность корректного решения. Рекомендуется 0.95 (95%)"
        >
          <InputNumber
            min={0.8}
            max={0.99}
            step={0.01}
            style={{ width: '100%' }}
          />
        </Form.Item>

        <Form.Item
          name="power"
          label="Мощность теста (1 − β)"
          tooltip="Способность обнаружить эффект, если он существует. Рекомендуется 0.8 (80%)"
        >
          <InputNumber
            min={0.5}
            max={0.95}
            step={0.05}
            style={{ width: '100%' }}
          />
        </Form.Item>

        <Form.Item
          name="minEffectSize"
          label="Минимальный детектируемый эффект (MDE)"
          tooltip="Минимальное изменение метрики, которое вы хотите обнаружить (например: 0.1 = 10%)"
        >
          <InputNumber
            min={0.01}
            max={1.0}
            step={0.01}
            style={{ width: '100%' }}
          />
        </Form.Item>

        <Form.Item
          name="datasetId"
          label="Синтетический датасет"
          tooltip="Опционально: сразу привязать датасет для симуляции"
        >
          <Select allowClear placeholder="Выберите датасет">
            {datasets.map((ds: any) => (
              <Option key={ds.id} value={ds.id}>
                ID: {ds.id} | {ds.dataset_name || 'Без имени'} | Записей: {ds.sample_count.toLocaleString('ru-RU')} | {new Date(ds.created_at).toLocaleDateString('ru-RU')}
              </Option>
            ))}
          </Select>
        </Form.Item>

        <Form.Item
          name="trafficSplitType"
          label="Стратегия разделения трафика"
          tooltip="fixed — стандартный равномерный A/B; adaptive — исследовательский режим (больше трафика на лучший вариант)"
        >
          <Select>
            <Option value="fixed">fixed — равномерное разделение (рекомендуется)</Option>
            <Option value="adaptive">adaptive — адаптивное разделение</Option>
          </Select>
        </Form.Item>

        <Form.Item
          name="simulationDurationMinutes"
          label="Длительность симуляции (минуты)"
          tooltip="Сколько минут займёт симуляция в реальном времени"
        >
          <InputNumber
            min={1}
            max={180}
            style={{ width: '100%' }}
          />
        </Form.Item>

        <Form.Item>
          <Button type="primary" htmlType="submit" loading={loading} block size="large">
            Создать A/B тест
          </Button>
        </Form.Item>
      </Form>

      <div style={{ marginTop: '20px', padding: '12px', background: '#f0f2f5', borderRadius: '4px' }}>
        <h4>📋 Рекомендации:</h4>
        <ul style={{ margin: 0, paddingLeft: '20px' }}>
          <li>Первый вариант (A) всегда считается контрольным</li>
          <li>Для финальной валидации гипотез используйте стратегию <strong>fixed</strong></li>
          <li>Если датасет не выбран, при запуске симуляции будет использован последний доступный синтетический датасет</li>
          <li>MDE 0.05 = вы хотите обнаружить эффект от 5% и выше</li>
        </ul>
      </div>

      {/* Модальное окно выбора шаблона */}
      <Modal
        title={
          <Space>
            <FileTextOutlined />
            <span>Выберите шаблон A/B теста</span>
          </Space>
        }
        open={templateModal}
        onCancel={() => setTemplateModal(false)}
        footer={null}
        width={800}
      >
        <Text type="secondary" style={{ marginBottom: 12, display: 'block' }}>
          Выберите готовый шаблон для быстрого заполнения формы. Все поля можно изменить после применения.
        </Text>
        <Table
          dataSource={templates}
          columns={templateColumns}
          rowKey="id"
          loading={loadingTemplates}
          pagination={{ pageSize: 8 }}
          size="small"
          locale={{ emptyText: 'Нет доступных шаблонов A/B тестов' }}
        />
      </Modal>
    </Card>
  );
};
