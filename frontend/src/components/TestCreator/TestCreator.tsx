import React, { useState, useEffect } from 'react';
import { Button, Form, Input, Select, InputNumber, Card, message } from 'antd';
import { abTestAPI, dataAPI } from '../../utils/api';

const { Option } = Select;

export const TestCreator: React.FC = () => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [datasets, setDatasets] = useState<any[]>([]);

  useEffect(() => {
    const loadDatasets = async () => {
      try {
        const response = await dataAPI.listGeneratedHistory(100);
        const syntheticDatasets = response.data.items.filter((d: any) => d.data_type === 'synthetic');
        setDatasets(syntheticDatasets);
      } catch (e) {
        console.error('Failed to load datasets', e);
      }
    };

    loadDatasets();
  }, []);

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
      };

      console.log('📤 Sending test data:', testData);

      const response = await abTestAPI.createTest(testData);

      message.success(`✅ Тест успешно создан! ID: ${response.data.test_id}`);
      form.resetFields();

    } catch (error: any) {
      console.error('❌ Error creating test:', error);
      const errorMsg = error.response?.data?.detail || error.message || 'Неизвестная ошибка';
      message.error(`Ошибка при создании теста: ${errorMsg}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card title="Создать новый A/B тест" style={{ maxWidth: 800, margin: '20px auto' }}>
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
          <Input placeholder="Например: Button Color Test" />
        </Form.Item>

        <Form.Item
          name="variants"
          label="Варианты (через запятую)"
          rules={[{ required: true, message: 'Введите варианты теста' }]}
          tooltip="Укажите названия вариантов через запятую. Первый вариант - контрольный (A)"
        >
          <Input placeholder="A, B, C" />
        </Form.Item>

        <Form.Item
          name="primaryMetric"
          label="Основная метрика"
          rules={[{ required: true, message: 'Введите основную метрику' }]}
          tooltip="Метрика для оценки успеха теста"
        >
          <Input placeholder="Например: conversion, revenue, click_rate" />
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
          tooltip="Оставьте пустым для автоматического расчета на основе MDE и мощности"
        >
          <InputNumber
            min={100}
            max={1000000}
            style={{ width: '100%' }}
            placeholder="Автоматический расчет"
          />
        </Form.Item>

        <Form.Item
          name="confidenceLevel"
          label="Уровень доверия (alpha)"
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
          label="Мощность теста (1-beta)"
          tooltip="Способность обнаружить эффект если он существует. Рекомендуется 0.8 (80%)"
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
                ID: {ds.id} | Samples: {ds.sample_count} | {new Date(ds.created_at).toLocaleDateString()}
              </Option>
            ))}
          </Select>
        </Form.Item>

        <Form.Item
          name="trafficSplitType"
          label="Стратегия трафика"
          tooltip="fixed — стандартный A/B; adaptive — исследовательский режим"
        >
          <Select>
            <Option value="fixed">fixed (рекомендуется)</Option>
            <Option value="adaptive">adaptive</Option>
          </Select>
        </Form.Item>

        <Form.Item
          name="simulationDurationMinutes"
          label="Длительность симуляции (минуты)"
          tooltip="Можно оставить по умолчанию — backend рассчитает динамически"
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
        <h4>    Рекомендации:</h4>
        <ul style={{ margin: 0, paddingLeft: '20px' }}>
          <li>Первый вариант (A) всегда считается контрольным</li>
          <li>Для финальной валидации гипотез используйте fixed split</li>
          <li>Если датасет не выбран, при запуске будет использован последний доступный synthetic датасет</li>
        </ul>
      </div>
    </Card>
  );
};
