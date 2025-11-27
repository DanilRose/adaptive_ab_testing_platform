import React, { useState } from 'react';
import { Button, Form, Input, Select, InputNumber, Card, message } from 'antd';
import { abTestAPI } from '../../utils/api';

const { Option } = Select;

export const TestCreator: React.FC = () => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);

  const onFinish = async (values: any) => {
    setLoading(true);
    try {
      // Правильно обрабатываем variants
      let variantsArray: string[];
      if (typeof values.variants === 'string') {
        variantsArray = values.variants.split(',').map((v: string) => v.trim());
      } else if (Array.isArray(values.variants)) {
        variantsArray = values.variants;
      } else {
        throw new Error('Variants must be string or array');
      }

      const response = await abTestAPI.createTest({
        test_name: values.testName,
        variants: variantsArray,  // ← УЖЕ МАССИВ
        primary_metric: values.primaryMetric,
        metric_type: values.metricType,
        description: values.description,
        sample_size: values.sampleSize,
        confidence_level: values.confidenceLevel,
        power: values.power,
        min_effect_size: values.minEffectSize,
      });

      message.success(`Тест создан! ID: ${response.data.test_id}`);
      form.resetFields();
    } catch (error) {
      message.error('Ошибка при создании теста');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card title="Создать новый A/B тест" style={{ maxWidth: 600, margin: '20px auto' }}>
      <Form
        form={form}
        layout="vertical"
        onFinish={onFinish}
        initialValues={{
          confidenceLevel: 0.95,
          power: 0.8,
          minEffectSize: 0.1,
          metricType: 'binary',
        }}
      >
        <Form.Item
          name="testName"
          label="Название теста"
          rules={[{ required: true, message: 'Введите название теста' }]}
        >
          <Input placeholder="Например: Button Color Test" />
        </Form.Item>

        <Form.Item
          name="variants"
          label="Варианты (через запятую)"
          rules={[{ required: true, message: 'Введите варианты теста' }]}
        >
          <Input placeholder="A, B, C" />
        </Form.Item>

        <Form.Item
          name="primaryMetric"
          label="Основная метрика"
          rules={[{ required: true, message: 'Введите основную метрику' }]}
        >
          <Input placeholder="Например: conversion_rate, revenue, engagement" />
        </Form.Item>

        <Form.Item
          name="metricType"
          label="Тип метрики"
          rules={[{ required: true }]}
        >
          <Select>
            <Option value="binary">Бинарная (конверсия)</Option>
            <Option value="continuous">Непрерывная (доход, время)</Option>
            <Option value="ratio">Отношение (CTR, CR)</Option>
          </Select>
        </Form.Item>

        <Form.Item
          name="sampleSize"
          label="Размер выборки (опционально)"
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
          label="Уровень доверия"
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
          label="Мощность теста"
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
          label="Минимальный размер эффекта"
        >
          <InputNumber 
            min={0.01} 
            max={1.0} 
            step={0.01} 
            style={{ width: '100%' }} 
          />
        </Form.Item>

        <Form.Item
          name="description"
          label="Описание теста"
        >
          <Input.TextArea rows={3} placeholder="Опишите цель и гипотезу теста" />
        </Form.Item>

        <Form.Item>
          <Button type="primary" htmlType="submit" loading={loading} block>
            Создать A/B тест
          </Button>
        </Form.Item>
      </Form>
    </Card>
  );
};