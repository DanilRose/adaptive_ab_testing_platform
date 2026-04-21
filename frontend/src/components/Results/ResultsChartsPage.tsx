import React, { useMemo, useState } from 'react';
import { Card, Col, Row, Select, Tag, Typography } from 'antd';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar,
} from 'recharts';
import { InfoCircleOutlined } from '@ant-design/icons';
import { useOutletContext } from 'react-router-dom';
import type { ResultsOutletContext } from './ResultsSectionPage';

const { Option } = Select;
const { Text, Paragraph } = Typography;

const CHART_DESCRIPTIONS: Record<string, { title: string; description: string; whatItShows: string; howToRead: string; impact: string }> = {
  cumulative: {
    title: 'Накопленная метрика',
    description: 'Суммарное значение целевой метрики по каждому варианту по мере роста объёма данных.',
    whatItShows: 'X — пользователи, Y — накопленная метрика, линия — вариант.',
    howToRead: 'Линия выше — выше совокупный эффект.',
    impact: 'Показывает абсолютный бизнес-эффект.',
  },
  mean: {
    title: 'Средняя метрика',
    description: 'Среднее значение метрики на пользователя в динамике.',
    whatItShows: 'X — пользователи, Y — средняя метрика на пользователя.',
    howToRead: 'Устойчивая линия выше контроля указывает на лучший вариант.',
    impact: 'Основа для расчёта uplift и выбора победителя.',
  },
  ci: {
    title: 'Доверительные интервалы',
    description: '95% доверительные интервалы оценки метрики по вариантам.',
    whatItShows: 'Нижняя/верхняя граница ДИ в динамике.',
    howToRead: 'Меньше пересечений ДИ — выше уверенность в различиях.',
    impact: 'Показывает надёжность оценок.',
  },
  pvalue: {
    title: 'p-value (raw)',
    description: 'Сырые p-значения как мониторинговый сигнал.',
    whatItShows: 'Линии p-value и порог 0.05.',
    howToRead: 'Ниже 0.05 — сигнал, но финальное решение по corrected p-value.',
    impact: 'Мониторинг статистической динамики.',
  },
  uplift: {
    title: 'Uplift vs control',
    description: 'Процентный прирост варианта относительно контроля.',
    whatItShows: 'X — пользователи, Y — uplift %.',
    howToRead: 'Стабильно выше нуля — положительный эффект.',
    impact: 'Ключевой бизнес-показатель прироста.',
  },
  power: {
    title: 'Power',
    description: 'Статистическая мощность по мере накопления выборки.',
    whatItShows: 'Y от 0 до 1, ориентир 0.8.',
    howToRead: 'Значения < 0.8 — риск недообнаружения эффекта.',
    impact: 'Оценивает достаточность данных.',
  },
  traffic: {
    title: 'Traffic split',
    description: 'Фактическое распределение трафика по вариантам.',
    whatItShows: 'Процент трафика по каждому варианту.',
    howToRead: 'Сильный дисбаланс может указывать на SRM.',
    impact: 'Критично для валидности эксперимента.',
  },
};

export const ResultsChartsPage: React.FC = () => {
  const { timeSeriesData } = useOutletContext<ResultsOutletContext>();
  const [chartType, setChartType] = useState<'cumulative' | 'mean' | 'ci' | 'pvalue' | 'uplift' | 'power' | 'traffic'>('cumulative');
  const [showChartInfo, setShowChartInfo] = useState(true);

  const chartData = useMemo(() => {
    if (!timeSeriesData || !timeSeriesData.data) return [];
    const groupedData: Record<number, any> = {};

    timeSeriesData.data.forEach((point) => {
      if (!groupedData[point.users_processed]) {
        groupedData[point.users_processed] = { users_processed: point.users_processed };
      }
      groupedData[point.users_processed][point.variant] = point;
    });

    return Object.values(groupedData).sort((a, b) => a.users_processed - b.users_processed);
  }, [timeSeriesData]);

  const getVariantColor = (variant: string) => {
    const variantColors: Record<string, string> = {
      A: '#1890ff',
      B: '#52c41a',
      C: '#faad14',
      D: '#f5222d',
    };
    return variantColors[variant] || '#722ed1';
  };

  const renderChartDescription = (type: string) => {
    const info = CHART_DESCRIPTIONS[type];
    if (!info || !showChartInfo) return null;

    return (
      <Card size="small" style={{ marginBottom: 12 }}>
        <Text strong>{info.title} — Что показывает этот график?</Text>
        <Paragraph style={{ marginTop: 8, marginBottom: 8 }}>{info.description}</Paragraph>
        <Row gutter={16}>
          <Col span={8}><Text strong>📊 Что отображено:</Text><br /><Text type="secondary">{info.whatItShows}</Text></Col>
          <Col span={8}><Text strong>🔍 Как читать:</Text><br /><Text type="secondary">{info.howToRead}</Text></Col>
          <Col span={8}><Text strong>💼 На что влияет:</Text><br /><Text type="secondary">{info.impact}</Text></Col>
        </Row>
      </Card>
    );
  };

  const renderCumulativeChart = () => (
    <Card title="📈 Накопленная метрика по вариантам" size="small" style={{ marginBottom: 16 }}>
      <ResponsiveContainer width="100%" height={380}>
        <AreaChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="users_processed" />
          <YAxis />
          <Tooltip />
          <Legend formatter={(value) => `Вариант ${value}`} />
          {timeSeriesData?.variants.map((variant) => (
            <Area
              key={variant}
              type="monotone"
              dataKey={(data: any) => data[variant]?.cumulative_metric || 0}
              name={variant}
              stroke={getVariantColor(variant)}
              fill={getVariantColor(variant)}
              fillOpacity={0.2}
            />
          ))}
        </AreaChart>
      </ResponsiveContainer>
    </Card>
  );

  const renderMeanMetricChart = () => (
    <Card title="📊 Средняя метрика на пользователя" size="small" style={{ marginBottom: 16 }}>
      <ResponsiveContainer width="100%" height={380}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="users_processed" />
          <YAxis />
          <Tooltip />
          <Legend formatter={(value) => `Вариант ${value}`} />
          {timeSeriesData?.variants.map((variant) => (
            <Line
              key={variant}
              type="monotone"
              dataKey={(data: any) => data[variant]?.mean_metric || 0}
              name={variant}
              stroke={getVariantColor(variant)}
              strokeWidth={2}
              dot={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </Card>
  );

  const renderConfidenceIntervalsChart = () => (
    <Card title="🎯 Доверительные интервалы (95%)" size="small" style={{ marginBottom: 16 }}>
      <ResponsiveContainer width="100%" height={380}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="users_processed" />
          <YAxis />
          <Tooltip />
          <Legend />
          {timeSeriesData?.variants.map((variant) => (
            <React.Fragment key={variant}>
              <Line
                type="monotone"
                dataKey={(data: any) => data[variant]?.confidence_interval_lower ?? null}
                name={`${variant} — нижняя граница ДИ`}
                stroke={getVariantColor(variant)}
                strokeDasharray="4 4"
                dot={false}
                connectNulls
              />
              <Line
                type="monotone"
                dataKey={(data: any) => data[variant]?.confidence_interval_upper ?? null}
                name={`${variant} — верхняя граница ДИ`}
                stroke={getVariantColor(variant)}
                strokeDasharray="4 4"
                dot={false}
                connectNulls
              />
            </React.Fragment>
          ))}
        </LineChart>
      </ResponsiveContainer>
    </Card>
  );

  const renderPValueChart = () => {
    const pValueData = chartData.map((d: any) => {
      const point: any = { users_processed: d.users_processed, threshold: 0.05 };
      timeSeriesData?.variants.forEach((variant) => {
        if (variant !== timeSeriesData.variants[0]) {
          point[variant] = d[variant]?.p_value ?? null;
        }
      });
      return point;
    });

    return (
      <Card title="📉 p-значение (сырое, мониторинг)" size="small" style={{ marginBottom: 16 }}>
        <ResponsiveContainer width="100%" height={380}>
          <LineChart data={pValueData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="users_processed" />
            <YAxis domain={[0, 1]} />
            <Tooltip />
            <Legend />
            <Line dataKey="threshold" name="Порог значимости (p = 0.05)" stroke="#ff4d4f" strokeDasharray="5 5" dot={false} />
            {timeSeriesData?.variants
              .filter((v) => v !== timeSeriesData.variants[0])
              .map((variant) => (
                <Line
                  key={variant}
                  type="monotone"
                  dataKey={variant}
                  name={`Вариант ${variant} vs ${timeSeriesData.variants[0]}`}
                  stroke={getVariantColor(variant)}
                  strokeWidth={2}
                  dot={false}
                  connectNulls
                />
              ))}
          </LineChart>
        </ResponsiveContainer>
      </Card>
    );
  };

  const renderUpliftChart = () => (
    <Card title="🚀 Прирост относительно контроля (%)" size="small" style={{ marginBottom: 16 }}>
      <ResponsiveContainer width="100%" height={380}>
        <LineChart data={timeSeriesData?.uplift_over_time || []}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="users_processed" />
          <YAxis />
          <Tooltip />
          <Legend formatter={(value) => `Вариант ${value} (uplift)`} />
          {(timeSeriesData?.variants || [])
            .filter((v) => v !== timeSeriesData?.variants?.[0])
            .map((variant) => (
              <Line
                key={variant}
                type="monotone"
                dataKey={variant}
                name={variant}
                stroke={getVariantColor(variant)}
                dot={false}
                connectNulls
              />
            ))}
        </LineChart>
      </ResponsiveContainer>
    </Card>
  );

  const renderPowerChart = () => (
    <Card title="⚡ Статистическая мощность" size="small" style={{ marginBottom: 16 }}>
      <ResponsiveContainer width="100%" height={380}>
        <LineChart data={timeSeriesData?.power_over_time || []}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="users_processed" />
          <YAxis domain={[0, 1]} />
          <Tooltip />
          <Legend formatter={(value) => `Вариант ${value} — power`} />
          {(timeSeriesData?.variants || [])
            .filter((v) => v !== timeSeriesData?.variants?.[0])
            .map((variant) => (
              <Line
                key={variant}
                type="monotone"
                dataKey={variant}
                name={variant}
                stroke={getVariantColor(variant)}
                dot={false}
                connectNulls
              />
            ))}
        </LineChart>
      </ResponsiveContainer>
    </Card>
  );

  const renderTrafficSplitChart = () => {
    const trafficData = Object.keys(timeSeriesData?.traffic_split?.variant_percentages || {}).map((variant) => ({
      variant: `Вариант ${variant}`,
      percent: timeSeriesData?.traffic_split?.variant_percentages?.[variant] || 0,
    }));

    return (
      <Card title="🛣️ Распределение трафика" size="small" style={{ marginBottom: 16 }}>
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={trafficData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="variant" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="percent" fill="#1890ff" name="Доля трафика, %" />
          </BarChart>
        </ResponsiveContainer>
      </Card>
    );
  };

  return (
    <>
      <Card style={{ marginBottom: 16 }}>
        <Row gutter={[16, 16]} align="middle" wrap>
          <Col>
            <Text strong>Тип графика:</Text>
          </Col>
          <Col flex="auto">
            <Select value={chartType} onChange={setChartType} style={{ width: 320 }}>
              <Option value="cumulative">📈 Накопленная метрика</Option>
              <Option value="mean">📊 Средняя метрика</Option>
              <Option value="ci">🎯 Доверительные интервалы (ДИ)</Option>
              <Option value="pvalue">📉 p-значение (сырое)</Option>
              <Option value="uplift">🚀 Прирост относительно контроля</Option>
              <Option value="power">⚡ Статистическая мощность</Option>
              <Option value="traffic">🛣️ Распределение трафика</Option>
            </Select>
          </Col>
          <Col>
            <Tag
              color={showChartInfo ? 'blue' : 'default'}
              style={{ cursor: 'pointer' }}
              onClick={() => setShowChartInfo(!showChartInfo)}
            >
              <InfoCircleOutlined /> {showChartInfo ? 'Скрыть описание' : 'Показать описание'}
            </Tag>
          </Col>
        </Row>
      </Card>

      {renderChartDescription(chartType)}
      {chartType === 'cumulative' && renderCumulativeChart()}
      {chartType === 'mean' && renderMeanMetricChart()}
      {chartType === 'ci' && renderConfidenceIntervalsChart()}
      {chartType === 'pvalue' && renderPValueChart()}
      {chartType === 'uplift' && renderUpliftChart()}
      {chartType === 'power' && renderPowerChart()}
      {chartType === 'traffic' && renderTrafficSplitChart()}
    </>
  );
};
