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
import { useTheme } from '@/context/ThemeContext';
import type { ResultsOutletContext } from './ResultsSectionPage';

const { Option } = Select;
const { Text, Paragraph } = Typography;

const CHART_DESCRIPTIONS: Record<string, { title: string; description: string; whatItShows: string; howToRead: string; impact: string }> = {
  cumulative: {
    title: 'Накопленная метрика',
    description: 'Суммарное значение целевой метрики по каждому варианту по мере роста объёма данных.',
    whatItShows: 'Ось X — пользователи, ось Y — накопленная метрика, линия — вариант.',
    howToRead: 'Линия выше означает больший совокупный эффект.',
    impact: 'Показывает абсолютный бизнес-эффект.',
  },
  mean: {
    title: 'Среднее значение метрики',
    description: 'Среднее значение метрики на одного пользователя в динамике.',
    whatItShows: 'Ось X — пользователи, ось Y — среднее значение метрики.',
    howToRead: 'Устойчивая линия выше контроля указывает на более сильный вариант.',
    impact: 'Основа для расчёта прироста и выбора победителя.',
  },
  ci: {
    title: 'Доверительные интервалы',
    description: '95% доверительные интервалы оценки метрики по вариантам.',
    whatItShows: 'Нижняя и верхняя границы доверительного интервала во времени.',
    howToRead: 'Чем меньше пересечений интервалов, тем выше уверенность в различиях.',
    impact: 'Показывает надёжность оценок.',
  },
  pvalue: {
    title: 'Сырое p-значение',
    description: 'Сырые p-значения как оперативный сигнал наблюдения.',
    whatItShows: 'Линии p-значения и порог 0.05.',
    howToRead: 'Ниже 0.05 — сигнал внимания, но решение по скорректированному p-значению.',
    impact: 'Нужно для мониторинга статистической динамики.',
  },
  uplift: {
    title: 'Прирост к контролю',
    description: 'Процентный прирост варианта относительно контрольной группы.',
    whatItShows: 'Ось X — пользователи, ось Y — прирост в процентах.',
    howToRead: 'Стабильно выше нуля — положительный эффект.',
    impact: 'Ключевой бизнес-показатель роста.',
  },
  power: {
    title: 'Статистическая мощность',
    description: 'Статистическая мощность по мере накопления выборки.',
    whatItShows: 'Ось Y от 0 до 1, ориентир 0.8.',
    howToRead: 'Значения ниже 0.8 указывают на риск недообнаружения эффекта.',
    impact: 'Оценивает достаточность данных для уверенного вывода.',
  },
  traffic: {
    title: 'Распределение трафика',
    description: 'Фактическое распределение пользователей между вариантами.',
    whatItShows: 'Процент трафика по каждому варианту.',
    howToRead: 'Сильный дисбаланс может указывать на перекос распределения.',
    impact: 'Критично для валидности эксперимента.',
  },
};

export const ResultsChartsPage: React.FC = () => {
  const { timeSeriesData } = useOutletContext<ResultsOutletContext>();
  const { theme } = useTheme();
  const isDark = theme === 'dark';

  const c = useMemo(
    () => ({
      panelBg: isDark ? '#1c1917' : '#ffffff',
      panelSoft: isDark ? '#171412' : '#f5f0e8',
      border: isDark ? '#292524' : '#e7e5e4',
      textPrimary: isDark ? '#fafaf9' : '#1c1917',
      textMuted: isDark ? '#a8a29e' : '#78716c',
      textSub: isDark ? '#57534e' : '#a8a29e',
      accent: '#d97706',
      accentSoft: isDark ? 'rgba(217,119,6,0.16)' : '#fef3c7',
      accentText: isDark ? '#fcd34d' : '#92400e',
      shadow: isDark ? '0 12px 38px rgba(0,0,0,0.38)' : '0 10px 30px rgba(28,25,23,0.08)',
    }),
    [isDark],
  );

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
      <div style={{ borderRadius: 14, border: `1px solid ${c.border}`, backgroundColor: c.panelBg, boxShadow: c.shadow, overflow: 'hidden', marginBottom: 12 }}>
        <div style={{ padding: '12px 14px', borderBottom: `1px solid ${c.border}`, backgroundColor: c.panelSoft, fontSize: 14, fontWeight: 700, color: c.textPrimary }}>
          {info.title} — пояснение графика
        </div>
        <div style={{ padding: 14 }}>
          <Paragraph style={{ marginTop: 0, marginBottom: 8 }}>{info.description}</Paragraph>
          <Row gutter={16}>
            <Col span={8}><Text strong>Что отображено:</Text><br /><Text type="secondary">{info.whatItShows}</Text></Col>
            <Col span={8}><Text strong>Как читать:</Text><br /><Text type="secondary">{info.howToRead}</Text></Col>
            <Col span={8}><Text strong>Практический смысл:</Text><br /><Text type="secondary">{info.impact}</Text></Col>
          </Row>
        </div>
      </div>
    );
  };

  const renderCumulativeChart = () => (
    <Card title="Накопленная метрика по вариантам" size="small" style={{ marginBottom: 16, borderRadius: 12, borderColor: c.border }}>
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
    <Card title="Среднее значение метрики на пользователя" size="small" style={{ marginBottom: 16, borderRadius: 12, borderColor: c.border }}>
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
    <Card title="Доверительные интервалы (95%)" size="small" style={{ marginBottom: 16, borderRadius: 12, borderColor: c.border }}>
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
                name={`${variant} — нижняя граница`}
                stroke={getVariantColor(variant)}
                strokeDasharray="4 4"
                dot={false}
                connectNulls
              />
              <Line
                type="monotone"
                dataKey={(data: any) => data[variant]?.confidence_interval_upper ?? null}
                name={`${variant} — верхняя граница`}
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
      <Card title="Сырое p-значение (мониторинг)" size="small" style={{ marginBottom: 16, borderRadius: 12, borderColor: c.border }}>
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
                  name={`Вариант ${variant} относительно ${timeSeriesData.variants[0]}`}
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
    <Card title="Прирост относительно контроля (%)" size="small" style={{ marginBottom: 16, borderRadius: 12, borderColor: c.border }}>
      <ResponsiveContainer width="100%" height={380}>
        <LineChart data={timeSeriesData?.uplift_over_time || []}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="users_processed" />
          <YAxis />
          <Tooltip />
          <Legend formatter={(value) => `Вариант ${value} (прирост)`} />
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
    <Card title="Статистическая мощность" size="small" style={{ marginBottom: 16, borderRadius: 12, borderColor: c.border }}>
      <ResponsiveContainer width="100%" height={380}>
        <LineChart data={timeSeriesData?.power_over_time || []}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="users_processed" />
          <YAxis domain={[0, 1]} />
          <Tooltip />
          <Legend formatter={(value) => `Вариант ${value} — мощность`} />
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
      <Card title="Распределение трафика" size="small" style={{ marginBottom: 16, borderRadius: 12, borderColor: c.border }}>
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


      <div style={{ borderRadius: 14, border: `1px solid ${c.border}`, backgroundColor: c.panelBg, boxShadow: c.shadow, overflow: 'hidden', marginBottom: 14 }}>
        <div style={{ padding: '12px 14px', borderBottom: `1px solid ${c.border}`, backgroundColor: c.panelSoft, fontSize: 14, fontWeight: 700, color: c.textPrimary }}>
          Настройки графиков
        </div>
        <div style={{ padding: 14 }}>
          <Row gutter={[16, 16]} align="middle" wrap>
            <Col>
              <Text strong>Тип графика:</Text>
            </Col>
            <Col flex="auto">
              <Select value={chartType} onChange={setChartType} style={{ width: 360 }}>
                <Option value="cumulative">Накопленная метрика</Option>
                <Option value="mean">Среднее значение метрики</Option>
                <Option value="ci">Доверительные интервалы (ДИ)</Option>
                <Option value="pvalue">Сырое p-значение</Option>
                <Option value="uplift">Прирост относительно контроля</Option>
                <Option value="power">Статистическая мощность</Option>
                <Option value="traffic">Распределение трафика</Option>
              </Select>
            </Col>
            <Col>
              <Tag
                color={showChartInfo ? 'blue' : 'default'}
                style={{ cursor: 'pointer' }}
                onClick={() => setShowChartInfo(!showChartInfo)}
              >
                <InfoCircleOutlined /> {showChartInfo ? 'Скрыть пояснение' : 'Показать пояснение'}
              </Tag>
            </Col>
          </Row>
        </div>
      </div>

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
