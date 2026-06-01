// frontend/src/components/Results/ResultsPage.tsx
import React, { useState, useEffect, useRef } from 'react';
import { Card, Row, Col, Select, Typography, Table, Tag, Statistic, Spin, Alert, Empty, Progress, Space } from 'antd';
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
import { resultsAPI, abTestAPI } from '../../utils/api';
import type { TimeSeriesResponse } from '../../types';
import {
  CheckCircleOutlined,
  WarningOutlined,
  InfoCircleOutlined,
} from '@ant-design/icons';

const { Option } = Select;
const { Title, Text, Paragraph } = Typography;

interface TimeSeriesDataPoint {
  users_processed: number;
  variant: string;
  cumulative_metric: number;
  mean_metric: number;
  sample_size: number;
  p_value: number | null;
  confidence_interval_lower: number | null;
  confidence_interval_upper: number | null;
}


interface TestSummary {
  test_id: string;
  test_name: string;
  status: string;
  primary_metric: string;
  variants: string[];
  total_users: number;
  completion_percentage: number;
  simulation_status?: string;
}

// Описания графиков для комиссии
const CHART_DESCRIPTIONS: Record<string, { title: string; description: string; whatItShows: string; howToRead: string; impact: string }> = {
  cumulative: {
    title: 'Накопленная метрика',
    description: 'График показывает суммарное значение целевой метрики (например, общий доход, общее количество конверсий) по каждому варианту теста по мере увеличения числа обработанных пользователей.',
    whatItShows: 'На оси X — количество пользователей, прошедших через тест. На оси Y — накопленная сумма метрики. Каждая линия соответствует одному варианту (A, B, C...). Заполненная область под линией показывает накопленный объём.',
    howToRead: 'Если линия варианта B идёт выше линии варианта A — это означает, что вариант B суммарно принёс больше конверсий/дохода при одинаковом числе пользователей. Чем круче наклон — тем быстрее растёт метрика.',
    impact: 'Используется для оценки абсолютного экономического эффекта. Показывает, какой вариант выгоднее в денежном выражении.',
  },
  mean: {
    title: 'Средняя метрика',
    description: 'График динамики среднего значения метрики на одного пользователя для каждого варианта по мере поступления новых данных.',
    whatItShows: 'На оси X — количество пользователей. На оси Y — среднее значение метрики (например, средняя конверсия 0.12 = 12%, средний доход 850 руб.). Каждая линия — отдельный вариант теста.',
    howToRead: 'Если линия варианта B стабилизировалась выше линии A — вариант B даёт в среднем лучший результат на пользователя. Важно смотреть на стабилизацию: в начале значения нестабильны из-за малой выборки. Доверять графику следует только при большом N (правая часть).',
    impact: 'Ключевой показатель для принятия решений. По нему определяется победитель. Влияет на расчёт uplift.',
  },
  ci: {
    title: 'Доверительные интервалы',
    description: 'График показывает диапазон значений, в котором с заданной вероятностью (95%) находится истинное среднее значение метрики для каждого варианта.',
    whatItShows: 'Две пунктирные линии одного цвета — нижняя и верхняя граница 95% доверительного интервала для каждого варианта. Чем уже "коридор" между ними — тем точнее оценка. Узкий интервал = большая выборка, достоверный результат.',
    howToRead: 'Если доверительные интервалы двух вариантов НЕ пересекаются — разница между ними статистически значима. Если интервалы сильно перекрываются — результаты пока не достоверны, нужно больше данных.',
    impact: 'Критически важен для оценки надёжности эксперимента. Позволяет понять, является ли наблюдаемый эффект случайным или реальным.',
  },
  pvalue: {
    title: 'p-значение (сырое, мониторинговое)',
    description: 'График изменения сырого (некорректированного) p-значения во времени. Используется для мониторинга динамики, но не как основной критерий решения при множественных сравнениях.',
    whatItShows: 'На оси X — количество пользователей. На оси Y — сырое p-значение от 0 до 1. Красная пунктирная линия — порог 0.05. Линии вариантов (B, C...) показывают сырые p-значения сравнения с контрольным вариантом A.',
    howToRead: 'Если сырое p-значение опускается ниже 0.05, это сигнал для внимания, но итоговое решение нужно принимать по скорректированным p-значениям (метод Холма—Бонферрони), особенно при 3+ вариантах.',
    impact: 'Служит мониторингом хода теста. Для финального решения используйте скорректированные p-значения (метод Холма), чтобы снизить риск ложноположительных выводов.',
  },
  uplift: {
    title: 'Прирост метрики относительно контроля',
    description: 'График показывает процентный прирост метрики варианта по сравнению с контрольным вариантом A в динамике.',
    whatItShows: 'На оси X — количество пользователей. На оси Y — прирост в процентах (например, +12% означает, что вариант B на 12% лучше варианта A). Нулевая линия — уровень контроля. Положительные значения = вариант лучше, отрицательные = хуже.',
    howToRead: 'Если линия устойчиво держится выше нуля — вариант B систематически лучше контроля. Резкие скачки в начале нормальны (мало данных). В конце теста значение стабилизируется и даёт реальную оценку прироста.',
    impact: 'Ключевой бизнес-показатель. Отвечает на вопрос: "На сколько процентов лучше новый вариант?" Используется для бизнес-обоснования внедрения изменений.',
  },
  power: {
    title: 'Статистическая мощность',
    description: 'График показывает вероятность обнаружить реальный эффект, если он существует. Рассчитывается по мере накопления данных.',
    whatItShows: 'На оси X — количество пользователей. На оси Y — мощность от 0 до 1 (0% до 100%). Значение 0.8 (80%) — стандартный минимум для надёжного теста. Каждая линия — мощность для обнаружения эффекта варианта B (C...) относительно A.',
    howToRead: 'Мощность растёт по мере увеличения выборки. Когда линия достигает 0.8 — у теста достаточно данных для обнаружения минимального детектируемого эффекта (MDE). Если мощность не достигает 0.8 — выборки недостаточно, результаты могут быть ненадёжными.',
    impact: 'Отвечает на вопрос: "Можем ли мы доверять результатам теста?" Низкая мощность (< 80%) означает высокий риск пропустить реальный эффект (ошибка II рода).',
  },
  traffic: {
    title: 'Распределение трафика (Traffic Split)',
    description: 'Столбчатая диаграмма, показывающая фактическое распределение пользователей по вариантам теста.',
    whatItShows: 'На оси X — варианты теста (A, B, C...). На оси Y — доля трафика в процентах. В идеале при равномерном разделении все столбцы должны иметь одинаковую высоту (50%/50% для двух вариантов).',
    howToRead: 'Если столбцы примерно одинаковы — трафик распределён равномерно, тест корректен. Если один из вариантов получил значительно больше трафика — это нарушение равномерности выборки (SRM), которое делает тест недействительным.',
    impact: 'Критически важен для валидности теста. Неравномерное распределение трафика приводит к смещённым результатам. Проверяется статистическим критерием хи-квадрат для SRM.',
  },
};

export const ResultsPage: React.FC = () => {
  const [tests, setTests] = useState<TestSummary[]>([]);
  const [selectedTestId, setSelectedTestId] = useState<string>('');
  const [timeSeriesData, setTimeSeriesData] = useState<TimeSeriesResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [chartType, setChartType] = useState<'cumulative' | 'mean' | 'ci' | 'pvalue' | 'uplift' | 'power' | 'traffic'>('cumulative');
  const [isSimulating, setIsSimulating] = useState(false);
  const [showChartInfo, setShowChartInfo] = useState(true);
  const pollingRef = useRef<number | null>(null);

  useEffect(() => {
    loadTests();

    return () => {
      if (pollingRef.current) {
        window.clearInterval(pollingRef.current);
      }
    };
  }, []);

  useEffect(() => {
    const selectedTest = tests.find((t) => t.test_id === selectedTestId);
    const shouldPollTests = isSimulating || selectedTest?.status === 'active' || selectedTest?.status === 'paused';

    if (!shouldPollTests) {
      return;
    }

    const testsInterval = window.setInterval(() => {
      loadTests();
    }, 5000);

    return () => {
      window.clearInterval(testsInterval);
    };
  }, [isSimulating, selectedTestId, tests]);

  useEffect(() => {
    if (pollingRef.current) {
      window.clearInterval(pollingRef.current);
    }

    if (!selectedTestId) {
      return;
    }

    const selectedTest = tests.find((t) => t.test_id === selectedTestId);
    const selectedTestIsRunning =
      selectedTest?.simulation_status === 'running' ||
      selectedTest?.status === 'active' ||
      selectedTest?.status === 'paused';

    if (!selectedTestIsRunning) {
      loadTimeSeriesData(selectedTestId, false);
      return;
    }

    const pollIntervalMs = selectedTest?.simulation_status === 'running' ? 3000 : 7000;

    pollingRef.current = window.setInterval(() => {
      loadTimeSeriesData(selectedTestId, false);
    }, pollIntervalMs);

    return () => {
      if (pollingRef.current) {
        window.clearInterval(pollingRef.current);
      }
    };
  }, [selectedTestId, tests]);

  useEffect(() => {
    const currentlySimulating = tests.some((t) => t.simulation_status === 'running');
    setIsSimulating(currentlySimulating);
  }, [tests]);

  const loadTests = async () => {
    try {
      const response = await abTestAPI.getAllTests();
      const allTests = [
        ...(response.data.active_tests || []),
        ...(response.data.paused_tests || []),
        ...(response.data.completed_tests || []),
      ].map((t: any) => ({
        test_id: t.test_id,
        test_name: t.test_name,
        status: t.status,
        primary_metric: t.primary_metric,
        variants: t.variants,
        total_users: t.total_users,
        completion_percentage: t.completion_percentage,
        simulation_status: t.simulation_status,
      }));
      setTests(allTests);

      const completedTest = allTests.find((t: any) => t.status === 'completed');
      if (completedTest && !selectedTestId) {
        setSelectedTestId(completedTest.test_id);
        loadTimeSeriesData(completedTest.test_id);
      }
    } catch (error) {
      console.error('Ошибка загрузки тестов:', error);
    }
  };

  const loadTimeSeriesData = async (testId: string, showLoading: boolean = true) => {
    if (showLoading) {
      setLoading(true);
    }
    try {
      const response = await resultsAPI.getTimeSeriesData(testId);
      setTimeSeriesData(response.data);
    } catch (error) {
      console.error('Ошибка загрузки временных рядов:', error);
      setTimeSeriesData(null);
    } finally {
      if (showLoading) {
        setLoading(false);
      }
    }
  };

  const handleTestChange = (testId: string) => {
    setSelectedTestId(testId);
    loadTimeSeriesData(testId);
  };

  const prepareChartData = () => {
    if (!timeSeriesData || !timeSeriesData.data) return [];

    const groupedData: Record<number, any> = {};

    timeSeriesData.data.forEach((point: TimeSeriesDataPoint) => {
      if (!groupedData[point.users_processed]) {
        groupedData[point.users_processed] = { users_processed: point.users_processed };
      }
      groupedData[point.users_processed][point.variant] = point;
    });

    return Object.values(groupedData).sort((a, b) => a.users_processed - b.users_processed);
  };

  const chartData = prepareChartData();

  const variantColors: Record<string, string> = {
    A: '#1890ff',
    B: '#52c41a',
    C: '#faad14',
    D: '#f5222d',
  };

  const getVariantColor = (variant: string) => variantColors[variant] || '#722ed1';

  // Блок описания графика
  const renderChartDescription = (type: string) => {
    const info = CHART_DESCRIPTIONS[type];
    if (!info || !showChartInfo) return null;
    return (
      <Alert
        style={{ marginBottom: 12 }}
        type="info"
        showIcon
        icon={<InfoCircleOutlined />}
        message={<Text strong>{info.title} — Что показывает этот график?</Text>}
        description={
          <div style={{ marginTop: 4 }}>
            <Paragraph style={{ marginBottom: 6 }}>{info.description}</Paragraph>
            <Row gutter={16}>
              <Col span={8}>
                <Text strong style={{ color: '#1890ff' }}>📊 Что отображено:</Text>
                <br />
                <Text style={{ fontSize: 12 }}>{info.whatItShows}</Text>
              </Col>
              <Col span={8}>
                <Text strong style={{ color: '#52c41a' }}>🔍 Как читать:</Text>
                <br />
                <Text style={{ fontSize: 12 }}>{info.howToRead}</Text>
              </Col>
              <Col span={8}>
                <Text strong style={{ color: '#fa8c16' }}>💼 На что влияет:</Text>
                <br />
                <Text style={{ fontSize: 12 }}>{info.impact}</Text>
              </Col>
            </Row>
          </div>
        }
      />
    );
  };

  const renderCumulativeChart = () => (
    <>
      {renderChartDescription('cumulative')}
      <Card title="📈 Накопленная метрика по вариантам" size="small" style={{ marginBottom: 16 }}>
        <ResponsiveContainer width="100%" height={380}>
          <AreaChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="users_processed"
              label={{ value: 'Количество пользователей', position: 'insideBottom', offset: -5 }}
            />
            <YAxis label={{ value: 'Накопленная метрика', angle: -90, position: 'insideLeft' }} />
            <Tooltip
              labelFormatter={(label) => `Пользователей обработано: ${label}`}
              formatter={(value: number, name: string) => [
                typeof value === 'number' ? value.toFixed(4) : value,
                `Вариант ${name}`,
              ]}
            />
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
    </>
  );

  const renderMeanMetricChart = () => (
    <>
      {renderChartDescription('mean')}
      <Card title="Средняя метрика на пользователя" size="small" style={{ marginBottom: 16 }}>
        <ResponsiveContainer width="100%" height={380}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="users_processed"
              label={{ value: 'Количество пользователей', position: 'insideBottom', offset: -5 }}
            />
            <YAxis label={{ value: 'Среднее значение метрики', angle: -90, position: 'insideLeft' }} />
            <Tooltip
              labelFormatter={(label) => `Пользователей обработано: ${label}`}
              formatter={(value: number, name: string) => [
                typeof value === 'number' ? value.toFixed(4) : value,
                `Вариант ${name}`,
              ]}
            />
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
    </>
  );

  const renderConfidenceIntervalsChart = () => {
    if (!timeSeriesData) return null;
    return (
      <>
        {renderChartDescription('ci')}
        <Card title="Доверительные интервалы (95%)" size="small" style={{ marginBottom: 16 }}>
          <ResponsiveContainer width="100%" height={380}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="users_processed"
                label={{ value: 'Количество пользователей', position: 'insideBottom', offset: -5 }}
              />
              <YAxis label={{ value: 'Значение метрики', angle: -90, position: 'insideLeft' }} />
              <Tooltip
                labelFormatter={(label) => `Пользователей обработано: ${label}`}
                formatter={(value: number, name: string) => [
                  typeof value === 'number' ? value.toFixed(4) : value,
                  name,
                ]}
              />
              <Legend />
              {timeSeriesData.variants.map((variant) => (
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
      </>
    );
  };

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
      <>
        {renderChartDescription('pvalue')}
        <Card title="p-значение (сырое, мониторинговое)" size="small" style={{ marginBottom: 16 }}>
          <ResponsiveContainer width="100%" height={380}>
            <LineChart data={pValueData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="users_processed"
                label={{ value: 'Количество пользователей', position: 'insideBottom', offset: -5 }}
              />
              <YAxis
                domain={[0, 1]}
                label={{ value: 'p-значение (0 — значимо, 1 — не значимо)', angle: -90, position: 'insideLeft' }}
              />
              <Tooltip
                labelFormatter={(label) => `Пользователей обработано: ${label}`}
                formatter={(value: number, name: string) => [
                  typeof value === 'number' ? value.toFixed(4) : value,
                  name,
                ]}
              />
              <Legend />
              <Line
                dataKey="threshold"
                name="Порог значимости (p = 0.05)"
                stroke="#ff4d4f"
                strokeDasharray="5 5"
                dot={false}
              />
              {timeSeriesData?.variants
                .filter((v) => v !== timeSeriesData.variants[0])
                .map((variant) => (
                  <Line
                    key={variant}
                    type="monotone"
                    dataKey={variant}
                    name={`Вариант ${variant} vs ${timeSeriesData.variants[0]} (контроль)`}
                    stroke={getVariantColor(variant)}
                    strokeWidth={2}
                    dot={false}
                    connectNulls
                  />
                ))}
            </LineChart>
          </ResponsiveContainer>
        </Card>
      </>
    );
  };

  const renderUpliftChart = () => (
    <>
      {renderChartDescription('uplift')}
      <Card title="Прирост метрики относительно контроля (%)" size="small" style={{ marginBottom: 16 }}>
        <ResponsiveContainer width="100%" height={380}>
          <LineChart data={timeSeriesData?.uplift_over_time || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="users_processed"
              label={{ value: 'Количество пользователей', position: 'insideBottom', offset: -5 }}
            />
            <YAxis label={{ value: 'Прирост относительно контроля, %', angle: -90, position: 'insideLeft' }} />
            <Tooltip
              labelFormatter={(label) => `Пользователей обработано: ${label}`}
              formatter={(value: number, name: string) => [
                `${typeof value === 'number' ? value.toFixed(2) : value}%`,
                `Вариант ${name}`,
              ]}
            />
            <Legend formatter={(value) => `Вариант ${value} (прирост, %)`} />
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
    </>
  );

  const renderPowerChart = () => (
    <>
      {renderChartDescription('power')}
      <Card title="Статистическая мощность теста" size="small" style={{ marginBottom: 16 }}>
        <ResponsiveContainer width="100%" height={380}>
          <LineChart data={timeSeriesData?.power_over_time || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="users_processed"
              label={{ value: 'Количество пользователей', position: 'insideBottom', offset: -5 }}
            />
            <YAxis
              domain={[0, 1]}
              label={{ value: 'Мощность (0.8 = 80% — рекомендуемый минимум)', angle: -90, position: 'insideLeft' }}
            />
            <Tooltip
              labelFormatter={(label) => `Пользователей обработано: ${label}`}
              formatter={(value: number, name: string) => [
                `${typeof value === 'number' ? (value * 100).toFixed(1) : value}%`,
                `Мощность варианта ${name}`,
              ]}
            />
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
    </>
  );

  const renderTrafficSplitChart = () => {
    const trafficData = Object.keys(timeSeriesData?.traffic_split?.variant_percentages || {}).map((variant) => ({
      variant: `Вариант ${variant}`,
      percent: timeSeriesData?.traffic_split?.variant_percentages?.[variant] || 0,
      users: timeSeriesData?.traffic_split?.variant_counts?.[variant] || 0,
    }));

    return (
      <>
        {renderChartDescription('traffic')}
        <Card title="Распределение трафика по вариантам" size="small" style={{ marginBottom: 16 }}>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={trafficData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="variant" label={{ value: 'Вариант', position: 'insideBottom', offset: -5 }} />
              <YAxis label={{ value: 'Доля трафика, %', angle: -90, position: 'insideLeft' }} />
              <Tooltip
                formatter={(value: number) => [
                  `${Number(value).toFixed(2)}%`,
                  'Доля трафика',
                ]}
                labelFormatter={(label) => `${label}`}
              />
              <Legend />
              <Bar dataKey="percent" fill="#1890ff" name="Доля трафика, %" />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </>
    );
  };

  const renderResultsTable = () => {
    if (!timeSeriesData || chartData.length === 0) return null;

    const finalData = chartData[chartData.length - 1];
    const controlVariant = timeSeriesData.variants[0];
    const controlMetric = finalData?.[controlVariant]?.mean_metric || 0;

    const tableData = timeSeriesData.variants.map((variant) => {
      const variantData = finalData?.[variant];
      const variantMean = Number(variantData?.mean_metric ?? 0);
      const variantCum = Number(variantData?.cumulative_metric ?? 0);
      const variantSample = Number(variantData?.sample_size ?? 0);
      const variantPValueRaw = variantData?.p_value ?? null;
      const variantPValueCorrected = variant === controlVariant
        ? null
        : (timeSeriesData?.p_values_corrected_latest?.[variant] ?? null);

      const uplift = controlMetric > 0
        ? ((variantMean - controlMetric) / controlMetric * 100)
        : 0;

      const significanceSource = variantPValueCorrected ?? variantPValueRaw;

      return {
        variant,
        sample_size: variantSample,
        mean_metric: variantMean.toFixed(4),
        cumulative_metric: variantCum.toFixed(2),
        uplift: uplift.toFixed(2),
        p_value_raw: variantPValueRaw !== null && variantPValueRaw !== undefined ? Number(variantPValueRaw).toFixed(4) : 'Н/Д',
        p_value_corrected: variantPValueCorrected !== null && variantPValueCorrected !== undefined ? Number(variantPValueCorrected).toFixed(4) : 'Н/Д',
        significant: significanceSource !== null && significanceSource !== undefined && significanceSource < 0.05,
        is_control: variant === controlVariant,
      };
    });

    const columns = [
      {
        title: 'Вариант',
        dataIndex: 'variant',
        key: 'variant',
        render: (variant: string, record: any) => (
          <Space>
            <Tag color={getVariantColor(variant)}>{variant}</Tag>
            {record.is_control && <Tag color="default">контроль</Tag>}
          </Space>
        ),
      },
      {
        title: 'Размер выборки',
        dataIndex: 'sample_size',
        key: 'sample_size',
        render: (v: number) => v.toLocaleString('ru-RU'),
      },
      {
        title: 'Средняя метрика',
        dataIndex: 'mean_metric',
        key: 'mean_metric',
        render: (v: string) => <Text strong>{v}</Text>,
      },
      {
        title: 'Накопленная метрика',
        dataIndex: 'cumulative_metric',
        key: 'cumulative_metric',
      },
      {
        title: 'Прирост относительно контроля',
        dataIndex: 'uplift',
        key: 'uplift',
        render: (uplift: string, record: any) => {
          if (record.is_control) return <Text type="secondary">—</Text>;
          return (
            <Text type={parseFloat(uplift) > 0 ? 'success' : 'danger'} strong>
              {parseFloat(uplift) > 0 ? '+' : ''}{uplift}%
            </Text>
          );
        },
      },
      {
        title: 'p-значение (метод Холма, скорректированное)',
        dataIndex: 'p_value_corrected',
        key: 'p_value_corrected',
        render: (v: string, record: any) => {
          if (record.is_control) return <Text type="secondary">—</Text>;
          return <Text strong>{v}</Text>;
        },
      },
      {
        title: 'p-значение (сырое)',
        dataIndex: 'p_value_raw',
        key: 'p_value_raw',
        render: (v: string, record: any) => {
          if (record.is_control) return <Text type="secondary">—</Text>;
          return <Text type="secondary">{v}</Text>;
        },
      },
      {
        title: 'Статистическая значимость',
        key: 'significant',
        render: (_: any, record: any) => {
          if (record.is_control) return <Tag color="default">Контроль</Tag>;
          return record.significant ? (
            <Tag icon={<CheckCircleOutlined />} color="success">Значимо (скорректированное p &lt; 0.05)</Tag>
          ) : (
            <Tag icon={<WarningOutlined />} color="warning">Незначимо (скорректированное p ≥ 0.05)</Tag>
          );
        },
      },
    ];

    return (
      <Card
        title="Итоговая таблица результатов A/B теста"
        size="small"
        extra={
          <Text type="secondary" style={{ fontSize: 12 }}>
            Данные на момент последней проверки
          </Text>
        }
      >
        <Alert
          type="info"
          showIcon
          style={{ marginBottom: 12 }}
          message="Как интерпретировать таблицу"
          description={
            <div>
              <Text style={{ fontSize: 12 }}>
                <strong>Средняя метрика</strong> — среднее значение на пользователя для каждого варианта.
                <strong> Прирост</strong> — процентный прирост относительно контроля (A).
                <strong> скорректированное p-значение (метод Холма) &lt; 0.05</strong> — основной критерий статистической значимости.
                <strong> сырое p-значение</strong> — мониторинговый сигнал; в многовариантных сценариях не используется как финальный критерий решения.
              </Text>
            </div>
          }
        />
        <Table
          columns={columns}
          dataSource={tableData}
          rowKey="variant"
          pagination={false}
          size="small"
        />
      </Card>
    );
  };

  const selectedTest = tests.find((t) => t.test_id === selectedTestId);

  const confidenceLabel: Record<string, { label: string; color: string }> = {
    low: { label: 'Низкая', color: 'red' },
    medium: { label: 'Средняя', color: 'orange' },
    high: { label: 'Высокая', color: 'green' },
  };

  const getAnalysisModeLabel = (mode?: string) => {
    if (mode === 'adaptive_bandit') return 'Адаптивный';
    if (mode === 'fixed_experiment') return 'Фиксированный';
    return mode || '—';
  };

  const formatEarlyStopReason = (reason: string | null): string => {
    if (!reason) return 'не указана';

    const observedEffectMatch = reason.match(
      /Observed effect\s*\(([-+]?\d+(?:\.\d+)?%)\)\s*<<\s*target MDE\s*\(([-+]?\d+(?:\.\d+)?%)\)/i,
    );

    if (observedEffectMatch) {
      const [, observedEffect, targetMde] = observedEffectMatch;
      return `Наблюдаемый эффект (${observedEffect}) значительно меньше целевого MDE (${targetMde})`;
    }

    const lookBoundaryMatch = reason.match(/Crossed O'Brien-Fleming boundary at look\s*(\d+)/i);
    if (lookBoundaryMatch) {
      return `Пересечена граница O'Brien-Fleming на проверке №${lookBoundaryMatch[1]}`;
    }

    if (reason === 'Too early for futility check') return 'Слишком рано для проверки на бесперспективность';
    if (reason === 'Continue experiment') return 'Продолжайте эксперимент';
    if (reason === 'Reached max looks, final check') return 'Достигнуто максимальное число проверок, выполнена финальная проверка';
    if (reason === 'No significance at max looks') return 'На финальной проверке статистическая значимость не достигнута';
    if (reason === 'No significance yet') return 'Статистическая значимость пока не достигнута';

    return reason;
  };

  return (
    <div style={{ padding: '20px' }}>
      <Title level={2}> Результаты A/B тестов</Title>

      {isSimulating && (
        <Alert
          message="🔄 Симуляция запущена"
          description="Данные обновляются в реальном времени каждые 3 секунды."
          type="info"
          showIcon
          style={{ marginBottom: 16 }}
        />
      )}

      <Card style={{ marginBottom: 16 }}>
        <Row gutter={[16, 16]} align="middle">
          <Col>
            <Text strong>Выберите тест:</Text>
          </Col>
          <Col flex="auto">
            <Select
              value={selectedTestId}
              onChange={handleTestChange}
              style={{ width: '100%' }}
              placeholder="Выберите A/B тест для просмотра результатов"
            >
              {tests.map((test) => (
                <Option key={test.test_id} value={test.test_id}>
                  {test.test_name} ({test.status === 'completed' ? 'завершён' : test.status === 'active' ? 'активен' : test.status}) — метрика: {test.primary_metric}
                </Option>
              ))}
            </Select>
          </Col>
        </Row>
      </Card>

      {!selectedTestId ? (
        <Empty description="Выберите тест для просмотра результатов" image={Empty.PRESENTED_IMAGE_SIMPLE} />
      ) : loading ? (
        <div style={{ textAlign: 'center', padding: 40 }}>
          <Spin size="large" tip="Загрузка результатов..." />
        </div>
      ) : !timeSeriesData || !timeSeriesData.data || timeSeriesData.data.length === 0 ? (
        <Empty
          description={
            isSimulating ? (
              <div>
                <Spin size="small" />
                <div style={{ marginTop: 8 }}>Симуляция запущена. Ожидание данных...</div>
              </div>
            ) : (
              'Нет данных для этого теста. Запустите симуляцию на странице управления тестами.'
            )
          }
          image={Empty.PRESENTED_IMAGE_SIMPLE}
        />
      ) : (
        <>
          {timeSeriesData.stopped_early && (
            <Alert
              style={{ marginBottom: 16 }}
              type="warning"
              showIcon
              message="⏹️ Тест был остановлен досрочно"
              description={`Причина: ${formatEarlyStopReason(timeSeriesData.early_stop_reason)} | Промежуточная проверка: ${timeSeriesData.current_sequential_look} из ${timeSeriesData.max_sequential_looks}`}
            />
          )}

          {timeSeriesData.analysis_mode === 'adaptive_bandit' && (
            <Alert
              style={{ marginBottom: 16 }}
              type="warning"
              showIcon
              message="Режим адаптивного бандита"
              description="Результаты используются только для исследования. Для финального причинно-следственного решения требуется фиксированный эксперимент."
            />
          )}

          {timeSeriesData.analysis_validity && timeSeriesData.analysis_validity !== 'valid_for_inference' && (
            <Alert
              style={{ marginBottom: 16 }}
              type="error"
              showIcon
              message={`Статус валидности: ${timeSeriesData.analysis_validity}`}
              description="Автоматическая валидация пометила результаты как невалидные для итогового статистического вывода."
            />
          )}

          {timeSeriesData.guardrails?.enabled && !timeSeriesData.guardrails?.passed && (
            <Alert
              style={{ marginBottom: 16 }}
              type="error"
              showIcon
              message="Guardrails нарушены"
              description={`Нарушенные метрики: ${(timeSeriesData.guardrails.failed_metrics || []).join(', ') || 'не указаны'}`}
            />
          )}

          {/* Ключевые метрики */}
          <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
            <Col span={6}>
              <Card size="small">
                <Statistic
                  title="Всего пользователей"
                  value={timeSeriesData.data[timeSeriesData.data.length - 1]?.users_processed || 0}
                  formatter={(v) => Number(v).toLocaleString('ru-RU')}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card size="small">
                <Statistic title="Вариантов в тесте" value={timeSeriesData.variants.length} />
              </Card>
            </Col>
            <Col span={6}>
              <Card size="small">
                <Statistic title="Точек измерения" value={timeSeriesData.total_snapshots} />
              </Card>
            </Col>
            <Col span={6}>
              <Card size="small">
                <Statistic
                  title="Завершённость теста"
                  value={timeSeriesData.completion_percentage ?? selectedTest?.completion_percentage ?? 0}
                  suffix="%"
                  precision={1}
                />
              </Card>
            </Col>
          </Row>

          {/* Единый блок рекомендаций и валидности */}
          <Card
            size="small"
            style={{ marginBottom: 16, borderRadius: 12 }}
            title="🚀 Рекомендация к внедрению"
            extra={
              <Tag color={timeSeriesData.winner ? 'success' : 'default'} style={{ fontWeight: 600 }}>
                {timeSeriesData.winner ? 'Решение готово' : 'Недостаточно оснований для внедрения'}
              </Tag>
            }
          >
            <Row gutter={[16, 16]} align="middle" style={{ marginBottom: 12 }}>
              <Col xs={24} md={10}>
                <Card
                  size="small"
                  style={{
                    background: 'linear-gradient(135deg, #f6ffed 0%, #ffffff 100%)',
                    borderColor: '#b7eb8f',
                    borderRadius: 10,
                  }}
                >
                  {timeSeriesData.winner ? (
                    <Space direction="vertical" size={4}>
                      <Text type="secondary">Победитель</Text>
                      <Tag color="success" style={{ fontSize: 16, padding: '6px 12px', width: 'fit-content' }}>
                        Вариант {timeSeriesData.winner}
                      </Tag>
                      <Text>
                        Прирост:{' '}
                        <Text strong type="success">+{timeSeriesData.winner_uplift_percent.toFixed(2)}%</Text>
                      </Text>
                      <Text>
                        Уровень уверенности:{' '}
                        <Tag color={confidenceLabel[timeSeriesData.winner_confidence]?.color}>
                          {confidenceLabel[timeSeriesData.winner_confidence]?.label || timeSeriesData.winner_confidence}
                        </Tag>
                      </Text>
                    </Space>
                  ) : (
                    <Space direction="vertical" size={4}>
                      <Text type="secondary">Победитель</Text>
                      <Tag color="default">Не определён</Tag>
                      <Text type="secondary" style={{ fontSize: 12 }}>
                        Недостаточно данных или статистической значимости
                      </Text>
                    </Space>
                  )}
                </Card>
              </Col>

              <Col xs={24} md={14}>
                <Row gutter={[12, 12]}>
                  <Col xs={24} sm={12}>
                    <Card size="small" style={{ borderRadius: 10 }}>
                      <Statistic
                        title="Валидность анализа"
                        value={timeSeriesData.analysis_validity === 'valid_for_inference' ? 'Валиден' : 'Ограничен'}
                        valueStyle={{
                          color: timeSeriesData.analysis_validity === 'valid_for_inference' ? '#389e0d' : '#cf1322',
                          fontSize: 18,
                        }}
                      />
                      <Text type="secondary" style={{ fontSize: 12 }}>
                        {timeSeriesData.analysis_validity === 'valid_for_inference'
                          ? 'Для итогового статистического вывода'
                          : 'Требуется дополнительная проверка'}
                      </Text>
                    </Card>
                  </Col>

                  <Col xs={24} sm={12}>
                    <Card size="small" style={{ borderRadius: 10 }}>
                      <Statistic
                        title="Равномерность распределения (SRM)"
                        value={
                          timeSeriesData.srm_check_passed === null
                            ? 'Нет данных'
                            : timeSeriesData.srm_check_passed
                              ? 'Норма'
                              : 'Нарушение'
                        }
                        valueStyle={{
                          color:
                            timeSeriesData.srm_check_passed === null
                              ? '#8c8c8c'
                              : timeSeriesData.srm_check_passed
                                ? '#389e0d'
                                : '#cf1322',
                          fontSize: 18,
                        }}
                      />
                      <Text type="secondary" style={{ fontSize: 12 }}>
                        p = {timeSeriesData.srm_p_value?.toFixed(4) || 'Н/Д'}
                      </Text>
                    </Card>
                  </Col>

                  <Col xs={24} sm={12}>
                    <Card size="small" style={{ borderRadius: 10 }}>
                      <Statistic
                        title="Guardrails"
                        value={timeSeriesData.guardrails?.enabled ? (timeSeriesData.guardrails?.passed ? 'Соблюдены' : 'Нарушены') : 'Не включены'}
                        valueStyle={{
                          color: !timeSeriesData.guardrails?.enabled
                            ? '#8c8c8c'
                            : timeSeriesData.guardrails?.passed
                              ? '#389e0d'
                              : '#cf1322',
                          fontSize: 18,
                        }}
                      />
                      <Text type="secondary" style={{ fontSize: 12 }}>
                        {timeSeriesData.guardrails?.enabled
                          ? ((timeSeriesData.guardrails?.failed_metrics || []).length > 0
                            ? `Нарушения: ${(timeSeriesData.guardrails?.failed_metrics || []).join(', ')}`
                            : 'Защитные ограничения соблюдены')
                          : 'Защитные ограничения не активированы'}
                      </Text>
                    </Card>
                  </Col>

                  <Col xs={24} sm={12}>
                    <Card size="small" style={{ borderRadius: 10 }}>
                      <Statistic
                        title="Общая оценка качества данных"
                        value={
                          timeSeriesData.quality_gate?.status === 'green'
                            ? 'Высокая'
                            : timeSeriesData.quality_gate?.status === 'yellow'
                              ? 'Средняя'
                              : timeSeriesData.quality_gate?.status === 'red'
                                ? 'Низкая'
                                : 'Нет данных'
                        }
                        valueStyle={{
                          color:
                            timeSeriesData.quality_gate?.status === 'green'
                              ? '#389e0d'
                              : timeSeriesData.quality_gate?.status === 'red'
                                ? '#cf1322'
                                : '#d48806',
                          fontSize: 18,
                        }}
                      />
                      <Text type="secondary" style={{ fontSize: 12 }}>
                        {timeSeriesData.quality_gate
                          ? `Проверки: ${timeSeriesData.quality_gate.passed_checks}/${timeSeriesData.quality_gate.total_checks}`
                          : 'Контроль качества ещё не рассчитан'}
                      </Text>
                    </Card>
                  </Col>
                </Row>
              </Col>
            </Row>

            <Alert
              type={timeSeriesData.winner && timeSeriesData.analysis_validity === 'valid_for_inference' && (timeSeriesData.srm_check_passed !== false) && (timeSeriesData.guardrails?.enabled ? timeSeriesData.guardrails?.passed : true) ? 'success' : 'warning'}
              showIcon
              message={timeSeriesData.winner
                ? `Победитель: вариант ${timeSeriesData.winner}`
                : 'Победитель пока не определён'}
              description={
                <Space direction="vertical" size={2}>
                  <Text>Скорректированное p-значение (метод Холма): {timeSeriesData.winner ? (timeSeriesData.p_values_corrected_latest?.[timeSeriesData.winner]?.toFixed(4) ?? 'Н/Д') : 'Н/Д'}</Text>
                  <Text>Режим A/B теста: {getAnalysisModeLabel(timeSeriesData.analysis_mode || 'fixed_experiment')}</Text>
                </Space>
              }
            />

            <div style={{ marginTop: 12 }}>
              <Card size="small" title="🔄 Последовательные проверки" style={{ borderRadius: 10 }}>
                <Space direction="vertical" size={6} style={{ width: '100%' }}>
                  <Tag color={timeSeriesData.early_stopping_enabled ? 'green' : 'default'}>
                    Ранняя остановка: {timeSeriesData.early_stopping_enabled ? 'включена' : 'выключена'}
                  </Tag>
                  <Progress
                    percent={Math.round((timeSeriesData.current_sequential_look / Math.max(1, timeSeriesData.max_sequential_looks)) * 100)}
                    format={() => `${timeSeriesData.current_sequential_look} / ${timeSeriesData.max_sequential_looks}`}
                  />
                </Space>
              </Card>
            </div>
          </Card>

          {/* Переключатель графиков */}
          <Card style={{ marginBottom: 16 }}>
            <Row gutter={[16, 16]} align="middle" wrap>
              <Col>
                <Text strong>Тип графика:</Text>
              </Col>
              <Col flex="auto">
                <Select
                  value={chartType}
                  onChange={setChartType}
                  style={{ width: 300 }}
                >
                  <Option value="cumulative">📈 Накопленная метрика</Option>
                  <Option value="mean">📊 Средняя метрика</Option>
                  <Option value="ci">🎯 Доверительные интервалы (ДИ)</Option>
                  <Option value="pvalue">📉 p-значение (сырое, мониторинг)</Option>
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
                  <InfoCircleOutlined /> {showChartInfo ? 'Скрыть описание' : 'Показать описание графика'}
                </Tag>
              </Col>
            </Row>
          </Card>

          {chartType === 'cumulative' && renderCumulativeChart()}
          {chartType === 'mean' && renderMeanMetricChart()}
          {chartType === 'ci' && renderConfidenceIntervalsChart()}
          {chartType === 'pvalue' && renderPValueChart()}
          {chartType === 'uplift' && renderUpliftChart()}
          {chartType === 'power' && renderPowerChart()}
          {chartType === 'traffic' && renderTrafficSplitChart()}

          {renderResultsTable()}
        </>
      )}
    </div>
  );
};
