// frontend/src/components/Results/ResultsPage.tsx
import React, { useState, useEffect, useRef } from 'react';
import { Card, Row, Col, Select, Typography, Table, Tag, Statistic, Spin, Alert, Empty, Progress } from 'antd';
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
import {
  CheckCircleOutlined,
  WarningOutlined,
} from '@ant-design/icons';

const { Option } = Select;
const { Title, Text } = Typography;

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

interface TimeSeriesResponse {
  test_id: string;
  variants: string[];
  data: TimeSeriesDataPoint[];
  total_snapshots: number;
  snapshots_per_variant: number;
  completion_percentage: number;
  stopped_early: boolean;
  early_stop_reason: string | null;
  current_sequential_look: number;
  max_sequential_looks: number;
  srm_check_passed: number | null;
  srm_p_value: number | null;
  traffic_split: {
    variant_counts: Record<string, number>;
    variant_percentages: Record<string, number>;
  };
  winner: string | null;
  winner_uplift_percent: number;
  winner_confidence: 'low' | 'medium' | 'high';
  power_over_time: Array<Record<string, number>>;
  uplift_over_time: Array<Record<string, number>>;
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

export const ResultsPage: React.FC = () => {
  const [tests, setTests] = useState<TestSummary[]>([]);
  const [selectedTestId, setSelectedTestId] = useState<string>('');
  const [timeSeriesData, setTimeSeriesData] = useState<TimeSeriesResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [chartType, setChartType] = useState<'cumulative' | 'mean' | 'ci' | 'pvalue' | 'uplift' | 'power' | 'traffic'>('cumulative');
  const [isSimulating, setIsSimulating] = useState(false);
  const pollingRef = useRef<number | null>(null);

  useEffect(() => {
    loadTests();

    const testsInterval = setInterval(() => {
      loadTests();
    }, 5000);

    return () => {
      if (pollingRef.current) {
        window.clearInterval(pollingRef.current);
      }
      if (testsInterval) {
        window.clearInterval(testsInterval);
      }
    };
  }, []);

  useEffect(() => {
    if (pollingRef.current) {
      window.clearInterval(pollingRef.current);
    }

    if (selectedTestId && isSimulating) {
      pollingRef.current = window.setInterval(() => {
        loadTimeSeriesData(selectedTestId, false);
      }, 3000);
    } else if (selectedTestId) {
      pollingRef.current = window.setInterval(() => {
        loadTimeSeriesData(selectedTestId, false);
      }, 7000);
    }

    return () => {
      if (pollingRef.current) {
        window.clearInterval(pollingRef.current);
      }
    };
  }, [selectedTestId, isSimulating]);

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
      console.error('Error loading tests:', error);
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
      console.error('Error loading time series data:', error);
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

  const renderCumulativeChart = () => (
    <Card title="📈 Накопленная метрика" size="small" style={{ marginBottom: 16 }}>
      <ResponsiveContainer width="100%" height={380}>
        <AreaChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="users_processed" />
          <YAxis />
          <Tooltip labelFormatter={(label) => `Пользователей: ${label}`} />
          <Legend />
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
    <Card title="📊 Средняя метрика" size="small" style={{ marginBottom: 16 }}>
      <ResponsiveContainer width="100%" height={380}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="users_processed" />
          <YAxis />
          <Tooltip labelFormatter={(label) => `Пользователей: ${label}`} />
          <Legend />
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

  const renderConfidenceIntervalsChart = () => {
    if (!timeSeriesData) return null;
    return (
      <Card title="🎯 Confidence Intervals" size="small" style={{ marginBottom: 16 }}>
        <ResponsiveContainer width="100%" height={380}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="users_processed" />
            <YAxis />
            <Tooltip labelFormatter={(label) => `Пользователей: ${label}`} />
            <Legend />
            {timeSeriesData.variants.map((variant) => (
              <React.Fragment key={variant}>
                <Line
                  type="monotone"
                  dataKey={(data: any) => data[variant]?.confidence_interval_lower ?? null}
                  name={`${variant} CI lower`}
                  stroke={getVariantColor(variant)}
                  strokeDasharray="4 4"
                  dot={false}
                  connectNulls
                />
                <Line
                  type="monotone"
                  dataKey={(data: any) => data[variant]?.confidence_interval_upper ?? null}
                  name={`${variant} CI upper`}
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
      <Card title="📉 P-value" size="small" style={{ marginBottom: 16 }}>
        <ResponsiveContainer width="100%" height={380}>
          <LineChart data={pValueData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="users_processed" />
            <YAxis domain={[0, 1]} />
            <Tooltip labelFormatter={(label) => `Пользователей: ${label}`} />
            <Legend />
            <Line dataKey="threshold" name="Порог 0.05" stroke="#ff4d4f" strokeDasharray="5 5" dot={false} />
            {timeSeriesData?.variants
              .filter((v) => v !== timeSeriesData.variants[0])
              .map((variant) => (
                <Line
                  key={variant}
                  type="monotone"
                  dataKey={variant}
                  name={`${variant} vs ${timeSeriesData.variants[0]}`}
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
    <Card title="🚀 Uplift % vs Control" size="small" style={{ marginBottom: 16 }}>
      <ResponsiveContainer width="100%" height={380}>
        <LineChart data={timeSeriesData?.uplift_over_time || []}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="users_processed" />
          <YAxis />
          <Tooltip labelFormatter={(label) => `Пользователей: ${label}`} />
          <Legend />
          {(timeSeriesData?.variants || [])
            .filter((v) => v !== timeSeriesData?.variants?.[0])
            .map((variant) => (
              <Line
                key={variant}
                type="monotone"
                dataKey={variant}
                name={`${variant} uplift %`}
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
    <Card title="⚡ Statistical Power" size="small" style={{ marginBottom: 16 }}>
      <ResponsiveContainer width="100%" height={380}>
        <LineChart data={timeSeriesData?.power_over_time || []}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="users_processed" />
          <YAxis domain={[0, 1]} />
          <Tooltip labelFormatter={(label) => `Пользователей: ${label}`} />
          <Legend />
          {(timeSeriesData?.variants || [])
            .filter((v) => v !== timeSeriesData?.variants?.[0])
            .map((variant) => (
              <Line
                key={variant}
                type="monotone"
                dataKey={variant}
                name={`${variant} power`}
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
      variant,
      percent: timeSeriesData?.traffic_split?.variant_percentages?.[variant] || 0,
      users: timeSeriesData?.traffic_split?.variant_counts?.[variant] || 0,
    }));

    return (
      <Card title="🛣️ Traffic Split" size="small" style={{ marginBottom: 16 }}>
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={trafficData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="variant" />
            <YAxis />
            <Tooltip formatter={(value: number) => [`${Number(value).toFixed(2)}%`, 'Доля трафика']} />
            <Legend />
            <Bar dataKey="percent" fill="#1890ff" name="Доля, %" />
          </BarChart>
        </ResponsiveContainer>
      </Card>
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
      const variantPValue = variantData?.p_value ?? null;

      const uplift = controlMetric > 0
        ? ((variantMean - controlMetric) / controlMetric * 100)
        : 0;

      return {
        variant,
        sample_size: variantSample,
        mean_metric: variantMean.toFixed(4),
        cumulative_metric: variantCum.toFixed(2),
        uplift: uplift.toFixed(2),
        p_value: variantPValue !== null && variantPValue !== undefined ? Number(variantPValue).toFixed(4) : 'N/A',
        significant: variantPValue !== null && variantPValue !== undefined && variantPValue < 0.05,
      };
    });

    const columns = [
      {
        title: 'Вариант',
        dataIndex: 'variant',
        key: 'variant',
        render: (variant: string) => (
          <Tag color={getVariantColor(variant)}>{variant}</Tag>
        ),
      },
      {
        title: 'Размер выборки',
        dataIndex: 'sample_size',
        key: 'sample_size',
      },
      {
        title: 'Средняя метрика',
        dataIndex: 'mean_metric',
        key: 'mean_metric',
      },
      {
        title: 'Накопленная метрика',
        dataIndex: 'cumulative_metric',
        key: 'cumulative_metric',
      },
      {
        title: 'Uplift vs Контроль',
        dataIndex: 'uplift',
        key: 'uplift',
        render: (uplift: string) => (
          <Text type={parseFloat(uplift) > 0 ? 'success' : 'danger'}>
            {parseFloat(uplift) > 0 ? '+' : ''}{uplift}%
          </Text>
        ),
      },
      {
        title: 'P-value',
        dataIndex: 'p_value',
        key: 'p_value',
      },
      {
        title: 'Значимость',
        key: 'significant',
        render: (_: any, record: any) => (
          record.significant ? (
            <Tag icon={<CheckCircleOutlined />} color="success">Значимо</Tag>
          ) : (
            <Tag icon={<WarningOutlined />} color="warning">Не значимо</Tag>
          )
        ),
      },
    ];

    return (
      <Card title="📋 Финальные результаты" size="small">
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

  return (
    <div style={{ padding: '20px' }}>
      <Title level={2}>📊 Результаты A/B тестов</Title>

      {isSimulating && (
        <Alert
          message="🔄 Симуляция запущена"
          description="Данные обновляются в реальном времени."
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
                  {test.test_name} ({test.status}) — {test.primary_metric}
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
          <Spin size="large" />
        </div>
      ) : !timeSeriesData || !timeSeriesData.data || timeSeriesData.data.length === 0 ? (
        <Empty
          description={
            isSimulating ? (
              <div>
                <Spin size="small" />
                <div style={{ marginTop: 8 }}>Симуляция запущена. Загрузка данных...</div>
              </div>
            ) : (
              'Нет данных временных рядов для этого теста. Запустите симуляцию.'
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
              message="Тест был остановлен досрочно"
              description={`Причина: ${timeSeriesData.early_stop_reason || 'N/A'}, look: ${timeSeriesData.current_sequential_look}/${timeSeriesData.max_sequential_looks}`}
            />
          )}

          <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
            <Col span={6}>
              <Card size="small">
                <Statistic title="Всего пользователей" value={timeSeriesData.data[timeSeriesData.data.length - 1]?.users_processed || 0} />
              </Card>
            </Col>
            <Col span={6}>
              <Card size="small">
                <Statistic title="Вариантов" value={timeSeriesData.variants.length} />
              </Card>
            </Col>
            <Col span={6}>
              <Card size="small">
                <Statistic title="Срезов данных" value={timeSeriesData.total_snapshots} />
              </Card>
            </Col>
            <Col span={6}>
              <Card size="small">
                <Statistic title="Завершено" value={timeSeriesData.completion_percentage ?? selectedTest?.completion_percentage ?? 0} suffix="%" />
              </Card>
            </Col>
          </Row>

          <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
            <Col span={8}>
              <Card size="small" title="🏆 Winner">
                {timeSeriesData.winner ? (
                  <Tag color="success">{timeSeriesData.winner} (+{timeSeriesData.winner_uplift_percent.toFixed(2)}%), confidence: {timeSeriesData.winner_confidence}</Tag>
                ) : (
                  <Tag color="default">Пока нет победителя</Tag>
                )}
              </Card>
            </Col>
            <Col span={8}>
              <Card size="small" title="🧪 Sequential Testing">
                <Progress
                  percent={Math.round((timeSeriesData.current_sequential_look / Math.max(1, timeSeriesData.max_sequential_looks)) * 100)}
                  format={() => `${timeSeriesData.current_sequential_look}/${timeSeriesData.max_sequential_looks}`}
                />
                <Text type="secondary">
                  Выполнено промежуточных проверок значимости (look) из запланированных.
                </Text>
              </Card>
            </Col>
            <Col span={8}>
              <Card size="small" title="🛡️ SRM Check">
                {timeSeriesData.srm_check_passed === null ? (
                  <Tag color="default">Нет данных</Tag>
                ) : timeSeriesData.srm_check_passed ? (
                  <Tag color="success">Passed (p={timeSeriesData.srm_p_value?.toFixed(4) || 'N/A'})</Tag>
                ) : (
                  <Tag color="error">Failed (p={timeSeriesData.srm_p_value?.toFixed(4) || 'N/A'})</Tag>
                )}
                <Text type="secondary">
                  Проверка равномерности распределения трафика по вариантам.
                </Text>
              </Card>
            </Col>
          </Row>

          <Card style={{ marginBottom: 16 }}>
            <Row gutter={[16, 16]} align="middle">
              <Col>
                <Text strong>Тип графика:</Text>
              </Col>
              <Col>
                <Select
                  value={chartType}
                  onChange={setChartType}
                  style={{ width: 260 }}
                >
                  <Option value="cumulative">Накопленная метрика</Option>
                  <Option value="mean">Средняя метрика</Option>
                  <Option value="ci">Confidence intervals</Option>
                  <Option value="pvalue">P-value</Option>
                  <Option value="uplift">Uplift vs control</Option>
                  <Option value="power">Statistical power</Option>
                  <Option value="traffic">Traffic split</Option>
                </Select>
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
