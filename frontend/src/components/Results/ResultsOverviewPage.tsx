import React, { useMemo } from 'react';
import { Alert, Col, Row, Space, Table, Tag, Typography } from 'antd';
import { CheckCircleOutlined, WarningOutlined } from '@ant-design/icons';
import { useOutletContext } from 'react-router-dom';
import { useTheme } from '@/context/ThemeContext';
import type { ResultsOutletContext } from './ResultsSectionPage';

const { Text } = Typography;

const confidenceLabel: Record<string, { label: string; color: string }> = {
  low: { label: 'Низкая', color: 'red' },
  medium: { label: 'Средняя', color: 'orange' },
  high: { label: 'Высокая', color: 'green' },
};

const recommendationLabel: Record<string, { label: string; color: string }> = {
  deploy: { label: 'Внедрять', color: 'success' },
  do_not_deploy: { label: 'Не внедрять', color: 'error' },
  need_more_data: { label: 'Требуется больше данных', color: 'warning' },
};

export const ResultsOverviewPage: React.FC = () => {
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
      successSoft: isDark ? 'rgba(34,197,94,0.12)' : '#ecfdf3',
      dangerSoft: isDark ? 'rgba(239,68,68,0.12)' : '#fef2f2',
      infoSoft: isDark ? 'rgba(59,130,246,0.12)' : '#eff6ff',
      shadow: isDark ? '0 12px 38px rgba(0,0,0,0.38)' : '0 10px 30px rgba(28,25,23,0.08)',
    }),
    [isDark],
  );

  const chartData = useMemo(() => {
    if (!timeSeriesData?.data) return [];

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
        title: 'Среднее значение',
        dataIndex: 'mean_metric',
        key: 'mean_metric',
        render: (v: string) => <Text strong>{v}</Text>,
      },
      {
        title: 'Накопленное значение',
        dataIndex: 'cumulative_metric',
        key: 'cumulative_metric',
      },
      {
        title: 'Прирост к контролю',
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
        title: 'Скорректированное p-значение (Холм)',
        dataIndex: 'p_value_corrected',
        key: 'p_value_corrected',
        render: (v: string, record: any) => {
          if (record.is_control) return <Text type="secondary">—</Text>;
          return <Text strong>{v}</Text>;
        },
      },
      {
        title: 'Сырое p-значение',
        dataIndex: 'p_value_raw',
        key: 'p_value_raw',
        render: (v: string, record: any) => {
          if (record.is_control) return <Text type="secondary">—</Text>;
          return <Text type="secondary">{v}</Text>;
        },
      },
      {
        title: 'Значимость',
        key: 'significant',
        render: (_: any, record: any) => {
          if (record.is_control) return <Tag color="default">Контроль</Tag>;
          return record.significant ? (
            <Tag icon={<CheckCircleOutlined />} color="success">Статистически значимо</Tag>
          ) : (
            <Tag icon={<WarningOutlined />} color="warning">Статистически незначимо</Tag>
          );
        },
      },
    ];

    return (
      <div style={{ borderRadius: 14, border: `1px solid ${c.border}`, backgroundColor: c.panelBg, boxShadow: c.shadow, overflow: 'hidden' }}>
        <div style={{ padding: '12px 14px', borderBottom: `1px solid ${c.border}`, backgroundColor: c.panelSoft, fontSize: 14, fontWeight: 700, color: c.textPrimary }}>
          Итоговая таблица результатов теста
        </div>
        <div style={{ padding: 14 }}>
          <Table
            columns={columns}
            dataSource={tableData}
            rowKey="variant"
            pagination={false}
            size="small"
            style={{ color: c.textPrimary }}
          />
        </div>
      </div>
    );
  };

  const recommendation = recommendationLabel[timeSeriesData?.recommendation_status || 'need_more_data']
    || recommendationLabel.need_more_data;

  return (
    <>

      <div style={{ color: c.textPrimary }}>
        <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
        <Col span={6}>
          <MetricCard title="Победитель" c={c}>
            {timeSeriesData?.winner ? (
              <Space direction="vertical" size={4}>
                <Tag color="success">Вариант {timeSeriesData.winner}</Tag>
                <Text strong type="success">Прирост: +{timeSeriesData.winner_uplift_percent.toFixed(2)}%</Text>
              </Space>
            ) : (
              <Tag color="default">Победитель не определён</Tag>
            )}
          </MetricCard>
        </Col>

        <Col span={6}>
          <MetricCard title="Уровень уверенности" c={c}>
            <Tag color={confidenceLabel[timeSeriesData?.winner_confidence || 'low']?.color || 'default'}>
              {confidenceLabel[timeSeriesData?.winner_confidence || 'low']?.label || 'Н/Д'}
            </Tag>
          </MetricCard>
        </Col>

        <Col span={6}>
          <MetricCard title="Валидность анализа" c={c}>
            <Tag color={timeSeriesData?.analysis_validity === 'valid_for_inference' ? 'success' : 'error'}>
              {timeSeriesData?.analysis_validity === 'valid_for_inference'
                ? 'Валиден для итогового вывода'
                : timeSeriesData?.analysis_validity === 'exploration_only'
                  ? 'Только исследовательский режим'
                  : timeSeriesData?.analysis_validity === 'invalid_srm'
                    ? 'Невалиден: перекос трафика'
                    : timeSeriesData?.analysis_validity === 'invalid_guardrails'
                      ? 'Невалиден: нарушены защитные метрики'
                      : 'Неизвестно'}
            </Tag>
          </MetricCard>
        </Col>

        <Col span={6}>
          <MetricCard title="Проверки корректности" c={c}>
            <Space direction="vertical" size={6} style={{ width: '100%' }}>
              <Tag color={timeSeriesData?.srm_check_passed ? 'success' : 'error'}>
                Равномерность распределения: {timeSeriesData?.srm_check_passed ? 'норма' : 'есть перекос'}
              </Tag>
              <Tag color={timeSeriesData?.guardrails?.passed ? 'success' : 'error'}>
                Защитные бизнес-ограничения: {timeSeriesData?.guardrails?.passed ? 'соблюдены' : 'нарушены'}
              </Tag>
              <Tag
                color={
                  timeSeriesData?.quality_gate?.status === 'green'
                    ? 'success'
                    : timeSeriesData?.quality_gate?.status === 'red'
                      ? 'error'
                      : 'warning'
                }
              >
                Общая оценка качества данных: {timeSeriesData?.quality_gate?.status === 'green'
                  ? 'высокая'
                  : timeSeriesData?.quality_gate?.status === 'red'
                    ? 'низкая'
                    : 'средняя'}
              </Tag>
            </Space>
          </MetricCard>
        </Col>
      </Row>

      <div style={{ borderRadius: 14, border: `1px solid ${c.border}`, backgroundColor: c.panelBg, boxShadow: c.shadow, overflow: 'hidden', marginBottom: 16 }}>
        <div style={{ padding: '12px 14px', borderBottom: `1px solid ${c.border}`, backgroundColor: c.panelSoft, fontSize: 14, fontWeight: 700, color: c.textPrimary }}>
          Рекомендация к внедрению
        </div>
        <div style={{ padding: 14 }}>
          <Space direction="vertical" size={10} style={{ width: '100%' }}>
            <Tag color={recommendation.color} style={{ width: 'fit-content', fontSize: 14, padding: '4px 12px' }}>
              {recommendation.label}
            </Tag>

            {(timeSeriesData?.recommendation_reason || []).length > 0 ? (
              <ul style={{ margin: 0, paddingLeft: 20 }}>
                {(timeSeriesData?.recommendation_reason || []).map((reason, idx) => (
                  <li key={`${reason}-${idx}`}>
                    <Text>{reason}</Text>
                  </li>
                ))}
              </ul>
            ) : (
              <Text type="secondary">Причины решения не предоставлены сервером</Text>
            )}

            {timeSeriesData?.rollout_hint && (
              <Alert
                type="info"
                showIcon
                message={`Подсказка по поэтапному запуску: ${timeSeriesData.rollout_hint}`}
              />
            )}
          </Space>
        </div>
      </div>

      {renderResultsTable()}
      </div>
    </>
  );
};

const MetricCard: React.FC<{ title: string; c: Record<string, string>; children: React.ReactNode }> = ({ title, c, children }) => (
  <div style={{ borderRadius: 12, border: `1px solid ${c.border}`, backgroundColor: c.panelBg, boxShadow: c.shadow, overflow: 'hidden', height: '100%' }}>
    <div style={{ padding: '10px 12px', borderBottom: `1px solid ${c.border}`, backgroundColor: c.panelSoft, fontSize: 13, fontWeight: 700, color: c.textPrimary }}>{title}</div>
    <div style={{ padding: 12, color: c.textPrimary }}>{children}</div>
  </div>
);
