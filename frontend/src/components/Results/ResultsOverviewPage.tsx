import React, { useMemo } from 'react';
import { Alert, Card, Col, Row, Space, Statistic, Table, Tag, Typography } from 'antd';
import { CheckCircleOutlined, WarningOutlined } from '@ant-design/icons';
import { useOutletContext } from 'react-router-dom';
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
  const { timeSeriesData, financialImpact } = useOutletContext<ResultsOutletContext>();

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
            <Tag icon={<CheckCircleOutlined />} color="success">Значимо (скорректированное p {'<'} 0.05)</Tag>
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
      >
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

  const recommendation = recommendationLabel[timeSeriesData?.recommendation_status || 'need_more_data']
    || recommendationLabel.need_more_data;

  return (
    <>
      <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
        <Col span={6}>
          <Card size="small" title="Победитель">
            {timeSeriesData?.winner ? (
              <Space direction="vertical" size={4}>
                <Tag color="success">Вариант {timeSeriesData.winner}</Tag>
                <Text strong type="success">Uplift: +{timeSeriesData.winner_uplift_percent.toFixed(2)}%</Text>
              </Space>
            ) : (
              <Tag color="default">Победитель не определён</Tag>
            )}
          </Card>
        </Col>

        <Col span={6}>
          <Card size="small" title="Confidence">
            <Tag color={confidenceLabel[timeSeriesData?.winner_confidence || 'low']?.color || 'default'}>
              {confidenceLabel[timeSeriesData?.winner_confidence || 'low']?.label || 'Н/Д'}
            </Tag>
          </Card>
        </Col>

        <Col span={6}>
          <Card size="small" title="Валидность анализа">
            <Tag color={timeSeriesData?.analysis_validity === 'valid_for_inference' ? 'success' : 'error'}>
              {timeSeriesData?.analysis_validity || 'unknown'}
            </Tag>
          </Card>
        </Col>

        <Col span={6}>
          <Card size="small" title="SRM / Guardrails / Quality Gate">
            <Space wrap>
              <Tag color={timeSeriesData?.srm_check_passed ? 'success' : 'error'}>
                SRM: {timeSeriesData?.srm_check_passed ? 'PASS' : 'FAIL'}
              </Tag>
              <Tag color={timeSeriesData?.guardrails?.passed ? 'success' : 'error'}>
                Guardrails: {timeSeriesData?.guardrails?.passed ? 'PASS' : 'FAIL'}
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
                QG: {(timeSeriesData?.quality_gate?.status || 'yellow').toUpperCase()}
              </Tag>
            </Space>
          </Card>
        </Col>
      </Row>

      <Card title="Рекомендация к внедрению" size="small" style={{ marginBottom: 16 }}>
        <Space direction="vertical" size={8} style={{ width: '100%' }}>
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
              message={`Подсказка по rollout: ${timeSeriesData.rollout_hint}`}
            />
          )}
        </Space>
      </Card>

      <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
        <Col span={8}>
          <Card size="small">
            <Statistic
              title="Инкрементальная выручка (significance-gated)"
              value={financialImpact?.financial_analysis?.incremental_revenue || 0}
              precision={2}
              suffix="₽"
            />
          </Card>
        </Col>
        <Col span={8}>
          <Card size="small">
            <Statistic
              title="Лучший наблюдаемый эффект"
              value={financialImpact?.financial_analysis?.best_observed_incremental_revenue || 0}
              precision={2}
              suffix="₽"
            />
          </Card>
        </Col>
        <Col span={8}>
          <Card size="small" title="Рекомендуемый вариант">
            <Tag color="blue">{financialImpact?.financial_analysis?.best_variant || 'Нет'}</Tag>
            <div style={{ marginTop: 8 }}>
              <Text type="secondary">Assumed ARPU: {financialImpact?.assumed_arpu ?? 100}</Text>
            </div>
          </Card>
        </Col>
      </Row>

      {renderResultsTable()}
    </>
  );
};
