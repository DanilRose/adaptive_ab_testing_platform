import React, { useMemo } from 'react';
import { Col, Row, Space, Table, Tag, Typography } from 'antd';
import { CheckCircleOutlined, RocketOutlined, WarningOutlined } from '@ant-design/icons';
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

const validityLabel: Record<string, { label: string; color: string }> = {
  valid_for_inference: { label: 'Валиден для итогового вывода', color: 'success' },
  exploration_only: { label: 'Только исследовательский режим', color: 'warning' },
  invalid_srm: { label: 'Невалиден: перекос трафика', color: 'error' },
  invalid_guardrails: { label: 'Невалиден: нарушены защитные метрики', color: 'error' },
};

const qualityLabel: Record<string, { label: string; color: string }> = {
  green: { label: 'Высокая', color: 'success' },
  yellow: { label: 'Средняя', color: 'warning' },
  red: { label: 'Низкая', color: 'error' },
};

type ValueTone = 'success' | 'warning' | 'danger' | 'secondary' | 'default';

const TONE_TO_TAG: Record<ValueTone, string> = {
  success: 'success',
  warning: 'warning',
  danger: 'error',
  secondary: 'default',
  default: 'default',
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

  const validity = validityLabel[timeSeriesData?.analysis_validity || 'exploration_only']
    || { label: 'Неизвестно', color: 'default' };

  const quality = qualityLabel[timeSeriesData?.quality_gate?.status || 'yellow']
    || { label: 'Средняя', color: 'warning' };

  const toTone = (color?: string): ValueTone => {
    if (!color) return 'default';
    if (['success', 'green'].includes(color)) return 'success';
    if (['warning', 'orange'].includes(color)) return 'warning';
    if (['error', 'red'].includes(color)) return 'danger';
    return color as ValueTone;
  };

  const correctedWinnerPValue =
    timeSeriesData?.winner && timeSeriesData?.p_values_corrected_latest?.[timeSeriesData.winner] !== undefined
      ? Number(timeSeriesData.p_values_corrected_latest[timeSeriesData.winner]).toFixed(4)
      : 'Н/Д';

  const winnerUplift =
    typeof timeSeriesData?.winner_uplift_percent === 'number'
      ? `${timeSeriesData.winner_uplift_percent >= 0 ? '+' : ''}${timeSeriesData.winner_uplift_percent.toFixed(2)}%`
      : 'Н/Д';

  const upliftTone: ValueTone = typeof timeSeriesData?.winner_uplift_percent === 'number'
    ? (timeSeriesData.winner_uplift_percent >= 0 ? 'success' : 'danger')
    : 'secondary';

  const pValueTone: ValueTone = correctedWinnerPValue !== 'Н/Д'
    ? (Number(correctedWinnerPValue) < 0.05 ? 'success' : 'danger')
    : 'secondary';

  const confidenceTone: ValueTone = toTone(confidenceLabel[timeSeriesData?.winner_confidence || 'low']?.color);

  const validityTone: ValueTone = toTone(validity.color);

  const srmTone: ValueTone = timeSeriesData?.srm_check_passed ? 'success' : 'danger';

  const guardrailsTone: ValueTone = timeSeriesData?.guardrails?.passed ? 'success' : 'danger';

  const qualityTone: ValueTone = toTone(quality.color);

  const rolloutTone: ValueTone = timeSeriesData?.rollout_hint ? 'warning' : 'secondary';

  const recommendationDetails = () => {
    if (timeSeriesData?.recommendation_status === 'deploy') {
      const rolloutHint = timeSeriesData?.rollout_hint ? `Рекомендуется поэтапная выкатка в прод: ${timeSeriesData.rollout_hint}.` : 'Можно внедрять сразу.';
      return ` ${rolloutHint} При запуске — мониторинг метрик и стоп-условия при ухудшении.`;
    }
    if (timeSeriesData?.recommendation_status === 'do_not_deploy') {
      return 'Внедрять не рекомендуется: прирост недостаточен/отрицательный либо есть риски по качеству данных и защитных ограничениях. Продолжайте тест или пересмотрите гипотезу.';
    }
    return 'Требуется больше данных: дождитесь достаточной статистики и подтвердите стабильность качества данных перед решением о внедрении.';
  };

  return (
    <div style={{ color: c.textPrimary }}>
      <div
        style={{
          borderRadius: 14,
          border: `1px solid ${c.border}`,
          backgroundColor: c.panelBg,
          boxShadow: c.shadow,
          overflow: 'hidden',
          marginBottom: 16,
        }}
      >
        <div style={{ padding: '12px 14px', borderBottom: `1px solid ${c.border}`, backgroundColor: c.panelSoft, fontSize: 14, fontWeight: 700, color: c.textPrimary }}>
          Итог эксперимента
        </div>
        <div style={{ padding: 16 }}>
          <Row gutter={[16, 16]} align="middle">
            <Col xs={24} md={14}>
              <Space direction="vertical" size={8}>
                <Text type="secondary" style={{ fontSize: 12 }}>Победитель</Text>
                {timeSeriesData?.winner ? (
                  <Space align="center" wrap>
                    <Tag color="success" style={{ fontSize: 15, padding: '4px 12px' }}>Вариант {timeSeriesData.winner}</Tag>
                    <Text strong style={{ fontSize: 24, color: c.textPrimary }}>{winnerUplift}</Text>
                  </Space>
                ) : (
                  <Tag color="default">Победитель не определён</Tag>
                )}
              </Space>
            </Col>

            <Col xs={24} md={10}>
              <Space direction="vertical" size={8} style={{ width: '100%' }}>
                <Text type="secondary" style={{ fontSize: 12 }}>Рекомендация</Text>
                <Tag color={recommendation.color} icon={<RocketOutlined />} style={{ width: 'fit-content', fontSize: 14, padding: '4px 12px' }}>
                  {recommendation.label}
                </Tag>
                <Text style={{ color: c.textMuted }}>{recommendationDetails()}</Text>
              </Space>
            </Col>
          </Row>

          {(timeSeriesData?.recommendation_reason || []).length > 0 && (
            <div style={{ marginTop: 14 }}>
              <Text type="secondary">Основания решения:</Text>
              <ul style={{ margin: '6px 0 0 0', paddingLeft: 18 }}>
                {(timeSeriesData?.recommendation_reason || []).map((reason, idx) => (
                  <li key={`${reason}-${idx}`}>
                    <Text>{reason}</Text>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>

      <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
        <Col xs={24} md={8}>
          <MetricCard title="Ключевые метрики" c={c}>
            <Space direction="vertical" size={8} style={{ width: '100%' }}>
              <MetricRow label="Прирост" value={winnerUplift} strong valueTone={upliftTone} />
              <MetricRow
                label="Скорректированное p-значение"
                value={correctedWinnerPValue}
                valueTone={pValueTone}
              />
              <MetricRow
                label="Уровень уверенности"
                value={confidenceLabel[timeSeriesData?.winner_confidence || 'low']?.label || 'Н/Д'}
                valueTone={confidenceTone}
              />
            </Space>
          </MetricCard>
        </Col>

        <Col xs={24} md={8}>
          <MetricCard title="Валидность и проверки" c={c}>
            <Space direction="vertical" size={8} style={{ width: '100%' }}>
              <MetricRow
                label="Итог валидности"
                value={validity.label}
                valueTone={validityTone}
              />
              <MetricRow
                label="Равномерность трафика (SRM)"
                value={timeSeriesData?.srm_check_passed ? 'Пройдена' : 'Есть перекос'}
                valueTone={srmTone}
              />
              <MetricRow
                label="Защитные ограничения"
                value={timeSeriesData?.guardrails?.passed ? 'Соблюдены' : 'Нарушены'}
                valueTone={guardrailsTone}
              />
            </Space>
          </MetricCard>
        </Col>

        <Col xs={24} md={8}>
          <MetricCard title="Качество данных и выкатка в прод" c={c}>
            <Space direction="vertical" size={8} style={{ width: '100%' }}>
              <MetricRow
                label="Оценка качества данных"
                value={quality.label}
                valueTone={qualityTone}
              />
              <MetricRow
                label="Подсказка по раскатке"
                value={timeSeriesData?.rollout_hint || 'Не задана'}
                valueTone={rolloutTone}
              />
            </Space>
          </MetricCard>
        </Col>
      </Row>

      {renderResultsTable()}
    </div>
  );
};

const MetricCard: React.FC<{ title: string; c: Record<string, string>; children: React.ReactNode }> = ({ title, c, children }) => (
  <div style={{ borderRadius: 12, border: `1px solid ${c.border}`, backgroundColor: c.panelBg, boxShadow: c.shadow, overflow: 'hidden', height: '100%' }}>
    <div style={{ padding: '10px 12px', borderBottom: `1px solid ${c.border}`, backgroundColor: c.panelSoft, fontSize: 13, fontWeight: 700, color: c.textPrimary }}>{title}</div>
    <div style={{ padding: 12, color: c.textPrimary }}>{children}</div>
  </div>
);

const MetricRow: React.FC<{
  label: string;
  value: string;
  valueTone?: ValueTone;
  strong?: boolean;
}> = ({
  label,
  value,
  valueTone = 'default',
}) => (
  <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
    <Text type="secondary" style={{ fontSize: 12 }}>{label}</Text>
    <Tag color={TONE_TO_TAG[valueTone]} style={{ width: 'fit-content', margin: 0 }}>
      {value}
    </Tag>
  </div>
);
