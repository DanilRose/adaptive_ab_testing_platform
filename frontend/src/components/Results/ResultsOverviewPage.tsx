import React, { useMemo } from 'react';
import { Col, Row, Space, Table, Tag, Tooltip, Typography } from 'antd';
import { CheckCircleOutlined, CloseCircleOutlined, RocketOutlined } from '@ant-design/icons';
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

type VariantStat = {
  variant: string;
  uplift: number | null;
  pValue: number | null;
};

const TONE_TO_TAG: Record<ValueTone, string> = {
  success: 'success',
  warning: 'warning',
  danger: 'error',
  secondary: 'default',
  default: 'default',
};

const MIN_SAMPLE_PER_VARIANT_FOR_SIGNIFICANCE = 300;

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

  const finalSnapshot = chartData.length > 0 ? chartData[chartData.length - 1] : null;
  const controlVariant = timeSeriesData?.variants?.[0];
  const controlMetric = controlVariant && finalSnapshot
    ? Number(finalSnapshot?.[controlVariant]?.mean_metric ?? 0)
    : 0;
  const nonControlVariants = controlVariant
    ? (timeSeriesData?.variants?.filter((variant) => variant !== controlVariant) ?? [])
    : [];
  const sampleSizesByVariant = timeSeriesData?.variants?.map((variant) => Number(finalSnapshot?.[variant]?.sample_size ?? 0)) ?? [];
  const minSampleSizeAcrossVariants = sampleSizesByVariant.length > 0 ? Math.min(...sampleSizesByVariant) : 0;
  const hasInsufficientSample = minSampleSizeAcrossVariants > 0 && minSampleSizeAcrossVariants < MIN_SAMPLE_PER_VARIANT_FOR_SIGNIFICANCE;
  const hasSrmIssue = timeSeriesData?.srm_check_passed === false;
  const variantStats: VariantStat[] = nonControlVariants.map((variant) => {
    const snapshot = finalSnapshot?.[variant];
    const mean = snapshot ? Number(snapshot?.mean_metric ?? 0) : null;
    const uplift = mean !== null && controlMetric > 0
      ? ((mean - controlMetric) / controlMetric * 100)
      : null;
    const pValue = timeSeriesData?.p_values_corrected_latest?.[variant] ?? snapshot?.p_value ?? null;
    return { variant, uplift, pValue };
  }).filter((stat) => stat.uplift !== null);

  const bestUpliftStat = variantStats.length > 0
    ? variantStats.reduce((best, current) => (current.uplift! > best.uplift! ? current : best), variantStats[0])
    : null;
  const bestUpliftValue = bestUpliftStat?.uplift ?? null;
  const bestUpliftDisplay = bestUpliftValue !== null
    ? `${bestUpliftValue >= 0 ? '+' : ''}${bestUpliftValue.toFixed(2)}%`
    : 'Н/Д';
  const bestUpliftLabelBase = bestUpliftStat
    ? `Лучший прирост к контролю (вариант ${bestUpliftStat.variant})`
    : 'Лучший прирост к контролю';

  const minPValueStat = variantStats.reduce<VariantStat | null>((best, current) => {
    if (current.pValue === null || current.pValue === undefined) return best;
    if (!best || best.pValue === null || best.pValue === undefined || current.pValue < best.pValue) return current;
    return best;
  }, null);
  const decisionPValue = minPValueStat?.pValue ?? null;
  const decisionPValueDisplay = decisionPValue !== null && decisionPValue !== undefined
    ? Number(decisionPValue).toFixed(4)
    : 'Н/Д';
  const completionPercentage = Number(timeSeriesData?.completion_percentage ?? 0);
  const isComplete = completionPercentage >= 100 || Boolean(timeSeriesData?.stopped_early);
  const hasEnoughProgressForPreliminary = completionPercentage >= 30;
  const allVariantsWorse = bestUpliftValue !== null && bestUpliftValue < 0;
  const bestUpliftLabel = allVariantsWorse && bestUpliftStat
    ? `Лучший из ухудшений (вариант ${bestUpliftStat.variant})`
    : bestUpliftLabelBase;

  const renderResultsTable = () => {
    if (!timeSeriesData || chartData.length === 0 || !finalSnapshot || !controlVariant) return null;

    const upliftValues = timeSeriesData.variants.map((variant) => {
      if (variant === controlVariant) return 0;
      const variantData = finalSnapshot?.[variant];
      const variantMean = Number(variantData?.mean_metric ?? 0);
      return controlMetric > 0
        ? ((variantMean - controlMetric) / controlMetric * 100)
        : 0;
    });
    const maxUpliftValue = upliftValues.length > 0 ? Math.max(...upliftValues) : 0;

    const tableData = timeSeriesData.variants.map((variant) => {
      const variantData = finalSnapshot?.[variant];
      const variantMean = Number(variantData?.mean_metric ?? 0);
      const variantCum = Number(variantData?.cumulative_metric ?? 0);
      const variantSample = Number(variantData?.sample_size ?? 0);
      const variantPValueRaw = variantData?.p_value ?? null;
      const variantPValueCorrected = variant === controlVariant
        ? null
        : (timeSeriesData?.p_values_corrected_latest?.[variant] ?? null);

      const upliftValue = controlMetric > 0
        ? ((variantMean - controlMetric) / controlMetric * 100)
        : 0;

      const significanceSource = variantPValueCorrected ?? variantPValueRaw;
      const winProbability = (pValue: number | null, uplift: number, isBestCurrent: boolean) => {
        if (pValue === null || pValue === undefined) return null;
        if (uplift <= 0) return Math.max(1, Math.round((1 - pValue) * 20));
        if (pValue < 0.05) return 95;
        if (isBestCurrent) return Math.max(55, Math.min(89, Math.round((1 - pValue) * 100)));
        return Math.max(25, Math.min(70, Math.round((1 - pValue) * 80)));
      };

      const isControl = variant === controlVariant;
      const isBestVariant = isComplete && !isControl && maxUpliftValue > 0 && Math.abs(upliftValue - maxUpliftValue) < 0.0001;
      const isWorse = !isControl && upliftValue < 0;
      const isLeader = isComplete && isControl && maxUpliftValue <= 0;

      return {
        variant,
        sample_size: variantSample,
        mean_metric: variantMean.toFixed(4),
        cumulative_metric: variantCum.toFixed(2),
        uplift: upliftValue.toFixed(2),
        uplift_value: upliftValue,
        win_probability: winProbability(
          significanceSource,
          upliftValue,
          !isControl && maxUpliftValue > 0 && Math.abs(upliftValue - maxUpliftValue) < 0.0001,
        ),
        p_value_raw: variantPValueRaw !== null && variantPValueRaw !== undefined ? Number(variantPValueRaw).toFixed(4) : 'Н/Д',
        p_value_corrected: variantPValueCorrected !== null && variantPValueCorrected !== undefined ? Number(variantPValueCorrected).toFixed(4) : 'Н/Д',
        significant: significanceSource !== null && significanceSource !== undefined && significanceSource < 0.05,
        waiting_data: hasInsufficientSample || !hasEnoughProgressForPreliminary,
        is_control: isControl,
        is_best: isBestVariant,
        is_worse: isWorse,
        is_leader: isLeader,
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
            {record.is_leader && <Tag color="green">Лидер</Tag>}
            {record.is_best && <Tag color="success">Лучше контроля</Tag>}
            {record.is_worse && <Tag color="error">Ухудшение</Tag>}
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
            <Text type={record.uplift_value > 0 ? 'success' : 'danger'} strong>
              {record.uplift_value > 0 ? '+' : ''}{uplift}%
            </Text>
          );
        },
      },
      {
        title: (
          <Tooltip title="Эвристическая оценка на основе p-value и направления прироста; не является байесовской вероятностью.">
            Вероятность выигрыша
          </Tooltip>
        ),
        dataIndex: 'win_probability',
        key: 'win_probability',
        render: (value: number | null, record: any) => {
          if (record.is_control) return <Text type="secondary">—</Text>;
          if (!isComplete && hasEnoughProgressForPreliminary) {
            return <Text type="secondary">Предварительная оценка</Text>;
          }
          if (value === null) return <Text type="secondary">Н/Д</Text>;
          return (
            <Text type={value >= 50 ? 'success' : 'danger'} strong>
              {value}%
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
          if (record.waiting_data) return <Tag color="default">Ожидание данных</Tag>;
          if (record.significant && record.uplift_value < 0) {
            return <Tag icon={<CloseCircleOutlined />} color="error">Значимо хуже</Tag>;
          }
          return record.significant ? (
            <Tag icon={<CheckCircleOutlined />} color="success">Статистически значимо</Tag>
          ) : (
            <Tag icon={<CloseCircleOutlined />} color="error">Статистически незначимо</Tag>
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
            onRow={(record: any) => ({
              style: !record.is_control && parseFloat(record.uplift) < 0
                ? { backgroundColor: c.dangerSoft }
                : undefined,
            })}
          />
          <Text type="secondary" style={{ display: 'block', marginTop: 8, fontSize: 12 }}>
            * Для финального вывода используется скорректированное p-значение (Холм).
          </Text>
        </div>
      </div>
    );
  };

  const recommendationFallback = recommendationLabel[timeSeriesData?.recommendation_status || 'need_more_data']
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

  const correctedWinnerPValue = hasInsufficientSample
    ? '(мало данных)'
    : decisionPValueDisplay;

  const winnerUpliftFromData =
    isComplete && typeof timeSeriesData?.winner_uplift_percent === 'number'
      ? `${timeSeriesData.winner_uplift_percent >= 0 ? '+' : ''}${timeSeriesData.winner_uplift_percent.toFixed(2)}%`
      : null;

  const winnerUplift = winnerUpliftFromData ?? bestUpliftDisplay;

  const upliftTone: ValueTone = bestUpliftValue !== null
    ? (bestUpliftValue >= 0 ? 'success' : 'danger')
    : 'secondary';

  const pValueTone: ValueTone = hasInsufficientSample
    ? 'secondary'
    : decisionPValue !== null && decisionPValue !== undefined
    ? (decisionPValue < 0.05 ? 'success' : 'danger')
    : 'secondary';

  const confidenceTone: ValueTone = toTone(confidenceLabel[timeSeriesData?.winner_confidence || 'low']?.color);

  const validityTone: ValueTone = toTone(validity.color);

  const srmTone: ValueTone = timeSeriesData?.srm_check_passed ? 'success' : 'danger';

  const guardrailsTone: ValueTone = timeSeriesData?.guardrails?.passed ? 'success' : 'danger';

  const qualityTone: ValueTone = hasInsufficientSample ? 'danger' : toTone(quality.color);
  const qualityValue = hasInsufficientSample ? 'Низкая' : quality.label;

  const hasStatSig = decisionPValue !== null && decisionPValue < 0.05;
  const pValueForNarrative = decisionPValue !== null
    ? (decisionPValue < 0.0001 ? 'p < 0.0001' : `p = ${decisionPValue.toFixed(4)}`)
    : 'p недоступно';

  const validitySummary = validity.color === 'success'
    ? 'Валидность анализа: Валиден'
    : validity.color === 'warning'
      ? 'Валидность анализа: Ограничен (исследовательский режим)'
      : 'Валидность анализа: Невалиден';

  const resultSummary = !hasEnoughProgressForPreliminary
    ? `Результат: Недостаточно данных для предварительного вывода (прогресс ${completionPercentage.toFixed(1)}%, требуется от 30%).`
    : !isComplete
    ? `Результат: Предварительный анализ доступен.`
    : decisionPValue !== null && decisionPValue !== undefined
      ? `Результат: ${decisionPValue < 0.05 ? 'Статистически значим' : 'Статистически незначим'} (min p=${decisionPValueDisplay}${minPValueStat?.variant ? `, вариант ${minPValueStat.variant}` : ''})`
      : 'Результат: Недостаточно данных';

  const preCompleteSummary = !hasEnoughProgressForPreliminary
    ? `Предварительный вывод недоступен: собрано только ${completionPercentage.toFixed(1)}% данных. Дождитесь минимум 30% объёма теста.`
    : hasInsufficientSample
    ? `Предупреждение: Слишком малая выборка. Минимум на вариант: ${MIN_SAMPLE_PER_VARIANT_FOR_SIGNIFICANCE}, сейчас минимум: ${minSampleSizeAcrossVariants}. Для выявления эффекта даже при крупном uplift требуется как минимум 1000-2000 наблюдений на вариант.`
    : decisionPValue !== null && decisionPValue !== undefined
    ? (decisionPValue < 0.05
      ? (bestUpliftValue !== null && bestUpliftValue < 0
        ? `Предварительно: значимое ухудшение ${bestUpliftDisplay} (min p=${decisionPValueDisplay})`
        : `Предварительно: значимый рост ${bestUpliftDisplay} (min p=${decisionPValueDisplay})`)
      : `Предварительно: различия незначимы (min p=${decisionPValueDisplay})`)
    : 'Идёт набор данных, выводы преждевременны.';

  const dynamicsSummary = bestUpliftValue !== null && controlVariant
    ? (allVariantsWorse
      ? `Динамика: Все варианты хуже ${controlVariant}; лучший из них ${bestUpliftStat?.variant} на ${Math.abs(bestUpliftValue).toFixed(2)}%`
      : bestUpliftValue > 0
        ? `Динамика: Лучший вариант ${bestUpliftStat?.variant} опережает ${controlVariant} на ${Math.abs(bestUpliftValue).toFixed(2)}%`
        : `Динамика: Нет отличий от контроля ${controlVariant}`)
    : null;

  const sampleSummary = `Выборка: минимум ${minSampleSizeAcrossVariants} наблюдений на вариант (порог для первичной значимости: ${MIN_SAMPLE_PER_VARIANT_FOR_SIGNIFICANCE}).`;
  const srmSummary = hasSrmIssue
    ? 'SRM: обнаружен перекос распределения трафика. Выводы ненадёжны.'
    : 'SRM: критичного перекоса не обнаружено.';

  const decisionBasis = [validitySummary, resultSummary, sampleSummary, srmSummary, dynamicsSummary].filter(Boolean) as string[];

  const recommendationDetailsFallback = () => {
    if (timeSeriesData?.recommendation_status === 'deploy') {
      const rolloutHint = timeSeriesData?.rollout_hint ? `Рекомендуется поэтапная выкатка в прод: ${timeSeriesData.rollout_hint}.` : 'Можно внедрять сразу.';
      return ` ${rolloutHint} При запуске — мониторинг метрик и стоп-условия при ухудшении.`;
    }
    if (timeSeriesData?.recommendation_status === 'do_not_deploy') {
      return 'Внедрять не рекомендуется: прирост недостаточен/отрицательный либо есть риски по качеству данных и защитных ограничениях. Продолжайте тест или пересмотрите гипотезу.';
    }
    return 'Требуется больше данных: дождитесь достаточной статистики и подтвердите стабильность качества данных перед решением о внедрении.';
  };

  const recommendationView = (() => {
    if (!hasEnoughProgressForPreliminary) {
      return {
        label: 'Ожидание данных до 30%',
        color: 'default',
        details: preCompleteSummary,
      };
    }
    if (hasInsufficientSample) {
      return {
        label: 'Предупреждение: Слишком малая выборка',
        color: 'error',
        details: preCompleteSummary,
      };
    }
    if (!isComplete && allVariantsWorse) {
      return {
        label: 'Предварительно: лучше оставить контроль',
        color: 'warning',
        details: 'На текущих данных все варианты хуже контроля. Продолжайте тест для подтверждения.',
      };
    }
    if (!isComplete) {
      return {
        label: 'Предварительный результат',
        color: 'warning',
        details: preCompleteSummary,
      };
    }
    if (allVariantsWorse) {
      return {
        label: 'Все варианты хуже контроля',
        color: 'error',
        details: isComplete
          ? (bestUpliftValue !== null
            ? `Тест завершён. Все варианты хуже контроля; лучший из альтернатив — вариант ${bestUpliftStat?.variant} (${bestUpliftDisplay}). Рекомендуется оставить контроль без изменений.`
            : 'Тест завершён. Значимого улучшения нет, рекомендуется оставить контроль без изменений.')
          : (bestUpliftValue !== null
            ? `Все наблюдаемые варианты хуже контроля; лучший из альтернатив — вариант ${bestUpliftStat?.variant} (${bestUpliftDisplay}). Предварительно рекомендуется оставить контроль.`
            : 'Предварительно рекомендуется оставить контроль.'),
      };
    }
    if (!isComplete && bestUpliftValue !== null && bestUpliftValue > 0 && decisionPValue !== null && decisionPValue >= 0.05) {
      return {
        label: 'Есть лидер, статистика пока не подтверждена',
        color: 'warning',
        details: `Предварительно лидирует вариант ${bestUpliftStat?.variant} с приростом ${bestUpliftDisplay}, но пока ${pValueForNarrative}. Продолжайте тест до завершения для финального решения.`,
      };
    }
    if (decisionPValue !== null && decisionPValue >= 0.05) {
      return {
        label: 'Нет статистически значимых различий',
        color: 'default',
        details: isComplete
          ? `Тест завершён. Различия не подтверждены статистически (${pValueForNarrative}${minPValueStat?.variant ? `, лучший сигнал у варианта ${minPValueStat.variant}` : ''}). Внедрение изменений не рекомендуется.`
          : `Различия пока не подтверждены статистически (${pValueForNarrative}${minPValueStat?.variant ? `, лучший сигнал у варианта ${minPValueStat.variant}` : ''}). Продолжайте набор данных.`,
      };
    }
    if (decisionPValue !== null && decisionPValue < 0.05 && bestUpliftValue !== null && bestUpliftValue > 0) {
      return {
        label: 'Есть статистически значимый рост',
        color: 'success',
        details: isComplete
          ? `Тест завершён. Вариант ${bestUpliftStat?.variant} показал статистически значимый прирост ${bestUpliftDisplay} (${pValueForNarrative}). Рекомендуется внедрить вариант ${bestUpliftStat?.variant} на 100% аудитории.`
          : `Предварительный сигнал: вариант ${bestUpliftStat?.variant} показывает рост ${bestUpliftDisplay} (${pValueForNarrative}). Подтвердите эффект к завершению теста.`,
      };
    }
    return { ...recommendationFallback, details: recommendationDetailsFallback() };
  })();

  const recommendationDetailsWithChecks = [
    recommendationView.details,
    hasSrmIssue
      ? ' Обнаружен перекос в распределении трафика (SRM). Продолжите сбор данных до выравнивания долей вариантов.'
      : null,
  ].filter(Boolean).join(' ');

  const hasFinalWinner = isComplete && decisionPValue !== null && decisionPValue < 0.05 && bestUpliftValue !== null && bestUpliftValue > 0;
  const shouldFallbackWinnerToControl = !isComplete && hasEnoughProgressForPreliminary && bestUpliftValue !== null && bestUpliftValue <= 0 && !!controlVariant;
  const resolvedWinnerVariant = isComplete
    ? (hasFinalWinner ? bestUpliftStat?.variant ?? null : null)
    : (hasEnoughProgressForPreliminary
      ? (bestUpliftValue !== null && bestUpliftValue > 0 ? bestUpliftStat?.variant ?? null : (shouldFallbackWinnerToControl ? controlVariant : null))
      : null);
  const resolvedWinnerLabel = isComplete
    ? (hasFinalWinner ? `Победитель: вариант ${resolvedWinnerVariant}` : 'Победителя нет')
    : resolvedWinnerVariant
      ? (shouldFallbackWinnerToControl
        ? `Предварительно: оставить контроль (вариант ${controlVariant})`
        : (bestUpliftValue !== null && bestUpliftValue < 5
          ? `Вариант ${resolvedWinnerVariant} выигрывает с минимальным отрывом. Рекомендуется оставить контрольный вариант.`
          : `Предварительный победитель: вариант ${resolvedWinnerVariant}`))
      : (hasEnoughProgressForPreliminary
        ? 'Предварительный победитель не определён'
        : `Предварительный победитель недоступен: прогресс ${completionPercentage.toFixed(1)}% из требуемых 30%`);
  const showControlFallback = Boolean(shouldFallbackWinnerToControl);
  const resolvedWinnerUplift = resolvedWinnerVariant && !showControlFallback ? winnerUplift : null;
  const winnerTagColor = isComplete && !hasFinalWinner
    ? 'warning'
    : showControlFallback
      ? 'warning'
      : 'success';

  const rolloutHintValue = (() => {
    if (!hasEnoughProgressForPreliminary || !isComplete) {
      return 'Подождать завершения';
    }
    if (hasFinalWinner && bestUpliftStat?.variant) {
      return `Раскатать вариант ${bestUpliftStat.variant} на 100%`;
    }
    return 'Оставить контроль (изменения не нужны)';
  })();

  const rolloutTone: ValueTone = hasFinalWinner ? 'success' : (isComplete ? 'warning' : 'secondary');

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
                {resolvedWinnerVariant ? (
                  <Space align="center" wrap>
                    <Tag color={winnerTagColor} style={{ fontSize: 15, padding: '4px 12px' }}>{resolvedWinnerLabel}</Tag>
                    {resolvedWinnerUplift && (
                      <Text strong style={{ fontSize: 24, color: c.textPrimary }}>{resolvedWinnerUplift}</Text>
                    )}
                  </Space>
                ) : (
                  <Tag color={isComplete ? 'warning' : 'default'}>{resolvedWinnerLabel}</Tag>
                )}
              </Space>
            </Col>

            <Col xs={24} md={10}>
              <Space direction="vertical" size={8} style={{ width: '100%' }}>
                <Text type="secondary" style={{ fontSize: 12 }}>Рекомендация</Text>
                <Tag color={recommendationView.color} icon={<RocketOutlined />} style={{ width: 'fit-content', fontSize: 14, padding: '4px 12px' }}>
                  {recommendationView.label}
                </Tag>
                <Text style={{ color: c.textMuted }}>{recommendationDetailsWithChecks}</Text>
              </Space>
            </Col>
          </Row>

          {decisionBasis.length > 0 && (
            <div style={{ marginTop: 14 }}>
              <Text type="secondary">Основания решения:</Text>
              <ul style={{ margin: '6px 0 0 0', paddingLeft: 18 }}>
                {decisionBasis.map((reason, idx) => (
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
              <MetricRow label={bestUpliftLabel} value={bestUpliftDisplay} strong valueTone={upliftTone} />
              <MetricRow
                label={`Минимальное p-значение${minPValueStat?.variant ? ` (вариант ${minPValueStat.variant})` : ''}`}
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
                value={qualityValue}
                valueTone={qualityTone}
              />
              <MetricRow
                label="Подсказка по раскатке"
                value={rolloutHintValue}
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
