import React, { useMemo } from 'react';
import { Alert, Card, Col, Empty, Row, Select, Spin } from 'antd';
import { BarChart3 } from 'lucide-react';
import { Link, Outlet, useLocation } from 'react-router-dom';
import { useTheme } from '@/context/ThemeContext';
import { useResultsData } from './hooks/useResultsData';

const { Option } = Select;

export interface ResultsOutletContext {
  selectedTestId: string;
  loading: boolean;
  timeSeriesData: ReturnType<typeof useResultsData>['timeSeriesData'];
  selectedTest: ReturnType<typeof useResultsData>['selectedTest'];
}

export const ResultsSectionPage: React.FC = () => {
  const location = useLocation();
  const { theme } = useTheme();
  const isDark = theme === 'dark';

  const {
    tests,
    selectedTestId,
    setSelectedTestId,
    selectedTest,
    timeSeriesData,
    loading,
    isSimulating,
  } = useResultsData();

  const c = useMemo(
    () => ({
      panelBg: isDark ? '#1c1917' : '#ffffff',
      panelSoft: isDark ? '#171412' : '#f5f0e8',
      border: isDark ? '#292524' : '#e7e5e4',
      textPrimary: isDark ? '#fafaf9' : '#1c1917',
      textMuted: isDark ? '#a8a29e' : '#78716c',
      textSub: isDark ? '#57534e' : '#a8a29e',
      inputBg: isDark ? '#292524' : '#fafaf9',
      accent: '#d97706',
      accentSoft: isDark ? 'rgba(217,119,6,0.16)' : '#fef3c7',
      accentText: isDark ? '#fcd34d' : '#92400e',
      shadow: isDark ? '0 10px 32px rgba(0,0,0,0.36)' : '0 8px 28px rgba(28,25,23,0.07)',
    }),
    [isDark],
  );

  const selectedMenu = location.pathname.includes('/charts') ? 'charts' : 'overview';
  const formatPercent = (value?: number | null) => {
    const safeValue = typeof value === 'number' && Number.isFinite(value) ? value : 0;
    return new Intl.NumberFormat('ru-RU', { maximumFractionDigits: 2 }).format(safeValue);
  };

  return (
    <div
      className="results-theme"
      style={{ color: c.textPrimary, fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 22, gap: 12, flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div
            style={{
              width: 40,
              height: 40,
              borderRadius: 10,
              backgroundColor: c.accentSoft,
              border: `1px solid ${isDark ? 'rgba(217,119,6,0.25)' : '#fde68a'}`,
              color: c.accentText,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <BarChart3 size={20} />
          </div>
          <div>
            <h1 style={{ margin: 0, fontSize: 22, fontWeight: 700, letterSpacing: '-0.4px', color: c.textPrimary }}>Результаты тестов</h1>
            <p style={{ margin: 0, fontSize: 13, color: c.textMuted }}>Сводка и динамика эксперимента</p>
          </div>
        </div>
      </div>

      {isSimulating && (
        <Alert
          message="Тест запущен"
          description="Данные обновляются в реальном времени каждые 3 секунды."
          type="info"
          showIcon
          style={{ marginBottom: 16, borderRadius: 12 }}
        />
      )}

      <div style={{ borderRadius: 14, border: `1px solid ${c.border}`, backgroundColor: c.panelBg, boxShadow: c.shadow, overflow: 'hidden', marginBottom: 14 }}>
        <div style={{ padding: '12px 14px', borderBottom: `1px solid ${c.border}`, backgroundColor: c.panelSoft, fontSize: 14, fontWeight: 700, color: c.textPrimary }}>
          Выбор эксперимента
        </div>
        <div style={{ padding: 14 }}>
          <Row gutter={[16, 16]} align="middle">
            <Col flex="auto">
              <Select
                value={selectedTestId || undefined}
                onChange={setSelectedTestId}
                style={{ width: '100%' }}
                placeholder="Выберите сплит-тест для просмотра результатов"
                dropdownStyle={{ borderRadius: 10 }}
              >
                {tests.map((test) => (
                   <Option key={test.test_id} value={test.test_id}>
                     {test.test_name} ({test.status === 'completed' ? 'завершён' : test.status === 'active' ? 'активен' : test.status}) — метрика: {test.primary_metric} — прогресс: {formatPercent(test.completion_percentage)}%
                   </Option>
))}
              </Select>
            </Col>
          </Row>
        </div>
      </div>

      <div style={{ display: 'flex', gap: 6, marginBottom: 14, flexWrap: 'wrap' }}>
        <Link to="/results/overview" style={{ textDecoration: 'none' }}>
          <button
            style={{
              padding: '7px 14px',
              borderRadius: 999,
              border: `1px solid ${selectedMenu === 'overview' ? c.accent : c.border}`,
              backgroundColor: selectedMenu === 'overview' ? c.accentSoft : c.panelBg,
              color: selectedMenu === 'overview' ? c.accentText : c.textMuted,
              fontSize: 13,
              fontWeight: selectedMenu === 'overview' ? 700 : 600,
              cursor: 'pointer',
            }}
          >
            Сводка и решение
          </button>
        </Link>

        <Link to="/results/charts" style={{ textDecoration: 'none' }}>
          <button
            style={{
              padding: '7px 14px',
              borderRadius: 999,
              border: `1px solid ${selectedMenu === 'charts' ? c.accent : c.border}`,
              backgroundColor: selectedMenu === 'charts' ? c.accentSoft : c.panelBg,
              color: selectedMenu === 'charts' ? c.accentText : c.textMuted,
              fontSize: 13,
              fontWeight: selectedMenu === 'charts' ? 700 : 600,
              cursor: 'pointer',
            }}
          >
            Графики и динамика
          </button>
        </Link>
      </div>

      {!selectedTestId ? (
        <Card style={{ borderRadius: 14, borderColor: c.border }}>
          <Empty description="Выберите тест для просмотра результатов" image={Empty.PRESENTED_IMAGE_SIMPLE} />
        </Card>
      ) : loading ? (
        <Card style={{ borderRadius: 14, borderColor: c.border }}>
          <div style={{ textAlign: 'center', padding: 40 }}>
            <Spin size="large" tip="Загрузка результатов…" />
          </div>
        </Card>
      ) : !timeSeriesData || !timeSeriesData.data || timeSeriesData.data.length === 0 ? (
        <Card style={{ borderRadius: 14, borderColor: c.border }}>
          <Empty
            description="Нет данных для этого теста. Запустите симуляцию на странице управления тестами."
            image={Empty.PRESENTED_IMAGE_SIMPLE}
          />
        </Card>
      ) : (
        <Outlet
          context={{
            selectedTestId,
            loading,
            timeSeriesData,
            selectedTest,
          } satisfies ResultsOutletContext}
        />
      )}
    </div>
  );
};
