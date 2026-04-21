import React from 'react';
import { Alert, Card, Col, Empty, Row, Select, Spin, Tag, Typography } from 'antd';
import { Link, Outlet, useLocation } from 'react-router-dom';
import { useResultsData } from './hooks/useResultsData';

const { Option } = Select;
const { Title, Text } = Typography;

export interface ResultsOutletContext {
  selectedTestId: string;
  loading: boolean;
  timeSeriesData: ReturnType<typeof useResultsData>['timeSeriesData'];
  financialImpact: ReturnType<typeof useResultsData>['financialImpact'];
  selectedTest: ReturnType<typeof useResultsData>['selectedTest'];
}

export const ResultsSectionPage: React.FC = () => {
  const location = useLocation();
  const {
    tests,
    selectedTestId,
    setSelectedTestId,
    selectedTest,
    timeSeriesData,
    financialImpact,
    loading,
    isSimulating,
  } = useResultsData();

  const selectedMenu = location.pathname.includes('/charts') ? 'charts' : 'overview';

  return (
    <div style={{ padding: '20px' }}>
      <Title level={2}>Результаты A/B тестов</Title>

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
              value={selectedTestId || undefined}
              onChange={setSelectedTestId}
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

      <Card style={{ marginBottom: 16 }}>
        <Row gutter={8}>
          <Col>
            <Link to="/results/overview">
              <Tag color={selectedMenu === 'overview' ? 'blue' : 'default'} style={{ cursor: 'pointer', padding: '6px 12px' }}>
                Сводка эксперимента и решение к внедрению
              </Tag>
            </Link>
          </Col>
          <Col>
            <Link to="/results/charts">
              <Tag color={selectedMenu === 'charts' ? 'blue' : 'default'} style={{ cursor: 'pointer', padding: '6px 12px' }}>
                Графики и динамика
              </Tag>
            </Link>
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
          description="Нет данных для этого теста. Запустите симуляцию на странице управления тестами."
          image={Empty.PRESENTED_IMAGE_SIMPLE}
        />
      ) : (
        <Outlet
          context={{
            selectedTestId,
            loading,
            timeSeriesData,
            financialImpact,
            selectedTest,
          } satisfies ResultsOutletContext}
        />
      )}
    </div>
  );
};
