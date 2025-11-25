// frontend/src/components/Dashboard/Dashboard.tsx
import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Statistic, List, Tag, Button } from 'antd';
import { abTestAPI, resultsAPI } from '../../utils/api';
import { TestSummary } from '../../types';

export const Dashboard: React.FC = () => {
  const [activeTests, setActiveTests] = useState<any[]>([]);
  const [platformStats, setPlatformStats] = useState<any>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    try {
      const [testsResponse, statsResponse] = await Promise.all([
        abTestAPI.getActiveTests(),
        resultsAPI.getPlatformStats()
      ]);

      setActiveTests(testsResponse.data.active_tests || []);
      setPlatformStats(statsResponse.data);
    } catch (error) {
      console.error('Error loading dashboard:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: '20px' }}>
      <Row gutter={[16, 16]} style={{ marginBottom: '20px' }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="Активных тестов"
              value={platformStats.active_tests || 0}
              valueStyle={{ color: '#3f8600' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Всего пользователей"
              value={platformStats.total_users || 0}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Средняя завершенность"
              value={platformStats.average_completion || 0}
              suffix="%"
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Тестов сегодня"
              value={platformStats.tests_today || 0}
            />
          </Card>
        </Col>
      </Row>

      <Card title="Активные A/B тесты" loading={loading}>
        <List
          itemLayout="horizontal"
          dataSource={activeTests}
          renderItem={(test: any) => (
            <List.Item
              actions={[
                <Button type="link" key="results">Результаты</Button>,
                <Button type="link" danger key="stop">Остановить</Button>
              ]}
            >
              <List.Item.Meta
                title={test.config?.test_id}
                description={
                  <div>
                    <div>{test.description}</div>
                    <div style={{ marginTop: '8px' }}>
                      <Tag color="blue">Варианты: {test.config?.variants?.join(', ')}</Tag>
                      <Tag color="green">Метрика: {test.config?.primary_metric}</Tag>
                      <Tag color="orange">Пользователей: {test.total_users}</Tag>
                      <Tag color={test.completion_percentage > 80 ? 'green' : 'orange'}>
                        Завершено: {test.completion_percentage}%
                      </Tag>
                    </div>
                  </div>
                }
              />
            </List.Item>
          )}
        />
      </Card>
    </div>
  );
};