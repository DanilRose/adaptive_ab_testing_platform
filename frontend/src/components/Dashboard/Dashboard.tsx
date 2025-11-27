// frontend/src/components/Dashboard/Dashboard.tsx
import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Statistic, List, Tag, Button, message } from 'antd';
import { abTestAPI, resultsAPI, dataAPI } from '../../utils/api';
import { TestSummary } from '../../types';
import { Modal } from 'antd';

export const Dashboard: React.FC = () => {
  const [activeTests, setActiveTests] = useState<any[]>([]);
  const [platformStats, setPlatformStats] = useState<any>({});
  const [loading, setLoading] = useState(true);
  const [simulationLoading, setSimulationLoading] = useState<string | null>(null);

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

  const handleRunSimulation = async (testId: string) => {
    setSimulationLoading(testId);
    try {
      await dataAPI.runABTestOnSynthetic({
        test_id: testId,
        user_count: 1000
      });
      message.success(`Симуляция запущена для теста ${testId}! Данные начнут поступать через несколько секунд.`);
      
      // Обновляем данные через 3 секунды
      setTimeout(() => {
        loadDashboardData();
      }, 3000);
      
    } catch (error: any) {
      message.error('Ошибка запуска симуляции: ' + (error.response?.data?.detail || error.message));
    } finally {
      setSimulationLoading(null);
    }
  };



  const handleViewResults = async (testId: string) => {
    try {
      const response = await abTestAPI.getResults(testId);
      
      // Открываем модальное окно с результатами
      Modal.info({
        title: `Результаты теста ${testId}`,
        width: 1000,
        content: (
          <div style={{ maxHeight: '60vh', overflow: 'auto' }}>
            <h4>Статистика по вариантам:</h4>
            <pre>{JSON.stringify(response.data.results, null, 2)}</pre>
            
            {response.data.statistical_significance && (
              <>
                <h4>Статистическая значимость:</h4>
                <pre>{JSON.stringify(response.data.statistical_significance, null, 2)}</pre>
              </>
            )}
            
            {response.data.summary && (
              <>
                <h4>Рекомендации:</h4>
                <pre>{JSON.stringify(response.data.summary, null, 2)}</pre>
              </>
            )}
          </div>
        ),
      });
      
    } catch (error: any) {
      message.error('Ошибка загрузки результатов: ' + (error.response?.data?.detail || error.message));
    }
  };

  const handleStopTest = async (testId: string) => {
    try {
      await abTestAPI.stopTest(testId, "Остановлено пользователем");
      message.success(`Тест ${testId} остановлен`);
      loadDashboardData(); // Обновляем список
    } catch (error: any) {
      message.error('Ошибка остановки теста: ' + (error.response?.data?.detail || error.message));
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
                <Button 
                  type="primary" 
                  key="simulate"
                  loading={simulationLoading === test.config?.test_id}
                  onClick={() => handleRunSimulation(test.config?.test_id)}
                >
                  Запустить симуляцию
                </Button>,
                <Button 
                  type="link" 
                  key="results"
                  onClick={() => handleViewResults(test.config?.test_id)}
                >
                  Результаты
                </Button>,
                <Button 
                  type="link" 
                  danger 
                  key="stop"
                  onClick={() => handleStopTest(test.config?.test_id)}
                >
                  Остановить
                </Button>
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
                      <Tag color="orange">Пользователей: {test.total_users || 0}</Tag>
                      <Tag color={test.completion_percentage > 80 ? 'green' : 'orange'}>
                        Завершено: {test.completion_percentage || 0}%
                      </Tag>
                      <Tag color="purple">Создан: {new Date(test.created_at).toLocaleDateString()}</Tag>
                    </div>
                  </div>
                }
              />
            </List.Item>
          )}
        />
        {activeTests.length === 0 && !loading && (
          <div style={{ textAlign: 'center', padding: '20px' }}>
            <p>Нет активных тестов</p>
            <p>Создайте первый A/B тест чтобы начать симуляцию</p>
          </div>
        )}
      </Card>
    </div>
  );
};