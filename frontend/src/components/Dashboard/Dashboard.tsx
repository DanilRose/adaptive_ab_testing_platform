// frontend/src/components/Dashboard/Dashboard.tsx
import React, { useState, useEffect, useRef } from 'react';
import { Card, Row, Col, Statistic, List, Tag, Button, message, Typography, Modal, Divider, Empty } from 'antd';
import { abTestAPI, dataAPI } from '../../utils/api';
import {
  PlayCircleOutlined,
  PauseCircleOutlined,
  DeleteOutlined,
  FolderOutlined,
} from '@ant-design/icons';

const { Title } = Typography;

interface Test {
  test_id: string;
  test_name: string;
  description?: string;
  status: string;
  simulation_status?: string;
  variants: string[];
  primary_metric: string;
  metric_type: string;
  sample_size?: number;
  total_users: number;
  completion_percentage: number;
  created_at: string;
  archive_reason?: string;
}

interface TestsByStatus {
  prepared_tests: Test[];
  active_tests: Test[];
  paused_tests: Test[];
  completed_tests: Test[];
  archived_tests: Test[];
  counts: {
    prepared: number;
    active: number;
    paused: number;
    completed: number;
    archived: number;
  };
}

export const Dashboard: React.FC = () => {
  const [testsData, setTestsData] = useState<TestsByStatus | null>(null);
  const [datasets, setDatasets] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [simulationLoading, setSimulationLoading] = useState<string | null>(null);
  const pollingRef = React.useRef<number | null>(null);

  useEffect(() => {
    loadDashboardData();
    loadDatasets();
    
    // Обновляем дашборд каждые 3 секунды для отслеживания статуса симуляции
    pollingRef.current = window.setInterval(() => {
      loadDashboardData();
    }, 3000);
    
    return () => {
      if (pollingRef.current) {
        window.clearInterval(pollingRef.current);
      }
    };
  }, []);

  const loadDashboardData = async () => {
    try {
      const response = await abTestAPI.getAllTests();
      setTestsData(response.data);
    } catch (error) {
      console.error('Error loading dashboard:', error);
      message.error('Ошибка загрузки данных дашборда');
    } finally {
      setLoading(false);
    }
  };

  const loadDatasets = async () => {
    try {
      const response = await dataAPI.listGeneratedHistory(100);
      const syntheticDatasets = response.data.items.filter((d: any) => d.data_type === 'synthetic');
      setDatasets(syntheticDatasets);
    } catch (error) {
      console.error('Error loading datasets:', error);
    }
  };

  const handleRunSimulation = async (testId: string) => {
    if (datasets.length === 0) {
      Modal.error({
        title: 'Нет синтетических данных',
        content: (
          <div>
            <p>Для запуска симуляции необходимо сначала сгенерировать синтетические данные.</p>
            <p>Перейдите в раздел "GAN Manager" и:</p>
            <ol>
              <li>Сгенерируйте реальные данные</li>
              <li>Обучите GAN модель</li>
              <li>Сгенерируйте синтетические данные</li>
            </ol>
          </div>
        ),
      });
      return;
    }

    setSimulationLoading(testId);
    try {
      await abTestAPI.startSimulation(testId, {});

      message.success({
        content: `✅ Симуляция запущена! Используются параметры, сохранённые в тесте.`,
        duration: 5,
      });

      setTimeout(() => {
        loadDashboardData();
      }, 3000);
    } catch (error: any) {
      console.error('❌ Simulation error:', error);
      const errorMsg = error.response?.data?.detail || error.message || 'Неизвестная ошибка';
      message.error(`Ошибка запуска симуляции: ${errorMsg}`);
    } finally {
      setSimulationLoading(null);
    }
  };

  const handlePauseTest = async (testId: string) => {
    Modal.confirm({
      title: 'Поставить тест на паузу?',
      content: 'Вы уверены, что хотите приостановить симуляцию?',
      okText: 'Да, на паузу',
      cancelText: 'Отмена',
      onOk: async () => {
        try {
          await abTestAPI.pauseTest(testId);
          message.success('Тест поставлен на паузу');
          loadDashboardData();
        } catch (error: any) {
          message.error(`Ошибка паузы теста: ${error.response?.data?.detail || error.message}`);
        }
      },
    });
  };

  const handleResumeTest = async (testId: string) => {
    try {
      await abTestAPI.resumeTest(testId);
      message.success('Тест продолжен');
      loadDashboardData();
    } catch (error: any) {
      message.error(`Ошибка продолжения теста: ${error.response?.data?.detail || error.message}`);
    }
  };

  const handleDeleteTest = async (testId: string, moveToArchived: boolean) => {
    try {
      await abTestAPI.deleteTestWithOption(testId, moveToArchived);
      message.success(moveToArchived ? 'Тест перемещен в архив' : 'Тест перемещен в подготовленные');
      loadDashboardData();
    } catch (error: any) {
      message.error(`Ошибка удаления теста: ${error.response?.data?.detail || error.message}`);
    }
  };

  const handleDeleteWithConfirm = (testId: string, testName: string) => {
    Modal.confirm({
      title: `Удалить тест "${testName}"?`,
      content: (
        <div>
          <p>Выберите действие:</p>
          <ul>
            <li><strong>Переместить в архив</strong> - тест будет сохранен в архиве</li>
            <li><strong>Переместить в подготовленные</strong> - тест можно будет запустить снова</li>
          </ul>
        </div>
      ),
      okText: 'Переместить в архив',
      cancelText: 'Отмена',
      icon: <DeleteOutlined style={{ color: '#ff4d4f' }} />,
      onOk: async () => {
        await handleDeleteTest(testId, true);
      },
      onCancel: async () => {
        // Пользователь нажал "Отмена" в модалке, но мы хотим дать возможность переместить в подготовленные
        Modal.confirm({
          title: 'Переместить в подготовленные?',
          content: 'Тест будет перемещен в раздел "Подготовленные тесты" и его можно будет запустить снова',
          okText: 'Да, переместить',
          cancelText: 'Нет, отмена',
          onOk: async () => {
            await handleDeleteTest(testId, false);
          },
        });
      },
    });
  };

  const handleArchiveTest = async (testId: string) => {
    try {
      await abTestAPI.archiveTest(testId);
      message.success('Тест перемещен в архив');
      loadDashboardData();
    } catch (error: any) {
      message.error(`Ошибка архивирования: ${error.response?.data?.detail || error.message}`);
    }
  };

  const handlePermanentlyDeleteTest = async (testId: string) => {
    Modal.confirm({
      title: 'Полностью удалить тест?',
      content: 'Это действие необратимо! Тест будет удален из базы данных.',
      okText: 'Да, удалить',
      okType: 'danger',
      cancelText: 'Отмена',
      onOk: async () => {
        try {
          await abTestAPI.permanentlyDeleteTest(testId);
          message.success('Тест полностью удален');
          loadDashboardData();
        } catch (error: any) {
          message.error(`Ошибка удаления: ${error.response?.data?.detail || error.message}`);
        }
      },
    });
  };

  const renderTestItem = (test: Test, type: 'prepared' | 'active' | 'paused' | 'completed' | 'archived') => {
    const isSimulating = simulationLoading === test.test_id;
    const isPaused = test.status === 'paused';

    // Определяем кнопки для каждого типа
    const getActions = () => {
      if (isPaused) {
        // Тест на паузе - показываем кнопку Продолжить
        return [
          <Button
            type="primary"
            key="resume"
            icon={<PlayCircleOutlined />}
            onClick={() => handleResumeTest(test.test_id)}
          >
            Продолжить
          </Button>,
          <Button
            type="default"
            danger
            key="delete"
            icon={<DeleteOutlined />}
            onClick={() => handleDeleteWithConfirm(test.test_id, test.test_name)}
          >
            Удалить
          </Button>,
        ];
      }

      switch (type) {
        case 'prepared':
          return [
            <Button
              type="primary"
              key="simulate"
              icon={<PlayCircleOutlined />}
              loading={isSimulating}
              onClick={() => handleRunSimulation(test.test_id)}
              disabled={datasets.length === 0}
            >
              Запустить симуляцию
            </Button>,
            <Button
              type="default"
              key="archive"
              icon={<FolderOutlined />}
              onClick={() => handleArchiveTest(test.test_id)}
            >
              В архив
            </Button>,
          ];
        case 'active':
          return [
            <Button
              type="default"
              key="pause"
              icon={<PauseCircleOutlined />}
              onClick={() => handlePauseTest(test.test_id)}
            >
              Пауза
            </Button>,
            <Button
              type="default"
              key="move-to-prepared"
              icon={<FolderOutlined />}
              onClick={() => handleDeleteTest(test.test_id, false)}
            >
              В подготовленные
            </Button>,
            <Button
              type="default"
              danger
              key="delete"
              icon={<DeleteOutlined />}
              onClick={() => handleDeleteWithConfirm(test.test_id, test.test_name)}
            >
              Удалить
            </Button>,
          ];
        case 'completed':
          return [
            <Button
              type="default"
              key="archive"
              icon={<FolderOutlined />}
              onClick={() => handleArchiveTest(test.test_id)}
            >
              В архив
            </Button>,
          ];
        case 'archived':
          return [
            <Button
              type="primary"
              danger
              key="delete"
              icon={<DeleteOutlined />}
              onClick={() => handlePermanentlyDeleteTest(test.test_id)}
            >
              Удалить
            </Button>,
          ];
        default:
          return [];
      }
    };

    const actions = getActions();

    // Стили для тестов на паузе
    const itemStyle = isPaused ? {
      background: '#fff7e6',
      border: '1px solid #ffd591',
      borderRadius: '4px',
      padding: '8px',
      marginBottom: '8px',
    } : {};

    return (
      <List.Item actions={actions} style={itemStyle}>
        <List.Item.Meta
          title={
            <div>
              <span style={{ fontSize: '16px', fontWeight: 500 }}>{test.test_name}</span>
              <Tag color={getStatusColor(test.status)} style={{ marginLeft: '8px' }}>
                {getStatusLabel(test.status)}
              </Tag>
              {test.simulation_status === 'running' && test.status !== 'archived' && (
                <Tag color="red" style={{ marginLeft: '8px' }}>
                  Симуляция запущена
                </Tag>
              )}
            </div>
          }
          description={
            <div>
              <div style={{ marginBottom: '8px' }}>{test.description || 'Нет описания'}</div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                <Tag color="blue">Варианты: {test.variants?.join(', ')}</Tag>
                <Tag color="green">Метрика: {test.primary_metric}</Tag>
                <Tag color="orange">Пользователей: {test.total_users || 0}</Tag>
                <Tag color={test.completion_percentage > 80 ? 'green' : 'orange'}>
                  Завершено: {test.completion_percentage || 0}%
                </Tag>
                <Tag color="purple">Создан: {new Date(test.created_at).toLocaleString()}</Tag>
              </div>
            </div>
          }
        />
      </List.Item>
    );
  };

  const getStatusColor = (status: string): string => {
    const colors: Record<string, string> = {
      prepared: 'blue',
      active: 'green',
      paused: 'orange',
      completed: 'purple',
      archived: 'default',
    };
    return colors[status] || 'default';
  };

  const getStatusLabel = (status: string): string => {
    const labels: Record<string, string> = {
      prepared: 'Подготовлен',
      active: 'Активен',
      paused: 'На паузе',
      completed: 'Завершен',
      archived: 'Архив',
    };
    return labels[status] || status;
  };

  const renderTestSection = (title: string, tests: Test[], type: 'prepared' | 'active' | 'paused' | 'completed' | 'archived') => (
    <Card 
      title={title} 
      size="small"
      style={{ marginBottom: '16px' }}
      extra={<Tag>{tests.length}</Tag>}
    >
      {tests.length > 0 ? (
        <List
          itemLayout="horizontal"
          dataSource={tests}
          renderItem={(test: Test) => renderTestItem(test, type)}
        />
      ) : (
        <Empty description={`Нет тестов`} image={Empty.PRESENTED_IMAGE_SIMPLE} />
      )}
    </Card>
  );

  return (
    <div style={{ padding: '20px' }}>
      <Row gutter={[16, 16]} style={{ marginBottom: '20px' }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="Подготовленные тесты"
              value={testsData?.counts.prepared || 0}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Активные тесты"
              value={(testsData?.counts.active || 0) + (testsData?.counts.paused || 0)}
              valueStyle={{ color: '#3f8600' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Завершенные тесты"
              value={testsData?.counts.completed || 0}
              valueStyle={{ color: '#722ed1' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Архив"
              value={testsData?.counts.archived || 0}
              valueStyle={{ color: '#8c8c8c' }}
            />
          </Card>
        </Col>
      </Row>

      {datasets.length === 0 && (
        <Card style={{ marginBottom: '16px', borderColor: '#ffd591' }}>
          <div style={{ padding: '12px', background: '#fff7e6', borderRadius: '4px' }}>
            <Typography.Text type="warning">
              ⚠️ <strong>Внимание:</strong> Для запуска симуляции необходимо создать синтетические данные в разделе "GAN Manager"
            </Typography.Text>
          </div>
        </Card>
      )}

      <Divider orientation="left">Активные тесты</Divider>
      {renderTestSection('Активные тесты', [...(testsData?.active_tests || []), ...(testsData?.paused_tests || [])], 'active')}

      <Divider orientation="left">Подготовленные тесты</Divider>
      {renderTestSection('Подготовленные тесты', testsData?.prepared_tests || [], 'prepared')}

      <Divider orientation="left">Завершенные тесты</Divider>
      {renderTestSection('Завершенные тесты', testsData?.completed_tests || [], 'completed')}

      <Divider orientation="left">Архив</Divider>
      {renderTestSection('Архив', testsData?.archived_tests || [], 'archived')}


      {loading && (
        <div style={{ textAlign: 'center', padding: '40px' }}>
          <Typography.Title level={4}>Загрузка данных...</Typography.Title>
        </div>
      )}
    </div>
  );
};
