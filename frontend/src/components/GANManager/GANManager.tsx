// frontend/src/components/GANManager/GANManager.tsx
import React, { useState, useEffect } from 'react';
import { Card, Button, Progress, Statistic, Row, Col, Descriptions, message, List, Tag, Modal } from 'antd';
import { dataAPI } from '../../utils/api';

export const GANManager: React.FC = () => {
  const [ganStatus, setGanStatus] = useState<any>({});
  const [checkpoints, setCheckpoints] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [training, setTraining] = useState(false);
  const [loadModalVisible, setLoadModalVisible] = useState(false);

  useEffect(() => {
    loadGANStatus();
    loadCheckpoints();
    const interval = setInterval(loadGANStatus, 3000);
    return () => clearInterval(interval);
  }, []);

  const loadGANStatus = async () => {
    try {
      const response = await dataAPI.getGANStatus();
      setGanStatus(response.data || {});
    } catch (error) {
      console.error('Error loading GAN status:', error);
      setGanStatus({ status: 'error', is_trained: false });
    }
  };

  const loadCheckpoints = async () => {
    try {
      console.log("🔄 Loading checkpoints from API...");
      const response = await dataAPI.getGANCheckpoints();
      console.log("📁 API Response:", response.data);
      setCheckpoints(response.data?.checkpoints || []);
      console.log("✅ Checkpoints set:", response.data?.checkpoints || []);
    } catch (error) {
      console.error("❌ Error loading checkpoints:", error);
      setCheckpoints([]);
    }
  };

  const handleTrainGAN = async () => {
    setTraining(true);
    try {
      await dataAPI.trainGAN({
        epochs: 50,
        real_data_samples: 50000,
        save_checkpoint: true
      });
      message.success('Обучение GAN запущено!');
    } catch (error: any) {
      message.error('Ошибка запуска обучения GAN: ' + (error.response?.data?.detail || error.message));
    } finally {
      setTraining(false);
    }
  };

  const handleGenerateData = async () => {
    setLoading(true);
    try {
      const response = await dataAPI.generateSynthetic({
        num_users: 10000,
        evaluation_metrics: true
      });
      message.success(`Сгенерировано ${response.data.synthetic_samples} синтетических пользователей!`);
    } catch (error: any) {
      message.error('Ошибка генерации данных: ' + (error.response?.data?.detail || error.message));
    } finally {
      setLoading(false);
    }
  };

  const handleLoadCheckpoint = async (checkpointName: string) => {
    try {
      // ДОБАВЬ ЭТИ СТРОКИ ДЛЯ ДЕБАГА:
      console.log("🔄 Loading checkpoint:", checkpointName);
      console.log("📁 Available checkpoints:", checkpoints);
      
      await dataAPI.loadGANCheckpoint(checkpointName);
      message.success(`Модель загружена из ${checkpointName}`);
      setLoadModalVisible(false);
      loadGANStatus();
    } catch (error: any) {
      // ДОБАВЬ ДЕБАГ ОШИБКИ:
      console.error("❌ Load checkpoint error:", error);
      message.error('Ошибка загрузки модели: ' + (error.response?.data?.detail || error.message));
    }
  };

  // ИСПРАВЛЕННЫЕ ФУНКЦИИ С ПРОВЕРКОЙ НА UNDEFINED
  const getStatusColor = (status: string | undefined) => {
    if (!status) return 'gray';
    if (status.includes('training')) return 'orange';
    if (status.includes('trained')) return 'green';
    if (status.includes('error')) return 'red';
    if (status.includes('loaded')) return 'blue';
    return 'gray';
  };

  const getProgressFromStatus = (status: string | undefined) => {
    if (!status) return 0;
    const match = status.match(/training_(\d+)%/);
    return match ? parseInt(match[1]) : 0;
  };

  const isTraining = ganStatus.status?.includes('training') || false;

  return (
    <div style={{ padding: '20px' }}>
      <Row gutter={[16, 16]}>
        <Col span={6}>
          <Card>
            <Statistic
              title="Статус GAN"
              value={ganStatus.status || 'Не инициализирован'}
              valueStyle={{ color: getStatusColor(ganStatus.status) }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Обучена"
              value={ganStatus.is_trained ? 'Да' : 'Нет'}
              valueStyle={{ color: ganStatus.is_trained ? 'green' : 'red' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Доступные чекпоинты"
              value={ganStatus.available_checkpoints || 0}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Эпох обучения"
              value={ganStatus.loss_history?.total_epochs || 0}
            />
          </Card>
        </Col>
      </Row>

      <Card title="Управление GAN" style={{ marginTop: '20px' }}>
        <Row gutter={16} style={{ marginBottom: '16px' }}>
          <Col>
            <Button 
              type="primary" 
              onClick={handleTrainGAN}
              loading={training}
              disabled={isTraining}
            >
              Обучить GAN с нуля
            </Button>
          </Col>
          <Col>
            <Button 
              onClick={() => setLoadModalVisible(true)}
              disabled={isTraining}
            >
              Загрузить чекпоинт
            </Button>
          </Col>
          <Col>
            <Button 
              onClick={handleGenerateData}
              loading={loading}
              disabled={!ganStatus.is_trained || isTraining}
            >
              Сгенерировать данные
            </Button>
          </Col>
          <Col>
            <Button onClick={loadGANStatus}>
              Обновить статус
            </Button>
          </Col>
        </Row>

        {isTraining && (
          <>
            <Progress 
              percent={getProgressFromStatus(ganStatus.status)} 
              status="active" 
              style={{ marginBottom: '16px' }}
            />
            <div style={{ marginBottom: '16px' }}>
              <Tag color="orange">Обучение в процессе</Tag>
              <span>Статус: {ganStatus.status}</span>
              <br />
              <span>Эпоха: {ganStatus.current_epoch}/{ganStatus.total_epochs}</span>
            </div>
          </>
        )}

        <Descriptions title="Детали модели" bordered>
          <Descriptions.Item label="Потери генератора">
            {ganStatus.loss_history?.g_losses?.length > 0 
              ? ganStatus.loss_history.g_losses[ganStatus.loss_history.g_losses.length - 1].toFixed(4)
              : 'N/A'
            }
          </Descriptions.Item>
          <Descriptions.Item label="Потери дискриминатора">
            {ganStatus.loss_history?.d_losses?.length > 0 
              ? ganStatus.loss_history.d_losses[ganStatus.loss_history.d_losses.length - 1].toFixed(4)
              : 'N/A'
            }
          </Descriptions.Item>
          <Descriptions.Item label="Всего эпох">
            {ganStatus.loss_history?.total_epochs || 0}
          </Descriptions.Item>
        </Descriptions>
      </Card>

      <Modal
        title="Выберите чекпоинт для загрузки"
        open={loadModalVisible}
        onCancel={() => setLoadModalVisible(false)}
        footer={null}
        width={800}
      >
      <List
        dataSource={checkpoints}
        renderItem={(checkpoint: any) => (
          <List.Item
            actions={[
              <Button 
                type="link" 
                onClick={() => handleLoadCheckpoint(checkpoint.name || checkpoint.filename)}
              >
                Загрузить
              </Button>
            ]}
          >
            <List.Item.Meta
              title={checkpoint.name || checkpoint.filename}
              description={
                <div>
                  <div>Размер: {checkpoint.size ? `${(checkpoint.size / 1024 / 1024).toFixed(2)} MB` : 'N/A'}</div>
                  <div>Изменен: {checkpoint.modified ? new Date(checkpoint.modified).toLocaleString() : 'N/A'}</div>
                </div>
              }
            />
          </List.Item>
        )}
      />
        {checkpoints.length === 0 && (
          <div style={{ textAlign: 'center', padding: '20px' }}>
            Нет доступных чекпоинтов
          </div>
        )}
      </Modal>
    </div>
  );
};