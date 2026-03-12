import React from 'react';
import { Card, Typography } from 'antd';

export const TemplatesPage: React.FC = () => {
  return (
    <Card>
      <Typography.Title level={4}>Шаблоны</Typography.Title>
      <Typography.Paragraph style={{ marginBottom: 0 }}>
        Разработка
      </Typography.Paragraph>
    </Card>
  );
};
