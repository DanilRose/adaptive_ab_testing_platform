// frontend/src/App.tsx
import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { Layout, Menu, theme } from 'antd';
import { Dashboard } from './components/Dashboard/Dashboard';
import { TestCreator } from './components/TestCreator/TestCreator';
import { GANManager } from './components/GANManager/GANManager';  

const { Header, Content, Sider } = Layout;

const App: React.FC = () => {
  const {
    token: { colorBgContainer },
  } = theme.useToken();

  return (
    <Router>
      <Layout style={{ minHeight: '100vh' }}>
        <Sider collapsible>
          <div style={{ 
            height: 32, 
            margin: 16, 
            background: 'rgba(255, 255, 255, 0.2)',
            borderRadius: 6,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'white',
            fontWeight: 'bold'
          }}>
            A/B Platform
          </div>
          <Menu theme="dark" mode="inline" defaultSelectedKeys={['1']}>
            <Menu.Item key="1">
              <Link to="/">Дашборд</Link>
            </Menu.Item>
            <Menu.Item key="2">
              <Link to="/create-test">Создать тест</Link>
            </Menu.Item>
            <Menu.Item key="3">
              <Link to="/gan-manager">GAN Менеджер</Link> {/* ИЗМЕНЕНО */}
            </Menu.Item>
            <Menu.Item key="4">
              <Link to="/results">Результаты</Link>
            </Menu.Item>
          </Menu>
        </Sider>
        
        <Layout>
          <Header style={{ padding: 0, background: colorBgContainer }} />
          <Content style={{ margin: '0 16px' }}>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/create-test" element={<TestCreator />} />
              <Route path="/gan-manager" element={<GANManager />} /> {/* ИЗМЕНЕНО */}
              <Route path="/results" element={<div>Results Page</div>} />
            </Routes>
          </Content>
        </Layout>
      </Layout>
    </Router>
  );
};

export default App;