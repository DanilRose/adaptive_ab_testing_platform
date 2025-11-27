// frontend/src/utils/api.ts
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api/v1';

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// A/B Tests API
export const abTestAPI = {
  createTest: (data: any) => api.post('/tests/', data),
  assignUser: (testId: string, data: any) => api.post(`/tests/${testId}/assign`, data),
  getResults: (testId: string) => api.get(`/tests/${testId}/results`),
  getActiveTests: () => api.get('/tests/'),
  stopTest: (testId: string, reason: string) => api.post(`/tests/${testId}/stop`, { reason }),
};

// Data Generation API
export const dataAPI = {
  generateRealData: (data: any) => api.post('/data/generate-real', data),
  trainGAN: (data: any) => api.post('/data/train-gan', data),
  generateSynthetic: (data: any) => api.post('/data/generate-synthetic', data),
  getGANStatus: () => api.get('/data/gan-status'),
  getGANCheckpoints: () => api.get('/data/gan-checkpoints'),
  loadGANCheckpoint: (checkpointName: string) => api.post('/data/gan-load-checkpoint', { checkpoint_name: checkpointName }),
  runABTestOnSynthetic: (data: any) => api.post('/data/run-ab-test-simulation', data),
};

// Results API
export const resultsAPI = {
  getDetailedResults: (testId: string) => api.get(`/results/${testId}/detailed`),
  getStatisticalSignificance: (testId: string, alpha: number = 0.05) => 
    api.get(`/results/${testId}/statistical-significance?alpha=${alpha}`),
  getPlatformStats: () => api.get('/results/platform/performance'),
};


// ДЕБАГ ДЛЯ ВСЕХ ЗАПРОСОВ - ДОБАВЬ В КОНЕЦ ФАЙЛА
api.interceptors.request.use(request => {
  console.log(`🚀 API REQUEST: ${request.method?.toUpperCase()} ${request.url}`, {
    data: request.data,
    params: request.params,
    headers: request.headers
  });
  return request;
});

api.interceptors.response.use(
  response => {
    console.log(`✅ API SUCCESS: ${response.status} ${response.config.url}`, {
      data: response.data,
      status: response.status
    });
    return response;
  },
  error => {
    console.error(`❌ API ERROR: ${error.response?.status || 'NO RESPONSE'} ${error.config?.url}`, {
      error: error.response?.data,
      message: error.message,
      config: error.config
    });
    return Promise.reject(error);
  }
);

// Экспорт для ручного тестирования в консоли
export const debugAPI = {
  testCheckpoints: () => api.get('/data/gan-checkpoints').then(r => {
    console.log('🔍 DEBUG Checkpoints response:', r.data);
    return r.data;
  }),
  testGANStatus: () => api.get('/data/gan-status').then(r => {
    console.log('🔍 DEBUG GAN Status response:', r.data);
    return r.data;
  })
};

// Глобальный дебаг объект
declare global {
  interface Window {
    debugAPI: any;
  }
}

if (typeof window !== 'undefined') {
  window.debugAPI = debugAPI;
}