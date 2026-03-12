// frontend/src/utils/api.ts
import axios from 'axios';
import type { LoginCredentials } from '../types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api/v1';

const clearAuthStorage = (): void => {
  localStorage.removeItem('access_token');
  localStorage.removeItem('user_role');
  localStorage.removeItem('user_full_name');
  localStorage.removeItem('username');
};

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('access_token');

  if (token) {
    config.headers = config.headers ?? {};
    config.headers.Authorization = `Bearer ${token}`;
  }

  return config;
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      clearAuthStorage();
      if (window.location.pathname !== '/login') {
        window.location.href = '/login';
      }
    }

    return Promise.reject(error);
  }
);

// A/B Tests API
export const abTestAPI = {
  createTest: (data: any) => api.post('/tests/', data),
  assignUser: (testId: string, data: any) => api.post(`/tests/${testId}/assign`, data),
  getResults: (testId: string) => api.get(`/tests/${testId}/results`),
  getAllTests: () => api.get('/tests/'),  // Новый endpoint для получения всех тестов по статусам
  stopTest: (testId: string, reason: string) => api.post(`/tests/${testId}/stop`, { reason }),
  // Новые endpoints для управления тестами
  startSimulation: (testId: string, data: any) => api.post(`/tests/${testId}/start-simulation`, data),
  pauseTest: (testId: string, reason?: string) => api.post(`/tests/${testId}/pause`, reason ? { reason } : {}),
  resumeTest: (testId: string) => api.post(`/tests/${testId}/resume`),
  deleteTestWithOption: (testId: string, moveToArchived: boolean) => api.post(`/tests/${testId}/delete-with-option`, { move_to_archived: moveToArchived }),
  archiveTest: (testId: string, reason?: string) => api.post(`/tests/${testId}/archive`, reason ? { reason } : {}),
  permanentlyDeleteTest: (testId: string) => api.delete(`/tests/${testId}/permanent`),
};

// Data Generation API
export const dataAPI = {
  generateRealData: (data: any) => api.post('/data/generate-real', data),
  trainGAN: (data: any) => api.post('/data/train-gan', data),
  stopGANTraining: () => api.post('/data/stop-gan-training'),
  resumeGANTraining: () => api.post('/data/resume-gan-training'),
  resetGANTraining: () => api.post('/data/reset-gan-training'),
  generateSynthetic: (data: any) => api.post('/data/generate-synthetic', data),
  getGANStatus: () => api.get('/data/gan-status'),
  getGANCheckpoints: () => api.get('/data/gan-checkpoints'),
  deleteGANCheckpoint: (checkpointId: number) => api.delete(`/data/gan-checkpoints/${checkpointId}`),
  getDatasetStats: () => api.get('/data/dataset-stats'),
  listGeneratedHistory: (limit: number = 50) => api.get(`/data/generated-history?limit=${limit}`),
  getFullDataset: (itemId: number) => api.get(`/data/generated-history/${itemId}/full`),
  deleteGeneratedHistoryItem: (itemId: number) => api.delete(`/data/generated-history/${itemId}`),
  loadGANCheckpoint: (checkpointName: string) => api.post('/data/gan-load-checkpoint', { checkpoint_name: checkpointName }),
  runABTestOnSynthetic: (data: any) => api.post('/data/run-ab-test-simulation', data),
};

// Results API
export const resultsAPI = {
  getDetailedResults: (testId: string) => api.get(`/results/${testId}/detailed`),
  getStatisticalSignificance: (testId: string, alpha: number = 0.05) =>
    api.get(`/results/${testId}/statistical-significance?alpha=${alpha}`),
  getPlatformStats: () => api.get('/results/platform/performance'),
  getTimeSeriesData: (testId: string) => api.get(`/results/${testId}/time-series-data`),
};

// Auth API
export const authAPI = {
  login: (credentials: LoginCredentials) => {
    const payload = new URLSearchParams();
    payload.append('username', credentials.username);
    payload.append('password', credentials.password);

    return api.post('/auth/login', payload, {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
    });
  },
  me: () => api.get('/auth/me'),
  logout: () => api.post('/auth/logout'),
};

// Debug-утилиты
if (import.meta.env.DEV) {
  api.interceptors.request.use((request) => {
    console.log(`🚀 API REQUEST: ${request.method?.toUpperCase()} ${request.url}`, {
      data: request.data,
      params: request.params,
    });
    return request;
  });

  api.interceptors.response.use(
    (response) => {
      console.log(`✅ API SUCCESS: ${response.status} ${response.config.url}`, {
        data: response.data,
        status: response.status,
      });
      return response;
    },
    (error) => {
      console.error(`❌ API ERROR: ${error.response?.status || 'NO RESPONSE'} ${error.config?.url}`, {
        error: error.response?.data,
        message: error.message,
      });
      return Promise.reject(error);
    }
  );

  const debugAPI = {
    testCheckpoints: () =>
      api.get('/data/gan-checkpoints').then((r) => {
        console.log('🔍 DEBUG Checkpoints response:', r.data);
        return r.data;
      }),
    testGANStatus: () =>
      api.get('/data/gan-status').then((r) => {
        console.log('🔍 DEBUG GAN Status response:', r.data);
        return r.data;
      }),
  };

  if (typeof window !== 'undefined') {
    (window as any).debugAPI = debugAPI;
  }
}
