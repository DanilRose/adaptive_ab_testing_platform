// frontend/src/utils/api.ts
import axios from 'axios';
import type {
  ABTestCreatePayload,
  LoginCredentials,
  StartSimulationPayload,
  SyntheticGenerationPayload,
  TimeSeriesResponse,
  UserAssignmentPayload,
  AdminUser,
  UserRole,
  AuthUser,
  ProfileUpdatePayload,
  AdminCreateUserPayload,
} from '../types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api/v1';

const clearAuthStorage = (): void => {
  localStorage.removeItem('access_token');
  localStorage.removeItem('user_role');
  localStorage.removeItem('user_full_name');
  localStorage.removeItem('username');
};

export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
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
  createTest: (data: ABTestCreatePayload) => api.post('/tests/', data),
  assignUser: (testId: string, data: UserAssignmentPayload) => api.post(`/tests/${testId}/assign`, data),
  getResults: (testId: string) => api.get(`/tests/${testId}/results`),
  getAllTests: () => api.get('/tests/'),
  stopTest: (testId: string, reason: string) => api.post(`/tests/${testId}/stop`, { reason }),
  startSimulation: (testId: string, data: StartSimulationPayload) => api.post(`/tests/${testId}/start-simulation`, data),
  pauseTest: (testId: string, reason?: string) => api.post(`/tests/${testId}/pause`, reason ? { reason } : {}),
  resumeTest: (testId: string) => api.post(`/tests/${testId}/resume`),
  deleteTestWithOption: (testId: string, moveToArchived: boolean) =>
    api.post(`/tests/${testId}/delete-with-option`, { move_to_archived: moveToArchived }),
  archiveTest: (testId: string, reason?: string) => api.post(`/tests/${testId}/archive`, reason ? { reason } : {}),
  permanentlyDeleteTest: (testId: string) => api.delete(`/tests/${testId}/permanent`),
};

// Data Generation API
export const dataAPI = {
  generateRealData: (data: { num_samples: number; save_to_file?: boolean; include_evaluation?: boolean; filters?: Record<string, unknown> }) =>
    api.post('/data/generate-real', data),
  trainGAN: (data: { epochs: number; real_data_samples: number; save_checkpoint: boolean; checkpoint_name?: string; gan_config?: Record<string, unknown> }) =>
    api.post('/data/train-gan', data),
  stopGANTraining: () => api.post('/data/stop-gan-training'),
  resumeGANTraining: () => api.post('/data/resume-gan-training'),
  resetGANTraining: () => api.post('/data/reset-gan-training'),
  generateSynthetic: (data: SyntheticGenerationPayload) =>
    api.post('/data/generate-synthetic', data, { timeout: 120000 }),
  getGANStatus: () => api.get('/data/gan-status'),
  getGANCheckpoints: () => api.get('/data/gan-checkpoints'),
  deleteGANCheckpoint: (checkpointId: number) => api.delete(`/data/gan-checkpoints/${checkpointId}`),
  getDatasetStats: () => api.get('/data/dataset-stats'),
  listGeneratedHistory: (limit: number = 50) => api.get(`/data/generated-history?limit=${limit}`),
  getFullDataset: (itemId: number) => api.get(`/data/generated-history/${itemId}/full`),
  deleteGeneratedHistoryItem: (itemId: number) => api.delete(`/data/generated-history/${itemId}`),
  loadGANCheckpoint: (checkpointName: string) => api.post('/data/gan-load-checkpoint', { checkpoint_name: checkpointName }),
  runABTestOnSynthetic: (data: StartSimulationPayload & { test_id: string }) => api.post('/data/run-ab-test-simulation', data),
};

// Templates API
export const templatesAPI = {
  listTemplates: (templateType?: string) =>
    api.get('/templates/', { params: templateType ? { template_type: templateType } : {} }),
  createTemplate: (data: Record<string, unknown>) => api.post('/templates/', data),
  getTemplate: (id: number) => api.get(`/templates/${id}`),
  updateTemplate: (id: number, data: Record<string, unknown>) => api.put(`/templates/${id}`, data),
  deleteTemplate: (id: number) => api.delete(`/templates/${id}`),
  seedDefaults: () => api.post('/templates/seed-defaults'),
};

// Results API
export const resultsAPI = {
  getDetailedResults: (testId: string) => api.get(`/results/${testId}/detailed`),
  getStatisticalSignificance: (testId: string, alpha: number = 0.05) =>
    api.get(`/results/${testId}/statistical-significance?alpha=${alpha}`),
  getPlatformStats: () => api.get('/results/platform/performance'),
  getTimeSeriesData: (testId: string) => api.get<TimeSeriesResponse>(`/results/${testId}/time-series-data`),
};

// Auth API
export const authAPI = {
  login: (credentials: LoginCredentials) => {
    const payload = new URLSearchParams();
    payload.append('username', credentials.username);
    payload.append('password', credentials.password);

    return api.post('/auth/login', payload, {
      timeout: 30000,
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
    });
  },
  me: () => api.get<AuthUser>('/auth/me', { timeout: 30000 }),
  updateProfile: (payload: ProfileUpdatePayload) => api.put<AuthUser>('/auth/me/profile', payload),
  uploadAvatar: (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    return api.post<AuthUser>('/auth/me/avatar', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 60000,
    });
  },
  getAvatarBlob: () => api.get<Blob>('/auth/me/avatar', { responseType: 'blob', timeout: 30000 }),
  logout: () => api.post('/auth/logout'),
};

export const adminAPI = {
  listUsers: () => api.get<{ items: AdminUser[]; count: number }>('/auth/admin/users'),
  updateUserRole: (userId: number, role: UserRole) =>
    api.put<AdminUser & { message: string }>(`/auth/admin/users/${userId}/role`, { role }),
  updateUserPermissions: (userId: number, permissions: string[]) =>
    api.put<AdminUser & { message: string }>(`/auth/admin/users/${userId}/permissions`, { permissions }),
  createUser: (payload: AdminCreateUserPayload) =>
    api.post<AdminUser & { message: string }>('/auth/admin/users', payload),
  getUserAvatarBlob: (userId: number) => api.get<Blob>(`/auth/admin/users/${userId}/avatar`, { responseType: 'blob', timeout: 30000 }),
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
    (window as typeof window & { debugAPI?: typeof debugAPI }).debugAPI = debugAPI;
  }
}
