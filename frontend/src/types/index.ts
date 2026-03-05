export interface TestConfig {
  test_id: string;
  variants: string[];
  primary_metric: string;
  metric_type: 'binary' | 'continuous' | 'ratio';
  sample_size?: number;
  confidence_level: number;
  power: number;
  min_effect_size: number;
}

export interface TestResult {
  variant: string;
  sample_size: number;
  mean: number;
  std: number;
  confidence_interval: [number, number];
  p_value?: number;
}

export interface TestSummary {
  test_id: string;
  results: Record<string, TestResult>;
  statistical_significance: Record<string, number>;
  summary: {
    best_variant: string;
    improvement_percentage: number;
    recommended_action: string;
    confidence_level: string;
  };
}

export type UserRole = 'developer' | 'analyst' | 'manager';

export interface AuthUser {
  id: number;
  username: string;
  role: UserRole;
  full_name: string;
}

export interface AuthTokenResponse {
  access_token: string;
  token_type: string;
  role: UserRole;
  full_name: string;
}

export interface LoginCredentials {
  username: string;
  password: string;
}

export interface GANConfigOverrides {
  [key: string]: string | number | boolean | number[];
}

export interface GANTrainingPayload {
  epochs: number;
  real_data_samples: number;
  save_checkpoint: boolean;
  checkpoint_name?: string;
  gan_config?: GANConfigOverrides;
}

export interface SyntheticGenerationPayload {
  num_users: number;
  evaluation_metrics: boolean;
  filters?: Record<string, any>;
  dataset_name?: string;
}
