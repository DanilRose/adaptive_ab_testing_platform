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

export type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonObject | JsonValue[];
export interface JsonObject {
  [key: string]: JsonValue;
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
  filters?: Record<string, JsonValue>;
  dataset_name?: string;
}

export interface ABTestCreatePayload {
  test_name: string;
  variants: string[];
  primary_metric: string;
  metric_type: 'binary' | 'continuous' | 'ratio';
  description?: string;
  sample_size?: number;
  confidence_level?: number;
  power?: number;
  min_effect_size?: number;
  dataset_id?: number;
  simulation_duration_minutes?: number;
  traffic_split_type?: 'fixed' | 'adaptive';
  variant_effects?: Record<string, Record<string, number>> | null;
  analysis_mode?: 'fixed_experiment' | 'adaptive_bandit';
  guardrails_config?: Record<string, { threshold: number; direction: 'max_increase' | 'max_decrease' | 'min_increase' | 'min_decrease' }> | null;
}

export interface StartSimulationPayload {
  dataset_id?: number;
  user_count?: number;
  strategy?: 'fixed' | 'adaptive';
  simulation_minutes?: number;
  variant_effects?: Record<string, Record<string, number>> | null;
}

export interface UserAssignmentPayload {
  user_id: string;
  user_context?: Record<string, JsonValue>;
}

export interface TimeSeriesPoint {
  users_processed: number;
  variant: string;
  cumulative_metric: number;
  mean_metric: number;
  sample_size: number;
  p_value: number | null;
  confidence_interval_lower: number | null;
  confidence_interval_upper: number | null;
}

export interface TimeSeriesResponse {
  test_id: string;
  variants: string[];
  data: TimeSeriesPoint[];
  total_snapshots: number;
  snapshots_per_variant: number;
  completion_percentage: number;
  stopped_early: boolean;
  early_stop_reason: string | null;
  current_sequential_look: number;
  max_sequential_looks: number;
  srm_check_passed: number | null;
  srm_p_value: number | null;
  traffic_split: {
    variant_counts: Record<string, number>;
    variant_percentages: Record<string, number>;
  };
  winner: string | null;
  winner_uplift_percent: number;
  winner_confidence: 'low' | 'medium' | 'high';
  power_over_time: Array<Record<string, number>>;
  uplift_over_time: Array<Record<string, number>>;
  analysis_mode?: 'fixed_experiment' | 'adaptive_bandit';
  analysis_validity?: 'valid_for_inference' | 'exploration_only' | 'invalid_srm' | 'invalid_guardrails';
  guardrails?: {
    enabled: boolean;
    passed: boolean;
    failed_metrics: string[];
    checks: Array<{
      metric: string;
      threshold: number;
      direction: string;
      observed: number;
      passed: boolean;
    }>;
  };
  quality_gate?: {
    status: 'green' | 'yellow' | 'red';
    passed: boolean;
    passed_checks: number;
    total_checks: number;
    checks: Array<{
      id: string;
      title: string;
      passed: boolean;
      actual: unknown;
      threshold: unknown;
      known?: boolean;
    }>;
  };
}

export interface GeneratedHistoryItem {
  id: number;
  data_type: 'real' | 'synthetic';
  sample_count: number;
  file_path: string | null;
  storage: string;
  dataset_name?: string;
  preview_json?: Array<Record<string, JsonValue>>;
  extra_metadata?: Record<string, JsonValue>;
  created_at?: string | null;
}
