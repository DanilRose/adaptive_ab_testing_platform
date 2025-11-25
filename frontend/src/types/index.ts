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