export interface HealthResponse {
  status: string;
  model_ready: boolean;
  model_path?: string | null;
}

export interface ModelInfoResponse {
  model_type?: string | null;
  subset?: string | null;
  trained_at?: string | null;
  n_features?: number | null;
  feature_names: string[];
  rul_clip?: number | null;
  validation_metrics?: { mae: number; rmse: number } | null;
  test_metrics?: { mae: number; rmse: number } | null;
  feature_importances: { feature: string; importance: number }[];
  model_params: Record<string, unknown>;
  status: string;
}

export interface FeatureContribution {
  feature: string;
  importance: number;
  value?: number | null;
}

export interface EvaluationPoint {
  engine_id: number;
  true_rul: number;
  predicted_rul: number;
  error: number;
}

export interface EvaluationHistogramBin {
  bin_start: number;
  bin_end: number;
  count: number;
}

export interface EvaluationDataResponse {
  points: EvaluationPoint[];
  histogram: EvaluationHistogramBin[];
  mae: number;
  rmse: number;
}

export interface FeatureStats {
  median: number;
  min: number;
  max: number;
}

export interface SampleEngine {
  id: number;
  category: string;
  final_rul: number;
  cycles: number[];
  feature_names: string[];
  features: number[][];
  true_rul: number[];
  predicted_rul?: number[];
}

export interface SampleEnginesResponse {
  feature_names: string[];
  feature_stats: Record<string, FeatureStats>;
  engines: SampleEngine[];
}

export interface PredictRequest {
  features?: Record<string, number>;
  feature_vector?: number[];
  unit_id?: number;
}

export interface PredictResponse {
  predicted_rul: number;
  health_status: "Normal" | "Warning" | "Critical";
  explanation: string;
  unit_id?: number | null;
  features_used: number;
  feature_contributions: FeatureContribution[];
  model_ready: boolean;
  thresholds: Record<string, number>;
  model_version: string;
  rul_low: number;
  rul_high: number;
}

export interface HistoryCyclePrediction {
  cycle: number;
  predicted_rul: number;
}

export interface HistoryEnginePrediction {
  engine_id: string;
  latest_cycle: number;
  predicted_rul: number;
  health_status: "Normal" | "Warning" | "Critical";
  rul_low: number;
  rul_high: number;
  rul_by_cycle: HistoryCyclePrediction[];
}

export interface PredictHistoryResponse {
  engines: HistoryEnginePrediction[];
  model_version: string;
  feature_names: string[];
}

export interface ApiError {
  detail: string | unknown;
}
