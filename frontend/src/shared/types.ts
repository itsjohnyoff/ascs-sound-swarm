export type SyncState = "unsynced" | "synced" | "drifting" | "clock_jump";
export type SoundClass = "drone" | "clap" | "other";

export interface Room {
  width_m: number; height_m: number; temperature_c: number;
  speaker_x_m: number; speaker_y_m: number;
}
export interface Phone {
  phone_id: string; label: string; x_m: number; y_m: number; sample_rate: number;
  connected: boolean; rms: number; sync_state: SyncState;
  residual_ms: number | null; drift_ppm: number | null; chirp_snr_db: number | null;
  last_sync_ts: number | null; dropped_frames?: number;
}
export interface SessionState {
  type: "session_state"; room: Room; phones: Phone[];
  gdop_warning: string | null; calibration_status: string;
}
export interface PairTdoa { i: string; j: string; tdoa_ms: number; weight: number; inlier: boolean }
export interface Detection {
  type: "detection"; event_id: string; timestamp_s: number; x_m: number; y_m: number;
  uncertainty_m: number; residual_ms: number; confidence: number;
  sound_class: SoundClass; is_drone: boolean; class_confidence: number;
  class_probs: Record<string, number>; phones_used: string[];
  pair_tdoas_ms: PairTdoa[];
  heatmap?: { nx: number; ny: number; values: number[] } | null;
  update_seq?: number; kind?: "transient" | "sustained";
}
export interface PhoneSyncReport {
  synced: boolean; residual_ms: number | null; drift_ppm: number | null;
  chirp_snr_db: number | null; error: string | null;
}
export interface CalibrationResult {
  type?: "calibration_result"; mode: string; success: boolean; synced_count: number;
  median_residual_ms: number | null; max_residual_ms: number | null;
  per_phone: Record<string, PhoneSyncReport>;
}
export interface Trial {
  trial_idx: number; label: string; expected_class: SoundClass; gt_x_m: number; gt_y_m: number;
  est_x_m: number | null; est_y_m: number | null; error_m: number | null; missed: boolean;
  pred_class: SoundClass | null; class_confidence: number | null; residual_ms: number | null;
  timestamp_s: number; inside_hull?: boolean;
}
export interface EvalSummary {
  total: number; detected: number; missed: number;
  mean_error_m: number | null; median_error_m: number | null;
  p90_error_m: number | null; max_error_m: number | null;
}
export interface PhoneStatus {
  type: "phone_status"; phone_id: string; synced: boolean;
  residual_ms: number | null; drift_ppm: number | null; state: SyncState;
}
