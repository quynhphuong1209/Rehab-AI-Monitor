export type User = {
  username: string;
  full_name?: string;
  email?: string;
  role: string;
  active?: boolean;
  must_change_password?: boolean;
};

export type AuditLogRecord = {
  id?: string;
  timestamp?: string;
  actor?: string;
  actor_role?: string;
  action?: string;
  target?: string;
  result?: string;
  metadata?: Record<string, unknown>;
};

export type VideoRecord = {
  username?: string;
  patient_username?: string;
  full_name?: string;
  subject_code?: string;
  video_name?: string;
  original_filename?: string;
  stored_filename?: string;
  video_path?: string;
  processed_path?: string | null;
  exercise?: string;
  status?: string;
  time?: string;
  accuracy?: number;
  metrics?: Record<string, unknown>;
};

export type PatientRecord = User & {
  subject_code?: string;
  assigned_doctor_username?: string;
  assigned_patient_usernames?: string[];
  team_usernames?: string[];
};

export type EvaluationRecord = {
  id?: string;
  patient_username?: string;
  video_name?: string;
  exercise?: string;
  doctor_username?: string;
  doctor_name?: string;
  doctor_result?: string;
  errors?: string[];
  comments?: string;
  comments_ncv?: string;
  plan?: string;
  time?: string;
};

export type SymptomRecord = {
  username?: string;
  patient_username?: string;
  full_name?: string;
  subject_code?: string;
  symptoms?: string;
  pain_score?: number;
  pain_level?: string;
  notes?: string;
  created_at?: string;
  timestamp?: string;
  time?: string;
  [key: string]: unknown;
};

export type CreateSymptomPayload = {
  full_name: string;
  patient_id: string;
  age: number;
  gender: string;
  exercise: string;
  symptoms: string;
  vas: number;
};

export type UploadVideoPayload = {
  file: File;
  full_name: string;
  exercise: string;
};

export type AnalysisJob = {
  job_id: string;
  run_id?: string;
  video_path?: string;
  username?: string;
  video_name?: string;
  exercise?: string;
  status: string;
  progress: number;
  elapsed?: number;
  start_time?: number;
  heartbeat?: number;
  updated_at?: string;
  status_msg?: string;
  error_msg?: string;
  result?: Record<string, unknown> | null;
  job_meta?: {
    requested_by?: string;
    action?: string;
    options?: AnalysisJobOptions;
    canceled_by?: string;
    canceled_at?: string;
    [key: string]: unknown;
  };
  steps?: Array<{
    key: string;
    label: string;
    status: 'done' | 'active' | 'pending' | 'error' | 'canceled' | string;
  }>;
};

export type AnalysisJobStartResult = {
  started: boolean;
  reason: string;
  job: AnalysisJob;
};

export type AnalysisJobActionResult = {
  ok?: boolean;
  reason?: string;
  job: AnalysisJob | null;
};

export type AnalysisJobHistoryResult = {
  items: AnalysisJob[];
  count: number;
};

export type AnalysisJobOptions = {
  model_type: 'MediaPipe Heavy' | 'MediaPipe Full' | 'MediaPipe Lite';
  skip_step: number;
  resize_width: number;
  min_confidence: number;
};

export type AnalysisArtifactItem = {
  kind: string;
  label: string;
  filename: string;
  available: boolean;
  size?: number;
  download_url: string;
};

export type AnalysisArtifactsResult = {
  video: {
    stored_filename: string;
    video_name: string;
    exercise: string;
    status: string;
    accuracy?: number;
  };
  metrics: Record<string, unknown>;
  items: AnalysisArtifactItem[];
  count: number;
};

export type ResultTimelineItem = {
  kind: string;
  label: string;
  time?: string;
  detail?: string;
  status?: string;
};

export type VideoResultDetail = {
  video: VideoRecord & {
    stored_filename: string;
    video_name: string;
  };
  evaluation: EvaluationRecord | null;
  latest_job: AnalysisJob | null;
  metrics: Record<string, unknown>;
  artifacts: AnalysisArtifactItem[];
  artifact_count: number;
  report_sent: boolean;
  ai_detail_allowed: boolean;
  report_status: {
    report_sent: boolean;
    report_status: string;
    ai_detail_allowed: boolean;
    sent_at?: string;
    sent_by?: string;
    message?: string;
  };
  phase_metrics: Record<string, PhaseMetricSummary>;
  summary: {
    patient?: string;
    video_name?: string;
    exercise?: string;
    status?: string;
    accuracy?: number;
    doctor_result?: string;
    doctor_plan?: string;
    doctor_comment?: string;
    analysis_status?: string;
    analysis_message?: string;
  };
  timeline: ResultTimelineItem[];
};

export type FrameLabel = 'ALL' | 'G1' | 'G2' | 'G3' | 'PASS' | 'NEAR' | 'FAIL';
export type RefLabel = 'PASS' | 'NEAR' | 'FAIL';

export type PhaseCountSummary = {
  total: number;
  PASS: number;
  NEAR: number;
  FAIL: number;
  threshold?: number;
};

export type PhaseMetricSummary = {
  threshold?: number;
  accuracy?: number | null;
  mae?: number | null;
  f1?: number | null;
  icc?: number | null;
};

export type MlFrameBadge = {
  label?: string;
  label_text?: string;
  confidence?: number | null;
  probabilities?: Record<string, number>;
};

export type AnalysisFrameItem = {
  index: number;
  timestamp?: string;
  label: RefLabel;
  phase?: 'G1' | 'G2' | 'G3';
  phase_label?: string;
  phase_threshold?: number;
  image_id: string;
  has_image: boolean;
  goc_vai?: number | null;
  goc_khuyu?: number | null;
  goc_vai_trai?: number | null;
  goc_vai_phai?: number | null;
  goc_khuyu_trai?: number | null;
  goc_khuyu_phai?: number | null;
  shoulder_ref?: number | null;
  elbow_ref?: number | null;
  shoulder_delta?: number | null;
  elbow_delta?: number | null;
  ml?: MlFrameBadge;
  detected?: boolean;
  filtered_stranger?: boolean;
};

export type AnalysisFramesResult = {
  items: AnalysisFrameItem[];
  summary: {
    total: number;
    PASS: number;
    NEAR: number;
    FAIL: number;
    phases?: Record<'G1' | 'G2' | 'G3', PhaseCountSummary>;
  };
  pagination: {
    page: number;
    page_size: number;
    total: number;
    total_pages: number;
  };
  filter: FrameLabel;
  segment_bounds?: number[];
  phase_ranges?: Record<'G1' | 'G2' | 'G3', { start: number; end: number; threshold: number }>;
  sources: {
    frames_json: boolean;
    frames_zip: boolean;
  };
};

export type ChartSeriesPoint = {
  index: number;
  frame?: number | string;
  timestamp?: string;
  label?: '' | 'PASS' | 'NEAR' | 'FAIL';
  phase?: 'G1' | 'G2' | 'G3';
  phase_threshold?: number;
  goc_vai?: number | null;
  goc_khuyu?: number | null;
  vai_chuan?: number | null;
  khuyu_chuan?: number | null;
};

export type ChartSeriesSummary = {
  count: number;
  min: number | null;
  max: number | null;
  avg: number | null;
};

export type AnalysisChartPreview = {
  source: 'csv' | 'frames-json' | 'none';
  source_label: string;
  filter: FrameLabel;
  total_rows: number;
  filtered_rows: number;
  sampled_rows: number;
  segment_bounds?: number[];
  columns: Array<'goc_vai' | 'goc_khuyu' | 'vai_chuan' | 'khuyu_chuan'>;
  summary: {
    series: Record<string, ChartSeriesSummary>;
    labels: {
      total: number;
      PASS: number;
      NEAR: number;
      FAIL: number;
    };
  };
  phase_summary?: {
    total: number;
    PASS: number;
    NEAR: number;
    FAIL: number;
    phases: Record<'G1' | 'G2' | 'G3', PhaseCountSummary>;
  };
  phase_metrics: Record<string, PhaseMetricSummary>;
  metrics: Record<string, unknown>;
  series: ChartSeriesPoint[];
};

export type CreateEvaluationPayload = {
  patient_username: string;
  video_name: string;
  exercise: string;
  doctor_result: string;
  errors: string[];
  comments: string;
  comments_ncv: string;
  plan: string;
};

export type ScheduleRecord = {
  id?: string;
  username?: string;
  patient_username?: string;
  patient_name?: string;
  full_name?: string;
  subject_code?: string;
  title?: string;
  type?: string;
  datetime?: string;
  date?: string;
  time?: string;
  status?: string;
  notes?: string;
  exercise_name?: string;
  frequency?: string;
  medication_name?: string;
  dosage?: string;
  [key: string]: unknown;
};

export type ResearchRecord = {
  id?: string;
  username?: string;
  patient_username?: string;
  full_name?: string;
  subject_code?: string;
  video_name?: string;
  exercise?: string;
  general_result?: string;
  doctor_result?: string;
  result?: string;
  created_at?: string;
  timestamp?: string;
  time?: string;
  [key: string]: unknown;
};

export type CreateSchedulePayload = {
  patient_username: string;
  type: string;
  title?: string;
  datetime?: string;
  notes?: string;
  exercise_name?: string;
  frequency?: string;
  medication_name?: string;
  dosage?: string;
};

export type CreateResearchPayload = {
  patient_username: string;
  subject_code: string;
  age: number;
  gender: string;
  diagnosis: string;
  exercise: string;
  general_result: string;
  plan: string;
  specialist_comment: string;
  recording_device: string;
  recording_angle: string;
};

export type LoginResult = {
  access_token: string;
  token_type: 'bearer';
  user: User;
};

export type RegisterPayload = {
  username: string;
  full_name: string;
  email: string;
  password: string;
  confirm_password: string;
};

export type ChangePasswordPayload = {
  old_password: string;
  new_password: string;
  confirm_password: string;
};

export type CreateUserPayload = {
  username: string;
  full_name: string;
  email: string;
  password: string;
  role: string;
  assigned_patient_usernames?: string[];
};

export type ResetUserPasswordPayload = {
  password: string;
  confirm_password: string;
};

export type UserAdminActionResult = {
  item: PatientRecord;
  revoked_sessions?: number;
};

export type RevokeSessionsResult = {
  ok: boolean;
  scope: 'user' | 'all';
  username?: string;
  revoked_sessions: number;
  global_session_version?: number;
};

export type ListResult<T> = {
  items: T[];
  count: number;
};

const API_BASE_URL = (import.meta.env.VITE_REHAB_API_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, options: RequestInit = {}, token?: string): Promise<T> {
  const headers = new Headers(options.headers);
  headers.set('Content-Type', 'application/json');
  if (token) {
    headers.set('Authorization', `Bearer ${token}`);
  }
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers,
  });
  if (!response.ok) {
    let message = 'Backend API error';
    try {
      const body = (await response.json()) as { detail?: string };
      message = body.detail || message;
    } catch {
      message = response.statusText || message;
    }
    throw new ApiError(message, response.status);
  }
  return (await response.json()) as T;
}

async function multipartRequest<T>(path: string, formData: FormData, token?: string): Promise<T> {
  const headers = new Headers();
  if (token) {
    headers.set('Authorization', `Bearer ${token}`);
  }
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: 'POST',
    headers,
    body: formData,
  });
  if (!response.ok) {
    let message = 'Backend API error';
    try {
      const body = (await response.json()) as { detail?: string };
      message = body.detail || message;
    } catch {
      message = response.statusText || message;
    }
    throw new ApiError(message, response.status);
  }
  return (await response.json()) as T;
}

async function blobRequest(path: string, token?: string): Promise<Blob> {
  const headers = new Headers();
  if (token) {
    headers.set('Authorization', `Bearer ${token}`);
  }
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers,
  });
  if (!response.ok) {
    let message = 'Backend API error';
    try {
      const body = (await response.json()) as { detail?: string };
      message = body.detail || message;
    } catch {
      message = response.statusText || message;
    }
    throw new ApiError(message, response.status);
  }
  return response.blob();
}

export const api = {
  baseUrl: API_BASE_URL,
  health: () => request<{ status: string }>('/health'),
  login: (username: string, password: string) =>
    request<LoginResult>('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ username, password }),
    }),
  register: (payload: RegisterPayload) =>
    request<LoginResult>('/auth/register', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  me: (token: string) => request<{ user: User }>('/auth/me', {}, token),
  changePassword: (token: string, payload: ChangePasswordPayload) =>
    request<{ user: User }>(
      '/auth/change-password',
      {
        method: 'POST',
        body: JSON.stringify(payload),
      },
      token,
    ),
  logout: (token: string) => request<{ ok: boolean }>('/auth/logout', { method: 'POST' }, token),
  adminUsers: (token: string) => request<ListResult<PatientRecord>>('/admin/users', {}, token),
  adminAuditLog: (token: string, limit = 100) =>
    request<ListResult<AuditLogRecord>>(`/admin/audit-log?limit=${encodeURIComponent(String(limit))}`, {}, token),
  createUser: (token: string, payload: CreateUserPayload) =>
    request<{ item: PatientRecord }>(
      '/admin/users',
      {
        method: 'POST',
        body: JSON.stringify(payload),
      },
      token,
    ),
  setUserActive: (token: string, username: string, active: boolean) =>
    request<UserAdminActionResult>(
      `/admin/users/${encodeURIComponent(username)}/active`,
      {
        method: 'POST',
        body: JSON.stringify({ active }),
      },
      token,
    ),
  resetUserPassword: (token: string, username: string, payload: ResetUserPasswordPayload) =>
    request<UserAdminActionResult>(
      `/admin/users/${encodeURIComponent(username)}/reset-password`,
      {
        method: 'POST',
        body: JSON.stringify(payload),
      },
      token,
    ),
  revokeUserSessions: (token: string, username: string, reason = 'admin') =>
    request<RevokeSessionsResult>(
      `/admin/users/${encodeURIComponent(username)}/sessions/revoke`,
      {
        method: 'POST',
        body: JSON.stringify({ reason }),
      },
      token,
    ),
  revokeAllSessions: (token: string, reason: string, confirm: string) =>
    request<RevokeSessionsResult>(
      '/admin/sessions/revoke',
      {
        method: 'POST',
        body: JSON.stringify({ reason, confirm }),
      },
      token,
    ),
  deleteUser: (token: string, username: string) =>
    request<{ ok: boolean; username: string }>(
      `/admin/users/${encodeURIComponent(username)}`,
      {
        method: 'DELETE',
      },
      token,
    ),
  videos: (token: string) => request<ListResult<VideoRecord>>('/videos', {}, token),
  videoBlob: (token: string, storedFilename: string) =>
    blobRequest(`/videos/media/${encodeURIComponent(storedFilename)}`, token),
  startAnalysisJob: (token: string, storedFilename: string, options?: Partial<AnalysisJobOptions>) =>
    request<AnalysisJobStartResult>(
      `/videos/${encodeURIComponent(storedFilename)}/analysis-jobs`,
      {
        method: 'POST',
        body: JSON.stringify(options || {}),
      },
      token,
    ),
  latestAnalysisJob: (token: string, storedFilename: string) =>
    request<{ job: AnalysisJob | null }>(`/videos/${encodeURIComponent(storedFilename)}/analysis-jobs/latest`, {}, token),
  analysisJobHistory: (token: string, storedFilename: string) =>
    request<AnalysisJobHistoryResult>(`/videos/${encodeURIComponent(storedFilename)}/analysis-jobs/history`, {}, token),
  cancelAnalysisJob: (token: string, storedFilename: string) =>
    request<AnalysisJobActionResult>(
      `/videos/${encodeURIComponent(storedFilename)}/analysis-jobs/cancel`,
      {
        method: 'POST',
        body: JSON.stringify({}),
      },
      token,
    ),
  retryAnalysisJob: (token: string, storedFilename: string) =>
    request<AnalysisJobStartResult>(
      `/videos/${encodeURIComponent(storedFilename)}/analysis-jobs/retry`,
      {
        method: 'POST',
        body: JSON.stringify({}),
      },
      token,
    ),
  rerunAnalysisJob: (token: string, storedFilename: string, options: Partial<AnalysisJobOptions>) =>
    request<AnalysisJobStartResult>(
      `/videos/${encodeURIComponent(storedFilename)}/analysis-jobs/rerun`,
      {
        method: 'POST',
        body: JSON.stringify(options),
      },
      token,
    ),
  videoResult: (token: string, storedFilename: string) =>
    request<VideoResultDetail>(`/videos/${encodeURIComponent(storedFilename)}/results`, {}, token),
  analysisFrames: (token: string, storedFilename: string, params: { page: number; pageSize: number; label: FrameLabel }) =>
    request<AnalysisFramesResult>(
      `/videos/${encodeURIComponent(storedFilename)}/analysis-frames?page=${encodeURIComponent(params.page)}&page_size=${encodeURIComponent(
        params.pageSize,
      )}&label=${encodeURIComponent(params.label)}`,
      {},
      token,
    ),
  analysisFrameBlob: (token: string, storedFilename: string, imageId: string) =>
    blobRequest(`/videos/${encodeURIComponent(storedFilename)}/analysis-frames/${encodeURIComponent(imageId)}`, token),
  analysisChart: (token: string, storedFilename: string, label: FrameLabel = 'ALL') =>
    request<AnalysisChartPreview>(`/videos/${encodeURIComponent(storedFilename)}/analysis-chart?label=${encodeURIComponent(label)}`, {}, token),
  analysisArtifacts: (token: string, storedFilename: string) =>
    request<AnalysisArtifactsResult>(`/videos/${encodeURIComponent(storedFilename)}/analysis-artifacts`, {}, token),
  artifactBlob: (token: string, storedFilename: string, kind: string) =>
    blobRequest(`/videos/${encodeURIComponent(storedFilename)}/analysis-artifacts/${encodeURIComponent(kind)}`, token),
  uploadVideo: (token: string, payload: UploadVideoPayload) => {
    const formData = new FormData();
    formData.set('file', payload.file);
    formData.set('full_name', payload.full_name);
    formData.set('exercise', payload.exercise);
    return multipartRequest<{ item: VideoRecord }>('/videos/upload', formData, token);
  },
  evaluations: (token: string) => request<ListResult<EvaluationRecord>>('/evaluations', {}, token),
  createEvaluation: (token: string, payload: CreateEvaluationPayload) =>
    request<{ item: EvaluationRecord }>(
      '/evaluations',
      {
        method: 'POST',
        body: JSON.stringify(payload),
      },
      token,
    ),
  deleteEvaluation: (token: string, id: string) =>
    request<{ ok: boolean; item?: EvaluationRecord }>(
      `/evaluations/${encodeURIComponent(id)}`,
      {
        method: 'DELETE',
      },
      token,
    ),
  patients: (token: string) => request<ListResult<PatientRecord>>('/patients', {}, token),
  symptoms: (token: string) => request<ListResult<SymptomRecord>>('/symptoms', {}, token),
  createSymptom: (token: string, payload: CreateSymptomPayload) =>
    request<{ item: SymptomRecord }>(
      '/symptoms',
      {
        method: 'POST',
        body: JSON.stringify(payload),
      },
      token,
    ),
  schedules: (token: string) => request<ListResult<ScheduleRecord>>('/schedules', {}, token),
  createSchedule: (token: string, payload: CreateSchedulePayload) =>
    request<{ item: ScheduleRecord }>(
      '/schedules',
      {
        method: 'POST',
        body: JSON.stringify(payload),
      },
      token,
    ),
  deleteSchedule: (token: string, id: string) =>
    request<{ ok: boolean; item?: ScheduleRecord }>(
      `/schedules/${encodeURIComponent(id)}`,
      {
        method: 'DELETE',
      },
      token,
    ),
  researchRecords: (token: string) => request<ListResult<ResearchRecord>>('/research-records', {}, token),
  createResearchRecord: (token: string, payload: CreateResearchPayload) =>
    request<{ item: ResearchRecord }>(
      '/research-records',
      {
        method: 'POST',
        body: JSON.stringify(payload),
      },
      token,
    ),
  deleteResearchRecord: (token: string, id: string) =>
    request<{ ok: boolean; item?: ResearchRecord }>(
      `/research-records/${encodeURIComponent(id)}`,
      {
        method: 'DELETE',
      },
      token,
    ),
};
