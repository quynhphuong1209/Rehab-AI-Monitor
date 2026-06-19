import {
  Activity,
  AlertCircle,
  ArrowRight,
  CalendarDays,
  CheckCircle2,
  ClipboardList,
  Cpu,
  Download,
  Eye,
  EyeOff,
  FileVideo,
  FlaskConical,
  History,
  KeyRound,
  LineChart,
  Lock,
  LogOut,
  Maximize2,
  MessageSquare,
  NotebookTabs,
  RefreshCw,
  Shield,
  Trash2,
  Unlock,
  UserX,
  X,
  UserPlus,
  UserRound,
  UsersRound,
} from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import { FormEvent, useEffect, useMemo, useRef, useState } from 'react';

import {
  api,
  AuditLogRecord,
  ApiError,
  CleanupTarget,
  CreateEvaluationPayload,
  CreateFeedbackPayload,
  CreateResearchPayload,
  CreateSchedulePayload,
  CreateSymptomPayload,
  CreateUserPayload,
  EvaluationRecord,
  FeedbackRecord,
  HfSyncJob,
  HfSyncStatus,
  PatientRecord,
  PoseClassifierJob,
  PoseClassifierModelStatus,
  ResearchRecord,
  ScheduleRecord,
  SymptomRecord,
  AnalysisJob,
  AnalysisJobOptions,
  AnalysisChartPreview,
  AnalysisFrameItem,
  AnalysisFramesResult,
  FrameLabel,
  VideoResultDetail,
  User,
  VideoRecord,
} from './api';

type Session = {
  token: string;
  user: User;
};

type VideoPreview = {
  key: string;
  url: string;
  label: string;
};

type FocusedFrame = {
  frame: AnalysisFrameItem;
  url: string;
};

type LoadState = 'idle' | 'loading' | 'ready' | 'error';
type AuthMode = 'login' | 'register';
type ViewId = 'home' | 'videos' | 'patients' | 'symptoms' | 'schedules' | 'research' | 'info' | 'users';
type RecordLike = Record<string, unknown>;

const PATIENT_ROLE = 'Bệnh nhân';
const DOCTOR_ROLE = 'Bác sĩ / KTV PHCN';
const RESEARCHER_ROLE = 'Nghiên cứu viên';
const ADMIN_ROLE = 'Quản trị viên';

function canStartAnalysis(role: string) {
  return role === ADMIN_ROLE || role === RESEARCHER_ROLE;
}

function canManagePoseClassifier(role: string) {
  return role === ADMIN_ROLE || role === RESEARCHER_ROLE;
}

function canManageHfSync(role: string) {
  return role === ADMIN_ROLE || role === RESEARCHER_ROLE;
}

function canViewClinicalResult(role: string) {
  return role === ADMIN_ROLE || role === DOCTOR_ROLE;
}

function canViewSchedules(role: string) {
  return role === ADMIN_ROLE || role === DOCTOR_ROLE || role === PATIENT_ROLE;
}

function canViewResearch(role: string) {
  return role === ADMIN_ROLE || role === DOCTOR_ROLE || role === RESEARCHER_ROLE;
}

function canCreateResearch(role: string) {
  return role === ADMIN_ROLE || role === DOCTOR_ROLE;
}

function canManageClinical(role: string) {
  return role === ADMIN_ROLE || role === DOCTOR_ROLE;
}

function canManageUsers(role: string) {
  return role === ADMIN_ROLE;
}

function canReviewFeedback(role: string) {
  return role === ADMIN_ROLE || role === RESEARCHER_ROLE;
}

function roleWorkspaceTitle(role: string) {
  if (role === PATIENT_ROLE) {
    return 'Không gian tập luyện của bệnh nhân';
  }
  if (role === DOCTOR_ROLE) {
    return 'Bàn làm việc bác sĩ / KTV PHCN';
  }
  if (role === RESEARCHER_ROLE) {
    return 'Không gian phân tích nghiên cứu';
  }
  if (role === ADMIN_ROLE) {
    return 'Bảng điều hành quản trị';
  }
  return 'Workspace phục hồi chức năng';
}

function roleWorkspaceEyebrow(role: string) {
  if (role === PATIENT_ROLE) {
    return 'Bệnh nhân';
  }
  if (role === DOCTOR_ROLE) {
    return 'Lâm sàng';
  }
  if (role === RESEARCHER_ROLE) {
    return 'Nghiên cứu';
  }
  if (role === ADMIN_ROLE) {
    return 'Quản trị';
  }
  return 'Dashboard dữ liệu';
}

function viewLabelForRole(view: ViewId, role: string) {
  const labels: Record<ViewId, string> = {
    home: 'Trang chủ',
    videos: role === PATIENT_ROLE ? 'Video tập luyện' : role === RESEARCHER_ROLE ? 'Hàng đợi AI' : 'Video bệnh nhân',
    patients: role === PATIENT_ROLE ? 'Hồ sơ của tôi' : 'Bệnh nhân',
    symptoms: role === PATIENT_ROLE ? 'Khai báo đau' : 'Triệu chứng',
    schedules: role === PATIENT_ROLE ? 'Lịch của tôi' : 'Lịch nhắc',
    research: role === RESEARCHER_ROLE ? 'Dữ liệu NCKH' : 'Phiếu nghiên cứu',
    info: 'Thông tin',
    users: 'Người dùng',
  };
  return labels[view];
}

function roleGuideSteps(role: string) {
  if (role === PATIENT_ROLE) {
    return [
      ['Chuẩn bị quay', 'Đứng cách camera 2-3 mét, đủ sáng, thấy rõ thân người và vùng vai/khuỷu.'],
      ['Gửi dữ liệu', 'Upload video bài tập, khai báo triệu chứng VAS và theo dõi lịch nhắc trong workspace.'],
      ['Xem phản hồi', 'Mở kết quả chi tiết sau khi nhóm điều trị phân tích và bác sĩ gửi nhận xét.'],
    ];
  }
  if (role === DOCTOR_ROLE) {
    return [
      ['Rà soát video', 'Xem video bệnh nhân trong phạm vi được phân công và đối chiếu kết quả AI khi đã được gửi.'],
      ['Đánh giá lâm sàng', 'Ghi nhận đúng/sai/gần đúng, lỗi kỹ thuật, nhận xét và kế hoạch tiếp theo.'],
      ['Theo dõi chăm sóc', 'Tạo lịch nhắc hẹn, lịch tập hoặc lịch thuốc để bệnh nhân duy trì tuân thủ.'],
    ];
  }
  if (role === RESEARCHER_ROLE) {
    return [
      ['Chuẩn hóa dữ liệu', 'Chạy job AI, xem artifact CSV/frame/chart và kiểm tra job history theo video.'],
      ['Huấn luyện ML', 'Dùng pose classifier ở chế độ dry-run trước khi train/apply thật cho dữ liệu đã phân tích.'],
      ['Đồng bộ nghiên cứu', 'Thực hiện HF sync/report có guard, không đồng bộ users.json và không đưa token lên frontend.'],
    ];
  }
  if (role === ADMIN_ROLE) {
    return [
      ['Quản lý truy cập', 'Tạo tài khoản, khóa/mở khóa, reset mật khẩu và thu hồi phiên khi cần.'],
      ['Vận hành dữ liệu', 'Xem audit log, backup/reset từng nhóm dữ liệu và theo dõi trạng thái backend.'],
      ['Giữ an toàn', 'Kiểm tra quyền, CORS, secrets server-side và chỉ thao tác destructive khi đã xác nhận.'],
    ];
  }
  return [['Bắt đầu', 'Đăng nhập đúng vai trò để xem luồng công việc phù hợp.']];
}

function feedbackCategoryLabel(category: unknown) {
  const labels: Record<string, string> = {
    general: 'Góp ý chung',
    bug: 'Lỗi hệ thống',
    workflow: 'Quy trình',
    content: 'Nội dung',
    safety: 'An toàn dữ liệu',
  };
  const key = String(category || '');
  return labels[key] || 'Góp ý chung';
}

const rehabKnowledge = [
  ['Bốn trụ cột y tế', 'Phục hồi chức năng bổ sung cho phòng bệnh, điều trị và nâng cao sức khỏe bằng mục tiêu khôi phục hoạt động hằng ngày.'],
  ['Lượng giá trước tập', 'Bài tập nên bắt đầu từ đánh giá tầm vận động, mức đau, bên tổn thương và khả năng phối hợp.'],
  ['Theo dõi an toàn', 'Ngưng hoặc giảm cường độ khi đau tăng nhanh, chóng mặt, tê lan hoặc mệt bất thường; báo lại cho nhân viên y tế.'],
  ['Duy trì tại nhà', 'Tập đều, ghi triệu chứng và bám lịch nhắc giúp nhóm điều trị điều chỉnh kế hoạch theo tiến triển thực tế.'],
];

const aiKnowledge = [
  ['MediaPipe Pose', 'Ước lượng 33 điểm mốc cơ thể để trích xuất góc vai/khuỷu và trạng thái chuyển động từ video.'],
  ['Artifact phân tích', 'CSV góc, JSON frame, ảnh frame và video xử lý là dữ liệu đối soát giữa AI, bác sĩ và nghiên cứu viên.'],
  ['Pose classifier', 'Mô hình ML gán nhãn đúng/gần đúng/sai trên dữ liệu đã trích xuất, có checksum sidecar trước khi apply.'],
  ['Human-in-the-loop', 'AI hỗ trợ đo lường; quyết định lâm sàng vẫn cần bác sĩ/KTV PHCN đánh giá trong bối cảnh từng bệnh nhân.'],
];

const researchInfoSections = [
  ['Đề tài', 'Phát triển mô hình thử nghiệm giám sát tập luyện phục hồi chức năng từ xa dựa trên AI và thị giác máy tính.'],
  ['Bài tập mục tiêu', 'Con lắc Codman, bài tập với gậy và bài tập với dây kháng lực cho người bệnh viêm quanh khớp vai.'],
  ['Dữ liệu thu thập', 'Video bài tập, góc khớp, nhãn đánh giá, phiếu nghiên cứu và phản hồi sử dụng trong phạm vi được phân quyền.'],
  ['Bảo mật', 'Không công bố thông tin định danh; dữ liệu nghiên cứu cần được giả danh trước khi đồng bộ hoặc xuất báo cáo.'],
];

const teamInfoSections = [
  ['Đơn vị triển khai', 'Bệnh viện Đa khoa Phạm Ngọc Thạch phối hợp với Trường Đại học Y tế Công cộng.'],
  ['Nhóm chuyên môn', 'Bác sĩ/KTV PHCN, nghiên cứu viên dữ liệu y sinh và quản trị hệ thống cùng vận hành workflow.'],
  ['Liên hệ hỗ trợ', 'Sử dụng form phản hồi trong hệ thống để gửi góp ý hoặc báo lỗi; admin/NCV sẽ rà soát trong workspace.'],
];

function textValue(...values: unknown[]) {
  for (const value of values) {
    if (typeof value === 'string' && value.trim()) {
      return value.trim();
    }
    if (typeof value === 'number' && Number.isFinite(value)) {
      return String(value);
    }
    if (typeof value === 'boolean') {
      return value ? 'Có' : 'Không';
    }
  }
  return 'N/A';
}

function patientLabel(record: RecordLike) {
  return textValue(record.full_name, record.subject_code, record.username, record.patient_username);
}

function pathBasename(value: unknown) {
  return String(value || '')
    .split(/[\\/]/)
    .filter(Boolean)
    .pop();
}

function mediaFilenameForVideo(video: VideoRecord) {
  return video.stored_filename || pathBasename(video.video_path) || pathBasename(video.processed_path) || pathBasename(video.video_name) || '';
}

function videoKey(video: VideoRecord, index: number) {
  return `${video.username || video.patient_username || 'video'}|${mediaFilenameForVideo(video) || index}|${video.exercise || ''}`;
}

function recordKey(prefix: string, record: RecordLike, index: number) {
  return `${prefix}|${textValue(record.username, record.patient_username, record.subject_code, index)}|${textValue(
    record.video_name,
    record.title,
    record.created_at,
    record.timestamp,
    record.time,
    index,
  )}`;
}

function matchingEvaluation(video: VideoRecord, evaluations: EvaluationRecord[]) {
  const username = video.username || video.patient_username;
  return evaluations.find((item) => {
    return (
      item.patient_username === username &&
      (!video.exercise || item.exercise === video.exercise) &&
      (!video.video_name || item.video_name === video.video_name)
    );
  });
}

function flattenValues(value: unknown): string[] {
  if (value === null || value === undefined) {
    return [];
  }
  if (Array.isArray(value)) {
    return value.flatMap(flattenValues);
  }
  if (typeof value === 'object') {
    return Object.values(value as RecordLike).flatMap(flattenValues);
  }
  return [String(value)];
}

function matchesQuery(record: RecordLike, query: string) {
  const text = query.trim().toLowerCase();
  if (!text) {
    return true;
  }
  return flattenValues(record).join(' ').toLowerCase().includes(text);
}

function statusClass(value: unknown) {
  const text = String(value || '').toLowerCase();
  if (text.includes('hoàn') || text.includes('ok') || text.includes('done') || text.includes('đã')) {
    return 'success';
  }
  if (text.includes('lỗi') || text.includes('fail') || text.includes('hủy')) {
    return 'danger';
  }
  if (text.includes('near') || text.includes('gần') || text.includes('gan')) {
    return 'warning';
  }
  return 'neutral';
}

function auditActionLabel(action: unknown) {
  const labels: Record<string, string> = {
    admin_create_user: 'Tạo tài khoản',
    admin_delete_user: 'Xóa tài khoản',
    admin_set_user_active: 'Khóa/mở khóa',
    admin_reset_password: 'Reset mật khẩu',
    admin_revoke_user_sessions: 'Thu hồi phiên user',
    admin_revoke_all_sessions: 'Thu hồi toàn bộ phiên',
    admin_cleanup_reset: 'Reset dữ liệu',
  };
  const key = String(action || '');
  return labels[key] || textValue(action);
}

function auditMetadataLabel(metadata: unknown) {
  if (!metadata || typeof metadata !== 'object') {
    return 'N/A';
  }
  const entries = Object.entries(metadata as Record<string, unknown>)
    .filter(([, value]) => value !== '' && value !== undefined && value !== null)
    .slice(0, 4);
  if (!entries.length) {
    return 'N/A';
  }
  return entries
    .map(([key, value]) => {
      if (Array.isArray(value)) {
        return `${key}: ${value.join(', ')}`;
      }
      return `${key}: ${String(value)}`;
    })
    .join(' · ');
}

function cleanupCountLabel(target: CleanupTarget) {
  const records = `${target.record_count} bản ghi`;
  if (target.file_count > 0) {
    return `${records} · ${target.file_count} file`;
  }
  return records;
}

function countLabel(count: number) {
  return count > 999 ? '999+' : String(count);
}

function analysisStatusLabel(status: unknown) {
  const text = String(status || '');
  if (text === 'ready_for_ai_worker') {
    return 'Sẵn sàng AI';
  }
  if (text === 'processing') {
    return 'Đang xử lý';
  }
  if (text === 'success') {
    return 'Hoàn tất';
  }
  if (text === 'error') {
    return 'Lỗi';
  }
  if (text === 'canceled') {
    return 'Đã hủy';
  }
  return text || 'Chưa chạy';
}

function analysisActionLabel(action: unknown) {
  const text = String(action || '');
  if (text === 'rerun') {
    return 'Rerun';
  }
  if (text === 'retry') {
    return 'Retry';
  }
  if (text === 'cancel') {
    return 'Hủy';
  }
  return 'Start';
}

function analysisRunLabel(job: AnalysisJob | null | undefined) {
  const runId = String(job?.run_id || '');
  return runId ? runId.replace(/^run_/, '') : 'N/A';
}

function jobTimeLabel(job: AnalysisJob | null | undefined) {
  const value = job?.updated_at || job?.heartbeat || job?.start_time;
  if (typeof value === 'number' && Number.isFinite(value)) {
    return new Date(value * 1000).toLocaleString('vi-VN');
  }
  return textValue(value);
}

function poseJobTimeLabel(job: PoseClassifierJob | null | undefined) {
  const value = job?.updated_at || job?.start_time;
  if (typeof value === 'number' && Number.isFinite(value)) {
    return new Date(value * 1000).toLocaleString('vi-VN');
  }
  return textValue(value);
}

function poseActionLabel(action: unknown) {
  const text = String(action || '');
  if (text === 'train') {
    return 'Train';
  }
  if (text === 'apply') {
    return 'Apply';
  }
  return text || 'N/A';
}

function hfActionLabel(action: unknown) {
  const text = String(action || '');
  if (text === 'sync') {
    return 'Sync';
  }
  if (text === 'upload') {
    return 'Upload';
  }
  if (text === 'report') {
    return 'Report';
  }
  return text || 'N/A';
}

function hfJobTimeLabel(job: HfSyncJob | null | undefined) {
  const value = job?.updated_at || job?.start_time;
  if (typeof value === 'number' && Number.isFinite(value)) {
    return new Date(value * 1000).toLocaleString('vi-VN');
  }
  return textValue(value);
}

function modelReadyLabel(model: PoseClassifierModelStatus | null) {
  if (!model) {
    return 'Chưa tải';
  }
  if (model.ready) {
    return 'Sẵn sàng';
  }
  if (model.checksum_ok === false && model.model_path) {
    return 'Checksum lỗi';
  }
  return 'Chưa có model';
}

function fileSizeLabel(size: unknown) {
  const value = typeof size === 'number' ? size : Number(size || 0);
  if (!Number.isFinite(value) || value <= 0) {
    return 'N/A';
  }
  if (value < 1024) {
    return `${value} B`;
  }
  if (value < 1024 * 1024) {
    return `${(value / 1024).toFixed(1)} KB`;
  }
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}

function metricLabel(key: string) {
  const labels: Record<string, string> = {
    do_chinh_xac: 'Độ chính xác',
    ty_le_tong_the: 'Tỷ lệ tổng thể',
    ai_accuracy: 'AI accuracy',
    f1_score: 'F1-score',
    mae_tong: 'MAE',
    icc: 'ICC',
    recall: 'Recall',
    precision: 'Precision',
    tb_goc_vai: 'Góc vai TB',
    tb_goc_khuyu: 'Góc khuỷu TB',
  };
  return labels[key] || key;
}

function metricValueLabel(value: unknown) {
  if (typeof value === 'number' && Number.isFinite(value)) {
    if (Math.abs(value) <= 1 && value !== 0) {
      return value.toFixed(2);
    }
    return Number.isInteger(value) ? String(value) : value.toFixed(1);
  }
  return textValue(value);
}

function accuracyLabel(value: unknown) {
  const number = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(number)) {
    return 'Chưa có';
  }
  return `${number.toFixed(1)}%`;
}

function patientResultMessage(result: VideoResultDetail) {
  if (!result.ai_detail_allowed && result.report_status.message) {
    return result.report_status.message;
  }
  const doctorResult = textValue(result.summary.doctor_result, result.evaluation?.doctor_result);
  const plan = textValue(result.summary.doctor_plan, result.evaluation?.plan);
  if (doctorResult !== 'N/A' && plan !== 'N/A') {
    return `Bác sĩ đánh giá bài tập: ${doctorResult}. Kế hoạch tiếp theo: ${plan}.`;
  }
  if (String(result.summary.status || '').includes('Đã phân tích')) {
    return 'AI đã phân tích video. Bác sĩ sẽ bổ sung nhận xét lâm sàng khi cần.';
  }
  if (result.latest_job?.status === 'processing') {
    return 'Video đang được phân tích. Bạn có thể quay lại sau vài phút để xem kết quả mới.';
  }
  return 'Chưa có kết quả chi tiết cho video này.';
}

function reportStatusLabel(result: VideoResultDetail) {
  if (result.report_sent) {
    const suffix = result.report_status.sent_at ? ` · ${result.report_status.sent_at}` : '';
    return `Đã gửi báo cáo${suffix}`;
  }
  return 'Chờ NCV gửi báo cáo';
}

function phaseLabel(label: FrameLabel) {
  const labels: Record<FrameLabel, string> = {
    ALL: 'Tất cả',
    G1: 'G1',
    G2: 'G2',
    G3: 'G3',
    PASS: 'PASS',
    NEAR: 'NEAR',
    FAIL: 'FAIL',
  };
  return labels[label];
}

function mlConfidenceLabel(value: unknown) {
  const number = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(number)) {
    return '';
  }
  const percent = number <= 1 ? number * 100 : number;
  return `${percent.toFixed(0)}%`;
}

function mlBadgeClass(label: unknown) {
  const text = String(label || '').toLowerCase();
  if ((text.includes('đúng') || text.includes('dung')) && !text.includes('gần') && !text.includes('gan')) {
    return 'success';
  }
  if (text.includes('gần') || text.includes('gan') || text.includes('near')) {
    return 'warning';
  }
  if (text.includes('sai') || text.includes('fail')) {
    return 'danger';
  }
  return 'neutral';
}

function phaseMetricValue(metrics: VideoResultDetail['phase_metrics'] | AnalysisChartPreview['phase_metrics'] | undefined, phase: 'G1' | 'G2' | 'G3', key: 'accuracy' | 'mae' | 'f1' | 'icc') {
  const value = metrics?.[phase]?.[key];
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function defaultPhaseThreshold(phase: 'G1' | 'G2' | 'G3') {
  if (phase === 'G1') {
    return 45;
  }
  if (phase === 'G2') {
    return 30;
  }
  return 15;
}

function metricPercentLabel(value: number | null) {
  if (value === null) {
    return 'N/A';
  }
  return `${value.toFixed(1)}%`;
}

const chartSeriesConfig = [
  { key: 'goc_vai', label: 'Góc vai', color: '#0284c7', dashed: false },
  { key: 'goc_khuyu', label: 'Góc khuỷu', color: '#059669', dashed: false },
  { key: 'vai_chuan', label: 'Chuẩn vai', color: '#7c3aed', dashed: true },
  { key: 'khuyu_chuan', label: 'Chuẩn khuỷu', color: '#dc2626', dashed: true },
] as const;

type ChartKey = (typeof chartSeriesConfig)[number]['key'];

function chartNumberLabel(value: unknown, suffix = '') {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return `${Number.isInteger(value) ? value : value.toFixed(1)}${suffix}`;
  }
  return 'N/A';
}

function chartPointValue(point: AnalysisChartPreview['series'][number], key: ChartKey) {
  const value = point[key];
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function chartPath(points: AnalysisChartPreview['series'], key: ChartKey, minY: number, maxY: number, width: number, height: number, pad: number) {
  const drawableWidth = width - pad * 2;
  const drawableHeight = height - pad * 2;
  const range = Math.max(1, maxY - minY);
  const usablePoints = points
    .map((point, index) => {
      const value = chartPointValue(point, key);
      if (value === null) {
        return null;
      }
      const x = pad + (points.length <= 1 ? 0 : (index / (points.length - 1)) * drawableWidth);
      const y = pad + ((maxY - value) / range) * drawableHeight;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .filter((value): value is string => Boolean(value));
  if (!usablePoints.length) {
    return '';
  }
  return `M ${usablePoints.join(' L ')}`;
}

function AngleChart({ chart }: { chart: AnalysisChartPreview }) {
  const width = 720;
  const height = 260;
  const pad = 34;
  const visibleSeries = chartSeriesConfig.filter((series) => chart.columns.includes(series.key));
  const values = chart.series.flatMap((point) =>
    visibleSeries
      .map((series) => chartPointValue(point, series.key))
      .filter((value): value is number => value !== null),
  );
  if (!values.length) {
    return <div className="empty-gallery">Dữ liệu CSV/JSON chưa có cột góc phù hợp để vẽ biểu đồ.</div>;
  }
  const minY = Math.min(0, Math.floor(Math.min(...values) / 10) * 10);
  const maxY = Math.max(180, Math.ceil(Math.max(...values) / 10) * 10);
  const gridValues = [minY, Math.round((minY + maxY) / 2), maxY];
  return (
    <div className="angle-chart-wrap">
      <svg className="angle-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Biểu đồ góc khớp">
        <rect x="0" y="0" width={width} height={height} rx="8" />
        {gridValues.map((value) => {
          const y = pad + ((maxY - value) / Math.max(1, maxY - minY)) * (height - pad * 2);
          return (
            <g key={value}>
              <line className="chart-grid-line" x1={pad} x2={width - pad} y1={y} y2={y} />
              <text className="chart-axis-label" x="8" y={y + 4}>
                {value}°
              </text>
            </g>
          );
        })}
        <line className="chart-axis-line" x1={pad} x2={width - pad} y1={height - pad} y2={height - pad} />
        {visibleSeries.map((series) => {
          const path = chartPath(chart.series, series.key, minY, maxY, width, height, pad);
          return path ? (
            <path
              className="chart-line"
              d={path}
              key={series.key}
              stroke={series.color}
              strokeDasharray={series.dashed ? '7 5' : undefined}
            />
          ) : null;
        })}
      </svg>
      <div className="chart-legend">
        {visibleSeries.map((series) => (
          <span key={series.key}>
            <i style={{ background: series.color }} />
            {series.label}
          </span>
        ))}
      </div>
    </div>
  );
}

type TableColumn<T extends RecordLike> = {
  key: string;
  label: string;
  render: (item: T, index: number) => React.ReactNode;
};

function DataTable<T extends RecordLike>({
  columns,
  items,
  emptyText,
  rowKey,
}: {
  columns: TableColumn<T>[];
  items: T[];
  emptyText: string;
  rowKey: (item: T, index: number) => string;
}) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column.key}>{column.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {items.map((item, index) => (
            <tr key={rowKey(item, index)}>
              {columns.map((column) => (
                <td key={column.key}>{column.render(item, index)}</td>
              ))}
            </tr>
          ))}
          {!items.length ? (
            <tr>
              <td colSpan={columns.length} className="empty-cell">
                {emptyText}
              </td>
            </tr>
          ) : null}
        </tbody>
      </table>
    </div>
  );
}

export function App() {
  const [session, setSession] = useState<Session | null>(() => {
    const token = window.localStorage.getItem('rehab_token');
    const userRaw = window.localStorage.getItem('rehab_user');
    if (!token || !userRaw) {
      return null;
    }
    try {
      return { token, user: JSON.parse(userRaw) as User };
    } catch {
      window.localStorage.removeItem('rehab_token');
      window.localStorage.removeItem('rehab_user');
      return null;
    }
  });
  const [authMode, setAuthMode] = useState<AuthMode>('login');
  const [activeView, setActiveView] = useState<ViewId>('home');
  const [health, setHealth] = useState<LoadState>('idle');
  const [videos, setVideos] = useState<VideoRecord[]>([]);
  const [evaluations, setEvaluations] = useState<EvaluationRecord[]>([]);
  const [patients, setPatients] = useState<PatientRecord[]>([]);
  const [users, setUsers] = useState<PatientRecord[]>([]);
  const [auditLog, setAuditLog] = useState<AuditLogRecord[]>([]);
  const [cleanupTargets, setCleanupTargets] = useState<CleanupTarget[]>([]);
  const [feedbackRecords, setFeedbackRecords] = useState<FeedbackRecord[]>([]);
  const [symptoms, setSymptoms] = useState<SymptomRecord[]>([]);
  const [schedules, setSchedules] = useState<ScheduleRecord[]>([]);
  const [researchRecords, setResearchRecords] = useState<ResearchRecord[]>([]);
  const [query, setQuery] = useState('');
  const [loadState, setLoadState] = useState<LoadState>('idle');
  const [formState, setFormState] = useState<LoadState>('idle');
  const [previewState, setPreviewState] = useState<LoadState>('idle');
  const [previewTargetKey, setPreviewTargetKey] = useState('');
  const [analysisJobs, setAnalysisJobs] = useState<Record<string, AnalysisJob | null>>({});
  const [analysisHistories, setAnalysisHistories] = useState<Record<string, AnalysisJob[]>>({});
  const [analysisOptions, setAnalysisOptions] = useState<AnalysisJobOptions>({
    model_type: 'MediaPipe Heavy',
    skip_step: 0,
    resize_width: 720,
    min_confidence: 0.5,
  });
  const [analysisState, setAnalysisState] = useState<LoadState>('idle');
  const [analysisTargetKey, setAnalysisTargetKey] = useState('');
  const [poseModelStatus, setPoseModelStatus] = useState<PoseClassifierModelStatus | null>(null);
  const [poseLatestJob, setPoseLatestJob] = useState<PoseClassifierJob | null>(null);
  const [poseHistory, setPoseHistory] = useState<PoseClassifierJob[]>([]);
  const [poseState, setPoseState] = useState<LoadState>('idle');
  const [poseMessage, setPoseMessage] = useState('');
  const [poseTargetKey, setPoseTargetKey] = useState('');
  const [poseDryRun, setPoseDryRun] = useState(true);
  const [poseMinSamples, setPoseMinSamples] = useState(10);
  const [poseSelectedVideoKey, setPoseSelectedVideoKey] = useState('');
  const [hfStatus, setHfStatus] = useState<HfSyncStatus | null>(null);
  const [hfLatestJob, setHfLatestJob] = useState<HfSyncJob | null>(null);
  const [hfHistory, setHfHistory] = useState<HfSyncJob[]>([]);
  const [hfState, setHfState] = useState<LoadState>('idle');
  const [hfMessage, setHfMessage] = useState('');
  const [hfTargetKey, setHfTargetKey] = useState('');
  const [hfDryRun, setHfDryRun] = useState(true);
  const [hfSelectedVideoKey, setHfSelectedVideoKey] = useState('');
  const [hfArtifactKind, setHfArtifactKind] = useState('angle-csv');
  const [hfSelectedFiles, setHfSelectedFiles] = useState<string[]>(['video_list.json', 'doctor_evaluations.json', 'research_data.json']);
  const [artifactState, setArtifactState] = useState<LoadState>('idle');
  const [artifactTargetKey, setArtifactTargetKey] = useState('');
  const [artifactMessage, setArtifactMessage] = useState('');
  const [selectedResult, setSelectedResult] = useState<VideoResultDetail | null>(null);
  const [framesState, setFramesState] = useState<LoadState>('idle');
  const [framesMessage, setFramesMessage] = useState('');
  const [framesPage, setFramesPage] = useState(1);
  const [framesLabel, setFramesLabel] = useState<FrameLabel>('ALL');
  const [analysisFrames, setAnalysisFrames] = useState<AnalysisFramesResult | null>(null);
  const [frameImageUrls, setFrameImageUrls] = useState<Record<string, string>>({});
  const [focusedFrame, setFocusedFrame] = useState<FocusedFrame | null>(null);
  const [chartState, setChartState] = useState<LoadState>('idle');
  const [chartMessage, setChartMessage] = useState('');
  const [analysisChart, setAnalysisChart] = useState<AnalysisChartPreview | null>(null);
  const [videoPreview, setVideoPreview] = useState<VideoPreview | null>(null);
  const [message, setMessage] = useState('');
  const [formMessage, setFormMessage] = useState('');
  const [previewMessage, setPreviewMessage] = useState('');
  const [analysisMessage, setAnalysisMessage] = useState('');
  const [adminActionKey, setAdminActionKey] = useState('');
  const previewUrlRef = useRef<string | null>(null);
  const frameUrlMapRef = useRef<Record<string, string>>({});
  const completedAnalysisRefreshRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    let active = true;
    api
      .health()
      .then(() => active && setHealth('ready'))
      .catch(() => active && setHealth('error'));
    return () => {
      active = false;
    };
  }, []);

  function clearLocalSession(nextMessage = '') {
    window.localStorage.removeItem('rehab_token');
    window.localStorage.removeItem('rehab_user');
    clearVideoPreview();
    resetFrameGallery();
    setSession(null);
    setVideos([]);
    setEvaluations([]);
    setPatients([]);
    setUsers([]);
    setAuditLog([]);
    setCleanupTargets([]);
    setFeedbackRecords([]);
    setSymptoms([]);
    setSchedules([]);
    setResearchRecords([]);
    setAnalysisJobs({});
    setAnalysisHistories({});
    setAnalysisState('idle');
    setAnalysisTargetKey('');
    setAnalysisMessage('');
    setPoseModelStatus(null);
    setPoseLatestJob(null);
    setPoseHistory([]);
    setPoseState('idle');
    setPoseMessage('');
    setPoseTargetKey('');
    setPoseSelectedVideoKey('');
    setHfStatus(null);
    setHfLatestJob(null);
    setHfHistory([]);
    setHfState('idle');
    setHfMessage('');
    setHfTargetKey('');
    setHfSelectedVideoKey('');
    setAdminActionKey('');
    setArtifactState('idle');
    setArtifactTargetKey('');
    setArtifactMessage('');
    setSelectedResult(null);
    resetChartPreview();
    completedAnalysisRefreshRef.current.clear();
    setMessage(nextMessage);
  }

  function clearFrameImages() {
    Object.values(frameUrlMapRef.current).forEach((url) => URL.revokeObjectURL(url));
    frameUrlMapRef.current = {};
    setFrameImageUrls({});
  }

  function resetFrameGallery() {
    setFocusedFrame(null);
    clearFrameImages();
    setFramesState('idle');
    setFramesMessage('');
    setFramesPage(1);
    setFramesLabel('ALL');
    setAnalysisFrames(null);
  }

  function resetChartPreview() {
    setChartState('idle');
    setChartMessage('');
    setAnalysisChart(null);
  }

  function clearSelectedResult() {
    setSelectedResult(null);
    resetFrameGallery();
    resetChartPreview();
  }

  async function loadDashboard(nextSession = session) {
    if (!nextSession) {
      return;
    }
    setLoadState('loading');
    setMessage('');
    const role = nextSession.user.role;
    try {
      const [
        videoResult,
        evaluationResult,
        patientResult,
        symptomResult,
        scheduleResult,
        researchResult,
        userResult,
        auditResult,
        cleanupResult,
        poseResult,
        poseHistoryResult,
        hfResult,
        hfHistoryResult,
        feedbackResult,
      ] = await Promise.all([
        api.videos(nextSession.token),
        api.evaluations(nextSession.token),
        api.patients(nextSession.token),
        api.symptoms(nextSession.token),
        canViewSchedules(role) ? api.schedules(nextSession.token) : Promise.resolve({ items: [], count: 0 }),
        canViewResearch(role) ? api.researchRecords(nextSession.token) : Promise.resolve({ items: [], count: 0 }),
        canManageUsers(role) ? api.adminUsers(nextSession.token) : Promise.resolve({ items: [], count: 0 }),
        canManageUsers(role) ? api.adminAuditLog(nextSession.token, 100) : Promise.resolve({ items: [], count: 0 }),
        canManageUsers(role) ? api.cleanupStatus(nextSession.token) : Promise.resolve({ items: [], count: 0 }),
        canManagePoseClassifier(role) ? api.poseClassifierStatus(nextSession.token) : Promise.resolve({ model: null as PoseClassifierModelStatus | null, latest_job: null }),
        canManagePoseClassifier(role) ? api.poseClassifierHistory(nextSession.token) : Promise.resolve({ items: [], count: 0 }),
        canManageHfSync(role) ? api.hfSyncStatus(nextSession.token) : Promise.resolve({ hf: null as HfSyncStatus | null, latest_job: null }),
        canManageHfSync(role) ? api.hfSyncHistory(nextSession.token) : Promise.resolve({ items: [], count: 0 }),
        canReviewFeedback(role) ? api.feedback(nextSession.token, 100) : Promise.resolve({ items: [], count: 0 }),
      ]);
      setVideos(videoResult.items);
      setEvaluations(evaluationResult.items);
      setPatients(patientResult.items);
      setUsers(userResult.items);
      setSymptoms(symptomResult.items);
      setSchedules(scheduleResult.items);
      setResearchRecords(researchResult.items);
      setAuditLog(auditResult.items);
      setCleanupTargets(cleanupResult.items);
      setPoseModelStatus(poseResult.model);
      setPoseLatestJob(poseResult.latest_job);
      setPoseHistory(poseHistoryResult.items);
      setHfStatus(hfResult.hf);
      setHfLatestJob(hfResult.latest_job);
      setHfHistory(hfHistoryResult.items);
      setFeedbackRecords(feedbackResult.items);
      setPreviewMessage('');
      setAnalysisMessage('');
      setLoadState('ready');
    } catch (error) {
      setLoadState('error');
      if (error instanceof ApiError && error.status === 401) {
        clearLocalSession('Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại.');
        return;
      }
      setMessage(error instanceof Error ? error.message : 'Không tải được dữ liệu.');
    }
  }

  async function handleCreateSymptom(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!session) {
      return;
    }
    const form = event.currentTarget;
    const formData = new FormData(form);
    const payload: CreateSymptomPayload = {
      full_name: String(formData.get('full_name') || session.user.full_name || session.user.username),
      patient_id: String(formData.get('patient_id') || session.user.username),
      age: Number(formData.get('age') || 0),
      gender: String(formData.get('gender') || ''),
      exercise: String(formData.get('exercise') || ''),
      symptoms: String(formData.get('symptoms') || ''),
      vas: Number(formData.get('vas') || 0),
    };
    setFormState('loading');
    setFormMessage('');
    try {
      await api.createSymptom(session.token, payload);
      form.reset();
      setFormState('ready');
      setFormMessage('Đã gửi khai báo triệu chứng.');
      await loadDashboard(session);
    } catch (error) {
      setFormState('error');
      if (error instanceof ApiError && error.status === 401) {
        clearLocalSession('Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại.');
        return;
      }
      setFormMessage(error instanceof Error ? error.message : 'Không gửi được khai báo triệu chứng.');
    }
  }

  async function handleCreateFeedback(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!session) {
      return;
    }
    const form = event.currentTarget;
    const formData = new FormData(form);
    const payload: CreateFeedbackPayload = {
      category: String(formData.get('category') || 'general'),
      message: String(formData.get('message') || ''),
      contact_ok: formData.get('contact_ok') === 'on',
      page: activeView,
    };
    setFormState('loading');
    setFormMessage('');
    try {
      await api.createFeedback(session.token, payload);
      form.reset();
      setFormState('ready');
      setFormMessage('Đã gửi phản hồi. Nhóm vận hành sẽ rà soát trong hệ thống.');
      if (canReviewFeedback(session.user.role)) {
        const result = await api.feedback(session.token, 100);
        setFeedbackRecords(result.items);
      }
    } catch (error) {
      setFormState('error');
      if (error instanceof ApiError && error.status === 401) {
        clearLocalSession('Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại.');
        return;
      }
      setFormMessage(error instanceof Error ? error.message : 'Không gửi được phản hồi.');
    }
  }

  async function handleUploadVideo(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!session) {
      return;
    }
    const form = event.currentTarget;
    const formData = new FormData(form);
    const file = formData.get('file');
    const exercise = String(formData.get('exercise') || '');
    if (!(file instanceof File) || !file.name) {
      setFormState('error');
      setFormMessage('Vui lòng chọn file video.');
      return;
    }
    setFormState('loading');
    setFormMessage('');
    try {
      await api.uploadVideo(session.token, {
        file,
        full_name: String(formData.get('full_name') || session.user.full_name || session.user.username),
        exercise,
      });
      form.reset();
      setFormState('ready');
      setFormMessage('Đã gửi video cho bác sĩ và nghiên cứu viên.');
      await loadDashboard(session);
    } catch (error) {
      setFormState('error');
      if (error instanceof ApiError && error.status === 401) {
        clearLocalSession('Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại.');
        return;
      }
      setFormMessage(error instanceof Error ? error.message : 'Không upload được video.');
    }
  }

  async function handleCreateEvaluation(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!session) {
      return;
    }
    const form = event.currentTarget;
    const formData = new FormData(form);
    const selectedVideoKey = String(formData.get('video_key') || '');
    const selectedVideo = videos.find((video, index) => videoKey(video, index) === selectedVideoKey);
    if (!selectedVideo) {
      setFormState('error');
      setFormMessage('Vui lòng chọn video cần đánh giá.');
      return;
    }
    const payload: CreateEvaluationPayload = {
      patient_username: textValue(selectedVideo.username, selectedVideo.patient_username),
      video_name: textValue(selectedVideo.video_name, selectedVideo.original_filename),
      exercise: textValue(selectedVideo.exercise),
      doctor_result: String(formData.get('doctor_result') || ''),
      errors: formData.getAll('errors').map(String),
      comments: String(formData.get('comments') || ''),
      comments_ncv: String(formData.get('comments_ncv') || ''),
      plan: String(formData.get('plan') || 'Tiếp tục'),
    };
    setFormState('loading');
    setFormMessage('');
    try {
      await api.createEvaluation(session.token, payload);
      form.reset();
      setFormState('ready');
      setFormMessage('Đã lưu đánh giá lâm sàng.');
      await loadDashboard(session);
    } catch (error) {
      setFormState('error');
      if (error instanceof ApiError && error.status === 401) {
        clearLocalSession('Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại.');
        return;
      }
      setFormMessage(error instanceof Error ? error.message : 'Không lưu được đánh giá.');
    }
  }

  async function handleDeleteEvaluation(record: EvaluationRecord) {
    if (!session || !record.id) {
      return;
    }
    setFormState('loading');
    setFormMessage('');
    try {
      await api.deleteEvaluation(session.token, record.id);
      setFormState('ready');
      setFormMessage('Đã xóa đánh giá.');
      await loadDashboard(session);
    } catch (error) {
      setFormState('error');
      setFormMessage(error instanceof Error ? error.message : 'Không xóa được đánh giá.');
    }
  }

  async function handleCreateSchedule(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!session) {
      return;
    }
    const form = event.currentTarget;
    const formData = new FormData(form);
    const type = String(formData.get('type') || 'appointment');
    const payload: CreateSchedulePayload = {
      patient_username: String(formData.get('patient_username') || ''),
      type,
      title: String(formData.get('title') || ''),
      datetime: String(formData.get('datetime') || ''),
      notes: String(formData.get('notes') || ''),
      exercise_name: String(formData.get('exercise_name') || ''),
      frequency: String(formData.get('frequency') || ''),
      medication_name: String(formData.get('medication_name') || ''),
      dosage: String(formData.get('dosage') || ''),
    };
    setFormState('loading');
    setFormMessage('');
    try {
      await api.createSchedule(session.token, payload);
      form.reset();
      setFormState('ready');
      setFormMessage('Đã thêm lịch nhắc.');
      await loadDashboard(session);
    } catch (error) {
      setFormState('error');
      setFormMessage(error instanceof Error ? error.message : 'Không thêm được lịch nhắc.');
    }
  }

  async function handleDeleteSchedule(record: ScheduleRecord) {
    if (!session || !record.id) {
      return;
    }
    setFormState('loading');
    setFormMessage('');
    try {
      await api.deleteSchedule(session.token, record.id);
      setFormState('ready');
      setFormMessage('Đã xóa lịch nhắc.');
      await loadDashboard(session);
    } catch (error) {
      setFormState('error');
      setFormMessage(error instanceof Error ? error.message : 'Không xóa được lịch nhắc.');
    }
  }

  async function handleCreateResearchRecord(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!session) {
      return;
    }
    const form = event.currentTarget;
    const formData = new FormData(form);
    const payload: CreateResearchPayload = {
      patient_username: String(formData.get('patient_username') || ''),
      subject_code: String(formData.get('subject_code') || ''),
      age: Number(formData.get('age') || 0),
      gender: String(formData.get('gender') || ''),
      diagnosis: String(formData.get('diagnosis') || ''),
      exercise: String(formData.get('exercise') || ''),
      general_result: String(formData.get('general_result') || ''),
      plan: String(formData.get('plan') || ''),
      specialist_comment: String(formData.get('specialist_comment') || ''),
      recording_device: String(formData.get('recording_device') || ''),
      recording_angle: String(formData.get('recording_angle') || ''),
    };
    setFormState('loading');
    setFormMessage('');
    try {
      await api.createResearchRecord(session.token, payload);
      form.reset();
      setFormState('ready');
      setFormMessage('Đã lưu phiếu nghiên cứu.');
      await loadDashboard(session);
    } catch (error) {
      setFormState('error');
      setFormMessage(error instanceof Error ? error.message : 'Không lưu được phiếu nghiên cứu.');
    }
  }

  async function handleDeleteResearchRecord(record: ResearchRecord) {
    if (!session || !record.id) {
      return;
    }
    setFormState('loading');
    setFormMessage('');
    try {
      await api.deleteResearchRecord(session.token, record.id);
      setFormState('ready');
      setFormMessage('Đã xóa phiếu nghiên cứu.');
      await loadDashboard(session);
    } catch (error) {
      setFormState('error');
      setFormMessage(error instanceof Error ? error.message : 'Không xóa được phiếu nghiên cứu.');
    }
  }

  async function handleCreateUser(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!session) {
      return;
    }
    const form = event.currentTarget;
    const formData = new FormData(form);
    const assignedText = String(formData.get('assigned_patient_usernames') || '');
    const payload: CreateUserPayload = {
      username: String(formData.get('username') || ''),
      full_name: String(formData.get('full_name') || ''),
      email: String(formData.get('email') || ''),
      password: String(formData.get('password') || ''),
      role: String(formData.get('role') || PATIENT_ROLE),
      assigned_patient_usernames: assignedText.split(',').map((item) => item.trim()).filter(Boolean),
    };
    setFormState('loading');
    setFormMessage('');
    try {
      await api.createUser(session.token, payload);
      form.reset();
      setFormState('ready');
      setFormMessage('Đã tạo tài khoản.');
      await loadDashboard(session);
    } catch (error) {
      setFormState('error');
      setFormMessage(error instanceof Error ? error.message : 'Không tạo được tài khoản.');
    }
  }

  async function handleDeleteUser(record: PatientRecord) {
    if (!session || !record.username) {
      return;
    }
    const confirm = window.prompt(`Nhập ${record.username} để xóa tài khoản`);
    if (confirm !== record.username) {
      setFormState('error');
      setFormMessage('Chưa xác nhận đúng tên tài khoản cần xóa.');
      return;
    }
    setFormState('loading');
    setFormMessage('');
    try {
      await api.deleteUser(session.token, record.username);
      setFormState('ready');
      setFormMessage('Đã xóa tài khoản.');
      await loadDashboard(session);
    } catch (error) {
      setFormState('error');
      setFormMessage(error instanceof Error ? error.message : 'Không xóa được tài khoản.');
    }
  }

  async function handleSetUserActive(record: PatientRecord, active: boolean) {
    if (!session || !record.username) {
      return;
    }
    const key = `active:${record.username}`;
    setAdminActionKey(key);
    setFormState('loading');
    setFormMessage('');
    try {
      const result = await api.setUserActive(session.token, record.username, active);
      setFormState('ready');
      setFormMessage(
        active
          ? `Đã mở khóa ${result.item.username}.`
          : `Đã khóa ${result.item.username} và thu hồi ${result.revoked_sessions || 0} phiên.`,
      );
      await loadDashboard(session);
    } catch (error) {
      setFormState('error');
      setFormMessage(error instanceof Error ? error.message : 'Không cập nhật được trạng thái tài khoản.');
    } finally {
      setAdminActionKey('');
    }
  }

  async function handleResetUserPassword(record: PatientRecord) {
    if (!session || !record.username) {
      return;
    }
    const password = window.prompt(`Mật khẩu tạm mới cho ${record.username}`);
    if (!password) {
      return;
    }
    const confirmPassword = window.prompt('Nhập lại mật khẩu tạm');
    if (password !== confirmPassword) {
      setFormState('error');
      setFormMessage('Mật khẩu nhập lại không khớp.');
      return;
    }
    const key = `reset:${record.username}`;
    setAdminActionKey(key);
    setFormState('loading');
    setFormMessage('');
    try {
      const result = await api.resetUserPassword(session.token, record.username, {
        password,
        confirm_password: confirmPassword,
      });
      setFormState('ready');
      setFormMessage(`Đã reset mật khẩu ${result.item.username}; người dùng sẽ phải đổi mật khẩu khi đăng nhập.`);
      await loadDashboard(session);
    } catch (error) {
      setFormState('error');
      setFormMessage(error instanceof Error ? error.message : 'Không reset được mật khẩu.');
    } finally {
      setAdminActionKey('');
    }
  }

  async function handleRevokeUserSessions(record: PatientRecord) {
    if (!session || !record.username) {
      return;
    }
    const key = `revoke:${record.username}`;
    setAdminActionKey(key);
    setFormState('loading');
    setFormMessage('');
    try {
      const result = await api.revokeUserSessions(session.token, record.username, 'admin user action');
      setFormState('ready');
      setFormMessage(`Đã thu hồi ${result.revoked_sessions} phiên của ${record.username}.`);
      await loadDashboard(session);
    } catch (error) {
      setFormState('error');
      setFormMessage(error instanceof Error ? error.message : 'Không thu hồi được phiên.');
    } finally {
      setAdminActionKey('');
    }
  }

  async function handleRevokeAllSessions() {
    if (!session) {
      return;
    }
    const confirm = window.prompt('Nhập REVOKE ALL SESSIONS để thu hồi toàn bộ phiên');
    if (confirm !== 'REVOKE ALL SESSIONS') {
      setFormState('error');
      setFormMessage('Chưa xác nhận thu hồi toàn bộ phiên.');
      return;
    }
    setAdminActionKey('revoke:all');
    setFormState('loading');
    setFormMessage('');
    try {
      const result = await api.revokeAllSessions(session.token, 'admin global action', confirm);
      clearLocalSession(`Đã thu hồi ${result.revoked_sessions} phiên. Vui lòng đăng nhập lại.`);
    } catch (error) {
      setFormState('error');
      setFormMessage(error instanceof Error ? error.message : 'Không thu hồi được toàn bộ phiên.');
    } finally {
      setAdminActionKey('');
    }
  }

  async function handleResetCleanupTarget(target: CleanupTarget) {
    if (!session) {
      return;
    }
    const confirm = window.prompt(`Nhập ${target.confirm} để reset ${target.label}`);
    if (confirm !== target.confirm) {
      setFormState('error');
      setFormMessage('Chưa xác nhận đúng chuỗi reset dữ liệu.');
      return;
    }
    setAdminActionKey(`cleanup:${target.target}`);
    setFormState('loading');
    setFormMessage('');
    try {
      const result = await api.resetCleanupTarget(session.token, target.target, confirm);
      setFormState('ready');
      setFormMessage(
        `Đã reset ${result.label}: ${result.cleared_records} bản ghi, ${result.deleted_files} file. Backup đã tạo trước thao tác.`,
      );
      await loadDashboard(session);
    } catch (error) {
      setFormState('error');
      setFormMessage(error instanceof Error ? error.message : 'Không reset được nhóm dữ liệu.');
    } finally {
      setAdminActionKey('');
    }
  }

  async function handleChangePassword(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!session) {
      return;
    }
    const form = event.currentTarget;
    const formData = new FormData(form);
    setFormState('loading');
    setFormMessage('');
    try {
      const result = await api.changePassword(session.token, {
        old_password: String(formData.get('old_password') || ''),
        new_password: String(formData.get('new_password') || ''),
        confirm_password: String(formData.get('confirm_password') || ''),
      });
      const nextSession = { ...session, user: result.user };
      window.localStorage.setItem('rehab_user', JSON.stringify(result.user));
      setSession(nextSession);
      form.reset();
      setFormState('ready');
      setFormMessage('Đã đổi mật khẩu.');
    } catch (error) {
      setFormState('error');
      setFormMessage(error instanceof Error ? error.message : 'Không đổi được mật khẩu.');
    }
  }

  function clearVideoPreview() {
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
      previewUrlRef.current = null;
    }
    setVideoPreview(null);
    setPreviewState('idle');
    setPreviewTargetKey('');
  }

  async function handlePreviewVideo(video: VideoRecord, index: number) {
    if (!session) {
      return;
    }
    const key = videoKey(video, index);
    if (videoPreview?.key === key) {
      clearVideoPreview();
      setPreviewMessage('');
      return;
    }
    const mediaFilename = mediaFilenameForVideo(video);
    if (!mediaFilename) {
      setPreviewState('error');
      setPreviewMessage('Video này chưa có file media để xem.');
      return;
    }
    setPreviewMessage('');
    clearVideoPreview();
    setPreviewState('loading');
    setPreviewTargetKey(key);
    try {
      const blob = await api.videoBlob(session.token, mediaFilename);
      const url = URL.createObjectURL(blob);
      previewUrlRef.current = url;
      setVideoPreview({
        key,
        url,
        label: textValue(video.video_name, video.original_filename, mediaFilename),
      });
      setPreviewState('ready');
    } catch (error) {
      setPreviewState('error');
      if (error instanceof ApiError && error.status === 401) {
        clearLocalSession('Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại.');
        return;
      }
      setPreviewMessage(error instanceof Error ? error.message : 'Không tải được video.');
    }
  }

  async function refreshAnalysisJob(video: VideoRecord, index: number, nextSession = session) {
    if (!nextSession) {
      return;
    }
    const mediaFilename = mediaFilenameForVideo(video);
    if (!mediaFilename) {
      return;
    }
    const key = videoKey(video, index);
    const result = await api.latestAnalysisJob(nextSession.token, mediaFilename);
    setAnalysisJobs((current) => ({ ...current, [key]: result.job }));
    if (result.job?.status === 'success' && !completedAnalysisRefreshRef.current.has(result.job.job_id)) {
      completedAnalysisRefreshRef.current.add(result.job.job_id);
      await loadDashboard(nextSession);
    }
  }

  function selectedAnalysisOptions(): AnalysisJobOptions {
    return {
      model_type: analysisOptions.model_type,
      skip_step: Math.max(0, Math.min(30, Number(analysisOptions.skip_step) || 0)),
      resize_width: Math.max(240, Math.min(2160, Number(analysisOptions.resize_width) || 720)),
      min_confidence: Math.max(0.1, Math.min(0.95, Number(analysisOptions.min_confidence) || 0.5)),
    };
  }

  async function handleStartAnalysis(video: VideoRecord, index: number) {
    if (!session) {
      return;
    }
    const mediaFilename = mediaFilenameForVideo(video);
    if (!mediaFilename) {
      setAnalysisState('error');
      setAnalysisMessage('Video này chưa có file media để phân tích.');
      return;
    }
    const key = videoKey(video, index);
    setAnalysisState('loading');
    setAnalysisTargetKey(key);
    setAnalysisMessage('');
    try {
      const result = await api.startAnalysisJob(session.token, mediaFilename, selectedAnalysisOptions());
      if (result.job?.job_id) {
        completedAnalysisRefreshRef.current.delete(result.job.job_id);
      }
      setAnalysisJobs((current) => ({ ...current, [key]: result.job }));
      setAnalysisState('ready');
      setAnalysisMessage(result.started ? 'Đã tạo job phân tích.' : 'Job phân tích đang chạy.');
    } catch (error) {
      setAnalysisState('error');
      if (error instanceof ApiError && error.status === 401) {
        clearLocalSession('Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại.');
        return;
      }
      setAnalysisMessage(error instanceof Error ? error.message : 'Không tạo được job phân tích.');
    }
  }

  async function handleRerunAnalysis(video: VideoRecord, index: number) {
    if (!session) {
      return;
    }
    const mediaFilename = mediaFilenameForVideo(video);
    if (!mediaFilename) {
      setAnalysisState('error');
      setAnalysisMessage('Video này chưa có file media để chạy lại.');
      return;
    }
    const key = videoKey(video, index);
    setAnalysisState('loading');
    setAnalysisTargetKey(key);
    setAnalysisMessage('');
    try {
      const result = await api.rerunAnalysisJob(session.token, mediaFilename, selectedAnalysisOptions());
      if (result.job?.job_id) {
        completedAnalysisRefreshRef.current.delete(result.job.job_id);
      }
      setAnalysisJobs((current) => ({ ...current, [key]: result.job }));
      setAnalysisState('ready');
      setAnalysisMessage('Đã tạo job chạy lại với cấu hình model hiện tại.');
    } catch (error) {
      setAnalysisState('error');
      if (error instanceof ApiError && error.status === 401) {
        clearLocalSession('Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại.');
        return;
      }
      setAnalysisMessage(error instanceof Error ? error.message : 'Không chạy lại được job phân tích.');
    }
  }

  async function handleRetryAnalysis(video: VideoRecord, index: number) {
    if (!session) {
      return;
    }
    const mediaFilename = mediaFilenameForVideo(video);
    if (!mediaFilename) {
      setAnalysisState('error');
      setAnalysisMessage('Video này chưa có file media để retry.');
      return;
    }
    const key = videoKey(video, index);
    setAnalysisState('loading');
    setAnalysisTargetKey(key);
    setAnalysisMessage('');
    try {
      const result = await api.retryAnalysisJob(session.token, mediaFilename);
      if (result.job?.job_id) {
        completedAnalysisRefreshRef.current.delete(result.job.job_id);
      }
      setAnalysisJobs((current) => ({ ...current, [key]: result.job }));
      setAnalysisState('ready');
      setAnalysisMessage('Đã retry job phân tích với cấu hình lần chạy gần nhất.');
    } catch (error) {
      setAnalysisState('error');
      if (error instanceof ApiError && error.status === 401) {
        clearLocalSession('Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại.');
        return;
      }
      setAnalysisMessage(error instanceof Error ? error.message : 'Không retry được job phân tích.');
    }
  }

  async function handleCancelAnalysis(video: VideoRecord, index: number) {
    if (!session) {
      return;
    }
    const mediaFilename = mediaFilenameForVideo(video);
    if (!mediaFilename) {
      setAnalysisState('error');
      setAnalysisMessage('Video này chưa có file media để hủy job.');
      return;
    }
    const key = videoKey(video, index);
    setAnalysisState('loading');
    setAnalysisTargetKey(key);
    setAnalysisMessage('');
    try {
      const result = await api.cancelAnalysisJob(session.token, mediaFilename);
      setAnalysisJobs((current) => ({ ...current, [key]: result.job }));
      setAnalysisState('ready');
      setAnalysisMessage(result.ok ? 'Đã hủy job phân tích.' : 'Không có job đang chạy để hủy.');
    } catch (error) {
      setAnalysisState('error');
      if (error instanceof ApiError && error.status === 401) {
        clearLocalSession('Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại.');
        return;
      }
      setAnalysisMessage(error instanceof Error ? error.message : 'Không hủy được job phân tích.');
    }
  }

  async function refreshPoseClassifier(nextSession = session) {
    if (!nextSession || !canManagePoseClassifier(nextSession.user.role)) {
      return;
    }
    try {
      const [statusResult, historyResult] = await Promise.all([
        api.poseClassifierStatus(nextSession.token),
        api.poseClassifierHistory(nextSession.token),
      ]);
      setPoseModelStatus(statusResult.model);
      setPoseLatestJob(statusResult.latest_job);
      setPoseHistory(historyResult.items);
    } catch (error) {
      if (error instanceof ApiError && error.status === 401) {
        clearLocalSession('Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại.');
        return;
      }
      setPoseState('error');
      setPoseMessage(error instanceof Error ? error.message : 'Không tải được trạng thái pose classifier.');
    }
  }

  async function pollPoseClassifierUntilSettled(nextSession = session): Promise<PoseClassifierJob | null> {
    if (!nextSession) {
      return null;
    }
    for (let attempt = 0; attempt < 20; attempt += 1) {
      await new Promise((resolve) => window.setTimeout(resolve, attempt < 4 ? 250 : 750));
      const statusResult = await api.poseClassifierStatus(nextSession.token);
      setPoseModelStatus(statusResult.model);
      setPoseLatestJob(statusResult.latest_job);
      if (
        statusResult.latest_job &&
        statusResult.latest_job.status !== 'queued' &&
        statusResult.latest_job.status !== 'processing'
      ) {
        const historyResult = await api.poseClassifierHistory(nextSession.token);
        setPoseHistory(historyResult.items);
        return statusResult.latest_job;
      }
    }
    return null;
  }

  async function handleRefreshPoseClassifier() {
    if (!session) {
      return;
    }
    setPoseState('loading');
    setPoseTargetKey('refresh');
    setPoseMessage('');
    await refreshPoseClassifier(session);
    setPoseState('ready');
    setPoseTargetKey('');
  }

  async function handleTrainPoseClassifier() {
    if (!session) {
      return;
    }
    setPoseState('loading');
    setPoseTargetKey('train');
    setPoseMessage('');
    try {
      const result = await api.trainPoseClassifier(session.token, {
        dry_run: poseDryRun,
        min_samples: Math.max(2, Math.min(10000, Number(poseMinSamples) || 10)),
      });
      setPoseLatestJob(result.job);
      setPoseState('ready');
      setPoseMessage(result.started ? (poseDryRun ? 'Đã tạo job dry-run train classifier.' : 'Đã tạo job train classifier.') : 'ML job đang chạy.');
      void pollPoseClassifierUntilSettled(session);
    } catch (error) {
      setPoseState('error');
      if (error instanceof ApiError && error.status === 401) {
        clearLocalSession('Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại.');
        return;
      }
      setPoseMessage(error instanceof Error ? error.message : 'Không tạo được job train classifier.');
    } finally {
      setPoseTargetKey('');
    }
  }

  async function handleApplyPoseClassifier() {
    if (!session) {
      return;
    }
    const selectedVideo = videos.find((video, index) => videoKey(video, index) === poseSelectedVideoKey);
    const mediaFilename = selectedVideo ? mediaFilenameForVideo(selectedVideo) : '';
    if (!selectedVideo || !mediaFilename) {
      setPoseState('error');
      setPoseMessage('Vui lòng chọn video đã phân tích có CSV để apply ML.');
      return;
    }
    setPoseState('loading');
    setPoseTargetKey('apply');
    setPoseMessage('');
    try {
      const result = await api.applyPoseClassifier(session.token, mediaFilename, { dry_run: poseDryRun });
      setPoseLatestJob(result.job);
      setPoseState('ready');
      setPoseMessage(result.started ? (poseDryRun ? 'Đã tạo job dry-run apply ML.' : 'Đã tạo job apply ML cho video.') : 'ML job đang chạy.');
      if (!poseDryRun) {
        void pollPoseClassifierUntilSettled(session).then(() => loadDashboard(session));
      } else {
        void pollPoseClassifierUntilSettled(session);
      }
    } catch (error) {
      setPoseState('error');
      if (error instanceof ApiError && error.status === 401) {
        clearLocalSession('Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại.');
        return;
      }
      setPoseMessage(error instanceof Error ? error.message : 'Không apply được pose classifier.');
    } finally {
      setPoseTargetKey('');
    }
  }

  async function refreshHfSync(nextSession = session, verify = false) {
    if (!nextSession || !canManageHfSync(nextSession.user.role)) {
      return;
    }
    try {
      const [statusResult, historyResult] = await Promise.all([
        api.hfSyncStatus(nextSession.token, { verify }),
        api.hfSyncHistory(nextSession.token),
      ]);
      setHfStatus(statusResult.hf);
      setHfLatestJob(statusResult.latest_job);
      setHfHistory(historyResult.items);
    } catch (error) {
      if (error instanceof ApiError && error.status === 401) {
        clearLocalSession('Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại.');
        return;
      }
      setHfState('error');
      setHfMessage(error instanceof Error ? error.message : 'Không tải được trạng thái HF sync.');
    }
  }

  async function pollHfSyncUntilSettled(nextSession = session): Promise<HfSyncJob | null> {
    if (!nextSession) {
      return null;
    }
    for (let attempt = 0; attempt < 20; attempt += 1) {
      await new Promise((resolve) => window.setTimeout(resolve, attempt < 4 ? 250 : 750));
      const statusResult = await api.hfSyncStatus(nextSession.token);
      setHfStatus(statusResult.hf);
      setHfLatestJob(statusResult.latest_job);
      if (
        statusResult.latest_job &&
        statusResult.latest_job.status !== 'queued' &&
        statusResult.latest_job.status !== 'processing'
      ) {
        const historyResult = await api.hfSyncHistory(nextSession.token);
        setHfHistory(historyResult.items);
        return statusResult.latest_job;
      }
    }
    return null;
  }

  async function handleRefreshHfSync() {
    if (!session) {
      return;
    }
    setHfState('loading');
    setHfTargetKey('refresh');
    setHfMessage('');
    await refreshHfSync(session, true);
    setHfState('ready');
    setHfTargetKey('');
  }

  async function handleStartHfMetadataSync() {
    if (!session) {
      return;
    }
    setHfState('loading');
    setHfTargetKey('sync');
    setHfMessage('');
    try {
      const result = await api.startHfSync(session.token, {
        dry_run: hfDryRun,
        files: hfSelectedFiles,
      });
      setHfLatestJob(result.job);
      setHfState('ready');
      setHfMessage(result.started ? (hfDryRun ? 'Đã tạo job dry-run sync metadata.' : 'Đã tạo job sync metadata.') : 'HF job đang chạy.');
      if (!hfDryRun) {
        void pollHfSyncUntilSettled(session).then(() => loadDashboard(session));
      } else {
        void pollHfSyncUntilSettled(session);
      }
    } catch (error) {
      setHfState('error');
      if (error instanceof ApiError && error.status === 401) {
        clearLocalSession('Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại.');
        return;
      }
      setHfMessage(error instanceof Error ? error.message : 'Không tạo được job HF sync.');
    } finally {
      setHfTargetKey('');
    }
  }

  async function handleUploadHfArtifact() {
    if (!session) {
      return;
    }
    const selectedVideo = videos.find((video, index) => videoKey(video, index) === hfSelectedVideoKey);
    const mediaFilename = selectedVideo ? mediaFilenameForVideo(selectedVideo) : '';
    if (!selectedVideo || !mediaFilename) {
      setHfState('error');
      setHfMessage('Vui lòng chọn video có artifact để upload.');
      return;
    }
    setHfState('loading');
    setHfTargetKey('upload');
    setHfMessage('');
    try {
      const result = await api.uploadHfArtifact(session.token, mediaFilename, hfArtifactKind, { dry_run: hfDryRun });
      setHfLatestJob(result.job);
      setHfState('ready');
      setHfMessage(result.started ? (hfDryRun ? 'Đã tạo job dry-run upload artifact.' : 'Đã tạo job upload artifact.') : 'HF job đang chạy.');
      void pollHfSyncUntilSettled(session);
    } catch (error) {
      setHfState('error');
      if (error instanceof ApiError && error.status === 401) {
        clearLocalSession('Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại.');
        return;
      }
      setHfMessage(error instanceof Error ? error.message : 'Không upload được artifact lên HF.');
    } finally {
      setHfTargetKey('');
    }
  }

  async function handleCreateHfReport() {
    if (!session) {
      return;
    }
    setHfState('loading');
    setHfTargetKey('report');
    setHfMessage('');
    try {
      const result = await api.createHfReport(session.token, { dry_run: hfDryRun });
      setHfLatestJob(result.job);
      setHfState('ready');
      setHfMessage(result.started ? (hfDryRun ? 'Đã tạo job dry-run báo cáo.' : 'Đã tạo job xuất báo cáo.') : 'HF job đang chạy.');
      void pollHfSyncUntilSettled(session);
    } catch (error) {
      setHfState('error');
      if (error instanceof ApiError && error.status === 401) {
        clearLocalSession('Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại.');
        return;
      }
      setHfMessage(error instanceof Error ? error.message : 'Không tạo được báo cáo HF.');
    } finally {
      setHfTargetKey('');
    }
  }

  function toggleHfSyncFile(file: string) {
    setHfSelectedFiles((current) =>
      current.includes(file) ? current.filter((item) => item !== file) : [...current, file],
    );
  }

  async function handleLoadAnalysisHistory(video: VideoRecord, index: number) {
    if (!session) {
      return;
    }
    const mediaFilename = mediaFilenameForVideo(video);
    if (!mediaFilename) {
      return;
    }
    const key = videoKey(video, index);
    setAnalysisState('loading');
    setAnalysisTargetKey(key);
    setAnalysisMessage('');
    try {
      const result = await api.analysisJobHistory(session.token, mediaFilename);
      setAnalysisHistories((current) => ({ ...current, [key]: result.items }));
      setAnalysisState('ready');
    } catch (error) {
      setAnalysisState('error');
      if (error instanceof ApiError && error.status === 401) {
        clearLocalSession('Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại.');
        return;
      }
      setAnalysisMessage(error instanceof Error ? error.message : 'Không tải được lịch sử job.');
    }
  }

  async function handleRefreshAnalysis(video: VideoRecord, index: number) {
    if (!session) {
      return;
    }
    const key = videoKey(video, index);
    setAnalysisState('loading');
    setAnalysisTargetKey(key);
    setAnalysisMessage('');
    try {
      await refreshAnalysisJob(video, index, session);
      setAnalysisState('ready');
    } catch (error) {
      setAnalysisState('error');
      if (error instanceof ApiError && error.status === 401) {
        clearLocalSession('Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại.');
        return;
      }
      setAnalysisMessage(error instanceof Error ? error.message : 'Không tải được tiến độ phân tích.');
    }
  }

  async function handleLoadArtifacts(video: VideoRecord, index: number) {
    if (!session) {
      return;
    }
    const mediaFilename = mediaFilenameForVideo(video);
    if (!mediaFilename) {
      setArtifactState('error');
      setArtifactMessage('Video này chưa có file media để tra cứu kết quả.');
      return;
    }
    const key = videoKey(video, index);
    setArtifactState('loading');
    setArtifactTargetKey(key);
    setArtifactMessage('');
    try {
      const result = await api.videoResult(session.token, mediaFilename);
      setSelectedResult(result);
      resetFrameGallery();
      resetChartPreview();
      setArtifactState('ready');
      setArtifactMessage('');
      if (result.ai_detail_allowed) {
        void loadAnalysisFrames(result.video.stored_filename, 1, 'ALL');
        void loadAnalysisChart(result.video.stored_filename, 'ALL');
      }
    } catch (error) {
      setArtifactState('error');
      if (error instanceof ApiError && error.status === 401) {
        clearLocalSession('Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại.');
        return;
      }
      setArtifactMessage(error instanceof Error ? error.message : 'Không tải được danh sách kết quả phân tích.');
    }
  }

  async function loadAnalysisChart(storedFilename: string, label: FrameLabel = framesLabel) {
    if (!session) {
      return;
    }
    setChartState('loading');
    setChartMessage('');
    try {
      const chart = await api.analysisChart(session.token, storedFilename, label);
      setAnalysisChart(chart);
      setChartState('ready');
    } catch (error) {
      setAnalysisChart(null);
      if (error instanceof ApiError && error.status === 401) {
        clearLocalSession('Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại.');
        return;
      }
      if (error instanceof ApiError && error.status === 404) {
        setChartState('idle');
        setChartMessage('');
        return;
      }
      setChartState('error');
      setChartMessage(error instanceof Error ? error.message : 'Không tải được dữ liệu biểu đồ.');
    }
  }

  async function loadAnalysisFrames(storedFilename: string, page = framesPage, label = framesLabel) {
    if (!session) {
      return;
    }
    setFramesState('loading');
    setFramesMessage('');
    try {
      const result = await api.analysisFrames(session.token, storedFilename, {
        page,
        pageSize: 12,
        label,
      });
      clearFrameImages();
      const imageEntries = await Promise.all(
        result.items
          .filter((item) => item.has_image && item.image_id)
          .map(async (item) => {
            try {
              const blob = await api.analysisFrameBlob(session.token, storedFilename, item.image_id);
              return [item.image_id, URL.createObjectURL(blob)] as const;
            } catch {
              return null;
            }
          }),
      );
      const nextUrls = Object.fromEntries(imageEntries.filter((entry): entry is readonly [string, string] => Boolean(entry)));
      frameUrlMapRef.current = nextUrls;
      setFrameImageUrls(nextUrls);
      setFocusedFrame(null);
      setAnalysisFrames(result);
      setFramesPage(result.pagination.page);
      setFramesLabel(result.filter);
      setFramesState('ready');
    } catch (error) {
      clearFrameImages();
      setFramesState('error');
      if (error instanceof ApiError && error.status === 401) {
        clearLocalSession('Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại.');
        return;
      }
      setFramesMessage(error instanceof Error ? error.message : 'Không tải được gallery frame.');
    }
  }

  function handleFrameLabelChange(nextLabel: FrameLabel) {
    setFramesLabel(nextLabel);
    setFramesPage(1);
    if (selectedResult?.video.stored_filename) {
      void loadAnalysisFrames(selectedResult.video.stored_filename, 1, nextLabel);
      void loadAnalysisChart(selectedResult.video.stored_filename, nextLabel);
    }
  }

  function handleFramePageChange(nextPage: number) {
    if (!selectedResult?.video.stored_filename || !analysisFrames) {
      return;
    }
    const boundedPage = Math.max(1, Math.min(analysisFrames.pagination.total_pages, nextPage));
    setFramesPage(boundedPage);
    void loadAnalysisFrames(selectedResult.video.stored_filename, boundedPage, framesLabel);
  }

  function openFocusedFrame(frame: AnalysisFrameItem) {
    const url = frame.image_id ? frameImageUrls[frame.image_id] : '';
    if (!url) {
      return;
    }
    setFocusedFrame({ frame, url });
  }

  async function handleDownloadArtifact(kind: string, filename: string) {
    if (!session || !selectedResult?.video.stored_filename) {
      return;
    }
    setArtifactState('loading');
    setArtifactMessage('');
    try {
      const blob = await api.artifactBlob(session.token, selectedResult.video.stored_filename, kind);
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = filename || `${kind}.dat`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
      setArtifactState('ready');
    } catch (error) {
      setArtifactState('error');
      setArtifactMessage(error instanceof Error ? error.message : 'Không tải được artifact.');
    }
  }

  useEffect(() => {
    if (session) {
      void loadDashboard(session);
    }
  }, [session?.token]);

  const filteredVideos = useMemo(() => videos.filter((video) => matchesQuery(video as RecordLike, query)), [query, videos]);
  const filteredPatients = useMemo(() => patients.filter((patient) => matchesQuery(patient as RecordLike, query)), [patients, query]);
  const filteredUsers = useMemo(() => users.filter((user) => matchesQuery(user as RecordLike, query)), [query, users]);
  const filteredSymptoms = useMemo(() => symptoms.filter((symptom) => matchesQuery(symptom, query)), [query, symptoms]);
  const filteredSchedules = useMemo(() => schedules.filter((schedule) => matchesQuery(schedule, query)), [query, schedules]);
  const filteredResearch = useMemo(
    () => researchRecords.filter((record) => matchesQuery(record, query)),
    [query, researchRecords],
  );

  useEffect(() => {
    if (!filteredVideos.length) {
      setPoseSelectedVideoKey('');
      setHfSelectedVideoKey('');
      return;
    }
    const firstWithMedia = filteredVideos.find((video) => mediaFilenameForVideo(video));
    if (!firstWithMedia) {
      return;
    }
    const firstKey = videoKey(firstWithMedia, filteredVideos.indexOf(firstWithMedia));
    if (!poseSelectedVideoKey || !filteredVideos.some((video, index) => videoKey(video, index) === poseSelectedVideoKey)) {
      setPoseSelectedVideoKey(firstKey);
    }
    if (!hfSelectedVideoKey || !filteredVideos.some((video, index) => videoKey(video, index) === hfSelectedVideoKey)) {
      setHfSelectedVideoKey(firstKey);
    }
  }, [filteredVideos, hfSelectedVideoKey, poseSelectedVideoKey]);

  useEffect(() => {
    if (!session || activeView !== 'videos' || !filteredVideos.length) {
      return;
    }
    let active = true;
    Promise.all(
      filteredVideos.slice(0, 20).map(async (video, index) => {
        const mediaFilename = mediaFilenameForVideo(video);
        if (!mediaFilename) {
          return null;
        }
        const result = await api.latestAnalysisJob(session.token, mediaFilename);
        return [videoKey(video, index), result.job] as const;
      }),
    )
      .then((entries) => {
        if (!active) {
          return;
        }
        setAnalysisJobs((current) => {
          const next = { ...current };
          for (const entry of entries) {
            if (entry) {
              next[entry[0]] = entry[1];
            }
          }
          return next;
        });
      })
      .catch(() => undefined);
    return () => {
      active = false;
    };
  }, [activeView, filteredVideos, session?.token]);

  useEffect(() => {
    if (!session) {
      return;
    }
    const hasProcessing = Object.values(analysisJobs).some((job) => job?.status === 'processing');
    if (!hasProcessing) {
      return;
    }
    const timer = window.setInterval(() => {
      filteredVideos.forEach((video, index) => {
        const key = videoKey(video, index);
        if (analysisJobs[key]?.status === 'processing') {
          void refreshAnalysisJob(video, index, session);
        }
      });
    }, 3000);
    return () => window.clearInterval(timer);
  }, [analysisJobs, filteredVideos, session?.token]);

  useEffect(() => {
    return () => {
      if (previewUrlRef.current) {
        URL.revokeObjectURL(previewUrlRef.current);
        previewUrlRef.current = null;
      }
      Object.values(frameUrlMapRef.current).forEach((url) => URL.revokeObjectURL(url));
      frameUrlMapRef.current = {};
    };
  }, []);

  const evaluatedCount = useMemo(() => {
    return videos.filter((video) => matchingEvaluation(video, evaluations)).length;
  }, [videos, evaluations]);

  const pendingVideos = useMemo(
    () => videos.filter((video) => !String(video.status || '').includes('Đã phân tích')).length,
    [videos],
  );

  const metricCards = useMemo(() => {
    const role = session?.user.role || '';
    if (role === PATIENT_ROLE) {
      return [
        { label: 'Video của tôi', value: videos.length },
        { label: 'Đã được đánh giá', value: evaluatedCount },
        { label: 'Khai báo đau', value: symptoms.length },
        { label: 'Lịch nhắc', value: schedules.length },
      ];
    }
    if (role === DOCTOR_ROLE) {
      return [
        { label: 'Bệnh nhân phụ trách', value: patients.length },
        { label: 'Video cần xem', value: videos.length },
        { label: 'Đã đánh giá', value: evaluations.length },
        { label: 'Lịch đã tạo', value: schedules.length },
      ];
    }
    if (role === RESEARCHER_ROLE) {
      return [
        { label: 'Video nghiên cứu', value: videos.length },
        { label: 'Chờ AI', value: pendingVideos },
        { label: 'Phiếu NCKH', value: researchRecords.length },
        { label: 'Backend', value: health === 'ready' ? 'OK' : 'Lỗi' },
      ];
    }
    if (role === ADMIN_ROLE) {
      return [
        { label: 'Người dùng', value: users.length },
        { label: 'Bệnh nhân', value: patients.length },
        { label: 'Video', value: videos.length },
        { label: 'Backend', value: health === 'ready' ? 'OK' : 'Lỗi' },
      ];
    }
    return [
      { label: 'Video', value: videos.length },
      { label: 'Đã đánh giá', value: evaluatedCount },
      { label: 'Bệnh nhân', value: patients.length },
      { label: 'Backend', value: health === 'ready' ? 'OK' : 'Lỗi' },
    ];
  }, [evaluatedCount, evaluations.length, health, patients.length, pendingVideos, researchRecords.length, schedules.length, session?.user.role, symptoms.length, users.length, videos.length]);

  const workflowCards = useMemo(() => {
    const role = session?.user.role || '';
    if (role === PATIENT_ROLE) {
      return [
        { view: 'videos' as ViewId, title: 'Gửi video tập luyện', body: 'Upload bài tập mới để bác sĩ và NCV theo dõi.' },
        { view: 'symptoms' as ViewId, title: 'Khai báo đau VAS', body: 'Ghi nhận triệu chứng trước hoặc sau buổi tập.' },
        { view: 'schedules' as ViewId, title: 'Xem lịch nhắc', body: 'Theo dõi lịch hẹn, lịch tập và lịch thuốc.' },
        { view: 'info' as ViewId, title: 'Đọc hướng dẫn', body: 'Xem quy trình tập và thông tin nghiên cứu.' },
      ];
    }
    if (role === DOCTOR_ROLE) {
      return [
        { view: 'videos' as ViewId, title: 'Đánh giá video', body: 'Ghi nhận ground truth, lỗi sai và chỉ định tiếp theo.' },
        { view: 'schedules' as ViewId, title: 'Tạo lịch nhắc', body: 'Gửi lịch hẹn, lịch tập hoặc thuốc cho bệnh nhân.' },
        { view: 'research' as ViewId, title: 'Lập phiếu nghiên cứu', body: 'Hoàn thiện dữ liệu lâm sàng phục vụ nghiên cứu.' },
        { view: 'info' as ViewId, title: 'Xem tài liệu', body: 'Tra cứu hướng dẫn, kiến thức PHCN và AI.' },
      ];
    }
    if (role === RESEARCHER_ROLE) {
      return [
        { view: 'videos' as ViewId, title: 'Chạy phân tích AI', body: 'Theo dõi hàng đợi video và tiến độ xử lý.' },
        { view: 'research' as ViewId, title: 'Rà soát dữ liệu NCKH', body: 'Xem phiếu nghiên cứu ở dạng đã giả danh.' },
        { view: 'patients' as ViewId, title: 'Danh sách đối tượng', body: 'Xem danh sách bệnh nhân ở dạng mã nghiên cứu.' },
        { view: 'info' as ViewId, title: 'Thông tin đề tài', body: 'Xem hướng dẫn, team và phản hồi người dùng.' },
      ];
    }
    if (role === ADMIN_ROLE) {
      return [
        { view: 'users' as ViewId, title: 'Cấp tài khoản', body: 'Tạo tài khoản bác sĩ, NCV, admin hoặc bệnh nhân.' },
        { view: 'videos' as ViewId, title: 'Giám sát dữ liệu', body: 'Xem toàn bộ video, job AI và đánh giá.' },
        { view: 'research' as ViewId, title: 'Kiểm tra phiếu NCKH', body: 'Rà soát dữ liệu nghiên cứu trong hệ thống.' },
        { view: 'info' as ViewId, title: 'Phản hồi hệ thống', body: 'Theo dõi nội dung hướng dẫn và góp ý vận hành.' },
      ];
    }
    return [];
  }, [session?.user.role]);

  const availableViews = useMemo(() => {
    const role = session?.user.role || '';
    const views: Array<{ id: ViewId; label: string; icon: LucideIcon; count: number }> = [
      { id: 'home', label: viewLabelForRole('home', role), icon: Activity, count: 0 },
      { id: 'videos', label: viewLabelForRole('videos', role), icon: FileVideo, count: videos.length },
      { id: 'patients', label: viewLabelForRole('patients', role), icon: UsersRound, count: patients.length },
      { id: 'symptoms', label: viewLabelForRole('symptoms', role), icon: ClipboardList, count: symptoms.length },
    ];
    if (canViewSchedules(role)) {
      views.push({ id: 'schedules', label: viewLabelForRole('schedules', role), icon: CalendarDays, count: schedules.length });
    }
    if (canViewResearch(role)) {
      views.push({ id: 'research', label: viewLabelForRole('research', role), icon: FlaskConical, count: researchRecords.length });
    }
    views.push({ id: 'info', label: viewLabelForRole('info', role), icon: NotebookTabs, count: canReviewFeedback(role) ? feedbackRecords.length : 0 });
    if (canManageUsers(role)) {
      views.push({ id: 'users', label: 'Người dùng', icon: UserPlus, count: users.length });
    }
    return views;
  }, [feedbackRecords.length, patients.length, researchRecords.length, schedules.length, session?.user.role, symptoms.length, users.length, videos.length]);

  useEffect(() => {
    if (!availableViews.some((view) => view.id === activeView)) {
      setActiveView(availableViews[0]?.id || 'home');
    }
  }, [activeView, availableViews]);

  const activeRole = session?.user.role || '';
  const activeCount = {
    home: videos.length + patients.length + symptoms.length + schedules.length + researchRecords.length + users.length,
    videos: filteredVideos.length,
    patients: filteredPatients.length,
    symptoms: filteredSymptoms.length,
    schedules: filteredSchedules.length,
    research: filteredResearch.length,
    info: canReviewFeedback(activeRole) ? feedbackRecords.length : 0,
    users: filteredUsers.length,
  }[activeView];

  async function handleLogin(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const formData = new FormData(event.currentTarget);
    const username = String(formData.get('username') || '');
    const password = String(formData.get('password') || '');
    setMessage('');
    setLoadState('loading');
    try {
      const login = await api.login(username, password);
      const nextSession = { token: login.access_token, user: login.user };
      window.localStorage.setItem('rehab_token', nextSession.token);
      window.localStorage.setItem('rehab_user', JSON.stringify(nextSession.user));
      setSession(nextSession);
      setActiveView('home');
      setLoadState('ready');
    } catch (error) {
      setLoadState('error');
      setMessage(error instanceof Error ? error.message : 'Đăng nhập thất bại.');
    }
  }

  async function handleRegister(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const formData = new FormData(event.currentTarget);
    const payload = {
      username: String(formData.get('username') || ''),
      full_name: String(formData.get('full_name') || ''),
      email: String(formData.get('email') || ''),
      password: String(formData.get('password') || ''),
      confirm_password: String(formData.get('confirm_password') || ''),
    };
    setMessage('');
    setLoadState('loading');
    try {
      const registration = await api.register(payload);
      const nextSession = { token: registration.access_token, user: registration.user };
      window.localStorage.setItem('rehab_token', nextSession.token);
      window.localStorage.setItem('rehab_user', JSON.stringify(nextSession.user));
      setSession(nextSession);
      setActiveView('home');
      setLoadState('ready');
    } catch (error) {
      setLoadState('error');
      setMessage(error instanceof Error ? error.message : 'Đăng ký thất bại.');
    }
  }

  async function handleLogout() {
    if (session) {
      try {
        await api.logout(session.token);
      } catch {
        // Local session cleanup is still the important part for this frontend slice.
      }
    }
    clearLocalSession();
  }

  if (!session) {
    return (
      <main className="auth-shell">
        <section className="auth-panel">
          <div className="brand-mark">
            <Activity size={28} />
          </div>
          <div>
            <p className="eyebrow">Rehab AI Monitor</p>
            <h1>{authMode === 'login' ? 'Đăng nhập hệ thống' : 'Đăng ký bệnh nhân'}</h1>
            <p className="muted">
              {authMode === 'login'
                ? 'Theo dõi video, lịch nhắc và kết quả phục hồi chức năng.'
                : 'Tạo tài khoản bệnh nhân để bắt đầu theo dõi phục hồi chức năng.'}
            </p>
          </div>
          <div className="auth-switch" role="tablist" aria-label="Chọn chế độ xác thực">
            <button className={authMode === 'login' ? 'active' : ''} onClick={() => setAuthMode('login')} type="button">
              Đăng nhập
            </button>
            <button className={authMode === 'register' ? 'active' : ''} onClick={() => setAuthMode('register')} type="button">
              Đăng ký
            </button>
          </div>
          {authMode === 'login' ? (
            <form className="login-form" onSubmit={handleLogin}>
              <label>
                Tên đăng nhập
                <input name="username" autoComplete="username" placeholder="admin / doctor / patient" required />
              </label>
              <label>
                Mật khẩu
                <input name="password" type="password" autoComplete="current-password" required />
              </label>
              <button type="submit" disabled={loadState === 'loading'}>
                {loadState === 'loading' ? <RefreshCw className="spin" size={18} /> : <Shield size={18} />}
                Đăng nhập
              </button>
            </form>
          ) : (
            <form className="login-form" onSubmit={handleRegister}>
              <label>
                Họ và tên
                <input name="full_name" autoComplete="name" placeholder="Nguyễn Văn A" required />
              </label>
              <label>
                Tên đăng nhập
                <input name="username" autoComplete="username" placeholder="patient01" minLength={3} required />
              </label>
              <label>
                Email liên hệ
                <input name="email" type="email" autoComplete="email" placeholder="email@example.com" required />
              </label>
              <label>
                Mật khẩu
                <input name="password" type="password" autoComplete="new-password" minLength={6} required />
              </label>
              <label>
                Xác nhận mật khẩu
                <input name="confirm_password" type="password" autoComplete="new-password" minLength={6} required />
              </label>
              <button type="submit" disabled={loadState === 'loading'}>
                {loadState === 'loading' ? <RefreshCw className="spin" size={18} /> : <UserRound size={18} />}
                Tạo tài khoản
              </button>
              <p className="form-note">Tài khoản bác sĩ, nghiên cứu viên và quản trị viên do quản trị viên cấp.</p>
            </form>
          )}
          <div className={`status-line ${health === 'ready' ? 'ok' : health === 'error' ? 'bad' : ''}`}>
            {health === 'ready' ? <CheckCircle2 size={16} /> : <AlertCircle size={16} />}
            Backend: {api.baseUrl}
          </div>
          {message ? <div className="alert">{message}</div> : null}
        </section>
      </main>
    );
  }

  const videoColumns: TableColumn<VideoRecord>[] = [
    { key: 'patient', label: 'Bệnh nhân', render: (video) => patientLabel(video as RecordLike) },
    {
      key: 'video',
      label: 'Video',
      render: (video) => (
        <span className="video-name">
          <FileVideo size={16} />
          {textValue(video.video_name)}
        </span>
      ),
    },
    { key: 'exercise', label: 'Bài tập', render: (video) => textValue(video.exercise) },
    {
      key: 'status',
      label: 'Trạng thái',
      render: (video) => <span className={`pill ${statusClass(video.status)}`}>{textValue(video.status, 'Chờ xử lý')}</span>,
    },
    {
      key: 'evaluation',
      label: 'Đánh giá',
      render: (video) => {
        const evaluation = matchingEvaluation(video, evaluations);
        return evaluation ? textValue(evaluation.doctor_result, 'Đã đánh giá') : 'Chưa có';
      },
    },
    {
      key: 'analysis',
      label: 'Phân tích',
      render: (video, index) => {
        const key = videoKey(video, index);
        const job = analysisJobs[key];
        const history = analysisHistories[key] || [];
        const progress = Math.round(Math.max(0, Math.min(1, job?.progress ?? 0)) * 100);
        const isLoading = analysisState === 'loading' && analysisTargetKey === key;
        const hasMedia = Boolean(mediaFilenameForVideo(video));
        const isProcessing = job?.status === 'processing';
        const canRetry = Boolean(job && job.status !== 'processing');
        return (
          <div className="analysis-cell">
            <div className="analysis-status">
              <span className={`pill ${statusClass(job?.status || video.status)}`}>{analysisStatusLabel(job?.status)}</span>
              <span>{job ? `${progress}%` : 'N/A'}</span>
            </div>
            {job ? (
              <div className="analysis-meta">
                <span>{analysisActionLabel(job.job_meta?.action)}</span>
                <span>{analysisRunLabel(job)}</span>
                <span>{textValue(job.job_meta?.options?.model_type)}</span>
              </div>
            ) : null}
            <div className="progress-track" aria-label="Tiến độ phân tích">
              <span className={`progress-fill ${statusClass(job?.status)}`} style={{ width: `${job ? progress : 0}%` }} />
            </div>
            {job?.steps?.length ? (
              <div className="analysis-steps" aria-label="Bốn bước xử lý AI">
                {job.steps.map((step) => (
                  <span className={step.status} key={step.key} title={step.label}>
                    {step.label}
                  </span>
                ))}
              </div>
            ) : null}
            <div className="analysis-actions">
              {canStartAnalysis(session.user.role) ? (
                <>
                  <button
                    className="table-action"
                    onClick={() => void handleStartAnalysis(video, index)}
                    disabled={!hasMedia || isLoading || isProcessing}
                    title={hasMedia ? 'Bắt đầu phân tích AI' : 'Chưa có file media'}
                    type="button"
                  >
                    {isLoading ? <RefreshCw className="spin" size={16} /> : <Cpu size={16} />}
                    Chạy
                  </button>
                  <button
                    className="table-action muted-action"
                    onClick={() => void handleRerunAnalysis(video, index)}
                    disabled={!hasMedia || isLoading || isProcessing}
                    title={hasMedia ? 'Chạy lại với cấu hình model hiện tại' : 'Chưa có file media'}
                    type="button"
                  >
                    <Cpu size={16} />
                    Rerun
                  </button>
                  <button
                    className="table-action muted-action"
                    onClick={() => void handleRetryAnalysis(video, index)}
                    disabled={!hasMedia || isLoading || !canRetry}
                    title={hasMedia ? 'Retry với cấu hình lần chạy gần nhất' : 'Chưa có file media'}
                    type="button"
                  >
                    <RefreshCw size={16} />
                    Retry
                  </button>
                  <button
                    className="table-action danger-action"
                    onClick={() => void handleCancelAnalysis(video, index)}
                    disabled={!hasMedia || isLoading || !isProcessing}
                    title={hasMedia ? 'Hủy job đang chạy' : 'Chưa có file media'}
                    type="button"
                  >
                    <X size={16} />
                    Hủy
                  </button>
                </>
              ) : null}
              <button
                className="table-action muted-action"
                onClick={() => void handleRefreshAnalysis(video, index)}
                disabled={!hasMedia || isLoading}
                title={hasMedia ? 'Cập nhật tiến độ' : 'Chưa có file media'}
                type="button"
              >
                <RefreshCw className={isLoading ? 'spin' : ''} size={16} />
                Cập nhật
              </button>
              <button
                className="table-action muted-action"
                onClick={() => void handleLoadAnalysisHistory(video, index)}
                disabled={!hasMedia || isLoading}
                title={hasMedia ? 'Xem lịch sử job' : 'Chưa có file media'}
                type="button"
              >
                <ClipboardList size={16} />
                History
              </button>
            </div>
            {history.length ? (
              <div className="analysis-history">
                {history.slice(-3).map((item) => (
                  <div key={item.run_id || `${item.job_id}-${item.start_time}`}>
                    <span className={`pill ${statusClass(item.status)}`}>{analysisStatusLabel(item.status)}</span>
                    <strong>{analysisActionLabel(item.job_meta?.action)}</strong>
                    <small>
                      {textValue(item.job_meta?.options?.model_type)} · {jobTimeLabel(item)}
                    </small>
                  </div>
                ))}
              </div>
            ) : null}
          </div>
        );
      },
    },
    {
      key: 'actions',
      label: 'Thao tác',
      render: (video, index) => {
        const key = videoKey(video, index);
        const isOpen = videoPreview?.key === key;
        const isLoading = previewState === 'loading' && previewTargetKey === key;
        const artifactLoading = artifactState === 'loading' && artifactTargetKey === key;
        const hasMedia = Boolean(mediaFilenameForVideo(video));
        return (
          <div className="row-actions">
            <button
              className="table-action"
              onClick={() => void handlePreviewVideo(video, index)}
              disabled={!hasMedia || isLoading}
              title={hasMedia ? (isOpen ? 'Ẩn video' : 'Xem video') : 'Chưa có file media'}
              type="button"
            >
              {isOpen ? <EyeOff size={16} /> : <Eye size={16} />}
              {isLoading ? 'Đang tải' : isOpen ? 'Ẩn' : 'Xem'}
            </button>
            <button
              className="table-action muted-action"
              onClick={() => void handleLoadArtifacts(video, index)}
              disabled={!hasMedia || artifactLoading}
              title={hasMedia ? 'Xem kết quả phân tích đã lưu' : 'Chưa có file media'}
              type="button"
            >
              {artifactLoading ? <RefreshCw className="spin" size={16} /> : <FlaskConical size={16} />}
              Kết quả
            </button>
          </div>
        );
      },
    },
  ];

  const patientColumns: TableColumn<PatientRecord>[] = [
    { key: 'name', label: 'Bệnh nhân', render: (patient) => patientLabel(patient as RecordLike) },
    { key: 'username', label: 'Mã/Tài khoản', render: (patient) => <span className="mono">{textValue(patient.subject_code, patient.username)}</span> },
    { key: 'doctor', label: 'Bác sĩ phụ trách', render: (patient) => textValue(patient.assigned_doctor_username) },
    {
      key: 'status',
      label: 'Trạng thái',
      render: (patient) => <span className={`pill ${patient.active === false ? 'danger' : 'success'}`}>{patient.active === false ? 'Tạm khóa' : 'Hoạt động'}</span>,
    },
  ];

  const symptomColumns: TableColumn<SymptomRecord>[] = [
    { key: 'patient', label: 'Bệnh nhân', render: (symptom) => patientLabel(symptom) },
    { key: 'symptoms', label: 'Triệu chứng', render: (symptom) => textValue(symptom.symptoms, symptom.pain_level) },
    { key: 'pain', label: 'Mức đau', render: (symptom) => textValue(symptom.pain_score, symptom.pain_level) },
    { key: 'time', label: 'Thời gian', render: (symptom) => textValue(symptom.created_at, symptom.timestamp, symptom.time) },
    { key: 'notes', label: 'Ghi chú', render: (symptom) => textValue(symptom.notes) },
  ];

  const evaluationColumns: TableColumn<EvaluationRecord>[] = [
    { key: 'patient', label: 'Bệnh nhân', render: (evaluation) => textValue(evaluation.patient_username) },
    { key: 'video', label: 'Video', render: (evaluation) => textValue(evaluation.video_name) },
    { key: 'result', label: 'Kết quả', render: (evaluation) => textValue(evaluation.doctor_result) },
    { key: 'plan', label: 'Chỉ định', render: (evaluation) => textValue(evaluation.plan) },
    { key: 'time', label: 'Thời gian', render: (evaluation) => textValue(evaluation.time) },
    {
      key: 'actions',
      label: 'Thao tác',
      render: (evaluation) =>
        canManageClinical(session.user.role) ? (
          <button className="table-action danger-action" onClick={() => void handleDeleteEvaluation(evaluation)} disabled={!evaluation.id || formState === 'loading'} type="button">
            <Trash2 size={16} />
            Xóa
          </button>
        ) : (
          'N/A'
        ),
    },
  ];

  const scheduleColumns: TableColumn<ScheduleRecord>[] = [
    { key: 'patient', label: 'Bệnh nhân', render: (schedule) => patientLabel(schedule) },
    { key: 'title', label: 'Nội dung', render: (schedule) => textValue(schedule.title, schedule.exercise_name, schedule.medication_name, schedule.type) },
    { key: 'time', label: 'Thời gian', render: (schedule) => textValue(schedule.datetime, schedule.date, schedule.time) },
    {
      key: 'status',
      label: 'Trạng thái',
      render: (schedule) => <span className={`pill ${statusClass(schedule.status)}`}>{textValue(schedule.status, 'Đang theo dõi')}</span>,
    },
    { key: 'notes', label: 'Ghi chú', render: (schedule) => textValue(schedule.notes) },
    {
      key: 'actions',
      label: 'Thao tác',
      render: (schedule) =>
        canManageClinical(session.user.role) ? (
          <button className="table-action danger-action" onClick={() => void handleDeleteSchedule(schedule)} disabled={!schedule.id || formState === 'loading'} type="button">
            <Trash2 size={16} />
            Xóa
          </button>
        ) : (
          'N/A'
        ),
    },
  ];

  const researchColumns: TableColumn<ResearchRecord>[] = [
    { key: 'subject', label: 'Đối tượng', render: (record) => patientLabel(record) },
    { key: 'result', label: 'Kết quả', render: (record) => textValue(record.general_result, record.doctor_result, record.result) },
    { key: 'exercise', label: 'Bài tập/Video', render: (record) => textValue(record.exercise, record.video_name) },
    { key: 'time', label: 'Thời gian', render: (record) => textValue(record.created_at, record.timestamp, record.time) },
    {
      key: 'actions',
      label: 'Thao tác',
      render: (record) =>
        session.user.role !== PATIENT_ROLE ? (
          <button className="table-action danger-action" onClick={() => void handleDeleteResearchRecord(record)} disabled={!record.id || formState === 'loading'} type="button">
            <Trash2 size={16} />
            Xóa
          </button>
        ) : (
          'N/A'
        ),
    },
  ];

  const userColumns: TableColumn<PatientRecord>[] = [
    { key: 'username', label: 'Tài khoản', render: (user) => <span className="mono">{textValue(user.username)}</span> },
    { key: 'name', label: 'Họ tên', render: (user) => textValue(user.full_name) },
    { key: 'role', label: 'Vai trò', render: (user) => textValue(user.role) },
    { key: 'email', label: 'Email', render: (user) => textValue(user.email) },
    {
      key: 'status',
      label: 'Trạng thái',
      render: (user) => (
        <div className="status-stack">
          <span className={`pill ${user.active === false ? 'danger' : 'success'}`}>{user.active === false ? 'Tạm khóa' : 'Hoạt động'}</span>
          {user.must_change_password ? <span className="pill warning">Cần đổi mật khẩu</span> : null}
        </div>
      ),
    },
    {
      key: 'actions',
      label: 'Thao tác',
      render: (user) => {
        const protectedUser = user.username === session.user.username || user.username === 'admin';
        const username = user.username || '';
        const busy = formState === 'loading';
        if (!username) {
          return 'N/A';
        }
        return (
          <div className="row-actions">
            <button
              className={user.active === false ? 'table-action' : 'table-action danger-action'}
              onClick={() => void handleSetUserActive(user, user.active === false)}
              disabled={protectedUser || busy}
              title={protectedUser ? 'Không thao tác trên tài khoản hệ thống/đang dùng' : user.active === false ? 'Mở khóa tài khoản' : 'Khóa tài khoản'}
              type="button"
            >
              {adminActionKey === `active:${username}` ? (
                <RefreshCw className="spin" size={16} />
              ) : user.active === false ? (
                <Unlock size={16} />
              ) : (
                <Lock size={16} />
              )}
              {user.active === false ? 'Mở' : 'Khóa'}
            </button>
            <button
              className="table-action muted-action"
              onClick={() => void handleResetUserPassword(user)}
              disabled={busy}
              title="Reset mật khẩu tạm và bắt đổi mật khẩu"
              type="button"
            >
              {adminActionKey === `reset:${username}` ? <RefreshCw className="spin" size={16} /> : <KeyRound size={16} />}
              Reset
            </button>
            <button
              className="table-action muted-action"
              onClick={() => void handleRevokeUserSessions(user)}
              disabled={busy}
              title="Thu hồi phiên đăng nhập của tài khoản"
              type="button"
            >
              {adminActionKey === `revoke:${username}` ? <RefreshCw className="spin" size={16} /> : <UserX size={16} />}
              Phiên
            </button>
            {protectedUser ? null : (
              <button className="table-action danger-action" onClick={() => void handleDeleteUser(user)} disabled={busy} type="button">
                <Trash2 size={16} />
                Xóa
              </button>
            )}
          </div>
        );
      },
    },
  ];

  const auditColumns: TableColumn<AuditLogRecord>[] = [
    { key: 'timestamp', label: 'Thời gian', render: (item) => textValue(item.timestamp) },
    { key: 'actor', label: 'Actor', render: (item) => <span className="mono">{textValue(item.actor)}</span> },
    { key: 'action', label: 'Hành động', render: (item) => auditActionLabel(item.action) },
    { key: 'target', label: 'Target', render: (item) => <span className="mono">{textValue(item.target)}</span> },
    {
      key: 'result',
      label: 'Kết quả',
      render: (item) => <span className={`pill ${statusClass(item.result)}`}>{textValue(item.result)}</span>,
    },
    { key: 'metadata', label: 'Metadata', render: (item) => auditMetadataLabel(item.metadata) },
  ];

  const feedbackColumns: TableColumn<FeedbackRecord>[] = [
    { key: 'timestamp', label: 'Thời gian', render: (item) => textValue(item.timestamp) },
    { key: 'actor', label: 'Người gửi', render: (item) => <span className="mono">{textValue(item.actor_username)}</span> },
    { key: 'role', label: 'Vai trò', render: (item) => textValue(item.actor_role) },
    { key: 'category', label: 'Nhóm', render: (item) => feedbackCategoryLabel(item.category) },
    { key: 'message', label: 'Nội dung', render: (item) => textValue(item.message) },
    {
      key: 'status',
      label: 'Trạng thái',
      render: (item) => <span className={`pill ${statusClass(item.status)}`}>{textValue(item.status, 'new')}</span>,
    },
  ];

  return (
    <main className="workspace">
      <aside className="sidebar">
        <div className="brand-row">
          <div className="brand-mark small">
            <Activity size={20} />
          </div>
          <div>
            <strong>Rehab AI</strong>
            <span>Monitor Workspace</span>
          </div>
        </div>
        <div className="user-panel">
          <UserRound size={22} />
          <div>
            <strong>{session.user.full_name || session.user.username}</strong>
            <span>{session.user.role}</span>
          </div>
        </div>
        <nav className="nav-list" aria-label="Khu vực dữ liệu">
          {availableViews.map((view) => {
            const Icon = view.icon;
            return (
              <button
                key={view.id}
                className={`nav-button ${activeView === view.id ? 'active' : ''}`}
                onClick={() => setActiveView(view.id)}
                type="button"
              >
                <Icon size={18} />
                <span>{view.label}</span>
                <strong>{countLabel(view.count)}</strong>
              </button>
            );
          })}
        </nav>
        <details className="sidebar-section">
          <summary>
            <KeyRound size={16} />
            Đổi mật khẩu
          </summary>
          <form className="mini-form" onSubmit={handleChangePassword}>
            <input name="old_password" type="password" autoComplete="current-password" placeholder="Mật khẩu hiện tại" required />
            <input name="new_password" type="password" autoComplete="new-password" placeholder="Mật khẩu mới" minLength={6} required />
            <input name="confirm_password" type="password" autoComplete="new-password" placeholder="Nhập lại mật khẩu mới" minLength={6} required />
            <button className="secondary-button" type="submit" disabled={formState === 'loading'}>
              <KeyRound size={16} />
              Cập nhật
            </button>
          </form>
        </details>
        <button className="secondary-button" onClick={() => void handleLogout()}>
          <LogOut size={18} />
          Đăng xuất
        </button>
      </aside>

      <section className="content">
        <header className="topbar">
          <div>
            <p className="eyebrow">{roleWorkspaceEyebrow(session.user.role)}</p>
            <h1>{roleWorkspaceTitle(session.user.role)}</h1>
          </div>
          <button className="icon-button" onClick={() => void loadDashboard()} disabled={loadState === 'loading'} title="Tải lại dữ liệu">
            <RefreshCw className={loadState === 'loading' ? 'spin' : ''} size={18} />
          </button>
        </header>

        <section className="metrics-grid">
          {metricCards.map((metric) => (
            <article className="metric-card" key={metric.label}>
              <span>{metric.label}</span>
              <strong className={metric.value === 'OK' ? 'success-text' : metric.value === 'Lỗi' ? 'danger-text' : ''}>{metric.value}</strong>
            </article>
          ))}
        </section>

        <section className="panel">
          <div className="panel-header">
            <div>
              <h2>{availableViews.find((view) => view.id === activeView)?.label || 'Dữ liệu'}</h2>
              <p>{activeCount} mục đang hiển thị</p>
            </div>
            {activeView === 'home' || activeView === 'info' ? null : (
              <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Lọc theo tên, mã, bài tập, trạng thái..." />
            )}
          </div>

          {message ? <div className="alert inline">{message}</div> : null}

          {activeView === 'home' ? (
            <section className="role-home">
              <div className="role-hero">
                <div>
                  <p className="eyebrow">{session.user.role}</p>
                  <h2>{roleWorkspaceTitle(session.user.role)}</h2>
                  <p>
                    {session.user.role === PATIENT_ROLE
                      ? 'Theo dõi buổi tập, khai báo triệu chứng và nhận lịch nhắc từ nhóm điều trị.'
                      : session.user.role === DOCTOR_ROLE
                        ? 'Xem video bệnh nhân, ghi nhận đánh giá lâm sàng và tạo lịch chăm sóc.'
                        : session.user.role === RESEARCHER_ROLE
                          ? 'Phân tích video, chuẩn hóa dữ liệu nghiên cứu và theo dõi kết quả AI.'
                          : 'Quản lý tài khoản, dữ liệu và trạng thái vận hành của hệ thống.'}
                  </p>
                </div>
              </div>
              <div className="workflow-grid">
                {workflowCards.map((card) => (
                  <button className="workflow-card" key={card.title} onClick={() => setActiveView(card.view)} type="button">
                    <span>{viewLabelForRole(card.view, session.user.role)}</span>
                    <strong>{card.title}</strong>
                    <p>{card.body}</p>
                    <ArrowRight size={18} />
                  </button>
                ))}
              </div>
            </section>
          ) : null}

          {activeView === 'videos' && session.user.role === PATIENT_ROLE ? (
            <form className="data-form" onSubmit={handleUploadVideo}>
              <div className="form-grid">
                <label>
                  Họ và tên
                  <input name="full_name" defaultValue={session.user.full_name || session.user.username} required />
                </label>
                <label>
                  Bài tập
                  <input name="exercise" placeholder="VD: Codman" required />
                </label>
                <label>
                  Video tập luyện
                  <input name="file" type="file" accept="video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,video/webm,.mp4,.mov,.avi,.mkv,.webm,.m4v" required />
                </label>
              </div>
              <div className="form-actions">
                <button className="primary-button" type="submit" disabled={formState === 'loading'}>
                  {formState === 'loading' ? <RefreshCw className="spin" size={18} /> : <FileVideo size={18} />}
                  Gửi video
                </button>
                <span className="form-help">Hỗ trợ MP4, MOV, AVI, MKV, WebM; tối đa 300MB.</span>
                {formMessage ? (
                  <span className={formState === 'error' ? 'form-status error' : 'form-status ok'}>{formMessage}</span>
                ) : null}
              </div>
            </form>
          ) : null}

          {activeView === 'videos' && canStartAnalysis(session.user.role) ? (
            <section className="analysis-config-panel" aria-label="Cấu hình phân tích AI">
              <div className="analysis-config-title">
                <Cpu size={18} />
                <div>
                  <h3>Cấu hình job AI</h3>
                  <span>Áp dụng cho nút Chạy/Rerun trong bảng video</span>
                </div>
              </div>
              <div className="analysis-config-grid">
                <label>
                  Model
                  <select
                    value={analysisOptions.model_type}
                    onChange={(event) =>
                      setAnalysisOptions((current) => ({
                        ...current,
                        model_type: event.target.value as AnalysisJobOptions['model_type'],
                      }))
                    }
                  >
                    <option value="MediaPipe Heavy">MediaPipe Heavy</option>
                    <option value="MediaPipe Full">MediaPipe Full</option>
                    <option value="MediaPipe Lite">MediaPipe Lite</option>
                  </select>
                </label>
                <label>
                  Skip step
                  <input
                    type="number"
                    min="0"
                    max="30"
                    value={analysisOptions.skip_step}
                    onChange={(event) =>
                      setAnalysisOptions((current) => ({
                        ...current,
                        skip_step: Number(event.target.value),
                      }))
                    }
                  />
                </label>
                <label>
                  Resize width
                  <input
                    type="number"
                    min="240"
                    max="2160"
                    step="80"
                    value={analysisOptions.resize_width}
                    onChange={(event) =>
                      setAnalysisOptions((current) => ({
                        ...current,
                        resize_width: Number(event.target.value),
                      }))
                    }
                  />
                </label>
                <label>
                  Confidence
                  <input
                    type="range"
                    min="0.1"
                    max="0.95"
                    step="0.05"
                    value={analysisOptions.min_confidence}
                    onChange={(event) =>
                      setAnalysisOptions((current) => ({
                        ...current,
                        min_confidence: Number(event.target.value),
                      }))
                    }
                  />
                  <strong>{Math.round(analysisOptions.min_confidence * 100)}%</strong>
                </label>
              </div>
            </section>
          ) : null}

          {activeView === 'videos' && canManagePoseClassifier(session.user.role) ? (
            <section className="analysis-config-panel pose-classifier-panel" aria-label="Pose classifier ML">
              <div className="analysis-config-title">
                <FlaskConical size={18} />
                <div>
                  <h3>Pose classifier ML</h3>
                  <span>
                    {modelReadyLabel(poseModelStatus)} · {poseModelStatus?.feature_count || 0} đặc trưng · checksum{' '}
                    {poseModelStatus?.checksum_ok ? 'OK' : 'chưa hợp lệ'}
                  </span>
                </div>
              </div>
              <div className="pose-status-grid">
                <div>
                  <span>Model</span>
                  <strong>{textValue(poseModelStatus?.model_path)}</strong>
                </div>
                <div>
                  <span>Cập nhật</span>
                  <strong>{textValue(poseModelStatus?.model_mtime)}</strong>
                </div>
                <div>
                  <span>Job gần nhất</span>
                  <strong>
                    {poseLatestJob
                      ? `${poseActionLabel(poseLatestJob.action)} · ${analysisStatusLabel(poseLatestJob.status)} · ${Math.round(
                          Math.max(0, Math.min(1, poseLatestJob.progress || 0)) * 100,
                        )}%`
                      : 'Chưa có'}
                  </strong>
                </div>
              </div>
              <div className="analysis-config-grid pose-config-grid">
                <label>
                  Video apply
                  <select value={poseSelectedVideoKey} onChange={(event) => setPoseSelectedVideoKey(event.target.value)}>
                    <option value="">Chọn video</option>
                    {filteredVideos.map((video, index) => (
                      <option value={videoKey(video, index)} key={videoKey(video, index)}>
                        {textValue(video.video_name, video.original_filename)} · {patientLabel(video as RecordLike)}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  Min samples
                  <input
                    type="number"
                    min="2"
                    max="10000"
                    value={poseMinSamples}
                    onChange={(event) => setPoseMinSamples(Number(event.target.value))}
                  />
                </label>
                <label className="checkbox-row">
                  <input checked={poseDryRun} onChange={(event) => setPoseDryRun(event.target.checked)} type="checkbox" />
                  Dry-run
                </label>
              </div>
              <div className="pose-actions">
                <button
                  className="table-action"
                  disabled={poseState === 'loading' && poseTargetKey === 'train'}
                  onClick={() => void handleTrainPoseClassifier()}
                  type="button"
                >
                  {poseState === 'loading' && poseTargetKey === 'train' ? <RefreshCw className="spin" size={16} /> : <Cpu size={16} />}
                  Train
                </button>
                <button
                  className="table-action muted-action"
                  disabled={!poseSelectedVideoKey || (poseState === 'loading' && poseTargetKey === 'apply')}
                  onClick={() => void handleApplyPoseClassifier()}
                  type="button"
                >
                  {poseState === 'loading' && poseTargetKey === 'apply' ? <RefreshCw className="spin" size={16} /> : <ArrowRight size={16} />}
                  Apply
                </button>
                <button
                  className="table-action muted-action"
                  disabled={poseState === 'loading' && poseTargetKey === 'refresh'}
                  onClick={() => void handleRefreshPoseClassifier()}
                  type="button"
                >
                  <RefreshCw className={poseState === 'loading' && poseTargetKey === 'refresh' ? 'spin' : ''} size={16} />
                  Cập nhật
                </button>
              </div>
              {poseLatestJob ? (
                <div className="progress-track" aria-label="Tiến độ ML">
                  <span
                    className={`progress-fill ${statusClass(poseLatestJob.status)}`}
                    style={{ width: `${Math.round(Math.max(0, Math.min(1, poseLatestJob.progress || 0)) * 100)}%` }}
                  />
                </div>
              ) : null}
              {poseMessage ? <div className={poseState === 'error' ? 'alert inline' : 'status-line inline ok'}>{poseMessage}</div> : null}
              {poseLatestJob?.status_msg || poseLatestJob?.error_msg ? (
                <p className={poseLatestJob.status === 'error' ? 'danger-text' : 'muted'}>
                  {textValue(poseLatestJob.status_msg, poseLatestJob.error_msg)}
                </p>
              ) : null}
              {poseHistory.length ? (
                <div className="analysis-history pose-history">
                  {poseHistory.slice(-3).map((job) => (
                    <div key={job.job_id}>
                      <span className={`pill ${statusClass(job.status)}`}>{analysisStatusLabel(job.status)}</span>
                      <strong>
                        {poseActionLabel(job.action)}
                        {job.dry_run ? ' · dry-run' : ''}
                      </strong>
                      <small>{poseJobTimeLabel(job)}</small>
                    </div>
                  ))}
                </div>
              ) : null}
            </section>
          ) : null}

          {activeView === 'symptoms' && session.user.role === PATIENT_ROLE ? (
            <form className="data-form" onSubmit={handleCreateSymptom}>
              <div className="form-grid">
                <label>
                  Họ và tên
                  <input name="full_name" defaultValue={session.user.full_name || session.user.username} required />
                </label>
                <label>
                  Mã định danh
                  <input name="patient_id" defaultValue={session.user.username} required />
                </label>
                <label>
                  Tuổi
                  <input name="age" type="number" min="0" max="120" defaultValue="22" required />
                </label>
                <label>
                  Giới tính
                  <select name="gender" required defaultValue="">
                    <option value="" disabled>
                      Chọn giới tính
                    </option>
                    <option value="Nam">Nam</option>
                    <option value="Nữ">Nữ</option>
                  </select>
                </label>
                <label>
                  Bài tập
                  <input name="exercise" placeholder="VD: Codman" required />
                </label>
                <label>
                  Mức đau VAS
                  <input name="vas" type="number" min="0" max="10" defaultValue="3" required />
                </label>
              </div>
              <label>
                Mô tả triệu chứng
                <textarea name="symptoms" rows={4} placeholder="VD: Đau nhói ở khớp vai khi nâng tay lên cao..." required />
              </label>
              <div className="form-actions">
                <button className="primary-button" type="submit" disabled={formState === 'loading'}>
                  {formState === 'loading' ? <RefreshCw className="spin" size={18} /> : <ClipboardList size={18} />}
                  Gửi khai báo
                </button>
                {formMessage ? (
                  <span className={formState === 'error' ? 'form-status error' : 'form-status ok'}>{formMessage}</span>
                ) : null}
              </div>
            </form>
          ) : null}

          {activeView === 'videos' && canManageClinical(session.user.role) ? (
            <form className="data-form" onSubmit={handleCreateEvaluation}>
              <div className="form-grid">
                <label>
                  Video cần đánh giá
                  <select name="video_key" required defaultValue="">
                    <option value="" disabled>
                      Chọn video
                    </option>
                    {filteredVideos.map((video, index) => (
                      <option key={videoKey(video, index)} value={videoKey(video, index)}>
                        {patientLabel(video as RecordLike)} - {textValue(video.exercise)} - {textValue(video.video_name)}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  Kết quả
                  <select name="doctor_result" required defaultValue="Gần đúng">
                    <option value="Đúng">Đúng</option>
                    <option value="Gần đúng">Gần đúng</option>
                    <option value="Sai">Sai</option>
                  </select>
                </label>
                <label>
                  Chỉ định
                  <select name="plan" required defaultValue="Tiếp tục">
                    <option value="Tiếp tục">Tiếp tục</option>
                    <option value="Chuyển bài">Chuyển bài</option>
                    <option value="Khám lại">Khám lại</option>
                  </select>
                </label>
              </div>
              <div className="check-row">
                {['Vị trí tay chưa đúng', 'Biên độ chưa đạt', 'Tốc độ quá nhanh/chậm', 'Sai tư thế thân người'].map((error) => (
                  <label key={error}>
                    <input name="errors" type="checkbox" value={error} />
                    {error}
                  </label>
                ))}
              </div>
              <div className="form-grid two">
                <label>
                  Nhận xét cho bệnh nhân
                  <textarea name="comments" rows={3} placeholder="Nhận xét ngắn gọn cho bệnh nhân" />
                </label>
                <label>
                  Ghi chú cho NCV
                  <textarea name="comments_ncv" rows={3} placeholder="Ghi chú chuyên môn cho nghiên cứu viên" />
                </label>
              </div>
              <div className="form-actions">
                <button className="primary-button" type="submit" disabled={formState === 'loading'}>
                  <ClipboardList size={18} />
                  Lưu đánh giá
                </button>
                {formMessage ? <span className={formState === 'error' ? 'form-status error' : 'form-status ok'}>{formMessage}</span> : null}
              </div>
            </form>
          ) : null}

          {activeView === 'schedules' && canManageClinical(session.user.role) ? (
            <form className="data-form" onSubmit={handleCreateSchedule}>
              <div className="form-grid">
                <label>
                  Bệnh nhân
                  <select name="patient_username" required defaultValue="">
                    <option value="" disabled>
                      Chọn bệnh nhân
                    </option>
                    {patients.map((patient, index) => (
                      <option key={recordKey('patient-option', patient as RecordLike, index)} value={patient.username}>
                        {patientLabel(patient as RecordLike)}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  Loại lịch
                  <select name="type" defaultValue="appointment">
                    <option value="appointment">Lịch hẹn khám</option>
                    <option value="exercise">Lịch tập luyện</option>
                    <option value="medication">Lịch uống thuốc</option>
                  </select>
                </label>
                <label>
                  Thời gian
                  <input name="datetime" type="datetime-local" />
                </label>
                <label>
                  Tiêu đề
                  <input name="title" placeholder="VD: Khám lại khớp vai" />
                </label>
                <label>
                  Bài tập
                  <input name="exercise_name" placeholder="VD: Codman" />
                </label>
                <label>
                  Thuốc / Liều
                  <input name="medication_name" placeholder="Tên thuốc" />
                </label>
                <label>
                  Tần suất
                  <input name="frequency" placeholder="VD: Hàng ngày" />
                </label>
                <label>
                  Liều lượng
                  <input name="dosage" placeholder="VD: 1 viên/lần" />
                </label>
              </div>
              <label>
                Ghi chú
                <textarea name="notes" rows={3} placeholder="Ghi chú cho bệnh nhân" />
              </label>
              <div className="form-actions">
                <button className="primary-button" type="submit" disabled={formState === 'loading'}>
                  <CalendarDays size={18} />
                  Thêm lịch
                </button>
                {formMessage ? <span className={formState === 'error' ? 'form-status error' : 'form-status ok'}>{formMessage}</span> : null}
              </div>
            </form>
          ) : null}

          {activeView === 'research' && canCreateResearch(session.user.role) ? (
            <form className="data-form" onSubmit={handleCreateResearchRecord}>
              <div className="form-grid">
                <label>
                  Bệnh nhân
                  <select name="patient_username" required defaultValue="">
                    <option value="" disabled>
                      Chọn bệnh nhân
                    </option>
                    {patients.map((patient, index) => (
                      <option key={recordKey('research-patient', patient as RecordLike, index)} value={patient.username}>
                        {patientLabel(patient as RecordLike)}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  Mã đối tượng
                  <input name="subject_code" placeholder="VD: BN001" />
                </label>
                <label>
                  Tuổi
                  <input name="age" type="number" min="0" max="120" defaultValue="40" />
                </label>
                <label>
                  Giới tính
                  <select name="gender" defaultValue="Nam">
                    <option value="Nam">Nam</option>
                    <option value="Nữ">Nữ</option>
                  </select>
                </label>
                <label>
                  Chẩn đoán
                  <input name="diagnosis" placeholder="VD: M75.0" />
                </label>
                <label>
                  Bài tập
                  <input name="exercise" placeholder="VD: Codman" />
                </label>
                <label>
                  Kết quả
                  <select name="general_result" defaultValue="Gần đúng">
                    <option value="Đúng">Đúng</option>
                    <option value="Gần đúng">Gần đúng</option>
                    <option value="Sai">Sai</option>
                  </select>
                </label>
                <label>
                  Chỉ định
                  <select name="plan" defaultValue="Tiếp tục">
                    <option value="Tiếp tục">Tiếp tục</option>
                    <option value="Chuyển bài">Chuyển bài</option>
                    <option value="Khám lại">Khám lại</option>
                  </select>
                </label>
                <label>
                  Thiết bị quay
                  <select name="recording_device" defaultValue="Điện thoại">
                    <option value="Điện thoại">Điện thoại</option>
                    <option value="Webcam">Webcam</option>
                    <option value="Khác">Khác</option>
                  </select>
                </label>
                <label>
                  Góc quay
                  <select name="recording_angle" defaultValue="Chính diện">
                    <option value="Chính diện">Chính diện</option>
                    <option value="Bên trái">Bên trái</option>
                    <option value="Bên phải">Bên phải</option>
                  </select>
                </label>
              </div>
              <label>
                Nhận xét chuyên môn
                <textarea name="specialist_comment" rows={3} placeholder="Nhận xét phục vụ nghiên cứu" />
              </label>
              <div className="form-actions">
                <button className="primary-button" type="submit" disabled={formState === 'loading'}>
                  <FlaskConical size={18} />
                  Lưu phiếu
                </button>
                {formMessage ? <span className={formState === 'error' ? 'form-status error' : 'form-status ok'}>{formMessage}</span> : null}
              </div>
            </form>
          ) : null}

          {activeView === 'research' && canManageHfSync(session.user.role) ? (
            <section className="analysis-config-panel hf-sync-panel" aria-label="Hugging Face sync">
              <div className="analysis-config-title">
                <FlaskConical size={18} />
                <div>
                  <h3>Hugging Face sync</h3>
                  <span>
                    {hfStatus?.configured ? textValue(hfStatus.dataset_id) : 'Chưa cấu hình Dataset'} · token{' '}
                    {hfStatus?.token_configured ? 'đã cấu hình' : 'chưa có'} · {hfStatus?.verify_ok ? 'kết nối OK' : 'dry-run trước'}
                  </span>
                </div>
              </div>
              <div className="pose-status-grid">
                <div>
                  <span>Dataset</span>
                  <strong>{textValue(hfStatus?.dataset_id)}</strong>
                </div>
                <div>
                  <span>Token fingerprint</span>
                  <strong>{textValue(hfStatus?.token_fingerprint)}</strong>
                </div>
                <div>
                  <span>Job gần nhất</span>
                  <strong>
                    {hfLatestJob
                      ? `${hfActionLabel(hfLatestJob.action)} · ${analysisStatusLabel(hfLatestJob.status)} · ${Math.round(
                          Math.max(0, Math.min(1, hfLatestJob.progress || 0)) * 100,
                        )}%`
                      : 'Chưa có'}
                  </strong>
                </div>
              </div>
              <div className="analysis-config-grid hf-config-grid">
                <label>
                  Video artifact
                  <select value={hfSelectedVideoKey} onChange={(event) => setHfSelectedVideoKey(event.target.value)}>
                    <option value="">Chọn video</option>
                    {filteredVideos.map((video, index) => (
                      <option value={videoKey(video, index)} key={videoKey(video, index)}>
                        {textValue(video.video_name, video.original_filename)} · {patientLabel(video as RecordLike)}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  Artifact
                  <select value={hfArtifactKind} onChange={(event) => setHfArtifactKind(event.target.value)}>
                    <option value="angle-csv">CSV góc</option>
                    <option value="frames-json">JSON frame</option>
                    <option value="frames-zip">ZIP frame</option>
                    <option value="processed-video">Video xử lý</option>
                  </select>
                </label>
                <label className="checkbox-row">
                  <input checked={hfDryRun} onChange={(event) => setHfDryRun(event.target.checked)} type="checkbox" />
                  Dry-run
                </label>
              </div>
              <div className="hf-file-grid" aria-label="File metadata cần sync">
                {(hfStatus?.allowed_sync_files || ['video_list.json', 'doctor_evaluations.json', 'research_data.json']).map((file) => (
                  <label className="checkbox-row hf-file-toggle" key={file}>
                    <input checked={hfSelectedFiles.includes(file)} onChange={() => toggleHfSyncFile(file)} type="checkbox" />
                    {file}
                  </label>
                ))}
              </div>
              <div className="pose-actions">
                <button className="table-action" disabled={hfState === 'loading' && hfTargetKey === 'sync'} onClick={() => void handleStartHfMetadataSync()} type="button">
                  {hfState === 'loading' && hfTargetKey === 'sync' ? <RefreshCw className="spin" size={16} /> : <RefreshCw size={16} />}
                  Sync metadata
                </button>
                <button className="table-action muted-action" disabled={!hfSelectedVideoKey || (hfState === 'loading' && hfTargetKey === 'upload')} onClick={() => void handleUploadHfArtifact()} type="button">
                  {hfState === 'loading' && hfTargetKey === 'upload' ? <RefreshCw className="spin" size={16} /> : <ArrowRight size={16} />}
                  Upload artifact
                </button>
                <button className="table-action muted-action" disabled={hfState === 'loading' && hfTargetKey === 'report'} onClick={() => void handleCreateHfReport()} type="button">
                  {hfState === 'loading' && hfTargetKey === 'report' ? <RefreshCw className="spin" size={16} /> : <ClipboardList size={16} />}
                  Report
                </button>
                <button className="table-action muted-action" disabled={hfState === 'loading' && hfTargetKey === 'refresh'} onClick={() => void handleRefreshHfSync()} type="button">
                  <RefreshCw className={hfState === 'loading' && hfTargetKey === 'refresh' ? 'spin' : ''} size={16} />
                  Kiểm tra
                </button>
              </div>
              {hfLatestJob ? (
                <div className="progress-track" aria-label="Tiến độ HF sync">
                  <span
                    className={`progress-fill ${statusClass(hfLatestJob.status)}`}
                    style={{ width: `${Math.round(Math.max(0, Math.min(1, hfLatestJob.progress || 0)) * 100)}%` }}
                  />
                </div>
              ) : null}
              {hfMessage ? <div className={hfState === 'error' ? 'alert inline' : 'status-line inline ok'}>{hfMessage}</div> : null}
              {hfStatus?.message ? <p className="muted">{hfStatus.message}</p> : null}
              {hfLatestJob?.status_msg || hfLatestJob?.error_msg ? (
                <p className={hfLatestJob.status === 'error' ? 'danger-text' : 'muted'}>
                  {textValue(hfLatestJob.status_msg, hfLatestJob.error_msg)}
                </p>
              ) : null}
              {hfHistory.length ? (
                <div className="analysis-history pose-history">
                  {hfHistory.slice(-3).map((job) => (
                    <div key={job.job_id}>
                      <span className={`pill ${statusClass(job.status)}`}>{analysisStatusLabel(job.status)}</span>
                      <strong>
                        {hfActionLabel(job.action)}
                        {job.dry_run ? ' · dry-run' : ''}
                      </strong>
                      <small>{hfJobTimeLabel(job)}</small>
                    </div>
                  ))}
                </div>
              ) : null}
            </section>
          ) : null}

          {activeView === 'users' && canManageUsers(session.user.role) ? (
            <form className="data-form" onSubmit={handleCreateUser}>
              <div className="form-grid">
                <label>
                  Tên đăng nhập
                  <input name="username" minLength={3} required />
                </label>
                <label>
                  Họ tên
                  <input name="full_name" required />
                </label>
                <label>
                  Email
                  <input name="email" type="email" />
                </label>
                <label>
                  Mật khẩu tạm
                  <input name="password" type="password" minLength={6} required />
                </label>
                <label>
                  Vai trò
                  <select name="role" defaultValue={DOCTOR_ROLE}>
                    <option value={DOCTOR_ROLE}>Bác sĩ / KTV PHCN</option>
                    <option value={RESEARCHER_ROLE}>Nghiên cứu viên</option>
                    <option value={ADMIN_ROLE}>Quản trị viên</option>
                    <option value={PATIENT_ROLE}>Bệnh nhân</option>
                  </select>
                </label>
                <label>
                  BN phụ trách
                  <input name="assigned_patient_usernames" placeholder="patient01, patient02" />
                </label>
              </div>
              <div className="form-actions">
                <button className="primary-button" type="submit" disabled={formState === 'loading'}>
                  <UserPlus size={18} />
                  Tạo tài khoản
                </button>
                {formMessage ? <span className={formState === 'error' ? 'form-status error' : 'form-status ok'}>{formMessage}</span> : null}
              </div>
            </form>
          ) : null}

          {activeView === 'users' && canManageUsers(session.user.role) ? (
            <div className="admin-ops">
              <div>
                <strong>Vận hành tài khoản</strong>
                <span>{auditLog.length} sự kiện audit gần nhất</span>
              </div>
              <div className="row-actions">
                <button
                  className="table-action muted-action"
                  onClick={() => void loadDashboard(session)}
                  disabled={loadState === 'loading' || formState === 'loading'}
                  type="button"
                  title="Tải lại người dùng và audit log"
                >
                  <RefreshCw className={loadState === 'loading' ? 'spin' : ''} size={16} />
                  Làm mới
                </button>
                <button
                  className="table-action danger-action"
                  onClick={() => void handleRevokeAllSessions()}
                  disabled={formState === 'loading'}
                  type="button"
                  title="Thu hồi toàn bộ phiên đăng nhập"
                >
                  {adminActionKey === 'revoke:all' ? <RefreshCw className="spin" size={16} /> : <Shield size={16} />}
                  Revoke all
                </button>
              </div>
            </div>
          ) : null}

          {activeView === 'users' && canManageUsers(session.user.role) ? (
            <div className="cleanup-panel">
              <div className="subpanel-header">
                <div>
                  <h3>Dọn dữ liệu</h3>
                  <span>Reset từng nhóm có backup và audit</span>
                </div>
                <Trash2 size={18} />
              </div>
              <div className="cleanup-grid">
                {cleanupTargets.map((target) => {
                  const empty = target.record_count === 0 && target.file_count === 0;
                  const busy = formState === 'loading' || adminActionKey === `cleanup:${target.target}`;
                  return (
                    <div className="cleanup-item" key={target.target}>
                      <div>
                        <strong>{target.label}</strong>
                        <span>{cleanupCountLabel(target)}</span>
                        <small className="mono">{target.confirm}</small>
                      </div>
                      <button
                        className="table-action danger-action"
                        disabled={empty || busy}
                        onClick={() => void handleResetCleanupTarget(target)}
                        title={empty ? 'Không có dữ liệu để reset' : `Nhập ${target.confirm} để reset`}
                        type="button"
                      >
                        {adminActionKey === `cleanup:${target.target}` ? <RefreshCw className="spin" size={16} /> : <Trash2 size={16} />}
                        Reset
                      </button>
                    </div>
                  );
                })}
                {!cleanupTargets.length ? <div className="empty-gallery">Chưa tải được trạng thái cleanup.</div> : null}
              </div>
            </div>
          ) : null}

          {activeView === 'videos' && videoPreview ? (
            <div className="video-preview">
              <div className="preview-header">
                <span className="video-name">
                  <FileVideo size={16} />
                  {videoPreview.label}
                </span>
                <button className="table-action" onClick={clearVideoPreview} type="button" title="Ẩn video">
                  <EyeOff size={16} />
                  Ẩn
                </button>
              </div>
              <video src={videoPreview.url} controls preload="metadata" />
            </div>
          ) : null}

          {activeView === 'videos' && selectedResult ? (
            <div className="result-panel">
              <div className="subpanel-header">
                <div>
                  <h3>Kết quả chi tiết</h3>
                  <span>
                    {selectedResult.video.video_name} - {textValue(selectedResult.video.exercise)}
                  </span>
                </div>
                <button className="table-action muted-action" onClick={clearSelectedResult} type="button">
                  <EyeOff size={16} />
                  Ẩn
                </button>
              </div>

              <div className="result-summary-grid">
                <div className="result-summary primary">
                  <span>{session.user.role === PATIENT_ROLE ? 'Dành cho bệnh nhân' : 'Tóm tắt'}</span>
                  <strong>{patientResultMessage(selectedResult)}</strong>
                  {selectedResult.summary.doctor_comment ? <p>{selectedResult.summary.doctor_comment}</p> : null}
                </div>
                <div className="result-summary">
                  <span>AI</span>
                  <strong>{selectedResult.ai_detail_allowed ? accuracyLabel(selectedResult.summary.accuracy) : 'Đang chờ'}</strong>
                  <p>{reportStatusLabel(selectedResult)}</p>
                </div>
                <div className="result-summary">
                  <span>Bác sĩ</span>
                  <strong>{textValue(selectedResult.summary.doctor_result, selectedResult.evaluation?.doctor_result)}</strong>
                  <p>{textValue(selectedResult.summary.doctor_plan, selectedResult.evaluation?.plan)}</p>
                </div>
              </div>

              <div className="result-sections">
                <section className="result-section">
                  <div className="section-title">
                    <ClipboardList size={16} />
                    <h4>Nhận xét lâm sàng</h4>
                  </div>
                  {selectedResult.evaluation ? (
                    <div className="clinical-notes">
                      <div>
                        <span>Kết quả</span>
                        <strong>{textValue(selectedResult.evaluation.doctor_result)}</strong>
                      </div>
                      <div>
                        <span>Kế hoạch</span>
                        <strong>{textValue(selectedResult.evaluation.plan)}</strong>
                      </div>
                      <div>
                        <span>Lỗi cần chú ý</span>
                        <strong>{selectedResult.evaluation.errors?.length ? selectedResult.evaluation.errors.join(', ') : 'Chưa ghi nhận'}</strong>
                      </div>
                      <p>{textValue(selectedResult.evaluation.comments)}</p>
                      {canViewClinicalResult(session.user.role) && selectedResult.evaluation.comments_ncv ? (
                        <p className="clinical-private">{selectedResult.evaluation.comments_ncv}</p>
                      ) : null}
                    </div>
                  ) : (
                    <p className="muted">Chưa có nhận xét bác sĩ cho video này.</p>
                  )}
                </section>

                <section className="result-section">
                  <div className="section-title">
                    <Cpu size={16} />
                    <h4>Chỉ số AI</h4>
                  </div>
                  {!selectedResult.ai_detail_allowed ? (
                    <div className="empty-gallery">{selectedResult.report_status.message || 'NCV chưa gửi báo cáo AI chính thức.'}</div>
                  ) : (
                    <>
                      <div className="artifact-metrics">
                        {Object.entries(selectedResult.metrics).slice(0, 8).map(([key, value]) => (
                          <div className="artifact-metric" key={key}>
                            <span>{metricLabel(key)}</span>
                            <strong>{metricValueLabel(value)}</strong>
                          </div>
                        ))}
                        {!Object.keys(selectedResult.metrics).length ? (
                          <div className="artifact-metric">
                            <span>Accuracy</span>
                            <strong>{accuracyLabel(selectedResult.video.accuracy)}</strong>
                          </div>
                        ) : null}
                      </div>
                      <div className="phase-metric-grid">
                        {(['G1', 'G2', 'G3'] as const).map((phase) => (
                          <div className="phase-metric-card" key={phase}>
                            <span>{phase} ±{selectedResult.phase_metrics?.[phase]?.threshold ?? defaultPhaseThreshold(phase)}°</span>
                            <strong>{metricPercentLabel(phaseMetricValue(selectedResult.phase_metrics, phase, 'accuracy'))}</strong>
                            <small>
                              MAE {chartNumberLabel(phaseMetricValue(selectedResult.phase_metrics, phase, 'mae'), '°')} · F1{' '}
                              {chartNumberLabel(phaseMetricValue(selectedResult.phase_metrics, phase, 'f1'))} · ICC{' '}
                              {chartNumberLabel(phaseMetricValue(selectedResult.phase_metrics, phase, 'icc'))}
                            </small>
                          </div>
                        ))}
                      </div>
                      {selectedResult.latest_job?.status_msg || selectedResult.latest_job?.error_msg ? (
                        <p className={selectedResult.latest_job.status === 'error' ? 'danger-text' : 'muted'}>
                          {textValue(selectedResult.latest_job.status_msg, selectedResult.latest_job.error_msg)}
                        </p>
                      ) : null}
                    </>
                  )}
                </section>
              </div>

              {selectedResult.ai_detail_allowed ? (
              <section className="chart-panel">
                <div className="subpanel-header">
                  <div>
                    <h3>Biểu đồ góc khớp</h3>
                    <span>
                      {analysisChart
                        ? `${analysisChart.source_label} · ${analysisChart.sampled_rows}/${analysisChart.filtered_rows} điểm (${phaseLabel(analysisChart.filter)})`
                        : 'Preview từ CSV hoặc JSON frame'}
                    </span>
                  </div>
                  <LineChart size={18} />
                </div>
                {chartState === 'loading' ? (
                  <div className="status-line">
                    <RefreshCw className="spin" size={16} />
                    Đang tải dữ liệu biểu đồ...
                  </div>
                ) : null}
                {chartState === 'error' ? <div className="alert inline">{chartMessage}</div> : null}
                {analysisChart ? (
                  <>
                    <div className="chart-summary">
                      <div>
                        <span>Góc vai TB</span>
                        <strong>{chartNumberLabel(analysisChart.summary.series.goc_vai?.avg, '°')}</strong>
                      </div>
                      <div>
                        <span>Góc khuỷu TB</span>
                        <strong>{chartNumberLabel(analysisChart.summary.series.goc_khuyu?.avg, '°')}</strong>
                      </div>
                      <div>
                        <span>PASS</span>
                        <strong>{analysisChart.summary.labels.PASS}</strong>
                      </div>
                      <div>
                        <span>NEAR/FAIL</span>
                        <strong>
                          {analysisChart.summary.labels.NEAR}/{analysisChart.summary.labels.FAIL}
                        </strong>
                      </div>
                    </div>
                    {analysisChart.phase_summary?.phases ? (
                      <div className="phase-summary-grid">
                        {(['G1', 'G2', 'G3'] as const).map((phase) => {
                          const phaseSummary = analysisChart.phase_summary?.phases[phase];
                          return (
                            <div key={phase}>
                              <span>{phase} ±{phaseSummary?.threshold ?? 'N/A'}°</span>
                              <strong>
                                {phaseSummary?.PASS ?? 0}/{phaseSummary?.total ?? 0} PASS
                              </strong>
                            </div>
                          );
                        })}
                      </div>
                    ) : null}
                    <AngleChart chart={analysisChart} />
                  </>
                ) : chartState !== 'loading' && chartState !== 'error' ? (
                  <div className="empty-gallery">Chưa có CSV/JSON đủ dữ liệu để preview biểu đồ.</div>
                ) : null}
              </section>
              ) : null}

              <div className="timeline-list">
                {selectedResult.timeline.map((item, index) => (
                  <div className="timeline-item" key={`${item.kind}-${index}`}>
                    <span className={`pill ${statusClass(item.status)}`}>{textValue(item.status, item.label)}</span>
                    <div>
                      <strong>{item.label}</strong>
                      <p>{textValue(item.detail)}</p>
                    </div>
                    <time>{textValue(item.time)}</time>
                  </div>
                ))}
              </div>

              {selectedResult.ai_detail_allowed ? (
              <section className="frame-gallery">
                <div className="subpanel-header">
                  <div>
                    <h3>Frame PASS / NEAR / FAIL</h3>
                    <span>
                      {analysisFrames
                        ? `${analysisFrames.pagination.total} frame trong bộ lọc`
                        : 'Dữ liệu frame tải theo trang'}
                    </span>
                  </div>
                  <div className="segmented-control frame-filter" aria-label="Lọc frame">
                    {(['ALL', 'G1', 'G2', 'G3', 'PASS', 'NEAR', 'FAIL'] as FrameLabel[]).map((label) => (
                      <button
                        className={framesLabel === label ? 'active' : ''}
                        key={label}
                        onClick={() => handleFrameLabelChange(label)}
                        type="button"
                      >
                        {phaseLabel(label)}
                      </button>
                    ))}
                  </div>
                </div>

                {analysisFrames ? (
                  <div className="frame-summary">
                    <div>
                      <span>Tổng</span>
                      <strong>{analysisFrames.summary.total}</strong>
                    </div>
                    <div>
                      <span>PASS</span>
                      <strong>{analysisFrames.summary.PASS}</strong>
                    </div>
                    <div>
                      <span>NEAR</span>
                      <strong>{analysisFrames.summary.NEAR}</strong>
                    </div>
                    <div>
                      <span>FAIL</span>
                      <strong>{analysisFrames.summary.FAIL}</strong>
                    </div>
                  </div>
                ) : null}
                {analysisFrames?.summary.phases ? (
                  <div className="phase-summary-grid">
                    {(['G1', 'G2', 'G3'] as const).map((phase) => {
                      const phaseSummary = analysisFrames.summary.phases?.[phase];
                      return (
                        <div key={phase}>
                          <span>{phase} ±{phaseSummary?.threshold ?? 'N/A'}°</span>
                          <strong>
                            {phaseSummary?.PASS ?? 0}/{phaseSummary?.total ?? 0} PASS
                          </strong>
                        </div>
                      );
                    })}
                  </div>
                ) : null}

                {framesState === 'loading' ? (
                  <div className="status-line">
                    <RefreshCw className="spin" size={16} />
                    Đang tải frame...
                  </div>
                ) : null}
                {framesState === 'error' ? <div className="alert inline">{framesMessage}</div> : null}

                {analysisFrames && !analysisFrames.items.length && framesState !== 'loading' ? (
                  <div className="empty-gallery">Chưa có frame phù hợp hoặc artifact frame chưa sẵn sàng.</div>
                ) : null}

                {analysisFrames?.items.length ? (
                  <>
                    <div className="frame-grid">
                      {analysisFrames.items.map((frame) => (
                        <article className="frame-card" key={`${frame.image_id || 'frame'}-${frame.index}`}>
                          <div className="frame-card-top">
                            <span>#{frame.index}</span>
                            <span className="frame-card-badges">
                              <span className={`pill ${statusClass(frame.label)}`}>{frame.phase ? `${frame.phase} · ${frame.label}` : frame.label}</span>
                              {frame.ml ? (
                                <span className={`pill ${mlBadgeClass(frame.ml.label_text || frame.ml.label)}`}>
                                  ML · {textValue(frame.ml.label_text, frame.ml.label)}
                                  {frame.ml.confidence !== undefined && frame.ml.confidence !== null ? ` ${mlConfidenceLabel(frame.ml.confidence)}` : ''}
                                </span>
                              ) : null}
                            </span>
                          </div>
                          <div className="frame-image-wrap">
                            {frame.image_id && frameImageUrls[frame.image_id] ? (
                              <button
                                aria-label={`Xem lớn frame ${frame.index}`}
                                className="frame-image-button"
                                onClick={() => openFocusedFrame(frame)}
                                title="Xem frame lớn"
                                type="button"
                              >
                                <img alt={`Frame ${frame.index} ${frame.label}`} src={frameImageUrls[frame.image_id]} />
                                <span>
                                  <Maximize2 size={16} />
                                </span>
                              </button>
                            ) : (
                              <span>Không có ảnh</span>
                            )}
                          </div>
                          <div className="frame-meta">
                            <span>
                              {textValue(frame.timestamp)} · REF ±{textValue(frame.phase_threshold)}°
                            </span>
                            <strong>
                              Vai {textValue(frame.goc_vai, frame.goc_vai_trai, frame.goc_vai_phai)} / Khuỷu{' '}
                              {textValue(frame.goc_khuyu, frame.goc_khuyu_trai, frame.goc_khuyu_phai)}
                            </strong>
                            <span>
                              Δ vai {textValue(frame.shoulder_delta)}° / Δ khuỷu {textValue(frame.elbow_delta)}°
                            </span>
                          </div>
                        </article>
                      ))}
                    </div>
                    <div className="pager">
                      <button
                        className="table-action muted-action"
                        disabled={framesState === 'loading' || analysisFrames.pagination.page <= 1}
                        onClick={() => handleFramePageChange(framesPage - 1)}
                        type="button"
                      >
                        Trước
                      </button>
                      <span>
                        Trang {analysisFrames.pagination.page}/{analysisFrames.pagination.total_pages}
                      </span>
                      <button
                        className="table-action muted-action"
                        disabled={framesState === 'loading' || analysisFrames.pagination.page >= analysisFrames.pagination.total_pages}
                        onClick={() => handleFramePageChange(framesPage + 1)}
                        type="button"
                      >
                        Sau
                      </button>
                    </div>
                  </>
                ) : null}
              </section>
              ) : null}

              {selectedResult.ai_detail_allowed ? (
              <div className="artifact-grid">
                {selectedResult.artifacts.map((item) => (
                  <div className={`artifact-item ${item.available ? '' : 'disabled'}`} key={item.kind}>
                    <div>
                      <strong>{item.label}</strong>
                      <span>{item.filename || 'Chưa có file'} · {item.available ? fileSizeLabel(item.size) : 'Chưa sẵn sàng'}</span>
                    </div>
                    <button
                      className="table-action"
                      disabled={!item.available || artifactState === 'loading'}
                      onClick={() => void handleDownloadArtifact(item.kind, item.filename)}
                      type="button"
                    >
                      {artifactState === 'loading' ? <RefreshCw className="spin" size={16} /> : <Download size={16} />}
                      Tải
                    </button>
                  </div>
                ))}
              </div>
              ) : null}

              {focusedFrame ? (
                <div className="modal-backdrop" role="presentation" onClick={() => setFocusedFrame(null)}>
                  <div className="frame-modal" role="dialog" aria-modal="true" aria-label={`Frame ${focusedFrame.frame.index}`} onClick={(event) => event.stopPropagation()}>
                    <div className="preview-header">
                      <span className="video-name">
                        <FileVideo size={16} />
                        Frame #{focusedFrame.frame.index} · {focusedFrame.frame.label}
                      </span>
                      <button className="table-action muted-action" onClick={() => setFocusedFrame(null)} type="button" title="Đóng">
                        <X size={16} />
                        Đóng
                      </button>
                    </div>
                    <img alt={`Frame ${focusedFrame.frame.index} ${focusedFrame.frame.label}`} src={focusedFrame.url} />
                    <div className="frame-modal-meta">
                      <span>{textValue(focusedFrame.frame.timestamp)}</span>
                      <strong>
                        Vai {textValue(focusedFrame.frame.goc_vai, focusedFrame.frame.goc_vai_trai, focusedFrame.frame.goc_vai_phai)} / Khuỷu{' '}
                        {textValue(focusedFrame.frame.goc_khuyu, focusedFrame.frame.goc_khuyu_trai, focusedFrame.frame.goc_khuyu_phai)}
                      </strong>
                    </div>
                  </div>
                </div>
              ) : null}
            </div>
          ) : null}

          {activeView === 'videos' && previewMessage ? <div className="alert inline">{previewMessage}</div> : null}
          {activeView === 'videos' && analysisMessage ? (
            <div className={analysisState === 'error' ? 'alert inline' : 'status-line inline ok'}>{analysisMessage}</div>
          ) : null}
          {activeView === 'videos' && artifactMessage ? (
            <div className={artifactState === 'error' ? 'alert inline' : 'status-line inline ok'}>{artifactMessage}</div>
          ) : null}

          {activeView === 'videos' ? (
            <>
              <DataTable columns={videoColumns} items={filteredVideos} emptyText="Chưa có video phù hợp." rowKey={videoKey} />
              {canManageClinical(session.user.role) ? (
                <div className="subpanel">
                  <div className="subpanel-header">
                    <h3>Đánh giá lâm sàng</h3>
                    <span>{evaluations.length} bản ghi</span>
                  </div>
                  <DataTable
                    columns={evaluationColumns}
                    items={evaluations.filter((evaluation) => matchesQuery(evaluation as RecordLike, query))}
                    emptyText="Chưa có đánh giá phù hợp."
                    rowKey={(evaluation, index) => recordKey('evaluation', evaluation as RecordLike, index)}
                  />
                </div>
              ) : null}
            </>
          ) : null}
          {activeView === 'patients' ? (
            <DataTable
              columns={patientColumns}
              items={filteredPatients}
              emptyText="Chưa có hồ sơ bệnh nhân phù hợp."
              rowKey={(patient, index) => recordKey('patient', patient as RecordLike, index)}
            />
          ) : null}
          {activeView === 'symptoms' ? (
            <DataTable
              columns={symptomColumns}
              items={filteredSymptoms}
              emptyText="Chưa có khai báo triệu chứng phù hợp."
              rowKey={(symptom, index) => recordKey('symptom', symptom, index)}
            />
          ) : null}
          {activeView === 'schedules' ? (
            <DataTable
              columns={scheduleColumns}
              items={filteredSchedules}
              emptyText="Chưa có lịch nhắc phù hợp."
              rowKey={(schedule, index) => recordKey('schedule', schedule, index)}
            />
          ) : null}
          {activeView === 'research' ? (
            <DataTable
              columns={researchColumns}
              items={filteredResearch}
              emptyText="Chưa có dữ liệu nghiên cứu phù hợp."
              rowKey={(record, index) => recordKey('research', record, index)}
            />
          ) : null}
          {activeView === 'info' ? (
            <section className="info-workspace" aria-label="Thông tin và hướng dẫn">
              <div className="info-band">
                <div>
                  <p className="eyebrow">Hướng dẫn theo vai trò</p>
                  <h3>{viewLabelForRole('info', session.user.role)}</h3>
                </div>
                <span>{session.user.role}</span>
              </div>
              <div className="info-step-grid">
                {roleGuideSteps(session.user.role).map(([title, body], index) => (
                  <article className="info-step" key={title}>
                    <span>{String(index + 1).padStart(2, '0')}</span>
                    <strong>{title}</strong>
                    <p>{body}</p>
                  </article>
                ))}
              </div>
              <div className="info-section-grid">
                <section className="info-section">
                  <h3>Kiến thức PHCN</h3>
                  {rehabKnowledge.map(([title, body]) => (
                    <div className="info-row-card" key={title}>
                      <strong>{title}</strong>
                      <p>{body}</p>
                    </div>
                  ))}
                </section>
                <section className="info-section">
                  <h3>Công nghệ AI</h3>
                  {aiKnowledge.map(([title, body]) => (
                    <div className="info-row-card" key={title}>
                      <strong>{title}</strong>
                      <p>{body}</p>
                    </div>
                  ))}
                </section>
              </div>
              <section className="info-section">
                <div className="subpanel-header">
                  <div>
                    <h3>Thông tin đề tài</h3>
                    <span>Nội dung nghiên cứu và nguyên tắc bảo mật</span>
                  </div>
                  <FlaskConical size={18} />
                </div>
                <div className="info-card-grid">
                  {researchInfoSections.map(([title, body]) => (
                    <article className="info-card" key={title}>
                      <strong>{title}</strong>
                      <p>{body}</p>
                    </article>
                  ))}
                </div>
              </section>
              <section className="info-section">
                <div className="subpanel-header">
                  <div>
                    <h3>Đội ngũ và hỗ trợ</h3>
                    <span>Thông tin chung, không hiển thị PII hoặc tài khoản mặc định</span>
                  </div>
                  <UsersRound size={18} />
                </div>
                <div className="info-card-grid compact">
                  {teamInfoSections.map(([title, body]) => (
                    <article className="info-card" key={title}>
                      <strong>{title}</strong>
                      <p>{body}</p>
                    </article>
                  ))}
                </div>
              </section>
              <form className="data-form feedback-form" onSubmit={handleCreateFeedback}>
                <div className="subpanel-header">
                  <div>
                    <h3>Gửi phản hồi</h3>
                    <span>Không gửi mật khẩu, token hoặc thông tin định danh nhạy cảm.</span>
                  </div>
                  <MessageSquare size={18} />
                </div>
                <div className="form-grid two feedback-grid">
                  <label>
                    Nhóm phản hồi
                    <select name="category" defaultValue="general">
                      <option value="general">Góp ý chung</option>
                      <option value="bug">Lỗi hệ thống</option>
                      <option value="workflow">Quy trình</option>
                      <option value="content">Nội dung</option>
                      <option value="safety">An toàn dữ liệu</option>
                    </select>
                  </label>
                  <label className="checkbox-row">
                    <input name="contact_ok" type="checkbox" />
                    Có thể liên hệ lại trong hệ thống
                  </label>
                </div>
                <label>
                  Nội dung
                  <textarea name="message" maxLength={2000} placeholder="Mô tả góp ý hoặc vấn đề cần hỗ trợ..." required />
                </label>
                <div className="form-actions">
                  <button className="primary-button" type="submit" disabled={formState === 'loading'}>
                    {formState === 'loading' ? <RefreshCw className="spin" size={18} /> : <MessageSquare size={18} />}
                    Gửi phản hồi
                  </button>
                  {formMessage ? (
                    <span className={formState === 'error' ? 'form-status error' : 'form-status ok'}>{formMessage}</span>
                  ) : null}
                </div>
              </form>
              {canReviewFeedback(session.user.role) ? (
                <section className="subpanel">
                  <div className="subpanel-header">
                    <div>
                      <h3>Phản hồi gần đây</h3>
                      <span>{feedbackRecords.length} phản hồi mới nhất</span>
                    </div>
                    <History size={18} />
                  </div>
                  <DataTable
                    columns={feedbackColumns}
                    items={feedbackRecords.filter((item) => matchesQuery(item as RecordLike, query))}
                    emptyText="Chưa có phản hồi nào."
                    rowKey={(item, index) => recordKey('feedback', item as RecordLike, index)}
                  />
                </section>
              ) : null}
            </section>
          ) : null}
          {activeView === 'users' ? (
            <>
              <DataTable
                columns={userColumns}
                items={filteredUsers}
                emptyText="Chưa có tài khoản phù hợp."
                rowKey={(user, index) => recordKey('user', user as RecordLike, index)}
              />
              {canManageUsers(session.user.role) ? (
                <div className="subpanel">
                  <div className="subpanel-header">
                    <div>
                      <h3>Audit log</h3>
                      <span>{auditLog.length} sự kiện gần nhất</span>
                    </div>
                    <History size={18} />
                  </div>
                  <DataTable
                    columns={auditColumns}
                    items={auditLog.filter((item) => matchesQuery(item as RecordLike, query))}
                    emptyText="Chưa có audit log phù hợp."
                    rowKey={(item, index) => recordKey('audit', item as RecordLike, index)}
                  />
                </div>
              ) : null}
            </>
          ) : null}
        </section>
      </section>
    </main>
  );
}
