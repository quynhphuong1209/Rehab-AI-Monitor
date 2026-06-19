import { expect, test } from '@playwright/test';
import { execFile, spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import { mkdirSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const webRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const repoRoot = path.resolve(webRoot, '..');
const scratchRoot = path.join(repoRoot, 'scratch', 'web-e2e-smoke');
const backendPort = 8010;
const frontendPort = 5183;
const backendUrl = `http://127.0.0.1:${backendPort}`;
const frontendUrl = `http://127.0.0.1:${frontendPort}`;

let backendProcess: ChildProcessWithoutNullStreams | undefined;
let frontendProcess: ChildProcessWithoutNullStreams | undefined;

function writeJson(filePath: string, data: unknown) {
  mkdirSync(path.dirname(filePath), { recursive: true });
  writeFileSync(filePath, JSON.stringify(data, null, 2), 'utf8');
}

function spawnServer(command: string, args: string[], options: { cwd: string; env?: NodeJS.ProcessEnv }) {
  const child = spawn(command, args, {
    cwd: options.cwd,
    env: { ...process.env, ...options.env },
    shell: process.platform === 'win32',
  });
  child.stdout.on('data', (data) => process.stdout.write(`[${path.basename(command)}] ${data}`));
  child.stderr.on('data', (data) => process.stderr.write(`[${path.basename(command)}] ${data}`));
  return child;
}

async function waitForOk(url: string, timeoutMs = 20_000) {
  const startedAt = Date.now();
  let lastError = '';
  while (Date.now() - startedAt < timeoutMs) {
    try {
      const response = await fetch(url);
      if (response.ok) {
        return;
      }
      lastError = `${response.status} ${response.statusText}`;
    } catch (error) {
      lastError = error instanceof Error ? error.message : String(error);
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  throw new Error(`Timed out waiting for ${url}: ${lastError}`);
}

async function stopServer(child: ChildProcessWithoutNullStreams | undefined) {
  if (!child || child.killed) {
    return;
  }
  if (process.platform === 'win32') {
    await new Promise<void>((resolve) => {
      execFile('taskkill.exe', ['/PID', String(child.pid), '/T', '/F'], () => resolve());
    });
    return;
  }
  await new Promise<void>((resolve) => {
    child.once('exit', () => resolve());
    child.kill('SIGTERM');
    setTimeout(() => {
      if (!child.killed) {
        child.kill('SIGKILL');
      }
      resolve();
    }, 2_000);
  });
}

test.beforeAll(async () => {
  rmSync(scratchRoot, { recursive: true, force: true });
  const databaseDir = path.join(scratchRoot, 'database');
  const processedDir = path.join(scratchRoot, 'processed_results');
  mkdirSync(databaseDir, { recursive: true });
  mkdirSync(processedDir, { recursive: true });

  writeJson(path.join(databaseDir, 'users.json'), {
    patient01: {
      username: 'patient01',
      full_name: 'Patient One',
      email: 'patient01@example.test',
      role: 'Bệnh nhân',
      active: true,
      password: '29d26d3fb01e2123ef8282260405597c7b4f8c1756984259142472d9519966cf',
      hash_version: 'sha256',
    },
  });
  writeJson(path.join(databaseDir, 'video_list.json'), [
    {
      username: 'patient01',
      full_name: 'Patient One',
      video_name: 'patient01_clip.mp4',
      stored_filename: 'patient01_clip.mp4',
      exercise: 'Codman',
      status: 'Chờ NCV phân tích',
      accuracy: 91.2,
      metrics: {
        do_chinh_xac: 91.2,
        f1_score: 0.88,
        metrics_g1: { do_chinh_xac: 100, mae_tong: 4.5, f1_score: 0.98, icc: 0.9 },
        metrics_g2: { do_chinh_xac: 72, mae_tong: 18, f1_score: 0.78, icc: 0.72 },
        metrics_g3: { do_chinh_xac: 0, mae_tong: 64, f1_score: 0, icc: 0.5 },
      },
      df_path: 'processed_results/patient01_clip_data.csv',
      all_frames_data_path: 'processed_results/patient01_clip_frames.json',
      time: '2026-06-19T07:00:00Z',
    },
  ]);
  writeJson(path.join(databaseDir, 'doctor_evaluations.json'), [
    {
      patient_username: 'patient01',
      video_name: 'patient01_clip.mp4',
      exercise: 'Codman',
      doctor_username: 'AI_Researcher',
      doctor_name: 'NCV: Researcher',
      doctor_result: 'Gần đúng',
      comments: 'Báo cáo AI chính thức',
      plan: 'Tiếp tục',
      time: '2026-06-19T07:04:00Z',
    },
    {
      patient_username: 'patient01',
      video_name: 'patient01_clip.mp4',
      exercise: 'Codman',
      doctor_username: 'doctor',
      doctor_result: 'Gần đúng',
      comments: 'Tiếp tục theo dõi',
      plan: 'Tiếp tục',
      time: '2026-06-19T07:05:00Z',
    },
  ]);
  writeJson(path.join(databaseDir, 'patient_symptoms.json'), [
    {
      username: 'patient01',
      full_name: 'Patient One',
      exercise: 'Codman',
      symptoms: 'Đau nhẹ',
      vas: 3,
      time: '2026-06-19T07:10:00Z',
    },
  ]);
  writeJson(path.join(databaseDir, 'schedules.json'), [
    {
      patient_username: 'patient01',
      title: 'Tập Codman',
      type: 'exercise',
      datetime: '2026-06-20T08:00',
      notes: 'Tập chậm',
    },
  ]);
  writeJson(path.join(databaseDir, 'research_data.json'), []);
  writeFileSync(path.join(processedDir, 'patient01_clip_data.csv'), 'frame,goc_vai\n1,90\n', 'utf8');
  writeJson(path.join(processedDir, 'patient01_clip_frames.json'), [
    {
      index: 1,
      timestamp: '00:01',
      path: 'processed_results/f_000001.jpg',
      goc_vai: 90,
      goc_khuyu: 170,
      dung: true,
      gan_dung: false,
      eval_info: { shoulder_ref: 90, elbow_ref: 170 },
      ml_label: 'dung',
      ml_label_text: 'Đúng',
      ml_confidence: 0.82,
    },
    {
      index: 2,
      timestamp: '00:02',
      path: 'processed_results/f_000002.jpg',
      goc_vai: 125,
      goc_khuyu: 135,
      dung: false,
      gan_dung: true,
      ml_label: 'gan_dung',
      ml_label_text: 'Gần đúng',
      ml_confidence: 0.42,
      ml_probabilities: { Sai: 0.2, 'Gần đúng': 0.42, Đúng: 0.38 },
    },
    {
      index: 3,
      timestamp: '00:03',
      path: 'processed_results/f_000003.jpg',
      goc_vai: 145,
      goc_khuyu: 120,
      dung: false,
      gan_dung: false,
    },
  ]);
  const tinyJpeg = Buffer.from('/9j/4AAQSkZJRgABAQAAAQABAAD/2w==', 'base64');
  writeFileSync(path.join(processedDir, 'f_000001.jpg'), tinyJpeg);
  writeFileSync(path.join(processedDir, 'f_000002.jpg'), tinyJpeg);
  writeFileSync(path.join(processedDir, 'f_000003.jpg'), tinyJpeg);

  backendProcess = spawnServer(
    path.join(repoRoot, '.venv', 'Scripts', 'python.exe'),
    ['-m', 'uvicorn', 'backend.main:app', '--host', '127.0.0.1', '--port', String(backendPort)],
    {
      cwd: repoRoot,
      env: {
        REHAB_REPO_ROOT: scratchRoot,
        REHAB_DATABASE_DIR: databaseDir,
        REHAB_BACKEND_CORS_ORIGINS: frontendUrl,
      },
    },
  );
  await waitForOk(`${backendUrl}/health`);

  frontendProcess = spawnServer('npm.cmd', ['run', 'dev', '--', '--host', '127.0.0.1', '--port', String(frontendPort)], {
    cwd: webRoot,
    env: {
      VITE_REHAB_API_URL: backendUrl,
    },
  });
  await waitForOk(frontendUrl);
});

test.afterAll(async () => {
  await stopServer(frontendProcess);
  await stopServer(backendProcess);
  rmSync(scratchRoot, { recursive: true, force: true });
});

test('patient can log in and see scoped dashboard data', async ({ page }) => {
  await page.goto(frontendUrl);
  const loginForm = page.locator('form').filter({ has: page.getByLabel('Tên đăng nhập') });
  await loginForm.getByLabel('Tên đăng nhập').fill('patient01');
  await loginForm.getByLabel('Mật khẩu').fill('patientpass');
  await loginForm.getByRole('button', { name: 'Đăng nhập' }).click();

  await expect(page.locator('h1', { hasText: 'Không gian tập luyện của bệnh nhân' })).toBeVisible();
  await expect(page.getByText('Patient One')).toBeVisible();
  await expect(page.getByText('Video của tôi').first()).toBeVisible();
  await expect(page.getByRole('button', { name: /Khai báo đau/ }).first()).toBeVisible();

  const navigation = page.getByRole('navigation', { name: 'Khu vực dữ liệu' });
  await navigation.getByRole('button', { name: /Video tập luyện/ }).click();
  await expect(page.getByText('patient01_clip.mp4')).toBeVisible();
  await expect(page.getByText('Codman')).toBeVisible();
  await page.getByRole('button', { name: 'Kết quả' }).click();
  await expect(page.getByRole('heading', { name: 'Kết quả chi tiết' })).toBeVisible();
  await expect(page.getByText('Bác sĩ đánh giá bài tập: Gần đúng')).toBeVisible();
  await expect(page.getByText('Tập Codman')).toBeVisible();
  await expect(page.getByText('F1-score')).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Biểu đồ góc khớp' })).toBeVisible();
  await expect(page.getByText('Góc vai TB')).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Frame PASS / NEAR / FAIL' })).toBeVisible();
  await expect(page.getByText('PASS').first()).toBeVisible();
  await expect(page.getByText(/ML · Đúng/).first()).toBeVisible();
  await page.getByTitle('Xem frame lớn').first().click();
  await expect(page.getByRole('dialog', { name: 'Frame 1' })).toBeVisible();
  await page.getByRole('button', { name: 'Đóng' }).click();
  await expect(page.getByText('NEAR').first()).toBeVisible();
  await expect(page.getByText(/ML · Gần đúng/).first()).toBeVisible();
  await page.getByRole('button', { name: 'G2', exact: true }).click();
  await expect(page.getByText('1 frame trong bộ lọc')).toBeVisible();
  await expect(page.getByText('G2 · NEAR')).toBeVisible();
  await page.getByRole('button', { name: 'G3', exact: true }).click();
  await expect(page.getByText('G3 · FAIL')).toBeVisible();
  await page.getByRole('button', { name: 'PASS', exact: true }).click();
  await expect(page.getByText('1 frame trong bộ lọc')).toBeVisible();

  await navigation.getByRole('button', { name: /Lịch của tôi/ }).click();
  await expect(page.getByText('Tập Codman')).toBeVisible();

  await navigation.getByRole('button', { name: /Thông tin/ }).click();
  await expect(page.getByRole('heading', { name: 'Kiến thức PHCN' })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Công nghệ AI' })).toBeVisible();
  await page.getByLabel('Nhóm phản hồi').selectOption('content');
  await page.getByRole('textbox', { name: 'Nội dung' }).fill('Cần thêm ví dụ về cách đặt camera.');
  await page.getByRole('button', { name: 'Gửi phản hồi' }).click();
  await expect(page.getByText('Đã gửi phản hồi')).toBeVisible();
});
