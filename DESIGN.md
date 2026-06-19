---
name: Antigravity AI Workspace Design System
version: 1.0.0
author: Core Engineering Team
colors:
  primary: "#0F172A"       # Slate 900 (Main text, deep background elements)
  secondary: "#475569"     # Slate 600 (Muted labels, secondary text)
  accent: "#0284C7"        # Sky 600 (Interactive links, active states, key CTAs)
  success: "#059669"       # Emerald 600 (Legal compliance green, successful tool execution)
  warning: "#D97706"       # Amber 600 (Cost estimation threshold warnings, pending events)
  danger: "#DC2626"        # Rose 600 (Error states, failed validation, critical logs)
  neutral-bg: "#F8FAFC"    # Slate 50 (Main interface background)
  card-bg: "#FFFFFF"       # White (Chat bubbles, panel backgrounds)
  terminal-bg: "#0B0F19"   # Ultra-dark slate for raw JSON logs and metadata rendering
typography:
  sans:
    fontFamily: "Inter, system-ui, sans-serif"
    sizes:
      base: "14px"
      md: "16px"
      lg: "20px"
      xl: "24px"
  mono:
    fontFamily: "JetBrains Mono, Fira Code, monospace"
    sizes:
      sm: "12px"
      base: "13px"
spacing:
  xs: "4px"
  sm: "8px"
  md: "16px"
  lg: "24px"
  xl: "32px"
breakpoints:
  tablet: "768px"
  desktop: "1280px"
---

# Quy chuẩn Thiết kế Hệ thống Giao diện Antigravity Workspace

Tài liệu này định nghĩa hệ thống token thị giác và nguyên tắc thiết kế giao diện cho toàn bộ hệ sinh thái Agent của Antigravity, bao gồm bảng điều khiển chat đa tác vụ, các module thu thập dữ liệu (InputCollector), ước tính chi phí (CostEstimator), và cố vấn tuân thủ pháp lý (LegalAdvisor). Các AI Coding Agent phải tuân thủ nghiêm ngặt các quy tắc này khi sinh mã UI.

## 1. Nguyên tắc Tổng quan (Design Philosophy)
- **Tập trung vào dữ liệu (Data-Centric):** Giao diện phải tối giản, sạch sẽ để nhường không gian hiển thị tối đa cho luồng hội thoại và dữ liệu phân tích.
- **Rõ ràng và Minh bạch (Transparency):** Trạng thái hoạt động của các Agent (đang suy nghĩ, đang gọi tool, gặp lỗi) phải được phân biệt trực quan bằng màu sắc và biểu tượng tương ứng.
- **Nhất quan về Cấu trúc:** Toàn bộ spacing phải tuân theo hệ lưới 8px (hoặc bội số của 4px) để đảm bảo tính cân đối trên mọi độ phân giải.

## 2. Hệ thống Màu sắc (Color Tokens Application)
- **Nền chính (Neutral Background - `#F8FAFC`):** Áp dụng cho toàn bộ vùng nền của workspace. Tạo cảm giác thoáng đãng và giảm mỏi mắt khi làm việc lâu.
- **Khung chứa (Card Background - `#FFFFFF`):** Dùng cho các bong bóng chat, các thẻ chứa thông tin của InputCollector và các bảng dữ liệu của CostEstimator. Mỗi thẻ cần có border mờ (`1px solid #E2E8F0`) và border-radius cố định ở mức `8px`.
- **Hộp thoại Logs & Metadata (Terminal Background - `#0B0F19`):** Dành riêng cho khu vực hiển thị log hệ thống, chuỗi JSON metadata (như Session ID, Tool Events, Prompt Hooks). Chữ bên trong bắt buộc dùng màu xanh neon nhạt (`#38BDF8`) hoặc trắng để đạt độ tương phản tối đa.

## 3. Thành phần Giao diện & Trạng thái Agent (Component Specs)

### Hội thoại & Bong bóng Chat (Chat Bubbles)
- **User Prompt:** Nằm bên phải, sử dụng background `#E2E8F0` với chữ `#0F172A`. Không dùng màu accent cho bong bóng chat của user để tránh gây nhiễu thị giác.
- **Agent Response:** Nằm bên trái, sử dụng background `#FFFFFF`. Phần header hiển thị tên Agent (ví dụ: `[LegalAdvisor]`) viết đậm, màu sắc thay đổi tùy theo vai trò hoặc trạng thái tuân thủ.

### Bảng hiển thị Chỉ số & Ước tính (Cost Estimator Metrics)
- Các số liệu tài chính, chi phí ước tính phải được đặt trong các thẻ số lớn (Font size: `24px` hoặc `20px` bold).
- Nếu chi phí vượt ngưỡng ngân sách thiết lập, màu chữ của số liệu phải tự động chuyển sang màu `warning` (`#D97706`) hoặc `danger` (`#DC2626`).

### Khu vực Log Sự kiện Hệ thống (System Event Logger)
- Định dạng hiển thị chuỗi JSON phải luôn được bọc trong block mã nguồn mã hóa theo font `mono` kích thước `13px`.
- Các trường dữ liệu quan trọng như `session_id`, `event_type`, `latency_ms` cần được highlight bằng màu `accent` (`#0284C7`) khi hiển thị trong bảng hoặc giao diện giám sát.

## 4. Quy định về Spacing và Bố cục (Layout Constraints)
- **Khoảng cách giữa các Chat Bubble:** Cố định ở mức `md` (`16px`).
- **Padding bên trong các ô nhập liệu (Input Fields):** `padding: 12px 16px` để tối ưu trải nghiệm nhập liệu cho InputCollector trên cả thiết bị di động và desktop.
- **Khoảng cách an toàn (Margin):** Giữa các phân vùng lớn (ví dụ: Khoảng cách từ Sidebar danh sách Agent đến Khung chat chính) luôn luôn là `lg` (`24px`).
