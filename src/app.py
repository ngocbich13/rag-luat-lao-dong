"""
app.py — GIAI ĐOẠN 7: WEB CHATBOT TRA CỨU BỘ LUẬT LAO ĐỘNG 2019 (FASTAPI + CLEAN LEGAL UI)
==========================================================================================
Ứng dụng Web Chatbot tra cứu Bộ luật Lao động 2019 đã tinh chỉnh theo 3 nhóm yêu cầu:
    - NHÓM 1: Bỏ badge trạng thái cạnh "Trợ lý pháp luật" và bỏ khung cảnh báo màu vàng.
              Khi thiếu văn bản hướng dẫn/Luật BHXH, LLM nêu bằng 1 câu thường ngay trong
              gạch đầu dòng liên quan. Khi ngoài phạm vi, trả lời thẳng:
              "Mình chưa tìm thấy căn cứ trong dữ liệu hiện có."
    - NHÓM 2: Câu trả lời ngắn gọn, đúng trọng tâm:
              + Dòng đầu là Kết luận (tối đa 2 câu, có ngay đáp án/con số, in đậm).
              + Tối đa 3 gạch đầu dòng chi tiết (mỗi gạch < 25 từ), có dòng "Cách tính:"
                khi tính từ công thức trong luật.
              + Chip "Hỏi tiếp" bên dưới cho các câu hỏi mở rộng liên quan.
    - NHÓM 3: Sửa toàn bộ lỗi giao diện:
              a) Khung "KHOẢN X · ĐƯỢC TRÍCH DẪN": width 100%, box-sizing border-box,
                 padding 14px 16px, nhãn margin-bottom 6px, không dùng pre-wrap gây dòng trống.
              b) Chip trích dẫn đặt cuối câu sau dấu chấm, không ngoặc, gộp chip trùng nhau,
                 một kiểu chip thống nhất (chỉ tô xanh đặc cho chip đang mở ở panel).
              c) Panel nguyên văn highlight tất cả các khoản được trích dẫn, thu gọn các khoản
                 không được trích thành hàng "Khoản N · nhấn để mở", tự cuộn khoản trích đầu tiên lên đầu.
              d) Ô nhập tách riêng bên dưới vùng cuộn chat, không đè nội dung cuối.
              e) Chỉ hiển thị nguồn căn cứ thực sự được trích dẫn, ẩn các điều phụ không chắc chắn.
              f) Lịch sử tra cứu có nút "Xóa toàn bộ" và nút xóa từng mục.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import sys
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generator import (
    DEFAULT_TOP_K,
    RAGGenerator,
    build_fallback_response,
    format_sources_for_ui,
)
from src.retriever import (
    DEFAULT_MIN_SCORE,
    HybridRetriever,
    deduplicate_by_article,
)

FEEDBACK_LOG_FILE = ROOT / "data" / "processed" / "feedback_log.jsonl"

_generator: RAGGenerator | None = None
_toc_cache: list[dict] | None = None


def get_generator() -> RAGGenerator:
    """Khởi tạo RAGGenerator 1 lần duy nhất (Singleton) và giữ nguyên trong RAM."""
    global _generator
    if _generator is None:
        retriever = HybridRetriever.from_files()
        _generator = RAGGenerator(retriever=retriever)
        _generator.warmup()
    return _generator


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """Nạp sẵn chỉ mục 258x3072 và kết nối Gemini một lần khi server khởi động."""
    get_generator()
    yield


app = FastAPI(
    title="Tra cứu Bộ luật Lao động 2019",
    description="Hệ thống tra cứu và hỏi đáp Bộ luật Lao động 2019 (Văn bản hợp nhất 18/VBHN-VPQH)",
    version="2.2.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------
# ĐỊNH NGHĨA DỮ LIỆU ĐẦU VÀO / ĐẦU RA CHO API
# --------------------------------------------------------------------------
class ChatRequest(BaseModel):
    query: str = Field(..., description="Câu hỏi tiếng Việt của người lao động")
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=10, description="Số điều luật tối đa đưa vào prompt (3-5)")
    use_vector: bool = Field(True, description="Có kết hợp tìm kiếm vector hay không")
    min_score: float = Field(DEFAULT_MIN_SCORE, ge=0.0, le=1.0, description="Ngưỡng điểm liên quan tối thiểu")


class FeedbackRequest(BaseModel):
    query: str = Field(..., description="Câu hỏi của người dùng")
    answer: str = Field("", description="Câu trả lời của hệ thống")
    rating: str = Field(..., description="'helpful' (Hữu ích) hoặc 'report' (Báo sai)")
    note: str = Field("", description="Ghi chú báo sai nếu có")
    cited_articles: list[int] = Field(default_factory=list, description="Danh sách số Điều được trích dẫn")


# --------------------------------------------------------------------------
# GIAO DIỆN WEB CHUẨN MOCKUP MỚI
# --------------------------------------------------------------------------
HTML_PAGE = """<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Trợ lý Tra cứu Bộ luật Lao động 2019</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-app: #f8f9fa;
      --bg-card: #ffffff;
      --bg-user: #0f172a;
      --text-main: #111827;
      --text-body: #1f2937;
      --text-muted: #4b5563;
      --text-subtle: #6b7280;
      --primary: #1d4ed8;
      --primary-hover: #1e40af;
      --primary-soft: #eff6ff;
      --primary-border: #bfdbfe;
      --border: #e5e7eb;
      --border-strong: #d1d5db;
      --green-bg: #ecfdf5;
      --green-border: #a7f3d0;
      --green-text: #065f46;
      --red-bg: #fef2f2;
      --red-border: #fecaca;
      --red-text: #991b1b;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    html, body {
      height: 100%;
      width: 100%;
      overflow: hidden;
    }

    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg-app);
      color: var(--text-main);
      display: flex;
      flex-direction: column;
      -webkit-font-smoothing: antialiased;
    }

    /* ==================== TOP HEADER ==================== */
    .topbar {
      min-height: 64px;
      padding: 10px 24px;
      background: #ffffff;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      flex-shrink: 0;
      z-index: 30;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      min-width: 0;
    }

    .brand-icon {
      width: 36px;
      height: 36px;
      border-radius: 10px;
      background: var(--primary);
      color: #ffffff;
      display: flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
    }

    .brand-info h1 {
      font-size: 1.02rem;
      font-weight: 700;
      color: var(--text-main);
      line-height: 1.25;
    }

    .brand-info p {
      font-size: 0.77rem;
      color: var(--text-muted);
      margin-top: 2px;
      line-height: 1.35;
    }

    .header-actions {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-shrink: 0;
    }

    .btn-pill {
      background: #ffffff;
      border: 1px solid var(--border-strong);
      color: var(--text-body);
      padding: 7px 15px;
      border-radius: 999px;
      font-size: 0.81rem;
      font-weight: 500;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: background 0.15s, border-color 0.15s;
    }

    .btn-pill:hover, .btn-pill:focus-visible {
      background: #f3f4f6;
      border-color: #9ca3af;
      outline: none;
    }

    .btn-danger-pill {
      background: var(--red-bg);
      border: 1px solid var(--red-border);
      color: var(--red-text);
      padding: 6px 13px;
      border-radius: 999px;
      font-size: 0.78rem;
      font-weight: 600;
      cursor: pointer;
    }

    .btn-danger-pill:hover {
      background: #fee2e2;
    }

    /* ==================== MAIN WORKSPACE (CHAT + RIGHT PANEL) ==================== */
    .workspace {
      flex: 1;
      display: flex;
      min-height: 0;
      overflow: hidden;
      position: relative;
    }

    /* Sửa lỗi 3d: chat-pane chia flex chuẩn, khu cuộn tách biệt hoàn toàn với ô nhập ở đáy */
    .chat-pane {
      flex: 1;
      display: flex;
      flex-direction: column;
      min-width: 0;
      min-height: 0;
      height: 100%;
      overflow: hidden;
    }

    .chat-scroll {
      flex: 1;
      min-height: 0;
      overflow-y: auto;
      padding: 28px 24px 36px;
      scroll-behavior: smooth;
    }

    .chat-container {
      max-width: 680px;
      width: 100%;
      margin: 0 auto;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }

    /* Hero gọn nhẹ */
    .hero-box {
      padding: 8px 0;
    }

    .hero-title {
      font-size: 1.12rem;
      font-weight: 700;
      color: var(--text-main);
      margin-bottom: 14px;
      text-align: center;
    }

    .sample-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 10px;
    }

    .sample-card {
      background: #ffffff;
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px 14px;
      text-align: left;
      cursor: pointer;
      transition: border-color 0.15s, box-shadow 0.15s, transform 0.15s;
      display: flex;
      flex-direction: column;
      gap: 5px;
    }

    .sample-card:hover, .sample-card:focus-visible {
      border-color: var(--primary-border);
      box-shadow: 0 4px 12px rgba(29, 78, 216, 0.08);
      transform: translateY(-1px);
      outline: none;
    }

    .sample-card-top {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 6px;
    }

    .sample-card-title {
      font-size: 0.84rem;
      font-weight: 600;
      color: var(--text-main);
    }

    .sample-card-art {
      font-size: 0.72rem;
      font-weight: 600;
      color: var(--primary);
      background: var(--primary-soft);
      padding: 2px 8px;
      border-radius: 999px;
      white-space: nowrap;
    }

    .sample-card-desc {
      font-size: 0.78rem;
      color: var(--text-muted);
      line-height: 1.4;
    }

    /* ==================== USER & BOT TURNS ==================== */
    .user-row {
      display: flex;
      justify-content: flex-end;
    }

    .user-pill {
      background: var(--bg-user);
      color: #ffffff;
      padding: 10px 20px;
      border-radius: 999px;
      font-size: 0.92rem;
      font-weight: 500;
      max-width: 85%;
      line-height: 1.45;
    }

    .bot-group {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .bot-card {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 20px 22px;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
    }

    /* NHÓM 1: Chỉ giữ nhãn "Trợ lý pháp luật", không có badge cạnh nhãn */
    .bot-card-header {
      display: flex;
      align-items: center;
      margin-bottom: 10px;
    }

    .bot-label {
      font-size: 0.84rem;
      font-weight: 700;
      color: var(--text-main);
    }

    .progress-steps {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 0.8rem;
      color: var(--text-muted);
      padding: 4px 0 8px;
    }

    .step-active {
      color: var(--primary);
      font-weight: 600;
    }

    /* NHÓM 2: Dòng đầu là Kết luận in đậm, bên dưới tối đa 3 gạch đầu dòng */
    .answer-content {
      font-size: 0.94rem;
      line-height: 1.68;
      color: var(--text-body);
    }

    .answer-lead {
      font-size: 1.03rem;
      font-weight: 700;
      color: var(--text-main);
      line-height: 1.55;
      margin-bottom: 12px;
    }

    .answer-content p {
      margin-bottom: 10px;
    }

    .answer-content ul {
      margin: 6px 0 12px 20px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .answer-content li {
      padding-left: 2px;
      line-height: 1.62;
    }

    /* Sửa lỗi 3b: Một kiểu chip trích dẫn thống nhất, chỉ tô xanh đặc cho chip đang mở ở panel */
    .cite-chip {
      display: inline-flex;
      align-items: center;
      padding: 3px 11px;
      margin-left: 6px;
      margin-right: 2px;
      border-radius: 999px;
      font-size: 0.79rem;
      font-weight: 600;
      font-family: inherit;
      background: var(--primary-soft);
      color: var(--primary);
      border: 1px solid var(--primary-border);
      cursor: pointer;
      transition: all 0.15s ease;
      vertical-align: baseline;
      line-height: 1.35;
    }

    .cite-chip:hover, .cite-chip:focus-visible {
      background: #dbeafe;
      border-color: var(--primary);
      outline: 2px solid rgba(29, 78, 216, 0.2);
    }

    .cite-chip.active {
      background: var(--primary);
      color: #ffffff;
      border-color: var(--primary);
    }

    /* Hàng Căn cứ + Hữu ích / Báo sai đồng bộ theo mockup mới */
    .card-meta-row {
      margin-top: 16px;
      padding-top: 14px;
      border-top: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
    }

    .sources-inline {
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 8px;
    }

    .source-label {
      font-size: 0.78rem;
      color: var(--text-muted);
      font-weight: 500;
      margin-right: 4px;
    }

    .feedback-group {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .action-pill {
      background: #ffffff;
      border: 1px solid var(--border-strong);
      color: var(--text-body);
      padding: 5px 13px;
      border-radius: 999px;
      font-size: 0.78rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.15s;
    }

    .action-pill:hover, .action-pill:focus-visible {
      background: #f3f4f6;
      border-color: #9ca3af;
      outline: none;
    }

    .action-pill.done-helpful {
      background: var(--green-bg);
      border-color: var(--green-border);
      color: var(--green-text);
    }

    .action-pill.done-report {
      background: var(--red-bg);
      border-color: var(--red-border);
      color: var(--red-text);
    }

    .disclaimer-row {
      margin-top: 10px;
      font-size: 0.75rem;
      color: var(--text-subtle);
    }

    /* Khu vực "Hỏi tiếp" */
    .followup-section {
      padding-left: 2px;
    }

    .followup-label {
      font-size: 0.77rem;
      font-weight: 500;
      color: var(--text-muted);
      margin-bottom: 8px;
    }

    .followup-chips {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }

    .followup-chip {
      background: #ffffff;
      border: 1px solid var(--border-strong);
      color: var(--text-main);
      padding: 7px 14px;
      border-radius: 999px;
      font-size: 0.81rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.15s;
    }

    .followup-chip:hover, .followup-chip:focus-visible {
      border-color: var(--primary);
      color: var(--primary);
      background: var(--primary-soft);
      outline: none;
    }

    /* ==================== BOTTOM COMPOSER (SỬA LỖI 3D: KHÔNG ĐÈ KHU CHAT) ==================== */
    .composer-area {
      position: static;
      flex-shrink: 0;
      padding: 12px 24px 18px;
      background: var(--bg-app);
      border-top: 1px solid rgba(229, 231, 235, 0.6);
      z-index: 10;
    }

    .composer-pill {
      max-width: 680px;
      margin: 0 auto;
      background: #ffffff;
      border: 1.5px solid var(--primary-border);
      border-radius: 999px;
      padding: 5px 6px 5px 18px;
      display: flex;
      align-items: center;
      gap: 10px;
      box-shadow: 0 2px 10px rgba(29, 78, 216, 0.06);
      transition: border-color 0.15s, box-shadow 0.15s;
    }

    .composer-pill:focus-within {
      border-color: var(--primary);
      box-shadow: 0 4px 16px rgba(29, 78, 216, 0.12);
    }

    #query-input {
      flex: 1;
      border: none;
      outline: none;
      font-size: 0.92rem;
      color: var(--text-main);
      background: transparent;
      padding: 7px 0;
      font-family: inherit;
    }

    #query-input::placeholder {
      color: var(--text-muted);
    }

    #send-btn {
      width: 36px;
      height: 36px;
      border-radius: 50%;
      border: none;
      background: var(--primary);
      color: #ffffff;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      flex-shrink: 0;
      transition: background 0.15s;
    }

    #send-btn:hover:not(:disabled) {
      background: var(--primary-hover);
    }

    #send-btn:disabled {
      background: #9ca3af;
      cursor: not-allowed;
    }

    .composer-subtext {
      max-width: 680px;
      margin: 7px auto 0;
      text-align: center;
      font-size: 0.75rem;
      color: var(--text-muted);
      line-height: 1.4;
    }

    /* ==================== RIGHT PANEL: NGUYÊN VĂN ĐIỀU LUẬT (SỬA LỖI 3A & 3C) ==================== */
    .law-panel {
      width: 410px;
      background: #ffffff;
      border-left: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      height: 100%;
      flex-shrink: 0;
    }

    .law-panel.hidden {
      display: none;
    }

    .law-panel-top {
      padding: 16px 20px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-shrink: 0;
    }

    .law-panel-label {
      font-size: 0.75rem;
      font-weight: 700;
      letter-spacing: 0.04em;
      color: var(--text-muted);
      text-transform: uppercase;
    }

    .btn-icon-close {
      background: transparent;
      border: none;
      color: var(--text-muted);
      width: 28px;
      height: 28px;
      border-radius: 6px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .btn-icon-close:hover {
      background: #f3f4f6;
      color: var(--text-main);
    }

    .law-panel-body {
      flex: 1;
      min-height: 0;
      overflow-y: auto;
      padding: 20px;
      scroll-behavior: smooth;
    }

    .law-art-heading {
      font-size: 1.12rem;
      font-weight: 700;
      color: var(--text-main);
      margin-bottom: 4px;
    }

    .law-art-meta {
      font-size: 0.77rem;
      color: var(--text-muted);
      margin-bottom: 14px;
      line-height: 1.4;
    }

    .clauses-list {
      display: flex;
      flex-direction: column;
      gap: 10px;
      width: 100%;
    }

    /* Sửa lỗi 3a: width 100%, box-sizing border-box, padding 14px 16px, nhãn margin-bottom 6px, không pre-wrap */
    .clause-block {
      width: 100%;
      box-sizing: border-box;
      font-size: 0.89rem;
      line-height: 1.65;
      color: var(--text-body);
      background: #ffffff;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 14px 16px;
    }

    .clause-block.highlighted {
      width: 100%;
      box-sizing: border-box;
      background: var(--primary-soft);
      border: 1px solid var(--primary-border);
      border-radius: 10px;
      padding: 14px 16px;
    }

    .clause-highlight-tag {
      display: block;
      font-size: 0.71rem;
      font-weight: 700;
      color: var(--primary);
      text-transform: uppercase;
      letter-spacing: 0.03em;
      margin: 0 0 6px 0;
      line-height: 1.3;
    }

    .clause-body-text {
      margin: 0;
      padding: 0;
      white-space: normal;
      word-break: break-word;
    }

    /* Sửa lỗi 3c: Hàng thu gọn cho các khoản không được trích hoặc khoản trích đang đóng */
    .clause-collapsed-bar {
      width: 100%;
      box-sizing: border-box;
      text-align: left;
      padding: 10px 14px;
      border-radius: 10px;
      font-size: 0.82rem;
      font-family: inherit;
      cursor: pointer;
      transition: all 0.14s ease;
      background: #ffffff;
      border: 1px solid var(--border);
      color: var(--text-muted);
      font-weight: 500;
    }

    .clause-collapsed-bar:hover {
      background: #f9fafb;
      border-color: var(--border-strong);
      color: var(--text-main);
    }

    .clause-collapsed-bar.cited-bar {
      background: var(--primary-soft);
      border: 1px solid var(--primary-border);
      color: var(--primary);
      font-weight: 600;
    }

    .clause-collapsed-bar.cited-bar:hover {
      background: #dbeafe;
    }

    .law-panel-footer {
      padding: 14px 20px;
      border-top: 1px solid var(--border);
      display: flex;
      align-items: center;
      gap: 10px;
      background: #ffffff;
      flex-shrink: 0;
    }

    /* ==================== MODAL MỤC LỤC LUẬT & LỊCH SỬ ==================== */
    .drawer-overlay {
      position: fixed;
      inset: 0;
      background: rgba(15, 23, 42, 0.35);
      z-index: 60;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 20px;
    }

    .drawer-overlay.open {
      display: flex;
    }

    .modal-box {
      background: #ffffff;
      border-radius: 16px;
      width: 680px;
      max-width: 100%;
      max-height: 82vh;
      display: flex;
      flex-direction: column;
      box-shadow: 0 20px 45px rgba(0, 0, 0, 0.15);
      overflow: hidden;
    }

    .modal-header {
      padding: 16px 20px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }

    .modal-header h3 {
      font-size: 1rem;
      font-weight: 700;
    }

    .modal-search {
      padding: 12px 20px;
      border-bottom: 1px solid var(--border);
      background: #f9fafb;
    }

    .modal-search input {
      width: 100%;
      padding: 9px 14px;
      border: 1px solid var(--border-strong);
      border-radius: 999px;
      font-size: 0.88rem;
      outline: none;
    }

    .modal-search input:focus {
      border-color: var(--primary);
    }

    .modal-body {
      flex: 1;
      overflow-y: auto;
      padding: 16px 20px;
    }

    .toc-chapter {
      margin-bottom: 16px;
    }

    .toc-chapter-title {
      font-size: 0.83rem;
      font-weight: 700;
      color: var(--primary);
      margin-bottom: 8px;
      text-transform: uppercase;
    }

    .toc-articles {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(270px, 1fr));
      gap: 6px;
    }

    .toc-art-btn {
      background: #f9fafb;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 7px 10px;
      text-align: left;
      font-size: 0.8rem;
      color: var(--text-body);
      cursor: pointer;
      transition: all 0.14s;
    }

    .toc-art-btn:hover {
      border-color: var(--primary);
      background: var(--primary-soft);
      color: var(--primary);
    }

    .history-item-row {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .history-item-row .toc-art-btn {
      flex: 1;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }

    .btn-del-item {
      width: 30px;
      height: 30px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: #ffffff;
      color: var(--text-muted);
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
    }

    .btn-del-item:hover {
      background: var(--red-bg);
      border-color: var(--red-border);
      color: var(--red-text);
    }

    @media (max-width: 920px) {
      .law-panel {
        position: fixed;
        left: 0;
        right: 0;
        bottom: 0;
        width: 100%;
        height: 68vh;
        border-left: none;
        border-top: 1px solid var(--border-strong);
        border-radius: 18px 18px 0 0;
        box-shadow: 0 -10px 30px rgba(0, 0, 0, 0.15);
        z-index: 50;
      }
    }
  </style>
</head>
<body>
  <!-- TOPBAR -->
  <header class="topbar">
    <div class="brand">
      <div class="brand-icon" aria-hidden="true">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M12 3v18"/>
          <path d="M5 7h14"/>
          <path d="M5 7l-3 6h6l-3-6z"/>
          <path d="M19 7l-3 6h6l-3-6z"/>
          <path d="M8 21h8"/>
        </svg>
      </div>
      <div class="brand-info">
        <h1>Tra cứu Bộ luật Lao động 2019</h1>
        <p>Dữ liệu: VBHN 18/VBHN-VPQH · Chưa có Luật BHXH và nghị định hướng dẫn · Cập nhật: 23/11/2026</p>
      </div>
    </div>
    <div class="header-actions">
      <button type="button" class="btn-pill" onclick="openHistoryModal()">Lịch sử</button>
      <button type="button" class="btn-pill" onclick="openTocModal()">Mục lục luật</button>
      <button type="button" class="btn-pill" onclick="resetConversation()">Hội thoại mới</button>
    </div>
  </header>

  <!-- MAIN WORKSPACE -->
  <div class="workspace">
    <div class="chat-pane">
      <main class="chat-scroll" id="main-scroll">
        <div class="chat-container" id="chat-box">
          <section class="hero-box" id="welcome-hero">
            <h2 class="hero-title">Chọn chủ đề tra cứu nhanh trong Bộ luật Lao động 2019</h2>
            <div class="sample-grid">
              <button type="button" class="sample-card" onclick="askSample('tuổi nghỉ hưu năm 2026 là bao nhiêu')">
                <div class="sample-card-top">
                  <span class="sample-card-title">Tuổi nghỉ hưu năm 2026</span>
                  <span class="sample-card-art">Điều 169</span>
                </div>
                <span class="sample-card-desc">Tính tuổi nghỉ hưu cụ thể năm 2026 của lao động nam và nữ</span>
              </button>

              <button type="button" class="sample-card" onclick="askSample('báo trước bao nhiêu ngày khi nghỉ việc')">
                <div class="sample-card-top">
                  <span class="sample-card-title">Thời hạn báo trước khi nghỉ việc</span>
                  <span class="sample-card-art">Điều 35</span>
                </div>
                <span class="sample-card-desc">Quy định báo trước 45 ngày, 30 ngày, 03 ngày và trường hợp miễn báo trước</span>
              </button>

              <button type="button" class="sample-card" onclick="askSample('trợ cấp thôi việc tính thế nào')">
                <div class="sample-card-top">
                  <span class="sample-card-title">Cách tính trợ cấp thôi việc</span>
                  <span class="sample-card-art">Điều 46</span>
                </div>
                <span class="sample-card-desc">Điều kiện làm việc từ đủ 12 tháng và mức hưởng 1/2 tháng lương mỗi năm</span>
              </button>

              <button type="button" class="sample-card" onclick="askSample('thời gian thử việc tối đa')">
                <div class="sample-card-top">
                  <span class="sample-card-title">Thời gian & Lương thử việc</span>
                  <span class="sample-card-art">Điều 25, 26</span>
                </div>
                <span class="sample-card-desc">Giới hạn 180 ngày, 60 ngày, 30 ngày, 06 ngày và mức lương tối thiểu 85%</span>
              </button>

              <button type="button" class="sample-card" onclick="askSample('làm thêm giờ được trả bao nhiêu')">
                <div class="sample-card-top">
                  <span class="sample-card-title">Tiền lương làm thêm giờ (OT)</span>
                  <span class="sample-card-art">Điều 98</span>
                </div>
                <span class="sample-card-desc">Mức hưởng ít nhất 150% ngày thường, 200% ngày nghỉ tuần, 300% lễ tết</span>
              </button>

              <button type="button" class="sample-card" onclick="askSample('người lao động tự ý bỏ việc bao nhiêu ngày thì bị xử lý kỷ luật sa thải')">
                <div class="sample-card-top">
                  <span class="sample-card-title">Các trường hợp kỷ luật sa thải</span>
                  <span class="sample-card-art">Điều 125</span>
                </div>
                <span class="sample-card-desc">Căn cứ áp dụng hình thức sa thải và trường hợp tự ý bỏ việc 05 ngày</span>
              </button>
            </div>
          </section>
        </div>
      </main>

      <!-- THANH NHẬP CÂU HỎI TÁCH RIÊNG Ở ĐÁY (KHÔNG ĐÈ NỘI DUNG CUỐI) -->
      <footer class="composer-area">
        <form class="composer-pill" onsubmit="sendChat(event)">
          <input
            id="query-input"
            type="text"
            placeholder="Hỏi về Bộ luật Lao động, hoặc nhập số Điều (ví dụ: Điều 35)"
            autocomplete="off"
          />
          <button id="send-btn" type="submit" aria-label="Gửi câu hỏi">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
              <line x1="22" y1="2" x2="11" y2="13"></line>
              <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
            </svg>
          </button>
        </form>
        <div class="composer-subtext">
          Nhấn vào chip Điều để xem nguyên văn ở khung bên phải
        </div>
      </footer>
    </div>

    <!-- PANEL CỐ ĐỊNH BÊN PHẢI: NGUYÊN VĂN ĐIỀU LUẬT -->
    <aside class="law-panel" id="article-popover" aria-label="Nguyên văn điều luật">
      <div class="law-panel-top">
        <span class="law-panel-label">NGUYÊN VĂN ĐIỀU LUẬT</span>
        <button type="button" class="btn-icon-close" onclick="closeLawPanel()" aria-label="Đóng khung nguyên văn">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <line x1="18" y1="6" x2="6" y2="18"></line>
            <line x1="6" y1="6" x2="18" y2="18"></line>
          </svg>
        </button>
      </div>
      <div class="law-panel-body" id="law-panel-body">
        <h3 class="law-art-heading" id="panel-art-title">Điều 169. Tuổi nghỉ hưu</h3>
        <div class="law-art-meta" id="panel-art-meta">Bộ luật Lao động 2019 · Chương XII</div>
        <div class="clauses-list" id="panel-clauses-list"></div>
      </div>
      <div class="law-panel-footer">
        <button type="button" class="btn-pill" id="btn-toggle-full" onclick="expandFullArticleInPanel()">Xem toàn bộ điều luật</button>
        <button type="button" class="btn-pill" id="btn-copy-cite" onclick="copyPanelCitation()">Sao chép trích dẫn</button>
      </div>
    </aside>
  </div>

  <!-- MODAL MỤC LỤC LUẬT -->
  <div class="drawer-overlay" id="toc-modal" onclick="if(event.target===this)closeTocModal()">
    <div class="modal-box">
      <div class="modal-header">
        <h3>Mục lục Bộ luật Lao động 2019 (17 Chương · 220 Điều)</h3>
        <button type="button" class="btn-icon-close" onclick="closeTocModal()">✕</button>
      </div>
      <div class="modal-search">
        <input type="text" id="toc-search-input" placeholder="Tìm nhanh theo số Điều hoặc từ khóa (ví dụ: Điều 35, thử việc, thai sản)..." oninput="filterTocList()" />
      </div>
      <div class="modal-body" id="toc-list-body">Đang tải mục lục...</div>
    </div>
  </div>

  <!-- MODAL LỊCH SỬ HỘI THOẠI (CÓ NÚT XÓA TOÀN BỘ & XÓA TỪNG MỤC) -->
  <div class="drawer-overlay" id="history-modal" onclick="if(event.target===this)closeHistoryModal()">
    <div class="modal-box">
      <div class="modal-header">
        <h3>Lịch sử câu hỏi đã tra cứu</h3>
        <div style="display:flex;align-items:center;gap:8px;">
          <button type="button" class="btn-danger-pill" id="btn-clear-history" onclick="clearAllHistory()">Xóa toàn bộ</button>
          <button type="button" class="btn-icon-close" onclick="closeHistoryModal()">✕</button>
        </div>
      </div>
      <div class="modal-body" id="history-list-body"></div>
    </div>
  </div>

  <script>
    const mainScroll = document.getElementById('main-scroll');
    const chatBox = document.getElementById('chat-box');
    const queryInput = document.getElementById('query-input');
    const sendBtn = document.getElementById('send-btn');
    const lawPanel = document.getElementById('article-popover');
    const panelTitle = document.getElementById('panel-art-title');
    const panelMeta = document.getElementById('panel-art-meta');
    const panelClauses = document.getElementById('panel-clauses-list');
    const btnToggleFull = document.getElementById('btn-toggle-full');

    const articleStore = {};
    // Map lưu toàn bộ các khoản được trích dẫn theo từng Điều của câu trả lời hiện tại: { 169: Set(['2', '3', '4']) }
    let currentTurnCitedClausesByArt = {};
    let currentPanelArticleNo = 169;
    let currentExpandedClauses = new Set(['2']);
    let currentCitedClausesSet = new Set(['2']);
    let showAllClausesMode = false;
    let currentCopyBuffer = '';
    let tocDataCache = null;

    function escapeHtml(str) {
      return (str || '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
    }

    function registerSources(sources) {
      if (!Array.isArray(sources)) return;
      for (const s of sources) {
        if (s && s.article_no) {
          articleStore[Number(s.article_no)] = s;
        }
      }
    }

    /* ==================== PARSE & FORMAT CÂU TRẢ LỜI (NHÓM 2 & 3B) ==================== */
    function parseClauseList(clauseRaw) {
      if (!clauseRaw) return [];
      return String(clauseRaw)
        .split(/[,và&\\s]+/i)
        .map(s => s.trim().toLowerCase())
        .filter(s => /^\\d+[a-z]?$/.test(s));
    }

    function formatStructuredAnswer(rawText) {
      if (!rawText) return '';

      let cleaned = rawText.trim();
      const lines = cleaned.split(/\\r?\\n/);
      while (lines.length > 1) {
        const last = lines[lines.length - 1].trim().toLowerCase();
        if (last.includes('không phải là tư vấn pháp lý chính thức') ||
            last.includes('không phải tư vấn pháp lý chính thức') ||
            last.includes('thông tin tham khảo') ||
            last === '') {
          lines.pop();
        } else {
          break;
        }
      }
      cleaned = lines.join('\\n').trim();

      let safe = escapeHtml(cleaned)
        .replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>')
        .replace(/\\*(.+?)\\*/g, '<em>$1</em>');

      // Khử trùng lặp chip trích dẫn trong cùng một câu trả lời và thu thập tập khoản được trích
      const seenChipKeys = new Set();
      const citedMap = {};

      // Bắt mọi dạng: [khoản 2 Điều 169], (Theo khoản 3, 4 Điều 169 Bộ luật Lao động 2019), khoản 1 và 2 Điều 35...
      const citeRegex = /(?:\\[|\\(\\s*)?(?:(?:Theo|Căn cứ|Căn cứ theo)\\s+)?(?:(các\\s+khoản|khoản)\\s+([0-9a-z]+(?:\\s*(?:,|và)\\s*(?:khoản\\s+)?[0-9a-z]+)*)\\s+)?Điều\\s+(\\d{1,3})(?:\\s+Bộ luật Lao động(?:\\s+2019)?)?\\s*(?:\\]|\\))?/gi;

      const rawLines = safe.split(/\\n/);
      let htmlOut = '';
      let inList = false;
      let leadRendered = false;

      for (let rawLine of rawLines) {
        let line = rawLine.trim();
        if (!line) {
          if (inList) { htmlOut += '</ul>'; inList = false; }
          continue;
        }
        const upperCheck = line.replace(/<[^>]+>/g, '').trim().toUpperCase();
        if (upperCheck === 'KẾT LUẬN:' || upperCheck === 'KẾT LUẬN' ||
            upperCheck === 'CHI TIẾT:' || upperCheck === 'CHI TIẾT') {
          continue;
        }

        // Tìm tất cả trích dẫn trong dòng này, gom thành duy nhất 1 chip đặt ở CUỐI dòng sau dấu chấm
        const lineMatches = [];
        let strippedLine = line.replace(citeRegex, (fullMatch, _khWord, clausesStr, artStr) => {
          const aNum = parseInt(artStr, 10);
          if (aNum < 1 || aNum > 500) return fullMatch;
          const cleanClauses = parseClauseList(clausesStr);
          lineMatches.push({ artNo: aNum, clauses: cleanClauses });
          return '';
        });

        // Dọn sạch dấu chấm/ngoặc thừa ở cuối câu sau khi rút trích dẫn ra cuối
        strippedLine = strippedLine
          .replace(/\\s+([.,;])/g, '$1')
          .replace(/\\(\\s*\\)|\\[\\s*\\]/g, '')
          .replace(/[,;:\\s]+$/g, '')
          .trim();

        if (!strippedLine) continue;
        // Đảm bảo câu kết thúc bằng dấu chấm trước khi đặt chip
        const plainEnd = strippedLine.replace(/<\\/strong>|<\\/em>/gi, '').trim();
        if (plainEnd && !/[.?!]$/.test(plainEnd)) {
          if (strippedLine.endsWith('</strong>')) {
            strippedLine = strippedLine.slice(0, -9) + '.</strong>';
          } else {
            strippedLine += '.';
          }
        }

        // Tạo tối đa 1 chip duy nhất ở cuối dòng (nếu chưa trùng với chip ở dòng trước)
        let chipHtml = '';
        if (lineMatches.length > 0) {
          const primaryArt = lineMatches[0].artNo;
          const mergedClauses = [];
          for (const m of lineMatches) {
            if (m.artNo === primaryArt) {
              for (const c of m.clauses) {
                if (!mergedClauses.includes(c)) mergedClauses.push(c);
              }
            }
          }
          if (!citedMap[primaryArt]) citedMap[primaryArt] = new Set();
          mergedClauses.forEach(c => citedMap[primaryArt].add(c));

          const clauseKey = mergedClauses.join(',');
          const chipKey = `${primaryArt}:${clauseKey}`;
          if (!seenChipKeys.has(chipKey)) {
            seenChipKeys.add(chipKey);
            const chipLabel = mergedClauses.length > 0
              ? `khoản ${mergedClauses.join(', ')} Điều ${primaryArt}`
              : `Điều ${primaryArt}`;
            chipHtml = ` <button type="button" class="cite-chip" data-article="${primaryArt}" data-clauses="${clauseKey}" onclick="showArticlePopover(event, ${primaryArt}, '${clauseKey}')">${chipLabel}</button>`;
          }
        }

        const isBullet = /^[-•*]\\s+/.test(strippedLine);
        if (isBullet) {
          if (!inList) { htmlOut += '<ul>'; inList = true; }
          const itemBody = strippedLine.replace(/^[-•*]\\s+/, '');
          htmlOut += `<li>${itemBody}${chipHtml}</li>`;
        } else {
          if (inList) { htmlOut += '</ul>'; inList = false; }
          if (!leadRendered) {
            leadRendered = true;
            const cleanLead = strippedLine.replace(/<\\/?strong>/gi, '');
            htmlOut += `<div class="answer-lead">${cleanLead}${chipHtml}</div>`;
          } else {
            htmlOut += `<p>${strippedLine}${chipHtml}</p>`;
          }
        }
      }

      if (inList) htmlOut += '</ul>';
      currentTurnCitedClausesByArt = citedMap;
      return htmlOut;
    }

    /* Sửa lỗi 3e: Chỉ hiển thị điều luật thực sự được trích dẫn hoặc căn cứ chính; ẩn điều phụ không chắc chắn */
    function renderCitedSources(sources, citedArticles) {
      if (!Array.isArray(sources) || sources.length === 0) return '';
      registerSources(sources);

      const citedSet = new Set((citedArticles || []).map(Number));
      let filtered = sources.filter(s => citedSet.has(Number(s.article_no)));
      if (filtered.length === 0) {
        filtered = [sources[0]];
      }

      const chips = filtered.map(s => {
        const aNo = Number(s.article_no);
        const cSet = currentTurnCitedClausesByArt[aNo];
        const clausesAttr = cSet && cSet.size > 0 ? Array.from(cSet).join(',') : '1';
        return `
          <button type="button" class="cite-chip source-ref-chip" data-article="${aNo}" data-clauses="${clausesAttr}" onclick="showArticlePopover(event, ${aNo}, '${clausesAttr}')">
            Điều ${aNo} - ${escapeHtml(s.article_title)}
          </button>
        `;
      }).join('');

      return `
        <div class="sources-inline">
          <span class="source-label">Căn cứ</span>
          ${chips}
        </div>
      `;
    }

    function renderFollowUps(followUps) {
      if (!Array.isArray(followUps) || followUps.length === 0) return '';
      const chips = followUps.map(q => {
        const safeQ = escapeHtml(q);
        const jsQ = q.replace(/'/g, "\\\\'");
        return `<button type="button" class="followup-chip" onclick="askSample('${jsQ}')">${safeQ}</button>`;
      }).join('');
      return `
        <div class="followup-section">
          <div class="followup-label">Hỏi tiếp</div>
          <div class="followup-chips">${chips}</div>
        </div>
      `;
    }

    /* ==================== PANEL NGUYÊN VĂN (SỬA LỖI 3A & 3C) ==================== */
    function splitArticleIntoClauses(rawText) {
      const lines = (rawText || '').split(/\\r?\\n/).map(l => l.trim()).filter(Boolean);
      const clauses = [];
      let current = null;

      for (const line of lines) {
        const m = line.match(/^(\\d+[a-z]?)\\.\\s+(.*)$/);
        if (m) {
          if (current) clauses.push(current);
          current = { clauseNo: m[1].toLowerCase(), lines: [line] };
        } else {
          if (!current) {
            current = { clauseNo: '1', lines: [line] };
          } else {
            current.lines.push(line);
          }
        }
      }
      if (current) clauses.push(current);
      return clauses;
    }

    function renderArticleInPanel(artData) {
      const artNo = Number(artData.article_no);
      currentPanelArticleNo = artNo;

      panelTitle.textContent = `Điều ${artNo}. ${artData.article_title || ''}`;
      const ch = artData.chapter_no ? `Chương ${artData.chapter_no}` : 'Bộ luật Lao động 2019';
      panelMeta.textContent = `Bộ luật Lao động 2019 · ${ch}`;

      const clauses = splitArticleIntoClauses(artData.text);
      let html = '';
      let firstCitedElId = '';
      const copyParts = [];

      clauses.forEach((cl, idx) => {
        const cNo = cl.clauseNo || String(idx + 1);
        const isCited = currentCitedClausesSet.has(cNo);
        const isExpanded = showAllClausesMode || currentExpandedClauses.has(cNo);
        const bodyHtml = cl.lines.map(l => escapeHtml(l)).join('<br>');

        if (isCited && !firstCitedElId) {
          firstCitedElId = `panel-clause-${cNo}`;
        }

        if (isExpanded) {
          copyParts.push(cl.lines.join('\\n'));
          const blockClass = isCited ? 'clause-block highlighted' : 'clause-block';
          const tagHtml = isCited
            ? `<span class="clause-highlight-tag">KHOẢN ${escapeHtml(cNo.toUpperCase())} · ĐƯỢC TRÍCH DẪN</span>`
            : '';
          html += `<div class="${blockClass}" id="panel-clause-${escapeHtml(cNo)}">${tagHtml}<div class="clause-body-text">${bodyHtml}</div></div>`;
        } else {
          // Khoản thu gọn thành 1 hàng bấm được
          const barClass = isCited ? 'clause-collapsed-bar cited-bar' : 'clause-collapsed-bar';
          const barText = isCited
            ? `Khoản ${cNo} · được trích dẫn · nhấn để mở`
            : `Khoản ${cNo} · nhấn để mở`;
          html += `<button type="button" class="${barClass}" id="panel-clause-${escapeHtml(cNo)}" onclick="toggleSingleClause('${escapeHtml(cNo)}')">${escapeHtml(barText)}</button>`;
        }
      });

      panelClauses.innerHTML = html;
      btnToggleFull.textContent = showAllClausesMode ? 'Thu gọn về khoản trích dẫn' : 'Xem toàn bộ điều luật';

      const citeHeader = `Điều ${artNo} Bộ luật Lao động 2019 (${artData.article_title})`;
      currentCopyBuffer = `${citeHeader}:\\n${copyParts.join('\\n\\n') || artData.text}`;

      if (firstCitedElId) {
        const firstEl = document.getElementById(firstCitedElId);
        if (firstEl) {
          setTimeout(() => firstEl.scrollIntoView({ behavior: 'smooth', block: 'start' }), 40);
        }
      }
    }

    function toggleSingleClause(clauseNo) {
      if (currentExpandedClauses.has(clauseNo)) {
        currentExpandedClauses.delete(clauseNo);
      } else {
        currentExpandedClauses.add(clauseNo);
      }
      const artData = articleStore[currentPanelArticleNo];
      if (artData) renderArticleInPanel(artData);
    }

    async function showArticlePopover(evt, articleNo, clausesCsv = '') {
      // Sửa lỗi 3b: chỉ tô xanh đặc cho đúng chip đang được mở ở panel
      document.querySelectorAll('.cite-chip.active').forEach(el => el.classList.remove('active'));
      if (evt && evt.currentTarget && evt.currentTarget.classList.contains('cite-chip')) {
        evt.currentTarget.classList.add('active');
      }

      lawPanel.classList.remove('hidden');
      const artNum = Number(articleNo);
      const clickedClauses = parseClauseList(clausesCsv);

      // Tập hợp tất cả các khoản được trích dẫn của Điều này trong câu trả lời
      const turnSet = currentTurnCitedClausesByArt[artNum];
      const allCited = new Set(turnSet ? Array.from(turnSet) : []);
      clickedClauses.forEach(c => allCited.add(c));
      if (allCited.size === 0) allCited.add('1');

      currentCitedClausesSet = allCited;
      // Mở khoản thuộc chip vừa bấm (hoặc tất cả khoản được trích nếu bấm từ nút Căn cứ)
      currentExpandedClauses = new Set(clickedClauses.length > 0 ? clickedClauses : Array.from(allCited));
      showAllClausesMode = false;

      if (articleStore[artNum]) {
        renderArticleInPanel(articleStore[artNum]);
        return;
      }

      panelTitle.textContent = `Điều ${artNum}`;
      panelClauses.innerHTML = '<div class="clause-block"><div class="clause-body-text">Đang tải nguyên văn điều luật...</div></div>';
      try {
        const res = await fetch(`/api/article/${artNum}`);
        if (!res.ok) throw new Error('Không tìm thấy điều luật');
        const data = await res.json();
        articleStore[artNum] = data;
        renderArticleInPanel(data);
      } catch (err) {
        panelClauses.innerHTML = `<div class="clause-block"><div class="clause-body-text">Không tìm thấy Điều ${artNum} trong dữ liệu hiện có.</div></div>`;
      }
    }

    function expandFullArticleInPanel() {
      const artData = articleStore[currentPanelArticleNo];
      if (!artData) return;
      showAllClausesMode = !showAllClausesMode;
      renderArticleInPanel(artData);
    }

    function closeLawPanel() {
      lawPanel.classList.add('hidden');
    }

    async function copyPanelCitation() {
      if (!currentCopyBuffer) return;
      const btn = document.getElementById('btn-copy-cite');
      try {
        await navigator.clipboard.writeText(currentCopyBuffer);
        const old = btn.textContent;
        btn.textContent = 'Đã sao chép';
        setTimeout(() => { btn.textContent = old; }, 1500);
      } catch (_) {}
    }

    /* ==================== FEEDBACK (HỮU ÍCH / BÁO SAI) ==================== */
    async function submitFeedback(btn, rating, encodedQuery, encodedAnswer) {
      const query = decodeURIComponent(encodedQuery);
      const answer = decodeURIComponent(encodedAnswer);
      let note = '';
      if (rating === 'report') {
        note = window.prompt('Vui lòng mô tả ngắn gọn điểm chưa chính xác để hệ thống cải thiện:', '') || 'Người dùng báo sai';
      }
      try {
        await fetch('/api/feedback', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query, answer, rating, note })
        });
        if (rating === 'helpful') {
          btn.classList.add('done-helpful');
          btn.textContent = 'Đã ghi nhận hữu ích';
        } else {
          btn.classList.add('done-report');
          btn.textContent = 'Đã ghi nhận báo sai';
        }
      } catch (_) {}
    }

    /* ==================== MỤC LỤC LUẬT & LỊCH SỬ (CÓ XÓA TOÀN BỘ / XÓA TỪNG MỤC) ==================== */
    async function openTocModal() {
      document.getElementById('toc-modal').classList.add('open');
      if (tocDataCache) return;
      try {
        const res = await fetch('/api/toc');
        tocDataCache = await res.json();
        renderTocChapters(tocDataCache.chapters || []);
      } catch (e) {
        document.getElementById('toc-list-body').textContent = 'Không tải được mục lục.';
      }
    }

    function closeTocModal() {
      document.getElementById('toc-modal').classList.remove('open');
    }

    function renderTocChapters(chapters) {
      const body = document.getElementById('toc-list-body');
      if (!chapters.length) {
        body.textContent = 'Không tìm thấy Điều luật khớp từ khóa.';
        return;
      }
      body.innerHTML = chapters.map(ch => `
        <div class="toc-chapter">
          <div class="toc-chapter-title">Chương ${escapeHtml(ch.chapter_no)} — ${escapeHtml(ch.chapter_title)}</div>
          <div class="toc-articles">
            ${ch.articles.map(a => `
              <button type="button" class="toc-art-btn" onclick="selectArticleFromToc(${a.article_no})">
                <strong>Điều ${a.article_no}.</strong> ${escapeHtml(a.article_title)}
              </button>
            `).join('')}
          </div>
        </div>
      `).join('');
    }

    function filterTocList() {
      if (!tocDataCache) return;
      const kw = (document.getElementById('toc-search-input').value || '').trim().toLowerCase();
      if (!kw) {
        renderTocChapters(tocDataCache.chapters);
        return;
      }
      const filtered = tocDataCache.chapters.map(ch => ({
        ...ch,
        articles: ch.articles.filter(a =>
          `điều ${a.article_no}`.includes(kw) ||
          String(a.article_no) === kw ||
          (a.article_title || '').toLowerCase().includes(kw)
        )
      })).filter(ch => ch.articles.length > 0);
      renderTocChapters(filtered);
    }

    function selectArticleFromToc(artNo) {
      closeTocModal();
      showArticlePopover(null, artNo, '1');
    }

    function saveHistoryItem(query) {
      try {
        const raw = localStorage.getItem('rag_labor_history') || '[]';
        const list = JSON.parse(raw).filter(x => x.query !== query);
        list.unshift({ query, time: new Date().toLocaleTimeString('vi-VN') });
        localStorage.setItem('rag_labor_history', JSON.stringify(list.slice(0, 30)));
      } catch (_) {}
    }

    function renderHistoryList() {
      const body = document.getElementById('history-list-body');
      const clearBtn = document.getElementById('btn-clear-history');
      try {
        const list = JSON.parse(localStorage.getItem('rag_labor_history') || '[]');
        if (!list.length) {
          if (clearBtn) clearBtn.style.display = 'none';
          body.innerHTML = '<p style="color:var(--text-muted);font-size:0.88rem;">Lịch sử tra cứu trống.</p>';
          return;
        }
        if (clearBtn) clearBtn.style.display = 'inline-flex';
        body.innerHTML = `<div style="display:flex;flex-direction:column;gap:8px;">` +
          list.map((item, idx) => {
            const jsQ = item.query.replace(/'/g, "\\\\'");
            return `
              <div class="history-item-row">
                <button type="button" class="toc-art-btn" onclick="closeHistoryModal();askSample('${jsQ}')">
                  <strong>${escapeHtml(item.query)}</strong>
                  <span style="color:var(--text-subtle);font-size:0.74rem;white-space:nowrap;">${escapeHtml(item.time)}</span>
                </button>
                <button type="button" class="btn-del-item" title="Xóa mục này" onclick="deleteHistoryItem(${idx})">✕</button>
              </div>
            `;
          }).join('') + `</div>`;
      } catch (_) {}
    }

    function openHistoryModal() {
      document.getElementById('history-modal').classList.add('open');
      renderHistoryList();
    }

    function deleteHistoryItem(index) {
      try {
        const list = JSON.parse(localStorage.getItem('rag_labor_history') || '[]');
        list.splice(index, 1);
        localStorage.setItem('rag_labor_history', JSON.stringify(list));
        renderHistoryList();
      } catch (_) {}
    }

    function clearAllHistory() {
      try {
        localStorage.removeItem('rag_labor_history');
        renderHistoryList();
      } catch (_) {}
    }

    function closeHistoryModal() {
      document.getElementById('history-modal').classList.remove('open');
    }

    /* ==================== GỬI CÂU HỎI & STREAMING ==================== */
    function askSample(text) {
      queryInput.value = text;
      sendChat(new Event('submit'));
    }

    function resetConversation() {
      chatBox.querySelectorAll('.user-row, .bot-group').forEach(el => el.remove());
      const hero = document.getElementById('welcome-hero');
      if (hero) hero.style.display = 'block';
      queryInput.focus();
    }

    async function sendChat(e) {
      if (e && e.preventDefault) e.preventDefault();
      const q = queryInput.value.trim();
      if (!q) return;

      const directArtMatch = q.match(/^điều\\s+(\\d{1,3})$/i);
      if (directArtMatch) {
        showArticlePopover(null, parseInt(directArtMatch[1], 10), '1');
      }

      saveHistoryItem(q);
      const hero = document.getElementById('welcome-hero');
      if (hero) hero.style.display = 'none';

      const userRow = document.createElement('div');
      userRow.className = 'user-row';
      userRow.innerHTML = `<div class="user-pill">${escapeHtml(q)}</div>`;
      chatBox.appendChild(userRow);

      queryInput.value = '';
      sendBtn.disabled = true;

      const botGroup = document.createElement('div');
      botGroup.className = 'bot-group';
      botGroup.innerHTML = `
        <div class="bot-card">
          <div class="bot-card-header">
            <span class="bot-label">Trợ lý pháp luật</span>
          </div>
          <div class="progress-steps">
            <span class="step-1 step-active">Đang tìm điều luật liên quan...</span>
            <span>→</span>
            <span class="step-2">Đang soạn câu trả lời</span>
          </div>
          <div class="answer-content"></div>
          <div class="card-meta-row" style="display:none;"></div>
          <div class="disclaimer-row" style="display:none;">Thông tin tham khảo, không phải tư vấn pháp lý chính thức</div>
        </div>
        <div class="followup-slot"></div>
      `;
      chatBox.appendChild(botGroup);
      mainScroll.scrollTop = mainScroll.scrollHeight;

      const progressEl = botGroup.querySelector('.progress-steps');
      const step1El = botGroup.querySelector('.step-1');
      const step2El = botGroup.querySelector('.step-2');
      const answerEl = botGroup.querySelector('.answer-content');
      const metaRow = botGroup.querySelector('.card-meta-row');
      const disclaimerRow = botGroup.querySelector('.disclaimer-row');
      const followupSlot = botGroup.querySelector('.followup-slot');

      let accumulatedText = '';

      try {
        const res = await fetch('/api/chat/stream', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: q, top_k: 3 })
        });

        const reader = res.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\\n');
          buffer = lines.pop() || '';

          for (const rawLine of lines) {
            const line = rawLine.trim();
            if (!line.startsWith('data:')) continue;
            const jsonStr = line.slice(5).trim();
            if (!jsonStr) continue;
            const evt = JSON.parse(jsonStr);

            if (evt.type === 'meta') {
              step1El.classList.remove('step-active');
              step2El.classList.add('step-active');
              registerSources(evt.sources || []);
            } else if (evt.type === 'token') {
              if (progressEl) progressEl.style.display = 'none';
              accumulatedText += (evt.delta || '');
              answerEl.innerHTML = formatStructuredAnswer(accumulatedText);
              mainScroll.scrollTop = mainScroll.scrollHeight;
            } else if (evt.type === 'done' && evt.result) {
              if (progressEl) progressEl.style.display = 'none';
              const r = evt.result;
              registerSources(r.sources || []);
              answerEl.innerHTML = formatStructuredAnswer(r.answer);

              if (r.status !== 'refused' && Array.isArray(r.sources) && r.sources.length > 0) {
                const sourcesHtml = renderCitedSources(r.sources, r.cited_articles);
                const encQ = encodeURIComponent(q);
                const encA = encodeURIComponent(r.answer || '');
                metaRow.style.display = 'flex';
                metaRow.innerHTML = `
                  ${sourcesHtml}
                  <div class="feedback-group">
                    <button type="button" class="action-pill" onclick="submitFeedback(this, 'helpful', '${encQ}', '${encA}')">Hữu ích</button>
                    <button type="button" class="action-pill" onclick="submitFeedback(this, 'report', '${encQ}', '${encA}')">Báo sai</button>
                  </div>
                `;
                disclaimerRow.style.display = 'block';
                followupSlot.innerHTML = renderFollowUps(r.follow_ups || []);

                // Tự động mở chip đầu tiên trong câu trả lời ở Panel bên phải
                const firstChip = answerEl.querySelector('.cite-chip');
                if (firstChip) {
                  firstChip.classList.add('active');
                  const aNo = parseInt(firstChip.getAttribute('data-article') || '0', 10);
                  const cCsv = firstChip.getAttribute('data-clauses') || '1';
                  if (aNo > 0) showArticlePopover(null, aNo, cCsv);
                  firstChip.classList.add('active');
                } else {
                  showArticlePopover(null, r.sources[0].article_no, '1');
                }
              }
            }
          }
        }
      } catch (err) {
        if (progressEl) progressEl.style.display = 'none';
        answerEl.textContent = 'Đã xảy ra lỗi kết nối máy chủ: ' + err.message;
      } finally {
        sendBtn.disabled = false;
        mainScroll.scrollTop = mainScroll.scrollHeight;
      }
    }

    window.addEventListener('DOMContentLoaded', () => {
      currentTurnCitedClausesByArt = { 169: new Set(['2', '3', '4']) };
      showArticlePopover(null, 169, '2');
    });
  </script>
</body>
</html>
"""


# --------------------------------------------------------------------------
# CÁC ĐIỂM CUỐI API (ENDPOINTS)
# --------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home_page() -> HTMLResponse:
    """Trang chủ giao diện Web Chatbot."""
    return HTMLResponse(content=HTML_PAGE)


@app.get("/api/health")
def health_check() -> dict:
    """Kiểm tra trạng thái hoạt động của server và số lượng chunk trong chỉ mục."""
    gen = get_generator()
    return {
        "status": "ok",
        "law_name": "Bộ luật Lao động 2019 (18/VBHN-VPQH)",
        "chunks_count": len(gen.retriever.chunks),
        "model": gen.model_name,
        "min_score": gen.min_score,
    }


@app.get("/api/article/{article_no}")
def api_get_article(article_no: int) -> dict:
    """
    Trả về đầy đủ 100% nguyên văn các khoản của một Điều luật bất kỳ theo số Điều (`article_no`).
    """
    gen = get_generator()
    raw_parts = gen.retriever.by_article.get(int(article_no), [])
    if not raw_parts:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy Điều {article_no} trong dữ liệu hiện có.")
    merged_list = deduplicate_by_article(raw_parts, gen.retriever.by_doc_article)
    ui_list = format_sources_for_ui(merged_list)
    return ui_list[0]


@app.get("/api/toc")
def api_get_toc() -> dict:
    """
    Trả về Mục lục toàn bộ 17 Chương và 220 Điều của Bộ luật Lao động 2019 để tra cứu nhanh.
    """
    global _toc_cache
    if _toc_cache is None:
        gen = get_generator()
        chapters_map: dict[str, dict] = {}
        seen_art: set[int] = set()
        for c in gen.retriever.chunks:
            art_no = int(c["article_no"])
            if art_no in seen_art:
                continue
            seen_art.add(art_no)
            ch_no = str(c.get("chapter_no") or "")
            ch_title = str(c.get("chapter_title") or "")
            if ch_no not in chapters_map:
                chapters_map[ch_no] = {
                    "chapter_no": ch_no,
                    "chapter_title": ch_title,
                    "articles": [],
                }
            chapters_map[ch_no]["articles"].append(
                {
                    "article_no": art_no,
                    "article_title": c.get("article_title", ""),
                }
            )
        _toc_cache = list(chapters_map.values())
    return {"chapters": _toc_cache}


@app.post("/api/feedback")
def api_submit_feedback(req: FeedbackRequest) -> dict:
    """
    Ghi nhận đánh giá 'Hữu ích' / 'Báo sai' của người dùng vào file data/processed/feedback_log.jsonl.
    """
    FEEDBACK_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": req.query,
        "rating": req.rating,
        "note": req.note,
        "cited_articles": req.cited_articles,
        "answer_preview": (req.answer or "")[:300],
    }
    with FEEDBACK_LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"status": "recorded", "rating": req.rating}


@app.post("/api/retrieve")
def api_retrieve(req: ChatRequest) -> dict:
    """
    API chỉ tìm kiếm và trả về các điều luật liên quan đã khử trùng lặp và gộp đủ nội dung.
    """
    q = (req.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống.")
    gen = get_generator()
    raw_results = gen.retriever.retrieve(
        q,
        top_k=max(req.top_k + 2, 5),
        use_vector=req.use_vector,
        min_score=req.min_score,
    )
    results = deduplicate_by_article(raw_results, gen.retriever.by_doc_article)[:req.top_k]
    return {
        "query": q,
        "count": len(results),
        "timings_ms": gen.retriever.last_timings,
        "results": results,
    }


@app.post("/api/chat")
def api_chat(req: ChatRequest) -> dict:
    """
    API hỏi đáp đồng bộ của Web Chatbot (trả về JSON hoàn chỉnh kèm `metrics.timings_ms`).
    """
    gen = get_generator()
    try:
        return gen.answer(
            req.query,
            top_k=req.top_k,
            use_vector=req.use_vector,
            min_score=req.min_score,
            fallback_on_error=True,
        )
    except Exception:
        q = (req.query or "").strip()
        raw_chunks = gen.retriever.retrieve(
            q,
            top_k=max(req.top_k + 2, 5),
            use_vector=False,
            min_score=req.min_score,
        )
        deduped = deduplicate_by_article(raw_chunks, gen.retriever.by_doc_article)[:req.top_k]
        return build_fallback_response(
            query=q,
            deduped_chunks=deduped,
            min_score=req.min_score,
            elapsed_seconds=0.0,
        )


@app.post("/api/chat/stream")
def api_chat_stream(req: ChatRequest) -> StreamingResponse:
    """
    API hỏi đáp theo luồng (Server-Sent Events Streaming) từ backend tới giao diện Web.
    """
    gen = get_generator()

    def event_stream() -> Iterator[str]:
        for evt in gen.stream_answer(
            req.query,
            top_k=req.top_k,
            use_vector=req.use_vector,
            min_score=req.min_score,
        ):
            payload = json.dumps(evt, ensure_ascii=False)
            yield f"data: {payload}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ==========================================================================
# KHỞI CHẠY SERVER KHI CHẠY `python src/app.py`
# ==========================================================================
def main() -> int:
    import uvicorn

    print("Đang khởi động Web Chatbot tại: http://127.0.0.1:8000")
    uvicorn.run("src.app:app", host="127.0.0.1", port=8000, reload=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
