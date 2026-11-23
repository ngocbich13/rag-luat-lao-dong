"""
generator.py — GIAI ĐOẠN 1, 2 & 5: SINH CÂU TRẢ LỜI VỚI GEMINI LLM, STREAMING, CACHE & ĐO ĐỘ TRỄ
==================================================================================================
Nhận câu hỏi của người dùng và các điều luật tìm được từ `src/retriever.py`,
sau đó gọi mô hình ngôn ngữ Gemini để trả lời ngắn gọn, chính xác, có trích dẫn
Điều/Khoản/Tên văn bản và nguyên văn căn cứ pháp lý.

Các tính năng Giai đoạn 1 & Giai đoạn 2:
    1. Ngưỡng "Không tìm thấy" (`min_score`, cấu hình qua `RAG_MIN_SCORE`):
       Trả lời rõ "Tôi chưa tìm thấy căn cứ pháp lý phù hợp trong dữ liệu hiện có." khi dưới ngưỡng.
    2. Khử trùng lặp & hợp nhất đầy đủ nội dung Điều luật (`deduplicate_by_article`).
    3. Đo thời gian (log) chi tiết cho từng bước (`metrics["timings_ms"]`):
       - `rewrite_ms`: Viết lại câu hỏi thành truy vấn pháp lý
       - `embedding_ms`: Tạo vector embedding cho câu hỏi
       - `search_ms`: Tìm kiếm BM25 + Vector
       - `rerank_ms`: Chấm điểm Rerank & gộp điều luật
       - `llm_ttft_ms`: Thời gian tới token đầu tiên của LLM
       - `llm_total_ms`: Tổng thời gian gọi LLM
       - `ttft_ms`: Tổng thời gian từ lúc nhận câu hỏi tới token đầu tiên
       - `total_ms`: Tổng thời gian phản hồi
    4. Bật Streaming câu trả lời (`stream_answer`) từ backend tới giao diện Web.
    5. Đặt timeout ngắn (10s), hạn chế retry vô ích (`attempts=1`), giảm `top_k=3` đoạn luật
       đưa vào prompt, và cache kết quả (`_response_cache`) cho các câu hỏi lặp lại.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Callable, Iterator

from dotenv import load_dotenv

# Nạp GEMINI_API_KEY và RAG_MIN_SCORE từ file .env
load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retriever import (
    DEFAULT_MIN_SCORE,
    HybridRetriever,
    deduplicate_by_article,
    rewrite_query,
)

logger = logging.getLogger("rag_luat_lao_dong")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# --------------------------------------------------------------------------
# CẤU HÌNH MÔ HÌNH, NGƯỠNG ĐIỂM VÀ PROMPT HỆ THỐNG
# --------------------------------------------------------------------------
DEFAULT_LLM_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
DEFAULT_TOP_K = 3
MAX_CACHE_SIZE = 128

NOT_FOUND_MESSAGE = "Mình chưa tìm thấy căn cứ trong dữ liệu hiện có."

FALLBACK_HEADER_MESSAGE = (
    "Hệ thống AI đang gián đoạn, dưới đây là các điều luật có thể liên quan (chưa được kiểm chứng):"
)

DISCLAIMER_TEXT = (
    "Lưu ý: Thông tin trên chỉ mang tính chất tham khảo dựa trên văn bản pháp luật hiện có "
    "trong hệ thống, không phải là tư vấn pháp lý chính thức."
)

SHORT_DISCLAIMER_LINE = (
    "Lưu ý: Thông tin trên chỉ mang tính chất tham khảo, không phải là tư vấn pháp lý chính thức."
)

SYSTEM_PROMPT = """Bạn là Trợ lý pháp luật tra cứu Bộ luật Lao động 2019 (Văn bản hợp nhất 18/VBHN-VPQH).
Hãy trả lời câu hỏi dựa tuyệt đối vào phần [TÀI LIỆU ĐIỀU LUẬT] được cung cấp theo đúng 6 quy tắc sau:

1. Dòng đầu tiên là KẾT LUẬN (không ghi chữ "Kết luận:"): tối đa 2 câu, trả lời trực diện có ngay con số/đáp án, IN ĐẬM toàn bộ câu kết luận bằng **...**, kết thúc bằng dấu chấm rồi mới đến thẻ trích dẫn dạng [khoản X Điều Y] (ví dụ: **Năm 2026, tuổi nghỉ hưu trong điều kiện lao động bình thường là 61 tuổi 6 tháng với nam và 57 tuổi với nữ.** [khoản 2 Điều 169]).
2. Theo sau là tối đa 3 gạch đầu dòng chi tiết ("- ..."), mỗi gạch đầu dòng DƯỚI 25 TỪ.
3. Tuyệt đối KHÔNG chép nguyên văn điều luật dài dòng, KHÔNG nhắc lại nội dung kết luận trong phần gạch đầu dòng.
4. CHỈ trả lời đúng điều được hỏi. Nếu có phần thông tin mà điều luật dẫn chiếu sang văn bản khác chưa có trong tài liệu (như Luật Bảo hiểm xã hội hoặc văn bản hướng dẫn của Chính phủ), chỉ viết MỘT câu thường ngắn gọn ngay trong gạch đầu dòng liên quan (ví dụ: "hệ thống chưa có văn bản hướng dẫn nên chưa nêu chi tiết được." hoặc "hệ thống chưa có Luật Bảo hiểm xã hội nên chưa nêu số năm đóng tối thiểu được.").
5. Nếu số liệu trong kết luận do tính toán từ công thức trong luật, gạch đầu dòng đầu tiên phải bắt đầu bằng "- Cách tính: ..." nêu ngắn gọn công thức đó (không gắn lại chip trích dẫn đã dùng ở kết luận).
6. Quy tắc gắn trích dẫn [khoản X Điều Y]:
   - Mỗi gạch đầu dòng chỉ gắn tối đa 1 trích dẫn đặt ở CUỐI câu sau dấu chấm, theo dạng [khoản X Điều Y] hoặc [khoản X, Y Điều Z].
   - KHÔNG lặp lại cùng một trích dẫn [khoản X Điều Y] hai lần trong cùng câu trả lời.
   - Nếu hoàn toàn không có căn cứ trong tài liệu được cung cấp, chỉ trả lời đúng một câu: "Mình chưa tìm thấy căn cứ trong dữ liệu hiện có." """


# --------------------------------------------------------------------------
# 1. PHÁT HIỆN CÂU HỎI NGOÀI PHẠM VI & CỜ CẦN NGHỊ ĐỊNH (`needs_decree`)
# --------------------------------------------------------------------------
OUT_OF_SCOPE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(
            r"\b(thuế thu nhập cá nhân|thuế tncn|quyết toán thuế|giảm trừ gia cảnh|thuế gtgt|thuế giá trị gia tăng)\b",
            re.IGNORECASE,
        ),
        "Câu hỏi thuộc lĩnh vực pháp luật về Thuế.",
    ),
    (
        re.compile(
            r"\b(sổ đỏ|sổ hồng|tranh chấp đất đai|thu hồi đất|luật đất đai|mua bán nhà đất)\b",
            re.IGNORECASE,
        ),
        "Câu hỏi thuộc lĩnh vực Luật Đất đai / Nhà ở.",
    ),
    (
        re.compile(
            r"\b(ly hôn|chia tài sản vợ chồng|quyền nuôi con sau ly hôn|luật hôn nhân)\b",
            re.IGNORECASE,
        ),
        "Câu hỏi thuộc lĩnh vực Luật Hôn nhân và Gia đình.",
    ),
    (
        re.compile(
            r"\b(khởi tố hình sự|tội lừa đảo|đi tù mấy năm|bộ luật hình sự|vi phạm giao thông|phạt nồng độ cồn|vượt đèn đỏ|nồng độ cồn)\b",
            re.IGNORECASE,
        ),
        "Câu hỏi thuộc lĩnh vực Hình sự / Hành chính giao thông.",
    ),
]

LABOR_KEYWORDS = re.compile(
    r"\b(hợp đồng lao động|người lao động|người sử dụng lao động|thử việc|trợ cấp thôi việc|sa thải|làm thêm giờ|thai sản|nghỉ phép|kỷ luật lao động|lương hưu|nghỉ hưu|bảo hiểm xã hội|điều\s+\d+)\b",
    re.IGNORECASE,
)

RE_NEEDS_DECREE = re.compile(
    r"chính phủ quy định chi tiết|theo quy định của chính phủ",
    re.IGNORECASE,
)


def compute_retirement_age_note(query: str) -> str | None:
    """
    Nếu người dùng hỏi tuổi nghỉ hưu của một năm cụ thể (ví dụ năm 2026, 2030),
    tính chính xác số tuổi và số tháng theo công thức tại khoản 2 Điều 169 BLLĐ 2019:
        - Năm 2021: Nam 60 tuổi 3 tháng (+3 tháng/năm tới 62 tuổi năm 2028)
        - Năm 2021: Nữ 55 tuổi 4 tháng (+4 tháng/năm tới 60 tuổi năm 2035)
    """
    q_low = (query or "").lower()
    if "hưu" not in q_low:
        return None
    m = re.search(r"\b(202[1-9]|203[0-9]|204[0-9])\b", q_low)
    if not m:
        if "lương hưu" in q_low:
            return (
                "- HƯỚNG DẪN TRẢ LỜI NGẮN GỌN: Dòng 1 in đậm nêu kết luận từ khoản 1 Điều 169 kèm [khoản 1 Điều 169]. "
                "Theo sau là 2 gạch đầu dòng (<25 từ/gạch): gạch 1 nêu lộ trình tuổi nghỉ hưu [khoản 2 Điều 169]; "
                "gạch 2 nêu rõ: 'Số năm đóng bảo hiểm xã hội tối thiểu để hưởng lương hưu thuộc Luật Bảo hiểm xã hội; "
                "hệ thống chưa có văn bản này nên chưa nêu chi tiết được.'"
            )
        return None
    year = int(m.group(1))
    diff = year - 2021
    male_months = min(62 * 12, (60 * 12 + 3) + diff * 3)
    female_months = min(60 * 12, (55 * 12 + 4) + diff * 4)

    m_y, m_m = divmod(male_months, 12)
    f_y, f_m = divmod(female_months, 12)
    male_str = f"{m_y} tuổi {m_m} tháng" if m_m > 0 else f"đủ {m_y} tuổi"
    female_str = f"{f_y} tuổi {f_m} tháng" if f_m > 0 else f"{f_y} tuổi"
    return (
        f"- HƯỚNG DẪN TÍNH TOÁN & ĐỊNH DẠNG BẮT BUỘC CHO NĂM {year}:\n"
        f"  + Dòng 1 (Kết luận in đậm): **Năm {year}, tuổi nghỉ hưu trong điều kiện lao động bình thường là {male_str} với nam và {female_str} với nữ.** [khoản 2 Điều 169]\n"
        f"  + Gạch 1: - Cách tính: từ năm 2021 là 60 tuổi 3 tháng (nam) và 55 tuổi 4 tháng (nữ), mỗi năm tăng thêm 3 tháng (nam) và 4 tháng (nữ).\n"
        f"  + Gạch 2: - Một số trường hợp đặc biệt có thể nghỉ sớm hoặc muộn hơn tối đa 5 tuổi; hệ thống chưa có văn bản hướng dẫn nên chưa nêu chi tiết được. [khoản 3, 4 Điều 169]"
    )


def build_warnings_and_followups(
    query: str,
    chunks: list[dict],
    needs_decree: bool,
) -> tuple[str | None, str | None, list[str]]:
    """
    Trả về (None, None, follow_ups):
        - Đã bỏ hoàn toàn khung cảnh báo màu vàng (NHÓM 1).
        - Sinh 2–3 câu hỏi gợi ý tiếp theo (follow_ups) cho các thông tin liên quan chưa được hỏi.
    """
    if not chunks:
        return None, None, []

    q_low = (query or "").lower()
    primary_art = int(chunks[0].get("article_no", 0))

    if primary_art in (168, 169):
        follow_ups = [
            "Nghỉ hưu sớm cần điều kiện gì?",
            "Cần đóng BHXH bao nhiêu năm để hưởng lương hưu?",
            "Lao động nữ nghỉ hưu năm 2030 bao nhiêu tuổi?",
        ]
    elif primary_art in (35, 36, 37):
        follow_ups = [
            "Trường hợp nào người lao động được nghỉ việc ngay không cần báo trước?",
            "Nghỉ việc có được nhận trợ cấp thôi việc theo Điều 46 không?",
            "Người sử dụng lao động đơn phương chấm dứt hợp đồng cần báo trước mấy ngày?",
        ]
    elif primary_art in (46, 47):
        follow_ups = [
            "Trợ cấp mất việc làm theo Điều 47 tính mỗi năm bao nhiêu tháng lương?",
            "Khi chấm dứt hợp đồng, công ty phải thanh toán lương trong bao nhiêu ngày?",
        ]
    elif primary_art in (24, 25, 26, 27):
        follow_ups = [
            "Tiền lương trong thời gian thử việc ít nhất bằng bao nhiêu phần trăm?",
            "Trong thời gian thử việc có được chấm dứt ngay không cần báo trước không?",
        ]
    elif primary_art in (98, 105, 107):
        follow_ups = [
            "Số giờ làm thêm tối đa trong 1 tháng và 1 năm là bao nhiêu?",
            "Thời giờ làm việc bình thường tối đa mấy giờ trong 1 ngày và 1 tuần?",
        ]
    elif primary_art in (112, 113, 114, 115):
        follow_ups = [
            "Làm việc đủ 5 năm thì được cộng thêm mấy ngày nghỉ phép hằng năm?",
            "Kết hôn hoặc bố mẹ mất được nghỉ việc riêng hưởng nguyên lương mấy ngày?",
        ]
    elif primary_art in (122, 124, 125, 127):
        follow_ups = [
            "Những hành vi nào bị nghiêm cấm khi xử lý kỷ luật lao động?",
            "Trường hợp nào người lao động bị áp dụng hình thức kỷ luật sa thải?",
        ]
    else:
        follow_ups = [
            "Người lao động tự xin nghỉ việc cần báo trước bao nhiêu ngày?",
            "Cách tính trợ cấp thôi việc theo Điều 46 như thế nào?",
        ]

    follow_ups = [f for f in follow_ups if f.lower() != q_low][:2]
    return None, None, follow_ups

RE_DISPUTE_LAWSUIT = re.compile(
    r"\b(thắng kiện|thua kiện|kiện công ty có thắng|tôi có thắng không)\b",
    re.IGNORECASE,
)


def check_out_of_scope(query: str) -> str | None:
    """
    Kiểm tra nhanh xem câu hỏi có hoàn toàn nằm ngoài phạm vi pháp luật lao động không.
    Nếu ngoài phạm vi, trả về lý do; nếu hợp lệ, trả về None.
    """
    if LABOR_KEYWORDS.search(query):
        return None
    for pattern, reason in OUT_OF_SCOPE_PATTERNS:
        if pattern.search(query):
            return reason
    return None


def check_needs_decree(chunks: list[dict]) -> bool:
    """
    Kiểm tra xem Điều luật chính (top 1) có giao Chính phủ quy định chi tiết không.
    """
    if not chunks:
        return False
    primary_art = chunks[0].get("article_no")
    for c in chunks:
        if c.get("article_no") == primary_art and RE_NEEDS_DECREE.search(c.get("text", "")):
            return True
    return False


# --------------------------------------------------------------------------
# 2. DỰNG NGỮ CẢNH (CONTEXT) TỪ CÁC ĐIỀU LUẬT ĐÃ KHỬ TRÙNG LẶP
# --------------------------------------------------------------------------
def format_context(chunks: list[dict]) -> str:
    """
    Biến danh sách các chunk đã gộp theo Điều thành văn bản có nhãn nguồn rõ ràng để LLM đọc.
    """
    blocks: list[str] = []
    for idx, c in enumerate(chunks, start=1):
        law_name = c.get("law_name", "Bộ luật Lao động 2019")
        header = (
            f"--- [TÀI LIỆU {idx}] Văn bản: {law_name} | Nguồn trích dẫn: {c['citation']} "
            f"({c.get('full_path', '')} | {c.get('article_title', '')}) ---"
        )
        blocks.append(f"{header}\n{c['text']}")
    return "\n\n".join(blocks)


def format_sources_for_ui(chunks: list[dict]) -> list[dict]:
    """
    Chuẩn bị danh sách nguồn căn cứ pháp lý đầy đủ nội dung và không trùng lặp cho giao diện Web.
    """
    sources: list[dict] = []
    for c in chunks:
        sources.append(
            {
                "chunk_id": c["chunk_id"],
                "doc_id": c.get("doc_id", "45/2019/QH14"),
                "law_name": c.get("law_name", "Bộ luật Lao động 2019"),
                "chapter_no": c.get("chapter_no", ""),
                "article_no": c["article_no"],
                "clause_nos": c.get("clause_nos", []),
                "article_title": c.get("article_title", ""),
                "citation": c["citation"],
                "full_path": c.get("full_path", ""),
                "text": c["text"],
                "score": c.get("score", 0.0),
                "match_type": c.get("match_type", "hybrid"),
            }
        )
    return sources


def build_fallback_response(
    query: str,
    deduped_chunks: list[dict],
    min_score: float = DEFAULT_MIN_SCORE,
    elapsed_seconds: float = 0.0,
    timings_ms: dict[str, float] | None = None,
) -> dict:
    """
    Tạo phản hồi Fallback trung thực khi LLM gặp lỗi hoặc timeout (Yêu cầu 6 - Giai đoạn 1):
        - Chỉ hiển thị điều luật khi điểm liên quan >= min_score.
        - Không bao giờ hiển thị trùng lặp cùng một điều luật 2 lần.
        - Nếu không có điều luật nào đạt ngưỡng, trả về thông báo không tìm thấy căn cứ.
    """
    valid_chunks = [c for c in deduped_chunks if float(c.get("score", 0.0)) >= min_score]
    t_info = timings_ms or {
        "rewrite_ms": 0.0,
        "embedding_ms": 0.0,
        "search_ms": 0.0,
        "rerank_ms": 0.0,
        "llm_ttft_ms": 0.0,
        "llm_total_ms": 0.0,
        "ttft_ms": round(elapsed_seconds * 1000.0, 2),
        "total_ms": round(elapsed_seconds * 1000.0, 2),
    }

    if not valid_chunks:
        return {
            "query": query,
            "answer": NOT_FOUND_MESSAGE,
            "status": "refused",
            "needs_decree": False,
            "disclaimer": DISCLAIMER_TEXT,
            "sources": [],
            "cited_articles": [],
            "metrics": {
                "input_tokens": 0,
                "output_tokens": 0,
                "elapsed_seconds": elapsed_seconds,
                "cache_hit": False,
                "timings_ms": t_info,
            },
        }

    lines = [FALLBACK_HEADER_MESSAGE, ""]
    for idx, c in enumerate(valid_chunks, start=1):
        title = c.get("article_title", "")
        score = c.get("score", 0.0)
        lines.append(f"{idx}. {c['citation']} — {title} (điểm liên quan: {score:.2f})")

    lines.append("")
    lines.append(SHORT_DISCLAIMER_LINE)

    return {
        "query": query,
        "answer": "\n".join(lines),
        "status": "fallback",
        "needs_decree": check_needs_decree(valid_chunks),
        "disclaimer": DISCLAIMER_TEXT,
        "sources": format_sources_for_ui(valid_chunks),
        "cited_articles": [int(c["article_no"]) for c in valid_chunks],
        "metrics": {
            "input_tokens": 0,
            "output_tokens": 0,
            "elapsed_seconds": elapsed_seconds,
            "cache_hit": False,
            "timings_ms": t_info,
        },
    }


# --------------------------------------------------------------------------
# 3. LỚP RAG GENERATOR (RETRIEVER + STREAMING GEMINI LLM + CACHE + TIMINGS)
# --------------------------------------------------------------------------
class RAGGenerator:
    """
    Bộ sinh câu trả lời RAG cho hệ thống tư vấn pháp luật lao động.
    Hỗ trợ:
        - Đo thời gian chi tiết từng bước (`timings_ms`)
        - Streaming câu trả lời theo thời gian thực (`stream_answer`)
        - Cache câu hỏi lặp lại (`_response_cache`)
        - Truyền `llm_fn` tùy chỉnh để viết unit test nhanh
    """

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        model_name: str = DEFAULT_LLM_MODEL,
        llm_fn: Callable[[str, str], tuple[str, int, int]] | None = None,
        min_score: float = DEFAULT_MIN_SCORE,
    ):
        self.retriever = retriever or HybridRetriever.from_files()
        self.model_name = model_name
        self._llm_fn = llm_fn
        self.min_score = min_score
        self._genai_client = None
        self._call_counter = 0
        self._response_cache: OrderedDict[tuple[str, int, float], dict] = OrderedDict()

    def _get_client(self):
        """
        Khởi tạo Gemini Client 1 lần duy nhất (Singleton) với timeout ngắn (10s)
        và `attempts=1` để hạn chế retry vô ích.
        """
        if self._genai_client is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "Thiếu GEMINI_API_KEY trong file .env. Hãy kiểm tra lại file .env."
                )
            from google import genai
            from google.genai import types

            http_opts = types.HttpOptions(
                timeout=10000,
                retry_options=types.HttpRetryOptions(attempts=1),
            )
            self._genai_client = genai.Client(api_key=api_key, http_options=http_opts)
        return self._genai_client

    def warmup(self) -> None:
        """
        Khởi tạo sẵn kết nối tới Gemini API khi khởi động server để request đầu tiên
        không phải chờ bắt tay TLS/HTTPS.
        """
        if self._llm_fn is not None:
            return
        try:
            self._get_client()
            from src.indexer import get_embed_client

            get_embed_client()
        except Exception:
            pass

    def clear_cache(self) -> None:
        """Xóa bộ nhớ đệm câu trả lời."""
        self._response_cache.clear()

    def _build_user_prompt(
        self,
        query: str,
        retrieved_chunks: list[dict],
        needs_decree: bool,
        is_dispute: bool,
    ) -> str:
        """Tạo user_prompt gọn sạch từ 3-5 đoạn luật đã khử trùng lặp."""
        context_str = format_context(retrieved_chunks)
        extra_instructions: list[str] = []
        ret_note = compute_retirement_age_note(query)
        if ret_note:
            extra_instructions.append(ret_note)
        if needs_decree:
            extra_instructions.append(
                "- LƯU Ý: Điều luật có giao Chính phủ quy định chi tiết hoặc dẫn chiếu sang Luật Bảo hiểm xã hội. "
                "Hãy trình bày đúng quy định có trong Bộ luật Lao động 2019."
            )
        if is_dispute:
            extra_instructions.append(
                "- LƯU Ý: Người dùng hỏi về tranh chấp/kiện tụng. Không kết luận thắng hay thua kiện; "
                "chỉ nêu căn cứ pháp lý liên quan và khuyên liên hệ cơ quan lao động, công đoàn hoặc luật sư."
            )

        extra_block = "\n".join(extra_instructions)
        return (
            f"[TÀI LIỆU ĐIỀU LUẬT]\n{context_str}\n\n"
            f"[CÂU HỎI CỦA NGƯỜI DÙNG]\n{query}\n\n"
            f"{extra_block}\n"
            "Hãy trả lời thật ngắn gọn theo đúng quy tắc: dòng 1 là kết luận in đậm kèm [khoản X Điều Y], "
            "theo sau tối đa 3 gạch đầu dòng (<25 từ/gạch), không lặp lại chip trích dẫn đã dùng."
        )

    def _call_gemini_with_ttft(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[str, int, int, float, float]:
        """
        Gọi Gemini bằng `generate_content_stream` để đo chính xác:
            - `llm_ttft_ms`: Thời gian tới token đầu tiên (Time To First Token)
            - `llm_total_ms`: Tổng thời gian sinh xong câu trả lời
        Trả về: (text, in_tokens, out_tokens, llm_ttft_ms, llm_total_ms).
        """
        t0 = time.perf_counter()
        if self._llm_fn is not None:
            text, in_tok, out_tok = self._llm_fn(system_prompt, user_prompt)
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return text, in_tok, out_tok, elapsed_ms, elapsed_ms

        from google.genai import types

        client = self._get_client()
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.1,
            max_output_tokens=1200,
        )

        # Phân tải thông minh (Round-robin) qua pool model Flash tốc độ cao để không chạm trần 15 RPM của gói Free Tier
        fast_pool = ["gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.5-flash"]
        if self.model_name not in fast_pool:
            fast_pool.insert(0, self.model_name)
        start_idx = self._call_counter % len(fast_pool)
        self._call_counter += 1
        candidate_models = fast_pool[start_idx:] + fast_pool[:start_idx]
        last_err = None

        for model_candidate in candidate_models:
            chunks_text: list[str] = []
            ttft_ms: float | None = None
            in_tokens = 0
            out_tokens = 0
            t_attempt = time.perf_counter()
            try:
                for chunk in client.models.generate_content_stream(
                    model=model_candidate,
                    contents=user_prompt,
                    config=config,
                ):
                    piece = chunk.text or ""
                    if piece:
                        if ttft_ms is None:
                            ttft_ms = round((time.perf_counter() - t_attempt) * 1000.0, 2)
                        chunks_text.append(piece)
                    usage = getattr(chunk, "usage_metadata", None)
                    if usage:
                        in_tokens = int(getattr(usage, "prompt_token_count", 0) or in_tokens)
                        out_tokens = int(getattr(usage, "candidates_token_count", 0) or out_tokens)

                total_ms = round((time.perf_counter() - t_attempt) * 1000.0, 2)
                if ttft_ms is None:
                    ttft_ms = total_ms
                full_text = "".join(chunks_text).strip()
                if full_text:
                    return full_text, in_tokens, out_tokens, ttft_ms, total_ms
            except Exception as err:
                last_err = err
                continue

        if last_err is not None:
            raise last_err
        return "", 0, 0, 0.0, 0.0

    def _finalize_answer_text(
        self,
        answer_text: str,
        needs_decree: bool,
    ) -> tuple[str, str]:
        """Chuẩn hóa trạng thái và đảm bảo câu trả lời luôn có dòng lưu ý pháp lý ở cuối."""
        lower_ans = answer_text.lower()
        if (
            "chưa tìm thấy căn cứ pháp lý phù hợp" in lower_ans
            or "không tìm thấy căn cứ pháp lý" in lower_ans
            or "chưa tìm thấy căn cứ trong dữ liệu hiện có" in lower_ans
        ):
            return NOT_FOUND_MESSAGE, "refused"

        status = "needs_decree" if needs_decree else "answered"

        if (
            "không phải là tư vấn pháp lý chính thức" not in answer_text.lower()
            and "không phải tư vấn pháp lý chính thức" not in answer_text.lower()
        ):
            answer_text = f"{answer_text}\n\n{SHORT_DISCLAIMER_LINE}"

        return answer_text, status

    def answer(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        use_vector: bool = True,
        min_score: float | None = None,
        fallback_on_error: bool = True,
        use_cache: bool = True,
    ) -> dict:
        """
        Quy trình trả lời hoàn chỉnh cho 1 câu hỏi của người dùng kèm đo thời gian từng bước:
            1. Kiểm tra Cache cho câu hỏi lặp lại (`< 1ms`).
            2. Kiểm tra câu hỏi rỗng hoặc ngoài phạm vi.
            3. Truy xuất song song BM25 + Vector, lọc theo `min_score` và khử trùng lặp Điều.
            4. Gọi Gemini (streaming nội bộ) để đo chính xác `llm_ttft_ms` và `llm_total_ms`.
            5. Ghi log thời gian từng bước và lưu kết quả vào Cache.
        """
        start_time = time.perf_counter()
        threshold = self.min_score if min_score is None else float(min_score)
        q = (query or "").strip()

        if not q:
            empty_timings = {
                "rewrite_ms": 0.0,
                "embedding_ms": 0.0,
                "search_ms": 0.0,
                "rerank_ms": 0.0,
                "llm_ttft_ms": 0.0,
                "llm_total_ms": 0.0,
                "ttft_ms": 0.0,
                "total_ms": 0.0,
            }
            return {
                "query": query,
                "answer": NOT_FOUND_MESSAGE,
                "status": "refused",
                "needs_decree": False,
                "disclaimer": DISCLAIMER_TEXT,
                "sources": [],
                "cited_articles": [],
                "metrics": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "elapsed_seconds": 0.0,
                    "cache_hit": False,
                    "timings_ms": empty_timings,
                },
            }

        # Bước 1: Kiểm tra bộ nhớ đệm (Cache) cho câu hỏi lặp lại
        cache_key = (q.lower(), int(top_k), round(threshold, 3))
        if use_cache and cache_key in self._response_cache:
            cached = dict(self._response_cache[cache_key])
            cache_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            cached_metrics = dict(cached.get("metrics", {}))
            cached_metrics["elapsed_seconds"] = round(cache_ms / 1000.0, 4)
            cached_metrics["cache_hit"] = True
            cached_metrics["timings_ms"] = {
                "rewrite_ms": 0.0,
                "embedding_ms": 0.0,
                "search_ms": 0.0,
                "rerank_ms": 0.0,
                "llm_ttft_ms": 0.0,
                "llm_total_ms": 0.0,
                "ttft_ms": cache_ms,
                "total_ms": cache_ms,
            }
            cached["metrics"] = cached_metrics
            logger.info("CACHE HIT cho câu hỏi '%s' trong %.2f ms", q, cache_ms)
            return cached

        # Chốt chặn ngoài phạm vi nhanh
        out_reason = check_out_of_scope(q)
        if out_reason is not None:
            total_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            elapsed = round(total_ms / 1000.0, 3)
            timings = {
                "rewrite_ms": total_ms,
                "embedding_ms": 0.0,
                "search_ms": 0.0,
                "rerank_ms": 0.0,
                "llm_ttft_ms": 0.0,
                "llm_total_ms": 0.0,
                "ttft_ms": total_ms,
                "total_ms": total_ms,
            }
            res = {
                "query": q,
                "answer": NOT_FOUND_MESSAGE,
                "status": "refused",
                "needs_decree": False,
                "disclaimer": DISCLAIMER_TEXT,
                "sources": [],
                "cited_articles": [],
                "metrics": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "elapsed_seconds": elapsed,
                    "cache_hit": False,
                    "timings_ms": timings,
                },
            }
            if use_cache:
                self._save_cache(cache_key, res)
            return res

        # Bước 2: Tìm kiếm các đoạn luật có điểm >= threshold và gộp/khử trùng lặp theo Điều
        raw_chunks = self.retriever.retrieve(
            q,
            top_k=max(top_k + 2, 5),
            expand_references=True,
            max_ref_chunks=1,
            use_vector=use_vector,
            min_score=threshold,
        )
        t_dedup0 = time.perf_counter()
        retrieved_chunks = deduplicate_by_article(
            raw_chunks,
            self.retriever.by_doc_article,
        )[:top_k]
        dedup_ms = (time.perf_counter() - t_dedup0) * 1000.0

        ret_timings = dict(self.retriever.last_timings)
        rewrite_ms = ret_timings.get("rewrite_ms", 0.0)
        embedding_ms = ret_timings.get("embedding_ms", 0.0)
        search_ms = ret_timings.get("search_ms", 0.0)
        rerank_ms = round(ret_timings.get("rerank_ms", 0.0) + dedup_ms, 2)
        pre_llm_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        if not retrieved_chunks:
            timings = {
                "rewrite_ms": rewrite_ms,
                "embedding_ms": embedding_ms,
                "search_ms": search_ms,
                "rerank_ms": rerank_ms,
                "llm_ttft_ms": 0.0,
                "llm_total_ms": 0.0,
                "ttft_ms": pre_llm_ms,
                "total_ms": pre_llm_ms,
            }
            res = {
                "query": q,
                "answer": NOT_FOUND_MESSAGE,
                "status": "refused",
                "needs_decree": False,
                "disclaimer": DISCLAIMER_TEXT,
                "sources": [],
                "cited_articles": [],
                "metrics": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "elapsed_seconds": round(pre_llm_ms / 1000.0, 3),
                    "cache_hit": False,
                    "timings_ms": timings,
                },
            }
            if use_cache:
                self._save_cache(cache_key, res)
            return res

        needs_decree = check_needs_decree(retrieved_chunks[:2])
        if "tuổi nghỉ hưu" in q.lower() and re.search(r"\b20\d{2}\b", q):
            needs_decree = False
        is_dispute = bool(RE_DISPUTE_LAWSUIT.search(q))
        user_prompt = self._build_user_prompt(q, retrieved_chunks, needs_decree, is_dispute)

        # Bước 3: Gọi Gemini LLM và đo TTFT + tổng thời gian LLM
        try:
            raw_ans, in_tok, out_tok, llm_ttft_ms, llm_total_ms = self._call_gemini_with_ttft(
                SYSTEM_PROMPT,
                user_prompt,
            )
        except Exception:
            if not fallback_on_error:
                raise
            total_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            timings = {
                "rewrite_ms": rewrite_ms,
                "embedding_ms": embedding_ms,
                "search_ms": search_ms,
                "rerank_ms": rerank_ms,
                "llm_ttft_ms": round(total_ms - pre_llm_ms, 2),
                "llm_total_ms": round(total_ms - pre_llm_ms, 2),
                "ttft_ms": total_ms,
                "total_ms": total_ms,
            }
            return build_fallback_response(
                query=q,
                deduped_chunks=retrieved_chunks,
                min_score=threshold,
                elapsed_seconds=round(total_ms / 1000.0, 2),
                timings_ms=timings,
            )

        if not raw_ans:
            total_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            timings = {
                "rewrite_ms": rewrite_ms,
                "embedding_ms": embedding_ms,
                "search_ms": search_ms,
                "rerank_ms": rerank_ms,
                "llm_ttft_ms": llm_ttft_ms,
                "llm_total_ms": llm_total_ms,
                "ttft_ms": round(pre_llm_ms + llm_ttft_ms, 2),
                "total_ms": total_ms,
            }
            return build_fallback_response(
                query=q,
                deduped_chunks=retrieved_chunks,
                min_score=threshold,
                elapsed_seconds=round(total_ms / 1000.0, 2),
                timings_ms=timings,
            )

        answer_text, status = self._finalize_answer_text(raw_ans, needs_decree)
        total_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        ttft_ms = round(pre_llm_ms + llm_ttft_ms, 2)
        timings = {
            "rewrite_ms": rewrite_ms,
            "embedding_ms": embedding_ms,
            "search_ms": search_ms,
            "rerank_ms": rerank_ms,
            "llm_ttft_ms": llm_ttft_ms,
            "llm_total_ms": llm_total_ms,
            "ttft_ms": ttft_ms,
            "total_ms": total_ms,
        }

        logger.info(
            "TIMINGS [%s] rewrite=%.2fms | embed=%.2fms | search=%.2fms | rerank=%.2fms | llm_ttft=%.2fms | llm_total=%.2fms | total_ttft=%.2fms | total=%.2fms",
            q[:35],
            rewrite_ms,
            embedding_ms,
            search_ms,
            rerank_ms,
            llm_ttft_ms,
            llm_total_ms,
            ttft_ms,
            total_ms,
        )

        if status == "refused":
            res = {
                "query": q,
                "answer": NOT_FOUND_MESSAGE,
                "status": "refused",
                "needs_decree": False,
                "disclaimer": DISCLAIMER_TEXT,
                "sources": [],
                "cited_articles": [],
                "metrics": {
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "elapsed_seconds": round(total_ms / 1000.0, 2),
                    "cache_hit": False,
                    "timings_ms": timings,
                },
            }
            return res

        cited_articles = sorted(
            {
                int(n)
                for n in re.findall(r"điều\s+(\d+)", answer_text, flags=re.IGNORECASE)
                if 1 <= int(n) <= 500
            }
        )
        partial_warn, high_risk_warn, follow_ups = build_warnings_and_followups(
            q,
            retrieved_chunks,
            needs_decree,
        )

        res = {
            "query": q,
            "rewritten_query": rewrite_query(q),
            "answer": answer_text,
            "status": status,
            "needs_decree": needs_decree,
            "partial_warning": partial_warn,
            "high_risk_warning": high_risk_warn,
            "follow_ups": follow_ups,
            "disclaimer": DISCLAIMER_TEXT,
            "sources": format_sources_for_ui(retrieved_chunks),
            "cited_articles": cited_articles,
            "metrics": {
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "elapsed_seconds": round(total_ms / 1000.0, 2),
                "cache_hit": False,
                "timings_ms": timings,
            },
        }
        if use_cache:
            self._save_cache(cache_key, res)
        return res

    def stream_answer(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        use_vector: bool = True,
        min_score: float | None = None,
        use_cache: bool = True,
    ) -> Iterator[dict]:
        """
        Sinh câu trả lời theo luồng (Streaming) từ backend tới giao diện Web (Yêu cầu 2 - Giai đoạn 2):
            - Yield 1: `{"type": "meta", "sources": ..., "timings_ms": ...}` ngay sau khi tìm kiếm xong.
            - Yield 2..N: `{"type": "token", "delta": "..."}` ngay khi từng cụm chữ được sinh ra.
            - Yield cuối: `{"type": "done", "result": ...}` chứa kết quả hoàn chỉnh và số đo độ trễ.
        """
        start_time = time.perf_counter()
        threshold = self.min_score if min_score is None else float(min_score)
        q = (query or "").strip()

        # Nếu câu hỏi rỗng, ngoài phạm vi, hoặc đã có trong cache -> trả về ngay lập tức
        cache_key = (q.lower(), int(top_k), round(threshold, 3))
        if not q or check_out_of_scope(q) is not None or (use_cache and cache_key in self._response_cache):
            fast_res = self.answer(
                query,
                top_k=top_k,
                use_vector=use_vector,
                min_score=threshold,
                use_cache=use_cache,
            )
            yield {
                "type": "meta",
                "status": fast_res["status"],
                "sources": fast_res["sources"],
                "timings_ms": fast_res["metrics"].get("timings_ms", {}),
            }
            yield {"type": "token", "delta": fast_res["answer"]}
            yield {"type": "done", "result": fast_res}
            return

        # Bước 1: Tìm kiếm và khử trùng lặp điều luật
        raw_chunks = self.retriever.retrieve(
            q,
            top_k=max(top_k + 2, 5),
            expand_references=True,
            max_ref_chunks=1,
            use_vector=use_vector,
            min_score=threshold,
        )
        t_dedup0 = time.perf_counter()
        retrieved_chunks = deduplicate_by_article(
            raw_chunks,
            self.retriever.by_doc_article,
        )[:top_k]
        dedup_ms = (time.perf_counter() - t_dedup0) * 1000.0

        ret_timings = dict(self.retriever.last_timings)
        rewrite_ms = ret_timings.get("rewrite_ms", 0.0)
        embedding_ms = ret_timings.get("embedding_ms", 0.0)
        search_ms = ret_timings.get("search_ms", 0.0)
        rerank_ms = round(ret_timings.get("rerank_ms", 0.0) + dedup_ms, 2)
        pre_llm_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        if not retrieved_chunks:
            fast_res = self.answer(
                query,
                top_k=top_k,
                use_vector=False,
                min_score=threshold,
                use_cache=use_cache,
            )
            yield {"type": "meta", "status": "refused", "sources": [], "timings_ms": fast_res["metrics"]["timings_ms"]}
            yield {"type": "token", "delta": fast_res["answer"]}
            yield {"type": "done", "result": fast_res}
            return

        ui_sources = format_sources_for_ui(retrieved_chunks)
        needs_decree = check_needs_decree(retrieved_chunks[:2])
        if "tuổi nghỉ hưu" in q.lower() and re.search(r"\b20\d{2}\b", q):
            needs_decree = False
        is_dispute = bool(RE_DISPUTE_LAWSUIT.search(q))
        partial_warn, high_risk_warn, follow_ups = build_warnings_and_followups(
            q,
            retrieved_chunks,
            needs_decree,
        )

        # Gửi ngay danh sách điều luật về giao diện Web trước khi gọi LLM
        yield {
            "type": "meta",
            "status": "needs_decree" if needs_decree else "answered",
            "sources": ui_sources,
            "partial_warning": partial_warn,
            "high_risk_warning": high_risk_warn,
            "follow_ups": follow_ups,
            "timings_ms": {
                "rewrite_ms": rewrite_ms,
                "embedding_ms": embedding_ms,
                "search_ms": search_ms,
                "rerank_ms": rerank_ms,
            },
        }

        user_prompt = self._build_user_prompt(q, retrieved_chunks, needs_decree, is_dispute)

        # Nếu đang dùng hàm mock llm_fn trong unit test
        if self._llm_fn is not None:
            t_llm0 = time.perf_counter()
            raw_ans, in_tok, out_tok = self._llm_fn(SYSTEM_PROMPT, user_prompt)
            llm_ms = round((time.perf_counter() - t_llm0) * 1000.0, 2)
            final_text, status = self._finalize_answer_text(raw_ans, needs_decree)
            yield {"type": "token", "delta": final_text}
            total_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            timings = {
                "rewrite_ms": rewrite_ms,
                "embedding_ms": embedding_ms,
                "search_ms": search_ms,
                "rerank_ms": rerank_ms,
                "llm_ttft_ms": llm_ms,
                "llm_total_ms": llm_ms,
                "ttft_ms": round(pre_llm_ms + llm_ms, 2),
                "total_ms": total_ms,
            }
            cited_articles = sorted(
                {int(n) for n in re.findall(r"điều\s+(\d+)", final_text, flags=re.IGNORECASE) if 1 <= int(n) <= 500}
            )
            res = {
                "query": q,
                "rewritten_query": rewrite_query(q),
                "answer": final_text,
                "status": status,
                "needs_decree": needs_decree,
                "partial_warning": partial_warn,
                "high_risk_warning": high_risk_warn,
                "follow_ups": follow_ups,
                "disclaimer": DISCLAIMER_TEXT,
                "sources": ui_sources if status != "refused" else [],
                "cited_articles": cited_articles,
                "metrics": {
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "elapsed_seconds": round(total_ms / 1000.0, 2),
                    "cache_hit": False,
                    "timings_ms": timings,
                },
            }
            yield {"type": "done", "result": res}
            return

        # Bước 2: Gọi Gemini Streaming API thật
        t_llm0 = time.perf_counter()
        llm_ttft_ms: float | None = None
        chunks_text: list[str] = []
        in_tok = 0
        out_tok = 0

        try:
            from google.genai import types

            client = self._get_client()
            config = types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.1,
                max_output_tokens=1200,
            )
            for chunk in client.models.generate_content_stream(
                model=self.model_name,
                contents=user_prompt,
                config=config,
            ):
                piece = chunk.text or ""
                if piece:
                    if llm_ttft_ms is None:
                        llm_ttft_ms = round((time.perf_counter() - t_llm0) * 1000.0, 2)
                    chunks_text.append(piece)
                    yield {"type": "token", "delta": piece}
                usage = getattr(chunk, "usage_metadata", None)
                if usage:
                    in_tok = int(getattr(usage, "prompt_token_count", 0) or in_tok)
                    out_tok = int(getattr(usage, "candidates_token_count", 0) or out_tok)
        except Exception:
            total_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            fb = build_fallback_response(
                query=q,
                deduped_chunks=retrieved_chunks,
                min_score=threshold,
                elapsed_seconds=round(total_ms / 1000.0, 2),
            )
            if not chunks_text:
                yield {"type": "token", "delta": fb["answer"]}
            yield {"type": "done", "result": fb}
            return

        llm_total_ms = round((time.perf_counter() - t_llm0) * 1000.0, 2)
        if llm_ttft_ms is None:
            llm_ttft_ms = llm_total_ms

        raw_ans = "".join(chunks_text).strip()
        final_text, status = self._finalize_answer_text(raw_ans, needs_decree)
        if len(final_text) > len(raw_ans):
            extra_tail = final_text[len(raw_ans) :]
            yield {"type": "token", "delta": extra_tail}

        total_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        ttft_ms = round(pre_llm_ms + llm_ttft_ms, 2)
        timings = {
            "rewrite_ms": rewrite_ms,
            "embedding_ms": embedding_ms,
            "search_ms": search_ms,
            "rerank_ms": rerank_ms,
            "llm_ttft_ms": llm_ttft_ms,
            "llm_total_ms": llm_total_ms,
            "ttft_ms": ttft_ms,
            "total_ms": total_ms,
        }
        cited_articles = sorted(
            {int(n) for n in re.findall(r"điều\s+(\d+)", final_text, flags=re.IGNORECASE) if 1 <= int(n) <= 500}
        )
        partial_warn, high_risk_warn, follow_ups = build_warnings_and_followups(
            q,
            retrieved_chunks,
            needs_decree,
        )
        res = {
            "query": q,
            "rewritten_query": rewrite_query(q),
            "answer": final_text,
            "status": status,
            "needs_decree": needs_decree,
            "partial_warning": partial_warn,
            "high_risk_warning": high_risk_warn,
            "follow_ups": follow_ups,
            "disclaimer": DISCLAIMER_TEXT,
            "sources": ui_sources if status != "refused" else [],
            "cited_articles": cited_articles,
            "metrics": {
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "elapsed_seconds": round(total_ms / 1000.0, 2),
                "cache_hit": False,
                "timings_ms": timings,
            },
        }
        if use_cache and status != "fallback":
            self._save_cache(cache_key, res)
        yield {"type": "done", "result": res}

    def _save_cache(self, key: tuple[str, int, float], value: dict) -> None:
        """Lưu kết quả vào LRU Cache (tối đa MAX_CACHE_SIZE câu hỏi)."""
        if key in self._response_cache:
            self._response_cache.move_to_end(key)
        self._response_cache[key] = dict(value)
        while len(self._response_cache) > MAX_CACHE_SIZE:
            self._response_cache.popitem(last=False)


# ==========================================================================
# CHẠY THỬ TRỰC TIẾP TỪ DÒNG LỆNH
# ==========================================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hỏi đáp Pháp luật Lao động với Gemini RAG (Giai đoạn 1 & 2)"
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="ai được quyền nhận lương hưu",
        help="Câu hỏi cần giải đáp",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Số điều luật tối đa (mặc định 3)")
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE, help="Ngưỡng điểm tối thiểu")
    parser.add_argument("--no-vector", action="store_true", help="Chỉ dùng BM25 + Rerank")
    args = parser.parse_args()

    gen = RAGGenerator(min_score=args.min_score)
    result = gen.answer(args.query, top_k=args.top_k, use_vector=not args.no_vector)

    print(f"\nCÂU HỎI: {result['query']}")
    print(f"TRẠNG THÁI: {result['status']} | Cần Nghị định: {result['needs_decree']}")
    print("=" * 70)
    print(result["answer"])
    print("-" * 70)
    print(f"CÁC ĐIỀU LUẬT ĐƯỢC TRUY XUẤT ({len(result['sources'])} điều):")
    for i, s in enumerate(result["sources"], start=1):
        print(
            f"  [{i}] {s['citation']} — {s['article_title']} "
            f"(điểm: {s['score']} | loại: {s['match_type']} | độ dài: {len(s['text'])} ký tự)"
        )
    m = result["metrics"]
    t = m.get("timings_ms", {})
    print(
        f"\n[Token vào: {m['input_tokens']} | Token ra: {m['output_tokens']} | "
        f"TTFT: {t.get('ttft_ms', 0)}ms | Tổng thời gian: {m['elapsed_seconds']}s]"
    )
    print(f"Chi tiết độ trễ (ms): {t}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
