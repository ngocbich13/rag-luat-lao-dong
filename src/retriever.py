"""
retriever.py — GIAI ĐOẠN 1 & 4: VIẾT LẠI TRUY VẤN, TÌM KIẾM LAI, RERANK & NGƯỠNG ĐIỂM
=======================================================================================
Nhận câu hỏi tiếng Việt của người dùng và trả về danh sách các đoạn luật (chunk)
liên quan nhất từ kho văn bản pháp luật (Bộ luật Lao động 2019 và các văn bản nạp thêm).

Các cải tiến trọng tâm ở Giai đoạn 1:
    1. Viết lại câu hỏi thành truy vấn pháp lý (`rewrite_query`):
       Chuyển ngôn ngữ đời thường sang thuật ngữ pháp lý chuẩn (ví dụ:
       "ai được quyền nhận lương hưu" -> "điều kiện hưởng lương hưu hằng tháng,
       tuổi nghỉ hưu, thời gian đóng bảo hiểm xã hội Điều 169 Điều 168").
    2. Tìm kiếm lai (BM25 từ ghép + Vector Cosine Similarity) kết hợp Rerank nhẹ
       (`_rerank_score`):
       - Khớp từ ghép pháp lý ('lương_hưu', 'nghỉ_hưu', 'báo_trước', 'thử_việc'...)
       - Cộng điểm khớp tiêu đề Điều (`article_title`)
       - Phạt nặng các trường hợp bắt nhầm từ đơn (ví dụ hỏi "lương hưu" nhưng
         điều luật chỉ nói về "trả lương" tại Điều 94 mà không có chữ "hưu").
    3. Ngưỡng điểm "Không tìm thấy" (`min_score`, cấu hình qua biến môi trường
       `RAG_MIN_SCORE`, mặc định 0.25):
       - Mọi kết quả có điểm thấp hơn ngưỡng đều bị loại bỏ để không bao giờ trả về
         điều luật không liên quan.
    4. Gộp và khử trùng lặp theo Điều luật (`deduplicate_by_article`):
       - Đảm bảo không bao giờ hiển thị trùng lặp cùng một điều luật 2 lần và mỗi
         điều luật đều có đầy đủ nội dung các khoản.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
import time
from typing import Callable

import numpy as np
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.indexer import (
    CHUNKS_FILE,
    DEFAULT_EMBED_MODEL,
    EMBEDDINGS_FILE,
    embed_texts,
    load_index,
    tokenize_vi,
)

# Ngưỡng điểm liên quan tối thiểu (cấu hình được qua .env: RAG_MIN_SCORE)
DEFAULT_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.25"))

# Các từ chung chung trong câu hỏi pháp luật, không mang đặc trưng phân biệt chủ đề
GENERIC_QUERY_WORDS = {
    "thủ", "tục", "thủ_tục", "quy", "định", "quy_định", "điều", "kiện", "điều_kiện",
    "trường", "hợp", "trường_hợp", "quyền", "nghĩa", "vụ", "nghĩa_vụ", "người",
    "năm", "tháng", "ngày", "tiền", "mức", "tính", "đóng", "hưởng", "nhận", "làm",
    "việc", "công", "ty", "công_ty", "luật", "bộ", "bộ_luật", "khoản", "điểm",
    "tối", "đa", "tối_đa", "tối_thiểu", "thiểu", "ít", "nhất", "ít_nhất", "phải",
    "không", "cần", "hỏi", "xin", "muốn", "bị", "được", "cho", "tôi", "mình",
}

# --------------------------------------------------------------------------
# 1. BƯỚC VIẾT LẠI CÂU HỎI THÀNH TRUY VẤN PHÁP LÝ (QUERY REWRITING)
# --------------------------------------------------------------------------
COLLOQUIAL_TO_LEGAL: list[tuple[tuple[str, ...], str]] = [
    (
        ("lương hưu", "nhận lương hưu", "hưởng lương hưu", "điều kiện hưởng lương hưu", "chế độ hưu trí"),
        "điều kiện hưởng lương hưu hằng tháng tuổi nghỉ hưu thời gian đóng bảo hiểm xã hội Điều 169 Điều 168",
    ),
    (
        ("tuổi nghỉ hưu", "về hưu", "hưu trí", "nghỉ hưu năm", "mấy tuổi nghỉ hưu"),
        "tuổi nghỉ hưu của người lao động trong điều kiện lao động bình thường lộ trình tăng tuổi nghỉ hưu nam 62 tuổi nữ 60 tuổi Điều 169",
    ),
    (
        ("báo trước", "tự nghỉ việc", "xin nghỉ việc", "nghỉ ngang", "bỏ việc", "muốn nghỉ việc"),
        "quyền đơn phương chấm dứt hợp đồng lao động của người lao động thời hạn báo trước 45 ngày 30 ngày 03 ngày làm việc Điều 35",
    ),
    (
        ("đuổi việc", "cho nghỉ việc", "cắt giảm nhân sự", "bị cho thôi việc", "công ty chấm dứt"),
        "quyền đơn phương chấm dứt hợp đồng lao động của người sử dụng lao động báo trước trợ cấp mất việc làm Điều 36 Điều 47",
    ),
    (
        ("trợ cấp thôi việc", "nghỉ việc được trợ cấp", "làm 3 năm rồi nghỉ", "tính trợ cấp thôi việc"),
        "trợ cấp thôi việc làm việc thường xuyên từ đủ 12 tháng trở lên mỗi năm làm việc nửa tháng tiền lương Điều 46",
    ),
    (
        ("trợ cấp mất việc", "mất việc làm"),
        "trợ cấp mất việc làm mỗi năm làm việc 01 tháng tiền lương ít nhất bằng 02 tháng tiền lương Điều 47",
    ),
    (
        ("sa thải", "bị sa thải"),
        "áp dụng hình thức xử lý kỷ luật sa thải hành vi trộm cắp tham ô tự ý bỏ việc 05 ngày cộng dồn Điều 125",
    ),
    (
        ("nghỉ đẻ", "sinh con", "mang bầu", "có thai", "con nhỏ", "thai sản"),
        "nghỉ thai sản lao động nữ sinh con được nghỉ trước và sau khi sinh con 06 tháng bảo vệ thai sản Điều 137 Điều 139",
    ),
    (
        ("tăng ca", "làm thêm", "ot", "ca đêm", "làm ban đêm", "làm thêm giờ"),
        "tiền lương làm thêm giờ làm việc vào ban đêm ít nhất bằng 150% 200% 300% số giờ làm thêm tối đa Điều 98 Điều 107",
    ),
    (
        ("thử việc", "lương thử việc", "hợp đồng thử việc", "thời gian thử việc"),
        "thử việc thời gian thử việc tối đa 180 ngày 60 ngày 30 ngày 06 ngày làm việc tiền lương thử việc ít nhất 85% Điều 24 Điều 25 Điều 26 Điều 27",
    ),
    (
        ("nghỉ phép", "phép năm", "ngày phép", "nghỉ hằng năm"),
        "nghỉ hằng năm 12 ngày làm việc 14 ngày 16 ngày ngày nghỉ hằng năm tăng thêm theo thâm niên 05 năm Điều 113 Điều 114",
    ),
    (
        ("nghỉ lễ", "nghỉ tết", "lễ tết"),
        "nghỉ lễ tết tết dương lịch tết âm lịch ngày chiến thắng ngày quốc khánh ngày quốc tế lao động giỗ tổ hùng vương hưởng nguyên lương Điều 112",
    ),
    (
        ("việc riêng", "đám cưới", "kết hôn", "đám tang", "bố mẹ mất"),
        "nghỉ việc riêng nghỉ không hưởng lương kết hôn nghỉ 03 ngày con kết hôn cha đẻ mẹ đẻ chết Điều 115",
    ),
    (
        ("giữ bằng", "giữ chứng minh", "giữ cccd", "căn cước công dân", "tiền cọc", "đặt cọc"),
        "hành vi người sử dụng lao động không được làm khi giao kết thực hiện hợp đồng lao động giữ bản chính giấy tờ tùy thân văn bằng chứng chỉ đặt cọc Điều 17",
    ),
    (
        ("nợ lương", "chậm lương", "trả lương trễ", "trả lương chậm"),
        "nguyên tắc trả lương kỳ hạn trả lương trả lương chậm từ 15 ngày trở lên đền bù lãi suất Điều 94 Điều 97",
    ),
    (
        ("trừ lương", "phạt tiền", "cắt lương", "khấu trừ lương"),
        "khấu trừ tiền lương tối đa 30% bồi thường thiệt hại hành vi bị nghiêm cấm khi xử lý kỷ luật phạt tiền cắt lương Điều 102 Điều 127",
    ),
    (
        ("thưởng tết", "lương tháng 13", "tiền thưởng"),
        "thưởng quy chế thưởng do người sử dụng lao động quyết định Điều 104",
    ),
    (
        ("bảo hiểm xã hội", "đóng bhxh", "tham gia bảo hiểm"),
        "tham gia bảo hiểm xã hội bảo hiểm y tế bảo hiểm thất nghiệp bắt buộc Điều 168",
    ),
    (
        ("hợp đồng lao động", "loại hợp đồng", "giao kết hợp đồng"),
        "các loại hợp đồng lao động không xác định thời hạn xác định thời hạn tối đa 36 tháng Điều 20",
    ),
    (
        ("giờ làm việc", "thời giờ làm việc", "ngày làm mấy tiếng", "tuần làm mấy giờ"),
        "thời giờ làm việc bình thường không quá 08 giờ trong 01 ngày và không quá 48 giờ trong 01 tuần Điều 105",
    ),
    (
        ("lương tối thiểu", "mức lương tối thiểu"),
        "mức lương tối thiểu vùng theo tháng theo giờ hội đồng tiền lương quốc gia Điều 91",
    ),
    (
        ("kỷ luật", "xử lý kỷ luật", "hình thức kỷ luật"),
        "hình thức xử lý kỷ luật lao động khiển trách kéo dài thời hạn nâng lương cách chức sa thải nguyên tắc trình tự Điều 122 Điều 124",
    ),
    (
        ("tai nạn lao động", "bệnh nghề nghiệp"),
        "an toàn lao động vệ sinh lao động tai nạn lao động bệnh nghề nghiệp trách nhiệm người sử dụng lao động Điều 38 Điều 132",
    ),
    (
        ("lao động chưa thành niên", "dưới 18 tuổi", "dưới 15 tuổi"),
        "lao động chưa thành niên người chưa đủ 18 tuổi thời giờ làm việc của người chưa thành niên Điều 143 Điều 146",
    ),
]


def rewrite_query(
    query: str,
    llm_rewrite_fn: Callable[[str], str] | None = None,
) -> str:
    """
    Viết lại câu hỏi của người dùng thành truy vấn pháp lý giàu thuật ngữ chuyên ngành.
    - Bước 1: Áp dụng quy tắc ánh xạ ngôn ngữ đời thường -> thuật ngữ pháp lý (< 0.1ms).
    - Bước 2: Nếu có truyền `llm_rewrite_fn` và câu hỏi chưa khớp quy tắc nào, gọi hàm
              viết lại câu hỏi nâng cao.
    """
    q_clean = (query or "").strip()
    if not q_clean:
        return ""

    q_lower = q_clean.lower()
    additions: list[str] = []
    for triggers, legal_terms in COLLOQUIAL_TO_LEGAL:
        if any(t in q_lower for t in triggers):
            additions.append(legal_terms)

    if additions:
        return q_clean + " (" + " ; ".join(additions) + ")"

    if llm_rewrite_fn is not None:
        try:
            rewritten = (llm_rewrite_fn(q_clean) or "").strip()
            if rewritten and rewritten != q_clean:
                return f"{q_clean} ({rewritten})"
        except Exception:
            pass

    return q_clean


# Giữ bí danh expand_query để tương thích ngược 100% với các test cũ
expand_query = rewrite_query


# --------------------------------------------------------------------------
# 2. NHẬN DIỆN SỐ ĐIỀU / KHOẢN ĐƯỢC HỎI TRỰC TIẾP TRONG CÂU HỎI
# --------------------------------------------------------------------------
RE_EXPLICIT_CLAUSE_ARTICLE = re.compile(
    r"khoản\s+(\d+[a-z]?)\s+điều\s+(\d+)", re.IGNORECASE
)
RE_EXPLICIT_ARTICLES = re.compile(
    r"điều\s+(\d+(?:\s*(?:,|và|đến|-)\s*\d+)*)", re.IGNORECASE
)


def parse_explicit_refs(query: str) -> dict:
    """
    Tìm các số Điều và cặp (Điều, Khoản) được người dùng nhắc trực tiếp trong câu hỏi.
    """
    articles: set[int] = set()
    clauses: list[tuple[int, str]] = []

    for m in RE_EXPLICIT_CLAUSE_ARTICLE.finditer(query):
        clause_no = m.group(1).lower()
        art_no = int(m.group(2))
        if 1 <= art_no <= 500:
            articles.add(art_no)
            if (art_no, clause_no) not in clauses:
                clauses.append((art_no, clause_no))

    for m in RE_EXPLICIT_ARTICLES.finditer(query):
        nums = [int(n) for n in re.findall(r"\d+", m.group(1))]
        for n in nums:
            if 1 <= n <= 500:
                articles.add(n)

    return {
        "articles": sorted(articles),
        "clauses": clauses,
    }


# --------------------------------------------------------------------------
# 3. GỘP CÁC CHUNK CÙNG MỘT ĐIỀU LUẬT & KHỬ TRÙNG LẶP
# --------------------------------------------------------------------------
def deduplicate_by_article(
    chunks: list[dict],
    by_article_map: dict[tuple[str, int], list[dict]] | None = None,
) -> list[dict]:
    """
    Khử trùng lặp danh sách kết quả theo `(doc_id, article_no)` để KHÔNG BAO GIỜ
    hiển thị cùng một điều luật 2 lần.
    Đồng thời, nếu một Điều luật bị tách thành nhiều chunk (ví dụ Điều 35 có Khoản 1
    và Khoản 2), tự động hợp nhất nội dung các chunk của Điều đó theo đúng thứ tự
    `part_index` (1 -> 2 -> ...) để người dùng và LLM luôn đọc được ĐIỀU LUẬT ĐẦY ĐỦ.
    """
    deduped: list[dict] = []
    seen_articles: set[tuple[str, int]] = set()

    for item in chunks:
        doc_id = str(item.get("doc_id", "45/2019/QH14"))
        art_no = int(item["article_no"])
        key = (doc_id, art_no)
        if key in seen_articles:
            continue
        seen_articles.add(key)

        merged = dict(item)
        # Nếu có bảng tra toàn bộ các phần của Điều này, ghép đủ nội dung các khoản theo thứ tự
        if by_article_map and key in by_article_map:
            siblings = sorted(
                by_article_map[key],
                key=lambda c: int(c.get("part_index", 1)),
            )
            if len(siblings) > 1 and not item.get("quoted_other_law"):
                full_text_parts: list[str] = []
                all_clauses: list[str] = []
                for sib in siblings:
                    sib_text = (sib.get("text") or "").strip()
                    if sib_text and sib_text not in full_text_parts:
                        full_text_parts.append(sib_text)
                    for cl in sib.get("clause_nos", []):
                        if str(cl) not in all_clauses:
                            all_clauses.append(str(cl))

                merged["text"] = "\n".join(full_text_parts)
                merged["clause_nos"] = all_clauses
                law_name = merged.get("law_name", "Bộ luật Lao động 2019")
                merged["citation"] = f"Điều {art_no} {law_name}"

        deduped.append(merged)

    return deduped


# --------------------------------------------------------------------------
# 4. LỚP HYBRID RETRIEVER (BM25 + VECTOR + RERANK NHẸ + NGƯỠNG ĐIỂM)
# --------------------------------------------------------------------------
class HybridRetriever:
    """
    Bộ tìm kiếm lai kết hợp:
        1. Viết lại truy vấn pháp lý (`rewrite_query`)
        2. Tìm kiếm từ khóa & từ ghép tiếng Việt (`BM25Okapi`)
        3. Tìm kiếm ngữ nghĩa (`Vector Cosine Similarity`) + Bộ nhớ đệm vector câu hỏi
        4. Rerank nhẹ theo đặc trưng pháp lý (tiêu đề Điều, cụm từ ghép, phạt lệch chủ đề)
        5. Lọc theo ngưỡng điểm tối thiểu (`min_score`) để từ chối câu hỏi không liên quan
    """

    def __init__(
        self,
        chunks: list[dict],
        embeddings: np.ndarray,
        embed_query_fn: Callable[[str], np.ndarray] | None = None,
    ):
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Số lượng chunks ({len(chunks)}) và embeddings ({embeddings.shape[0]}) không khớp."
            )
        self.chunks = chunks
        self.embeddings = embeddings
        self._embed_query_fn = embed_query_fn
        self._query_vec_cache: dict[str, np.ndarray] = {}
        self._executor = ThreadPoolExecutor(max_workers=2)
        self.last_timings: dict[str, float] = {
            "rewrite_ms": 0.0,
            "embedding_ms": 0.0,
            "search_ms": 0.0,
            "rerank_ms": 0.0,
        }

        # Bảng tra nhanh theo chunk_id, theo article_no và theo (doc_id, article_no)
        self.by_id: dict[str, dict] = {c["chunk_id"]: c for c in chunks}
        self.by_article: dict[int, list[dict]] = {}
        self.by_doc_article: dict[tuple[str, int], list[dict]] = {}
        for c in chunks:
            art_no = int(c["article_no"])
            doc_id = str(c.get("doc_id", "45/2019/QH14"))
            self.by_article.setdefault(art_no, []).append(c)
            self.by_doc_article.setdefault((doc_id, art_no), []).append(c)

        # Khởi tạo chỉ mục BM25 và tập token của từng chunk + tiêu đề Điều để Rerank nhanh
        self.tokenized_corpus: list[list[str]] = []
        self.chunk_token_sets: list[set[str]] = []
        self.title_token_sets: list[set[str]] = []
        for c in self.chunks:
            toks = tokenize_vi(c["text_for_embedding"])
            self.tokenized_corpus.append(toks)
            self.chunk_token_sets.append(set(toks))
            title_toks = tokenize_vi(f"{c.get('article_title', '')} {c.get('chapter_title', '')}")
            self.title_token_sets.append(set(title_toks))

        self.bm25 = BM25Okapi(self.tokenized_corpus)

    @classmethod
    def from_files(
        cls,
        chunks_path: str | Path = CHUNKS_FILE,
        embeddings_path: str | Path = EMBEDDINGS_FILE,
        embed_query_fn: Callable[[str], np.ndarray] | None = None,
    ) -> "HybridRetriever":
        """Tải chunks.jsonl và embeddings.npz từ ổ cứng để khởi tạo bộ tìm kiếm."""
        chunks, embeddings = load_index(chunks_path, embeddings_path)
        return cls(chunks=chunks, embeddings=embeddings, embed_query_fn=embed_query_fn)

    def _encode_query(self, query_text: str) -> np.ndarray | None:
        """
        Biến câu hỏi thành vector 1 chiều đã chuẩn hóa (có cache trong RAM để hỏi lại tốn 0ms).
        """
        if query_text in self._query_vec_cache:
            return self._query_vec_cache[query_text]

        if self._embed_query_fn is not None:
            vec = np.asarray(self._embed_query_fn(query_text), dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(vec))
            res = vec / norm if norm > 0 else vec
            self._query_vec_cache[query_text] = res
            return res

        try:
            mat = embed_texts([query_text], model=DEFAULT_EMBED_MODEL, batch_size=1)
            res = mat[0]
            self._query_vec_cache[query_text] = res
            return res
        except Exception:
            return None

    def _rerank_score(
        self,
        idx: int,
        chunk: dict,
        orig_tokens: list[str],
        expanded_tokens: list[str],
        bm25_raw: float,
        bm25_max: float,
        vec_sim: float | None,
        hinted_articles: set[int],
    ) -> float:
        """
        Hàm Rerank nhẹ (< 0.05ms/chunk) tính điểm liên quan chuẩn hóa trong [0.0, 1.0]:
            - Kết hợp điểm ngữ nghĩa Vector, điểm từ khóa BM25, độ phủ từ khóa gốc
              (`lexical_coverage`), độ khớp tiêu đề Điều (`title_overlap`), và Điều gợi ý
              từ bước viết lại truy vấn pháp lý (`hinted_articles`).
            - Phạt nặng khi câu hỏi hỏi về "lương hưu / nghỉ hưu" mà chunk lại thuộc
              Chương TIỀN LƯƠNG (như Điều 94) và không hề chứa từ "hưu".
        """
        doc_tokens = self.chunk_token_sets[idx]
        title_tokens = self.title_token_sets[idx]

        # 1. Độ phủ từ khóa đặc trưng (loại bỏ các từ chung chung như 'thủ tục', 'quy định')
        orig_set = {t for t in orig_tokens if not t.isdigit()}
        specific_set = {
            t for t in orig_set
            if t not in GENERIC_QUERY_WORDS
            and not (
                "_" in t
                and all(part in GENERIC_QUERY_WORDS for part in t.split("_"))
            )
        }
        orig_bigrams = {t for t in specific_set if "_" in t}
        orig_unigrams = {t for t in specific_set if "_" not in t}

        art_no = int(chunk["article_no"])

        # Nếu câu hỏi có từ khóa đặc trưng (ví dụ 'ly hôn', 'sổ đỏ', 'thuế') mà chunk không khớp
        # bất kỳ từ đặc trưng nào và cũng không nằm trong Điều gợi ý -> điểm rất thấp (< ngưỡng)
        if specific_set and len(specific_set & doc_tokens) == 0 and art_no not in hinted_articles:
            return 0.05

        uni_overlap = len(orig_unigrams & doc_tokens) / max(1, len(orig_unigrams))
        bg_overlap = (
            len(orig_bigrams & doc_tokens) / len(orig_bigrams)
            if orig_bigrams
            else uni_overlap
        )
        title_overlap = len(specific_set & title_tokens) / max(1, len(specific_set))

        # Phạt trường hợp chỉ khớp ngẫu nhiên 1 âm tiết đơn lẻ (ví dụ hỏi "ly hôn" khớp chữ "hôn" trong "kết hôn")
        if (
            orig_bigrams
            and len(orig_bigrams & doc_tokens) == 0
            and uni_overlap <= 0.5
            and art_no not in hinted_articles
        ):
            if vec_sim is None or float(vec_sim) < 0.62:
                return 0.12

        # 2. Điểm BM25 chuẩn hóa theo hàm bão hòa tuyệt đối (không chia cho bm25_max để tránh
        # câu hỏi ngoài phạm vi khớp 1 chữ chung chung cũng được 1.0)
        bm25_comp = bm25_raw / (bm25_raw + 16.0) if bm25_raw > 0 else 0.0

        # 3. Kết hợp với điểm Vector Cosine Similarity (nếu có)
        if vec_sim is not None:
            vec_norm = max(0.0, min(1.0, (float(vec_sim) - 0.48) / 0.32))
            base_score = (
                0.38 * bm25_comp
                + 0.32 * vec_norm
                + 0.18 * bg_overlap
                + 0.12 * title_overlap
            )
        else:
            base_score = (
                0.50 * bm25_comp
                + 0.22 * uni_overlap
                + 0.16 * bg_overlap
                + 0.12 * title_overlap
            )

        # 4. Cộng điểm nếu Điều luật nằm trong nhóm Điều gợi ý của bước Query Rewriting
        if art_no in hinted_articles:
            part_idx = int(chunk.get("part_index", 1))
            # Ưu tiên Điều khớp trực tiếp tiêu đề với câu hỏi (ví dụ Điều 169 đứng trên Điều 168 khi hỏi lương hưu/nghỉ hưu)
            title_bonus = 0.06 if title_overlap > 0 or ("hưu" in " ".join(orig_tokens) and art_no == 169) else 0.0
            base_score = max(base_score, 0.72) + 0.18 + title_bonus - 0.01 * part_idx

        # 5. Chốt chặn chống nhầm lẫn ngữ nghĩa ("lương hưu" vs "tiền lương / trả lương" Điều 94)
        if ("lương_hưu" in orig_set or "nghỉ_hưu" in orig_set or "hưu_trí" in orig_set or "hưu" in orig_set):
            if art_no not in hinted_articles and "hưu" not in doc_tokens and "lương_hưu" not in doc_tokens and "nghỉ_hưu" not in doc_tokens:
                base_score *= 0.10

        return float(min(0.99, max(0.0, base_score)))

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        expand_references: bool = True,
        max_ref_chunks: int = 2,
        use_vector: bool = True,
        min_score: float = DEFAULT_MIN_SCORE,
    ) -> list[dict]:
        """
        Tìm kiếm và xếp hạng lại (rerank) các chunk điều luật cho câu hỏi `query`.
        - Mọi chunk có điểm < `min_score` đều bị loại bỏ.
        - Trả về danh sách chunk kèm `score`, `match_type`, và `rewritten_query`.
        """
        query = (query or "").strip()
        if not query:
            self.last_timings = {
                "rewrite_ms": 0.0,
                "embedding_ms": 0.0,
                "search_ms": 0.0,
                "rerank_ms": 0.0,
            }
            return []

        # 1. Nhận diện số Điều/Khoản và viết lại câu hỏi thành truy vấn pháp lý
        t_rw0 = time.perf_counter()
        explicit = parse_explicit_refs(query)
        explicit_articles = set(explicit["articles"])
        explicit_clauses = explicit["clauses"]

        rewritten_query = rewrite_query(query)
        rewritten_refs = parse_explicit_refs(rewritten_query)
        hinted_articles = set(rewritten_refs["articles"]) - explicit_articles
        rewrite_ms = round((time.perf_counter() - t_rw0) * 1000.0, 2)

        # Mặc định bỏ qua Điều 219 (quoted_other_law), trừ khi hỏi trực tiếp Điều 219
        allow_quoted_other_law = 219 in explicit_articles

        # 2. Tìm kiếm từ khóa BM25 trên tập văn bản luật
        t_s0 = time.perf_counter()
        orig_tokens = tokenize_vi(query)
        expanded_tokens = tokenize_vi(rewritten_query)

        if expanded_tokens:
            bm25_scores = np.asarray(self.bm25.get_scores(expanded_tokens), dtype=np.float32)
        else:
            bm25_scores = np.zeros(len(self.chunks), dtype=np.float32)

        bm25_max = float(np.max(bm25_scores)) if bm25_scores.size > 0 else 0.0
        bm25_search_ms = (time.perf_counter() - t_s0) * 1000.0

        # 3. Kết hợp Vector Search trên chỉ mục ma trận L2 (258 x 3072):
        #    - Nếu vector câu hỏi đã có trong cache hoặc có hàm embed_query_fn cục bộ: chạy ngay (<0.2ms).
        #    - Nếu câu hỏi chưa khớp Điều gợi ý từ bước viết lại truy vấn (câu hỏi mơ hồ/mới):
        #      gọi Gemini Embedding API để lấy vector ngữ nghĩa.
        #    - Nếu câu hỏi đã khớp chính xác Điều gợi ý (hinted_articles) hoặc explicit_articles:
        #      bỏ qua bước gọi mạng Embedding dư thừa để giảm TTFT xuống < 1.5 giây.
        embedding_ms = 0.0
        q_vec: np.ndarray | None = None
        if use_vector and not explicit_articles:
            if rewritten_query in self._query_vec_cache or self._embed_query_fn is not None:
                t_emb0 = time.perf_counter()
                q_vec = self._encode_query(rewritten_query)
                embedding_ms = round((time.perf_counter() - t_emb0) * 1000.0, 2)
            elif not hinted_articles:
                t_emb0 = time.perf_counter()
                try:
                    vec_future = self._executor.submit(self._encode_query, rewritten_query)
                    q_vec = vec_future.result(timeout=1.5)
                except (FuturesTimeoutError, Exception):
                    q_vec = None
                embedding_ms = round((time.perf_counter() - t_emb0) * 1000.0, 2)

        t_v0 = time.perf_counter()
        vec_scores: np.ndarray | None = None
        if q_vec is not None and q_vec.shape[0] == self.embeddings.shape[1]:
            # Nhân ma trận chỉ mục đã chuẩn hóa L2 (258 x 3072) trong < 0.2ms
            vec_scores = self.embeddings @ q_vec
        search_ms = round(bm25_search_ms + (time.perf_counter() - t_v0) * 1000.0, 2)

        # 4. Chấm điểm Rerank cho từng chunk và lọc theo ngưỡng min_score
        t_rr0 = time.perf_counter()
        explicit_clause_map: dict[int, set[str]] = {}
        for art_no, cl_no in explicit_clauses:
            explicit_clause_map.setdefault(art_no, set()).add(cl_no)

        candidates: list[tuple[float, str, dict]] = []
        for idx, chunk in enumerate(self.chunks):
            if chunk.get("quoted_other_law") and not allow_quoted_other_law:
                continue

            art_no = int(chunk["article_no"])
            v_sim = float(vec_scores[idx]) if vec_scores is not None else None

            if art_no in explicit_articles:
                # Người dùng hỏi đích danh số Điều/Khoản
                wanted_clauses = explicit_clause_map.get(art_no)
                chunk_clauses = {str(c).lower() for c in chunk.get("clause_nos", [])}
                part_idx = int(chunk.get("part_index", 1))
                if wanted_clauses:
                    if wanted_clauses & chunk_clauses:
                        score = 0.99
                    elif part_idx == 1:
                        score = 0.92
                    else:
                        score = 0.85
                else:
                    score = round(0.99 - 0.01 * (part_idx - 1), 4)
                match_type = "explicit"
            else:
                score = self._rerank_score(
                    idx=idx,
                    chunk=chunk,
                    orig_tokens=orig_tokens,
                    expanded_tokens=expanded_tokens,
                    bm25_raw=float(bm25_scores[idx]),
                    bm25_max=bm25_max,
                    vec_sim=v_sim,
                    hinted_articles=hinted_articles,
                )
                match_type = "hybrid" if vec_scores is not None else "bm25"

            if score >= min_score:
                candidates.append((score, match_type, chunk))

        # Sắp xếp giảm dần theo điểm rerank
        candidates.sort(key=lambda x: x[0], reverse=True)
        if not candidates:
            self.last_timings = {
                "rewrite_ms": rewrite_ms,
                "embedding_ms": embedding_ms,
                "search_ms": search_ms,
                "rerank_ms": round((time.perf_counter() - t_rr0) * 1000.0, 2),
            }
            return []

        # Lọc bỏ các kết quả phụ có điểm quá thấp so với kết quả chính (ví dụ ẩn Điều 46 điểm 0.50 khi Điều 169 đạt 0.95)
        top_score = candidates[0][0]
        relative_floor = max(min_score, top_score * 0.62) if top_score >= 0.75 else max(min_score, top_score * 0.50)

        selected: list[dict] = []
        seen_ids: set[str] = set()

        for score, m_type, chunk in candidates:
            if len(selected) >= top_k:
                break
            if score < relative_floor:
                continue
            item = dict(chunk)
            item["score"] = round(score, 4)
            item["match_type"] = m_type
            item["rewritten_query"] = rewritten_query
            selected.append(item)
            seen_ids.add(chunk["chunk_id"])

        # 6. Mở rộng ngữ cảnh cùng Điều (đặt phần dẫn P01 ngay cạnh phần được chọn)
        with_leads: list[dict] = []
        for item in selected:
            if int(item.get("part_index", 1)) > 1:
                art_no = int(item["article_no"])
                for sibling in self.by_article.get(art_no, []):
                    if int(sibling.get("part_index", 1)) == 1 and sibling["chunk_id"] not in seen_ids:
                        lead_copy = dict(sibling)
                        lead_copy["score"] = round(item["score"] * 0.95, 4)
                        lead_copy["match_type"] = "same_article_lead"
                        lead_copy["rewritten_query"] = rewritten_query
                        with_leads.append(lead_copy)
                        seen_ids.add(sibling["chunk_id"])
            with_leads.append(item)
        selected = with_leads

        # 7. Mở rộng Điều tham chiếu trong trường `references` của chunk đứng đầu
        if expand_references and selected and max_ref_chunks > 0:
            ref_added = 0
            for primary in list(selected[:2]):
                for ref in primary.get("references", []):
                    if ref_added >= max_ref_chunks:
                        break
                    if isinstance(ref, int):
                        ref_art_no = ref
                        ref_clauses: set[str] = set()
                    elif isinstance(ref, dict):
                        ref_art_no = ref.get("article")
                        ref_clauses = {str(c).lower() for c in ref.get("clauses", [])}
                    else:
                        continue
                    if not ref_art_no:
                        continue

                    ref_candidates = self.by_article.get(int(ref_art_no), [])
                    chosen_ref_chunk = None
                    if ref_clauses:
                        for rc in ref_candidates:
                            rc_clauses = {str(c).lower() for c in rc.get("clause_nos", [])}
                            if ref_clauses & rc_clauses:
                                chosen_ref_chunk = rc
                                break
                    if chosen_ref_chunk is None and ref_candidates:
                        chosen_ref_chunk = ref_candidates[0]

                    if (
                        chosen_ref_chunk is not None
                        and chosen_ref_chunk["chunk_id"] not in seen_ids
                        and not chosen_ref_chunk.get("quoted_other_law")
                    ):
                        ref_copy = dict(chosen_ref_chunk)
                        ref_copy["score"] = round(max(min_score, primary["score"] * 0.6), 4)
                        ref_copy["match_type"] = "reference"
                        ref_copy["rewritten_query"] = rewritten_query
                        selected.append(ref_copy)
                        seen_ids.add(chosen_ref_chunk["chunk_id"])
                        ref_added += 1

        self.last_timings = {
            "rewrite_ms": rewrite_ms,
            "embedding_ms": embedding_ms,
            "search_ms": search_ms,
            "rerank_ms": round((time.perf_counter() - t_rr0) * 1000.0, 2),
        }
        return selected


# ==========================================================================
# CHẠY THỬ TRỰC TIẾP TỪ DÒNG LỆNH
# ==========================================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tìm kiếm điều luật liên quan cho câu hỏi tiếng Việt"
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="ai được quyền nhận lương hưu",
        help="Câu hỏi cần tra cứu",
    )
    parser.add_argument("--top-k", type=int, default=4, help="Số lượng chunk chính cần lấy")
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE, help="Ngưỡng điểm tối thiểu")
    parser.add_argument("--no-vector", action="store_true", help="Chỉ dùng BM25 + Rerank")
    args = parser.parse_args()

    retriever = HybridRetriever.from_files()
    results = retriever.retrieve(
        args.query,
        top_k=args.top_k,
        use_vector=not args.no_vector,
        min_score=args.min_score,
    )
    deduped = deduplicate_by_article(results, retriever.by_doc_article)

    print(f"Câu hỏi gốc      : {args.query}")
    print(f"Truy vấn pháp lý : {rewrite_query(args.query)}")
    print(f"Số điều luật đạt ngưỡng (>= {args.min_score}): {len(deduped)}\n" + "=" * 65)
    for i, r in enumerate(deduped, start=1):
        print(
            f"[{i}] {r['citation']} — {r.get('article_title', '')} | "
            f"Loại: {r['match_type']} | Điểm: {r['score']}"
        )
        preview = r["text"].replace("\n", " ")
        if len(preview) > 180:
            preview = preview[:180] + "..."
        print(f"    {preview}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
