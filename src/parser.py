"""
parser.py — GIAI ĐOẠN 1: XỬ LÝ VĂN BẢN PHÁP LUẬT
=================================================
Biến file Word "Bộ luật Lao động 2019" thành 1 file JSON có cấu trúc:
    Chương > (Mục) > Điều > Khoản > Điểm

8 bước (mỗi bước là 1 nhóm hàm bên dưới):
    1. Kiểm tra file đầu vào        -> check_input_file()
    2. Đọc file DOCX                -> read_docx()
    3. Làm sạch văn bản             -> clean_document()
    4. Nhận diện cấu trúc           -> classify_line()
    5. Tổ chức dữ liệu cha-con      -> build_tree()
    6. Gắn metadata                 -> attach_metadata()
    7. Kiểm tra dữ liệu             -> validate()
    8. Xuất dữ liệu JSON            -> export_outputs()

Cách chạy (từ thư mục gốc rag-luat-lao-dong/):
    python src/parser.py
    python src/parser.py data/raw/luatLaoDong2019.docx
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
import zipfile
from pathlib import Path

from docx import Document                      # đọc file Word
from docx.oxml.ns import qn                    # tên thẻ XML của Word (w:p, w:tbl)
from docx.table import Table
from docx.text.paragraph import Paragraph

# --------------------------------------------------------------------------
# CẤU HÌNH
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent          # thư mục gốc dự án
DEFAULT_INPUT = ROOT / "data" / "raw" / "luatLaoDong2019.docx"
OUTPUT_DIR = ROOT / "data" / "processed"

EXPECTED_ARTICLES = 220        # Bộ luật Lao động 2019 có 220 Điều (dùng để cảnh báo)

# Thứ tự chữ cái dùng cho "Điểm" trong văn bản Việt Nam (có đ, không có f, j, w, z
# trong thực tế, nhưng ta cứ dùng bảng đầy đủ để kiểm tra thứ tự tăng dần).
POINT_ORDER = "abcdđefghijklmnopqrstuvwxyz"


# ==========================================================================
# BƯỚC 1 — KIỂM TRA FILE ĐẦU VÀO
# ==========================================================================
def check_input_file(path: str | Path) -> Path:
    """Kiểm tra file hợp lệ. Sai điều kiện nào thì báo lỗi rõ ràng điều đó."""
    p = Path(path)

    # File "~$xxx.docx" là file khóa tạm Word tự tạo khi đang mở file thật.
    # Nó KHÔNG phải văn bản luật nên phải từ chối.
    if p.name.startswith("~$"):
        raise ValueError(
            f"'{p.name}' là file tạm của Word (đang mở file gốc). "
            "Hãy đóng Word và dùng file luật thật, không có ký tự '~$'."
        )
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {p}")
    if not p.is_file():
        raise ValueError(f"Đây không phải là file: {p}")
    if p.suffix.lower() != ".docx":
        raise ValueError(f"Cần file .docx, nhưng nhận được '{p.suffix}'")
    if p.stat().st_size == 0:
        raise ValueError(f"File rỗng (0 byte): {p}")
    # .docx thực chất là file zip; nếu không phải zip thì file hỏng
    if not zipfile.is_zipfile(p):
        raise ValueError(f"File .docx bị hỏng hoặc không đúng định dạng: {p}")
    return p


# ==========================================================================
# BƯỚC 2 — ĐỌC FILE DOCX
# ==========================================================================
def read_docx(path: Path) -> tuple[list[dict], list[dict]]:
    """
    Đọc toàn bộ file Word, giữ nguyên thứ tự.

    Trả về:
        lines  : [{"line_no": 1, "text": "..."}, ...]   (mỗi đoạn văn = 1 dòng)
        tables : [{"after_line": 1344, "rows": [[ô, ô], ...]}, ...]

    Vì sao phải duyệt thủ công thay vì doc.paragraphs?
    -> doc.paragraphs BỎ QUA các bảng. File này có 2 bảng (ở đầu: số văn bản hợp
       nhất, ở cuối: xác thực), nên ta duyệt từng phần tử của thân văn bản.
    """
    doc = Document(str(path))                  # mở file Word
    lines: list[dict] = []
    tables: list[dict] = []
    line_no = 0

    for child in doc.element.body.iterchildren():
        if child.tag == qn("w:p"):             # w:p = đoạn văn (paragraph)
            line_no += 1
            text = Paragraph(child, doc).text  # lấy chữ của đoạn
            lines.append({"line_no": line_no, "text": text})
        elif child.tag == qn("w:tbl"):         # w:tbl = bảng (table)
            table = Table(child, doc)
            rows = [[cell.text for cell in row.cells] for row in table.rows]
            tables.append({"after_line": line_no, "rows": rows})

    return lines, tables


# ==========================================================================
# BƯỚC 3 — LÀM SẠCH VĂN BẢN
# ==========================================================================
# Chú thích bị dính vào chữ có 2 dạng:
#   dạng A: "3.3[3] Người..."  hoặc "a)2[2]Thành..."  -> số lặp lại + [số]
#   dạng B: "Bộ luật Lao động[1]."                      -> chỉ có [số]
RE_FOOTNOTE_GLUED = re.compile(r"(\d+)\[\1\]")   # \1 = đúng số vừa khớp ở trước
RE_FOOTNOTE_ONLY = re.compile(r"\[(\d+)\]")
# Dòng bắt đầu phần "chú thích cuối trang" của file (dòng đầu tiên là "[1] Luật ...")
RE_FOOTNOTE_SECTION_START = re.compile(r"^\[1\]\s+\S")


def clean_line(raw: str) -> tuple[str, list[int]]:
    """
    Làm sạch 1 dòng. Trả về (dòng_sạch, danh_sách_số_chú_thích_đã_gỡ).
    Chỉ sửa thứ CHẮC CHẮN là lỗi, không đụng vào nội dung luật.
    """
    # 1. Chuẩn hóa Unicode về dạng NFC (chữ có dấu tiếng Việt thống nhất 1 kiểu)
    s = unicodedata.normalize("NFC", raw)

    # 2. Ký tự đặc biệt: \xa0 (khoảng trắng cứng) -> khoảng trắng thường;
    #    các ký tự vô hình (zero-width, BOM) -> xóa bỏ
    s = s.replace("\xa0", " ").replace(" ", " ")
    s = s.replace("​", "").replace("﻿", "")

    # 3. Gỡ dấu chú thích, nhưng GHI LẠI số chú thích để không mất dấu
    marks = [int(m) for m in RE_FOOTNOTE_ONLY.findall(s)]
    s = RE_FOOTNOTE_GLUED.sub("", s)       # "3.3[3] Người" -> "3. Người"
    s = RE_FOOTNOTE_ONLY.sub("", s)        # "lao động[1]." -> "lao động."

    # 4. Gộp nhiều khoảng trắng thành 1, bỏ khoảng trắng đầu/cuối dòng
    s = re.sub(r"[ \t]+", " ", s).strip()
    return s, marks


def clean_document(lines: list[dict]) -> tuple[list[dict], list[str], dict]:
    """
    Làm sạch toàn bộ.

    Trả về:
        body      : các dòng sạch của PHẦN LUẬT (đã bỏ dòng trống, bỏ chú thích cuối)
                    [{"line_no", "text", "footnotes": [số, ...]}, ...]
        footnotes : các dòng chú thích cuối file (giữ nguyên để tra cứu)
        stats     : thống kê để báo cáo

    Ghi chú: trong file DOCX, mỗi đoạn văn (paragraph) đã là 1 dòng logic nên
    KHÔNG cần "nối dòng bị ngắt giữa chừng" như khi đọc PDF. Nối bừa còn có thể
    làm dính 2 Khoản vào nhau, nên bước này cố ý không nối.
    """
    # Tìm nơi bắt đầu phần chú thích cuối file ("[1] Luật Công nghiệp ...")
    cut = len(lines)
    for i, ln in enumerate(lines):
        plain = ln["text"].replace("\xa0", " ").strip()
        if RE_FOOTNOTE_SECTION_START.match(plain):
            cut = i
            break

    body: list[dict] = []
    empty_removed = 0
    for ln in lines[:cut]:
        text, marks = clean_line(ln["text"])
        if not text:                       # dòng trống -> bỏ
            empty_removed += 1
            continue
        body.append({"line_no": ln["line_no"], "text": text, "footnotes": marks})

    footnotes = [clean_line(ln["text"])[0] for ln in lines[cut:]]
    footnotes = [f for f in footnotes if f]

    stats = {
        "raw_lines": len(lines),
        "clean_lines": len(body),
        "empty_lines_removed": empty_removed,
        "footnote_lines": len(footnotes),
    }
    return body, footnotes, stats


# ==========================================================================
# BƯỚC 4 — NHẬN DIỆN CẤU TRÚC (regex)
# ==========================================================================
RE_CHAPTER = re.compile(r"^Chương\s+([IVXLC]+)$")            # "Chương I"
RE_SECTION = re.compile(r"^Mục\s+(\d+)\.\s+(.+)$")           # "Mục 1. GIAO KẾT ..."
RE_ARTICLE = re.compile(r"^Điều\s+(\d+)\.\s+(.+)$")          # "Điều 3. Giải thích từ ngữ"
RE_CLAUSE = re.compile(r"^(\d+[a-z]?)\.\s+(.+)$")            # "1. ..." hoặc "8a. ..."
RE_POINT = re.compile(r"^([a-zđ])\)\s*(.+)$")                # "a) ..." hoặc "đ) ..."


def classify_line(text: str) -> dict:
    """
    Cho 1 dòng sạch, xác định nó thuộc loại nào.
    Thử từ cấp cao xuống thấp. Trả về dict có khóa "type":
        chapter | section | article | clause | point | text
    "text" = dòng nối tiếp (không bắt đầu bằng mẫu nào) -> thuộc mục phía trước.
    """
    m = RE_CHAPTER.match(text)
    if m:
        return {"type": "chapter", "number": m.group(1)}
    m = RE_SECTION.match(text)
    if m:
        return {"type": "section", "number": m.group(1), "title": m.group(2)}
    m = RE_ARTICLE.match(text)
    if m:
        return {"type": "article", "number": int(m.group(1)), "title": m.group(2)}
    m = RE_CLAUSE.match(text)
    if m:
        return {"type": "clause", "number": m.group(1), "text": m.group(2)}
    m = RE_POINT.match(text)
    if m:
        return {"type": "point", "number": m.group(1), "text": m.group(2)}
    return {"type": "text"}


# ==========================================================================
# BƯỚC 5 — TỔ CHỨC DỮ LIỆU CHA-CON (máy trạng thái)
# ==========================================================================
def _closes_quote(text: str) -> bool:
    """Dòng có kết thúc một đoạn trích dẫn không? (kết thúc bằng ” hoặc ”. ”; ”,)"""
    return bool(re.search(r"”\s*[.;,]*\s*$", text))


def build_tree(body: list[dict]) -> tuple[dict, list[str]]:
    """
    Đọc tuần tự từng dòng sạch, dựng cây:
        chapters[] > articles[] > clauses[] > points[]
    Mục (Mục 1, Mục 2...) lưu ở chapter["sections"], và mỗi Điều ghi rõ nó thuộc Mục nào.

    Hai tình huống đặc biệt của file này:
      (a) Phần mở đầu trước "Chương I" -> lưu riêng vào "preamble".
      (b) Điều 219 TRÍCH NGUYÊN VĂN điều luật của luật khác, bao trong “ ... ”.
          Bên trong đó cũng có "Điều 54.", "1.", "a)" nhưng đó KHÔNG phải cấu trúc
          của Bộ luật Lao động -> ta coi cả đoạn trích là nội dung của mục đang mở.
    """
    preamble: list[dict] = []
    chapters: list[dict] = []
    warnings: list[str] = []

    # "Con trỏ" cho biết đang đứng ở đâu trong cây
    chapter = section = article = clause = point = None
    waiting_chapter_title = False      # vừa gặp "Chương X", dòng kế là tên chương
    in_quote = False                   # đang ở trong đoạn trích dẫn “ ... ”

    def touch(n: int, marks: list[int]) -> None:
        """Cập nhật dòng cuối + số chú thích cho mọi nút đang mở."""
        for node in (chapter, article, clause, point):
            if node is not None:
                node["line_end"] = n
        target = point or clause or article
        if target is not None:
            for m in marks:
                if m not in target["footnote_refs"]:
                    target["footnote_refs"].append(m)

    def continuation_target() -> dict:
        """
        Dòng nối tiếp (không có số/chữ đầu dòng) gắn vào đâu?
          - Nếu đang ở Điểm và Điểm chưa kết thúc (không có . hoặc ;) -> gắn vào Điểm
          - Ngược lại, nếu đang ở Khoản -> gắn vào Khoản
          - Ngược lại -> gắn vào phần dẫn (intro) của Điều
        """
        if point is not None and not point["text"].rstrip().endswith((".", ";")):
            return point
        if clause is not None:
            return clause
        return article

    def append_text(node: dict, text: str) -> None:
        key = "intro" if node is article and node is not clause else "text"
        node[key] = f"{node[key]}\n{text}" if node[key] else text

    for ln in body:
        text, n, marks = ln["text"], ln["line_no"], ln["footnotes"]
        info = classify_line(text)

        # ---- (a) Phần mở đầu: mọi thứ trước dòng "Chương I" ----
        if chapter is None and info["type"] != "chapter":
            preamble.append({"line_no": n, "text": text})
            continue

        # ---- Dòng ngay sau "Chương X" là tên chương ----
        if waiting_chapter_title:
            waiting_chapter_title = False
            if info["type"] == "text":
                chapter["chapter_title"] = text
                chapter["line_end"] = n
                continue
            # không phải tên chương -> chương này không có tên, xử lý dòng bình thường

        # ---- (b) Đang trong đoạn trích dẫn: không nhận diện cấu trúc ----
        if in_quote:
            if article is None:
                warnings.append(f"Dòng {n}: trích dẫn nhưng chưa có Điều")
            else:
                append_text(continuation_target(), text)
                touch(n, marks)
            if _closes_quote(text):
                in_quote = False
            continue

        # ---- Xử lý theo loại dòng ----
        kind = info["type"]

        if kind == "chapter":
            chapter = {"chapter_no": info["number"], "chapter_title": "",
                       "sections": [], "articles": [],
                       "line_start": n, "line_end": n}
            chapters.append(chapter)
            section = article = clause = point = None
            waiting_chapter_title = True

        elif kind == "section":
            section = {"section_no": info["number"], "section_title": info["title"],
                       "line_start": n}
            chapter["sections"].append(section)
            article = clause = point = None
            chapter["line_end"] = n

        elif kind == "article":
            article = {
                "article_no": info["number"], "article_title": info["title"],
                "section_no": section["section_no"] if section else None,
                "section_title": section["section_title"] if section else None,
                "intro": "", "clauses": [],
                "footnote_refs": [], "line_start": n, "line_end": n,
            }
            chapter["articles"].append(article)
            clause = point = None
            touch(n, marks)

        elif kind == "clause":
            if article is None:
                warnings.append(f"Dòng {n}: Khoản nằm ngoài Điều: {text[:50]}")
                continue
            clause = {"clause_no": info["number"], "text": info["text"], "points": [],
                      "footnote_refs": [], "line_start": n, "line_end": n}
            article["clauses"].append(clause)
            point = None
            touch(n, marks)

        elif kind == "point":
            if article is None:
                warnings.append(f"Dòng {n}: Điểm nằm ngoài Điều: {text[:50]}")
                continue
            if clause is None:
                # Điểm nằm thẳng dưới Điều (không có Khoản) -> tạo Khoản ngầm định
                clause = {"clause_no": None, "text": "", "points": [],
                          "footnote_refs": [], "line_start": n, "line_end": n}
                article["clauses"].append(clause)
                warnings.append(f"Dòng {n}: Điểm không có Khoản, đã tạo Khoản ngầm định")
            point = {"point_no": info["number"], "text": info["text"],
                     "footnote_refs": [], "line_start": n, "line_end": n}
            clause["points"].append(point)
            touch(n, marks)

        else:  # kind == "text": dòng nối tiếp
            if article is None:
                warnings.append(f"Dòng {n}: dòng lẻ ngoài Điều: {text[:50]}")
                continue
            append_text(continuation_target(), text)
            touch(n, marks)
            # Mở đầu một đoạn trích dẫn nhiều dòng: “... (chưa đóng ”)
            if text.startswith("“") and not _closes_quote(text):
                in_quote = True

    if in_quote:
        warnings.append("Đoạn trích dẫn “ ... ” không được đóng ở cuối văn bản")

    return {"preamble": preamble, "chapters": chapters}, warnings


# ==========================================================================
# BƯỚC 6 — GẮN METADATA
# ==========================================================================
def _iso_date(d: str, m: str, y: str) -> str:
    return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"


def extract_document_info(preamble: list[dict], tables: list[dict]) -> dict:
    """Lấy số hiệu, tên luật, ngày ban hành, ngày hiệu lực từ phần mở đầu và bảng."""
    info = {"doc_id": None, "title": None, "issued_date": None,
            "effective_date": None, "consolidated_no": None,
            "consolidated_date": None}

    # Dòng kiểu: "Bộ luật Lao động số 45/2019/QH14 ngày 20 tháng 11 năm 2019 ...
    #             có hiệu lực kể từ ngày 01 tháng 01 năm 2021, ..."
    pat = re.compile(
        r"^(.+?)\s+số\s+(\S+)\s+ngày\s+(\d+)\s+tháng\s+(\d+)\s+năm\s+(\d+).*?"
        r"hiệu lực kể từ ngày\s+(\d+)\s+tháng\s+(\d+)\s+năm\s+(\d+)"
    )
    for p in preamble:
        m = pat.match(p["text"])
        if m:
            info["title"] = m.group(1)
            info["doc_id"] = m.group(2)
            info["issued_date"] = _iso_date(m.group(3), m.group(4), m.group(5))
            info["effective_date"] = _iso_date(m.group(6), m.group(7), m.group(8))
            break

    # Bảng đầu file: "Số: 18/VBHN-VPQH" và "Hà Nội, ngày 12 tháng 02 năm 2026"
    if tables:
        flat = " ".join(c for row in tables[0]["rows"] for c in row)
        m = re.search(r"Số:\s*(\S+)", flat)
        if m:
            info["consolidated_no"] = m.group(1)
        m = re.search(r"ngày\s+(\d+)\s+tháng\s+(\d+)\s+năm\s+(\d+)", flat)
        if m:
            info["consolidated_date"] = _iso_date(*m.groups())
    return info


def attach_metadata(tree: dict, info: dict, source_file: str) -> dict:
    """
    Ghép thông tin văn bản + gắn "full_path" và "citation" cho mọi Điều/Khoản/Điểm.
    full_path : "Chương I > Điều 3 > Khoản 1 > Điểm a"  (để hiển thị, lọc)
    citation  : "điểm a khoản 1 Điều 3 Bộ luật Lao động 2019" (đúng thể thức Việt Nam,
                 dùng ở Giai đoạn 5 để trích nguồn)
    """
    year = (info["issued_date"] or "")[:4]
    law_name = f'{info["title"]} {year}'.strip()

    for ch in tree["chapters"]:
        for art in ch["articles"]:
            base_path = [f'Chương {ch["chapter_no"]}']
            if art["section_no"]:
                base_path.append(f'Mục {art["section_no"]}')
            base_path.append(f'Điều {art["article_no"]}')
            art["full_path"] = " > ".join(base_path)
            art["citation"] = f'Điều {art["article_no"]} {law_name}'

            for cl in art["clauses"]:
                if cl["clause_no"] is not None:
                    cl_path = base_path + [f'Khoản {cl["clause_no"]}']
                    cl_cite = f'khoản {cl["clause_no"]} Điều {art["article_no"]} {law_name}'
                else:                                  # Khoản ngầm định
                    cl_path, cl_cite = base_path, art["citation"]
                cl["full_path"] = " > ".join(cl_path)
                cl["citation"] = cl_cite

                for pt in cl["points"]:
                    pt["full_path"] = " > ".join(cl_path + [f'Điểm {pt["point_no"]}'])
                    pt["citation"] = f'điểm {pt["point_no"]} {cl_cite}'

    return {
        "doc_id": info["doc_id"],
        "title": info["title"],
        "law_name": law_name,
        "issued_date": info["issued_date"],
        "effective_date": info["effective_date"],
        "consolidated_no": info["consolidated_no"],        # số văn bản hợp nhất
        "consolidated_date": info["consolidated_date"],
        "source_file": source_file,
        "preamble": [p["text"] for p in tree["preamble"]],
        "chapters": tree["chapters"],
    }


# ==========================================================================
# BƯỚC 7 — KIỂM TRA DỮ LIỆU
# ==========================================================================
ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


def roman_to_int(s: str) -> int:
    total = 0
    for i, ch in enumerate(s):
        v = ROMAN[ch]
        total += -v if i + 1 < len(s) and ROMAN[s[i + 1]] > v else v
    return total


def _alnum_count(texts) -> int:
    """Đếm số chữ + số (bỏ khoảng trắng, dấu câu) — dùng để so sánh trước/sau."""
    return sum(ch.isalnum() for t in texts for ch in t)


def _tree_texts(doc: dict):
    """Dựng lại toàn bộ chữ từ cây JSON (kèm nhãn Chương/Điều/Khoản/Điểm)."""
    yield from doc["preamble"]
    for ch in doc["chapters"]:
        yield f'Chương {ch["chapter_no"]}'
        yield ch["chapter_title"]
        for sec in ch["sections"]:
            yield f'Mục {sec["section_no"]}. {sec["section_title"]}'
        for art in ch["articles"]:
            yield f'Điều {art["article_no"]}. {art["article_title"]}'
            yield art["intro"]
            for cl in art["clauses"]:
                if cl["clause_no"] is not None:
                    yield f'{cl["clause_no"]}. {cl["text"]}'
                else:
                    yield cl["text"]
                for pt in cl["points"]:
                    yield f'{pt["point_no"]}) {pt["text"]}'


def validate(doc: dict, body: list[dict], build_warnings: list[str]) -> dict:
    """Chạy các kiểm tra, trả về báo cáo {counts, errors, warnings, ok}."""
    errors: list[str] = []
    warnings: list[str] = list(build_warnings)
    chapters = doc["chapters"]
    articles = [a for ch in chapters for a in ch["articles"]]

    # --- 1. Chương: số La Mã phải liên tục 1..N ---
    nums = [roman_to_int(ch["chapter_no"]) for ch in chapters]
    if nums != list(range(1, len(nums) + 1)):
        errors.append(f"Số Chương không liên tục: {nums}")

    # --- 2. Điều: phải liên tục 1..N, không thiếu, không trùng ---
    a_nums = [a["article_no"] for a in articles]
    if a_nums != list(range(1, len(a_nums) + 1)):
        missing = sorted(set(range(1, max(a_nums, default=0) + 1)) - set(a_nums))
        dup = sorted({x for x in a_nums if a_nums.count(x) > 1})
        errors.append(f"Số Điều không liên tục. Thiếu: {missing}; trùng: {dup}")
    if len(articles) != EXPECTED_ARTICLES:
        warnings.append(f"Số Điều = {len(articles)}, kỳ vọng {EXPECTED_ARTICLES}")

    # --- 3. Từng Điều: không rỗng; Khoản liên tục; Điểm đúng thứ tự ---
    n_clauses = n_points = 0
    for a in articles:
        label = f'Điều {a["article_no"]}'
        if not a["intro"] and not a["clauses"]:
            errors.append(f"{label} rỗng, không có nội dung")

        prev = 0
        for cl in a["clauses"]:
            n_clauses += 1
            no = cl["clause_no"]
            if no is not None:
                base = int(re.match(r"\d+", no).group())
                has_suffix = no != str(base)           # "8a", "1b" ...
                # Khoản thường phải = trước+1; Khoản có chữ (1a) giữ nguyên số trước
                ok = (base == prev) if has_suffix else (base == prev + 1)
                if not ok:
                    errors.append(f"{label}: Khoản '{no}' sai thứ tự (trước đó là {prev})")
                prev = base

            last_idx = -1
            for pt in cl["points"]:
                n_points += 1
                idx = POINT_ORDER.index(pt["point_no"])
                if idx <= last_idx:
                    errors.append(
                        f'{label}, Khoản {no}: Điểm {pt["point_no"]}) sai thứ tự')
                last_idx = idx

    # --- 4. Không còn ký tự lỗi trong phần luật ---
    for ln in body:
        if "\xa0" in ln["text"] or RE_FOOTNOTE_ONLY.search(ln["text"]):
            errors.append(f'Dòng {ln["line_no"]} còn ký tự lỗi / dấu chú thích')

    # --- 5. Không mất dữ liệu: số chữ+số trong JSON phải bằng trong text sạch ---
    before = _alnum_count(ln["text"] for ln in body)
    after = _alnum_count(_tree_texts(doc))
    if before != after:
        errors.append(f"Mất/thừa dữ liệu: text sạch {before} ký tự, JSON {after} ký tự")

    report = {
        "counts": {
            "chapters": len(chapters),
            "sections": sum(len(c["sections"]) for c in chapters),
            "articles": len(articles),
            "clauses": n_clauses,
            "points": n_points,
            "preamble_lines": len(doc["preamble"]),
            "alnum_chars_clean_text": before,
            "alnum_chars_json": after,
        },
        "errors": errors,
        "warnings": warnings,
        "ok": not errors,
    }
    return report


# ==========================================================================
# BƯỚC 8 — XUẤT DỮ LIỆU JSON
# ==========================================================================
def export_outputs(doc: dict, body: list[dict], footnotes: list[str],
                   tables: list[dict], report: dict, out_dir: Path) -> dict:
    """Ghi 3 file: clean.txt, structured_law.json, validation_report.json."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Văn bản sạch (để đối chiếu bằng mắt khi debug)
    clean_path = out_dir / "clean.txt"
    clean_path.write_text("\n".join(ln["text"] for ln in body), encoding="utf-8")

    # 2) Dữ liệu có cấu trúc — đầu vào của Giai đoạn 2 (Chunking)
    out = dict(doc)
    out["footnotes_raw"] = footnotes       # nội dung chú thích cuối file
    out["tables_raw"] = tables             # 2 bảng đầu/cuối file
    law_path = out_dir / "structured_law.json"
    law_path.write_text(
        json.dumps(out, ensure_ascii=False, indent=2),   # ensure_ascii=False: giữ tiếng Việt
        encoding="utf-8",
    )

    # 3) Báo cáo kiểm tra
    report_path = out_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    return {"clean": clean_path, "law": law_path, "report": report_path}


# ==========================================================================
# CHẠY TOÀN BỘ 8 BƯỚC
# ==========================================================================
def run(input_path: str | Path = DEFAULT_INPUT, out_dir: Path = OUTPUT_DIR) -> dict:
    path = check_input_file(input_path)                            # Bước 1
    lines, tables = read_docx(path)                                # Bước 2
    body, footnotes, _ = clean_document(lines)                     # Bước 3
    tree, build_warnings = build_tree(body)                        # Bước 4 + 5
    info = extract_document_info(tree["preamble"], tables)         # Bước 6
    doc = attach_metadata(tree, info, path.name)
    report = validate(doc, body, build_warnings)                   # Bước 7
    paths = export_outputs(doc, body, footnotes, tables, report, out_dir)  # Bước 8
    return {"doc": doc, "report": report, "paths": paths}


def main() -> int:
    input_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    try:
        result = run(input_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"LỖI ĐẦU VÀO: {e}")
        return 1

    rep = result["report"]
    print("Số lượng:", json.dumps(rep["counts"], ensure_ascii=False))
    for w in rep["warnings"]:
        print("CẢNH BÁO:", w)
    for e in rep["errors"]:
        print("LỖI:", e)
    print("Kết quả:", "ĐẠT" if rep["ok"] else "KHÔNG ĐẠT")
    for k, v in result["paths"].items():
        print(f"  đã ghi {k}: {v}")
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())