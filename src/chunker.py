"""
chunker.py — GIAI ĐOẠN 2: CHIA CHUNK
====================================
Đọc data/processed/structured_law.json (đầu ra Giai đoạn 1) và chia thành các
"chunk" để tạo embedding ở Giai đoạn 3. Mỗi chunk là 1 dòng trong chunks.jsonl.

Chiến lược: "chia theo cấu trúc pháp lý, chỉ cắt khi quá dài"
    1. Mặc định MỖI ĐIỀU = 1 CHUNK (Điều là đơn vị ý nghĩa hoàn chỉnh).
    2. Điều quá dài (> MAX_WORDS) -> cắt theo KHOẢN, gom các Khoản liền kề cho đầy chunk.
    3. Khoản quá dài -> cắt theo ĐIỂM (lặp lại câu dẫn của Khoản ở mỗi chunk).
    4. Điểm/đoạn quá dài -> cắt theo CÂU, có gối đầu (overlap) một câu ngắn.
    5. Câu dài hơn MAX_WORDS -> cắt cửa sổ theo từ (hiếm, có overlap).
    Không bao giờ cắt ngang giữa câu trừ trường hợp 5.

6 bước:
    1. Đọc dữ liệu Giai đoạn 1        -> load_structured_law()
    2. Dựng dòng nội dung từ cây       -> clause_lines(), leaf_lines()
    3. Cắt Khoản/Điểm/câu quá dài      -> split_clause(), pack_text()
    4. Gom thành chunk + metadata      -> chunk_article(), build_chunk()
    5. Kiểm tra                        -> validate_chunks()
    6. Xuất JSONL                      -> export_outputs()

Cách chạy (từ thư mục gốc dự án):
    python src/chunker.py
    python src/chunker.py --max-words 250 --overlap 30
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# CẤU HÌNH
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "data" / "processed" / "structured_law.json"
OUTPUT_DIR = ROOT / "data" / "processed"

# Đơn vị đo: "từ" = cụm ký tự cách nhau bởi khoảng trắng. Tiếng Việt mỗi từ ~ 1 âm tiết,
# 300 từ ~ 450-600 token của các mô hình embedding đa ngôn ngữ phổ biến.
# Dữ liệu thật: Điều trung vị ~150 từ, chỉ ~29/220 Điều dài hơn 300 từ.
MAX_WORDS = 300        # kích thước tối đa phần nội dung của 1 chunk (chưa tính header)
OVERLAP_WORDS = 30     # câu cuối được lặp lại ở chunk sau nếu ngắn hơn ngưỡng này
N_GRAM = 8             # độ dài cụm từ dùng để kiểm tra phủ nội dung (với câu rất dài)


def wc(text: str) -> int:
    """Đếm từ (cụm cách nhau bởi khoảng trắng)."""
    return len(text.split())


# ==========================================================================
# BƯỚC 1 — ĐỌC DỮ LIỆU GIAI ĐOẠN 1
# ==========================================================================
def load_structured_law(path: str | Path = INPUT_FILE) -> dict:
    """Đọc structured_law.json và kiểm tra có đúng dạng cần thiết không."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Không thấy {p}. Hãy chạy Giai đoạn 1 trước: python src/parser.py")
    doc = json.loads(p.read_text(encoding="utf-8"))
    for key in ("doc_id", "law_name", "chapters"):
        if not doc.get(key):
            raise ValueError(
                f"structured_law.json có '{key}' rỗng hoặc thiếu. "
                "Hãy chạy lại Giai đoạn 1 bằng parser.py mới: python src/parser.py")
    return doc


# ==========================================================================
# BƯỚC 2 — DỰNG DÒNG NỘI DUNG TỪ CÂY
# ==========================================================================
def clause_lines(clause: dict) -> tuple[list[str], list[dict]]:
    """
    Tách 1 Khoản thành:
        lead   : các dòng của chính Khoản ("1. Người lao động có quyền...")
        blocks : mỗi Điểm là 1 khối {"no": "a", "lines": ["a) ...", ...], ...}
    Nhãn "1." và "a)" được gắn lại để chunk đọc được như văn bản luật.
    """
    lead: list[str] = clause["text"].split("\n") if clause["text"] else []
    if lead and clause["clause_no"] is not None:
        lead[0] = f'{clause["clause_no"]}. {lead[0]}'
    blocks = []
    for pt in clause["points"]:
        lines = pt["text"].split("\n")
        lines[0] = f'{pt["point_no"]}) {lines[0]}'
        blocks.append({"no": pt["point_no"], "lines": lines, "node": pt})
    return lead, blocks


def leaf_lines(article: dict) -> list[str]:
    """Toàn bộ dòng nội dung của 1 Điều theo thứ tự (không gồm dòng tiêu đề Điều)."""
    lines: list[str] = []
    if article["intro"]:
        lines += article["intro"].split("\n")
    for cl in article["clauses"]:
        lead, blocks = clause_lines(cl)
        lines += lead
        for b in blocks:
            lines += b["lines"]
    return lines


def _make_piece(kind, lines, clause_nos, point_nos, footnotes, ls, le,
                lead_n=0, lead_key=None) -> dict:
    """
    "Mảnh" = đơn vị nhỏ nhất được gom vào chunk. Mỗi mảnh <= MAX_WORDS.
    kind: intro | clause | points | window
    lead_n/lead_key: số dòng đầu là câu dẫn của Khoản được lặp lại (để khỏi lặp 2 lần
                     trong cùng 1 chunk).
    """
    return {"kind": kind, "lines": lines, "words": wc(" ".join(lines)),
            "clause_nos": clause_nos, "point_nos": point_nos,
            "footnotes": footnotes, "line_start": ls, "line_end": le,
            "lead_n": lead_n, "lead_key": lead_key}


# ==========================================================================
# BƯỚC 3 — CẮT KHOẢN / ĐIỂM / CÂU QUÁ DÀI
# ==========================================================================
RE_LABEL_ONLY = re.compile(r"^\d+[a-z]?\.$")      # nhãn Khoản đứng một mình: "1." "8a."


def split_sentences(text: str) -> list[str]:
    """
    Tách câu: sau dấu . ; : rồi đến khoảng trắng. (Dùng chung với bước kiểm tra.)
    Nhãn Khoản ở đầu ("1.") không phải dấu hết câu nên được nối vào câu liền sau.
    """
    parts = [s for s in re.split(r"(?<=[.;:])\s+", text.strip()) if s]
    if len(parts) > 1 and RE_LABEL_ONLY.match(parts[0]):
        parts = [parts[0] + " " + parts[1]] + parts[2:]
    return parts


def hard_windows(words: list[str], max_words: int, overlap: int) -> list[str]:
    """Cắt cửa sổ trượt theo từ cho 1 câu dài hơn max_words (có gối đầu)."""
    step = max(1, max_words - overlap)
    out, i = [], 0
    while True:
        out.append(" ".join(words[i:i + max_words]))
        if i + max_words >= len(words):
            break
        i += step
    return out


def pack_text(text: str, max_words: int, overlap: int) -> list[str]:
    """
    Chia 1 đoạn dài thành các mảnh <= max_words, theo ranh giới câu.
    Nếu câu cuối của mảnh trước ngắn (<= overlap từ) thì lặp lại ở đầu mảnh sau
    để giữ mạch ý.
    """
    out: list[str] = []
    cur: list[str] = []
    curw = 0

    def flush():
        nonlocal cur, curw
        if cur:
            out.append(" ".join(cur))
        cur, curw = [], 0

    for s in split_sentences(text):
        w = wc(s)
        if w > max_words:                      # câu quá dài: cắt cửa sổ theo từ
            flush()
            out.extend(hard_windows(s.split(), max_words, overlap))
            continue
        if cur and curw + w > max_words:
            last = cur[-1]
            flush()
            if wc(last) <= overlap and wc(last) + w <= max_words:
                cur, curw = [last], wc(last)   # gối đầu
        cur.append(s)
        curw += w
    flush()
    return out


def split_clause(cl: dict, max_words: int, overlap: int) -> list[dict]:
    """
    Biến 1 Khoản thành 1 hoặc nhiều mảnh (mỗi mảnh <= max_words).
      - Khoản vừa -> 1 mảnh nguyên vẹn.
      - Khoản dài có Điểm -> gom các Điểm; câu dẫn của Khoản lặp lại ở mỗi mảnh.
      - Điểm hoặc Khoản (không có Điểm) quá dài -> cắt theo câu.
    """
    no = cl["clause_no"]
    cnos = [no] if no is not None else []
    lead, blocks = clause_lines(cl)
    refs_all = list(cl["footnote_refs"]) + [m for b in blocks for m in b["node"]["footnote_refs"]]
    all_lines = lead + [ln for b in blocks for ln in b["lines"]]

    if wc(" ".join(all_lines)) <= max_words:                 # vừa: giữ nguyên
        return [_make_piece("clause", all_lines, cnos, [], refs_all,
                            cl["line_start"], cl["line_end"])]

    pieces: list[dict] = []
    lead_text = " ".join(lead)

    if not blocks:                                            # dài nhưng không có Điểm
        for part in pack_text(lead_text, max_words, overlap):
            pieces.append(_make_piece("window", [part], cnos, [], list(cl["footnote_refs"]),
                                      cl["line_start"], cl["line_end"]))
        return pieces

    # Câu dẫn ngắn (<= 1/3 giới hạn) thì lặp ở mỗi mảnh; dài thì tách riêng 1 lần.
    repeat_lead = bool(lead) and wc(lead_text) <= max_words // 3
    if lead and not repeat_lead:
        for part in pack_text(lead_text, max_words, overlap):
            pieces.append(_make_piece("window", [part], cnos, [], list(cl["footnote_refs"]),
                                      cl["line_start"], cl["line_start"]))
        prefix: list[str] = []
        budget = max_words
    else:
        prefix = lead
        budget = max_words - wc(lead_text) if repeat_lead else max_words
    lead_n = len(prefix)

    group: list[dict] = []
    gw = 0

    def emit():
        nonlocal group, gw
        if not group:
            return
        lines = prefix + [ln for b in group for ln in b["lines"]]
        refs = (list(cl["footnote_refs"]) if prefix else []) + \
               [m for b in group for m in b["node"]["footnote_refs"]]
        ls = cl["line_start"] if prefix else group[0]["node"]["line_start"]
        pieces.append(_make_piece("points", lines, cnos, [b["no"] for b in group], refs,
                                  ls, group[-1]["node"]["line_end"], lead_n, no))
        group, gw = [], 0

    for b in blocks:
        bw = wc(" ".join(b["lines"]))
        if bw > budget:                                       # 1 Điểm quá dài: cắt theo câu
            emit()
            for part in pack_text(" ".join(b["lines"]), budget, overlap):
                pieces.append(_make_piece(
                    "window", prefix + [part], cnos, [b["no"]],
                    list(b["node"]["footnote_refs"]),
                    b["node"]["line_start"], b["node"]["line_end"], lead_n, no))
            continue
        if group and gw + bw > budget:
            emit()
        group.append(b)
        gw += bw
    emit()
    return pieces


def pack_pieces(pieces: list[dict], max_words: int) -> list[list[dict]]:
    """Gom các mảnh liền kề thành nhóm sao cho tổng từ <= max_words."""
    groups: list[list[dict]] = []
    cur: list[dict] = []
    curw = 0
    for p in pieces:
        if cur and curw + p["words"] > max_words:
            groups.append(cur)
            cur, curw = [], 0
        cur.append(p)
        curw += p["words"]
    if cur:
        groups.append(cur)
    return groups


# ==========================================================================
# BƯỚC 4 — GOM THÀNH CHUNK + METADATA
# ==========================================================================
# Tham chiếu chéo "Điều 35": chỉ lấy khi đó là Điều của CHÍNH Bộ luật này.
RE_ARTICLE_REF = re.compile(r"Điều\s+(\d+)")
RE_OTHER_LAW = re.compile(
    r"\s+của\s+(?:Luật|Bộ luật|Nghị định|Nghị quyết|Pháp lệnh|Thông tư|Hiến pháp)"
    r"(?!\s+này)(?!\s+Lao động(?!\s+số))")


# Dòng mở đầu một đoạn trích luật khác: “... (có thể sau nhãn "1." hoặc "a)")
RE_QUOTE_START = re.compile(r"^\s*(?:\d+[a-z]?\.|[a-zđ]\))?\s*“")


def extract_references(text: str, own_article: int) -> list[int]:
    """
    Tìm các Điều khác trong cùng Bộ luật được nhắc tới, ví dụ "khoản 2 Điều 35".
    Bỏ qua: Điều của luật khác ("Điều 54 của Luật Bảo hiểm xã hội") và chính Điều đang xét.
    Là best-effort (không bắt "các điều 11, 28 và 29" dạng liệt kê).
    """
    refs = set()
    for m in RE_ARTICLE_REF.finditer(text):
        if RE_OTHER_LAW.match(text[m.end(1):m.end(1) + 80]):
            continue
        n = int(m.group(1))
        if n != own_article:
            refs.add(n)
    return sorted(refs)


def _unique(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def make_citation(law_name: str, art_no: int, level: str, has_intro: bool,
                  clause_nos: list[str], point_nos: list[str]) -> str:
    """Trích dẫn theo thể thức Việt Nam: 'điểm a khoản 1 Điều 3 Bộ luật Lao động 2019'."""
    base = f"Điều {art_no} {law_name}"
    if level == "article" or has_intro or not clause_nos:
        return base
    if len(clause_nos) == 1:
        c = clause_nos[0]
        if point_nos:
            word = "điểm" if len(point_nos) == 1 else "các điểm"
            return f'{word} {", ".join(point_nos)} khoản {c} {base}'
        return f"khoản {c} {base}"
    return f'các khoản {", ".join(clause_nos)} {base}'


def build_chunk(group: list[dict], level: str, ctx: dict, part: int) -> dict:
    """Ghép 1 nhóm mảnh thành 1 chunk hoàn chỉnh với đủ metadata."""
    art, ch, doc = ctx["article"], ctx["chapter"], ctx["doc"]

    # Ghép dòng; bỏ câu dẫn lặp nếu 2 mảnh liền nhau cùng Khoản
    lines: list[str] = []
    prev = None
    for p in group:
        ls = p["lines"]
        if prev and p["lead_n"] and prev["lead_n"] and p["lead_key"] == prev["lead_key"]:
            ls = ls[p["lead_n"]:]
        lines += ls
        prev = p
    body = "\n".join(lines)

    has_intro = any(p["kind"] == "intro" for p in group)
    clause_nos = _unique(c for p in group for c in p["clause_nos"])
    point_nos = _unique(n for p in group if p["kind"] in ("points", "window")
                        for n in p["point_nos"])

    # Dòng ngữ cảnh gắn trước nội dung khi tạo embedding (giúp tìm đúng Điều/Chương)
    head = [doc["law_name"], f'Chương {ch["chapter_no"]}. {ch["chapter_title"]}']
    if art["section_no"]:
        head.append(f'Mục {art["section_no"]}. {art["section_title"]}')
    header = " | ".join(head) + f'\nĐiều {art["article_no"]}. {art["article_title"]}'

    # Đường dẫn
    if level == "article" or has_intro or not clause_nos:
        full_path = art["full_path"]
    elif len(clause_nos) == 1:
        c0 = next(c for c in art["clauses"] if c["clause_no"] == clause_nos[0])
        full_path = c0["full_path"] + (f' > Điểm {", ".join(point_nos)}' if point_nos else "")
    else:
        full_path = f'{art["full_path"]} > Khoản {", ".join(clause_nos)}'

    footnotes = sorted(set(art["footnote_refs"]) | {m for p in group for m in p["footnotes"]})
    quoted = ctx["quoted"]
    return {
        "chunk_id": f'{doc["slug"]}_D{art["article_no"]:03d}_P{part:02d}',
        "doc_id": doc["doc_id"],
        "law_name": doc["law_name"],
        "chapter_no": ch["chapter_no"],
        "chapter_title": ch["chapter_title"],
        "section_no": art["section_no"],
        "section_title": art["section_title"],
        "article_no": art["article_no"],
        "article_title": art["article_title"],
        "level": level,                      # article | clauses | points | window
        "part_index": part,
        "part_total": 0,                     # điền sau khi biết tổng số phần
        "clause_nos": clause_nos,
        "point_nos": point_nos,
        "full_path": full_path,
        "citation": make_citation(doc["law_name"], art["article_no"], level,
                                  has_intro, clause_nos, point_nos),
        "text": body,
        "text_for_embedding": f"{header}\n{body}",
        "word_count": wc(body),
        "footnote_refs": footnotes,
        "references": [] if quoted else extract_references(body, art["article_no"]),
        "quoted_other_law": quoted,          # Điều chứa đoạn trích luật khác (Điều 219)
        "line_start": min(p["line_start"] for p in group),
        "line_end": max(p["line_end"] for p in group),
    }


def article_pieces(article: dict, max_words: int, overlap: int) -> list[dict]:
    """Toàn bộ mảnh của 1 Điều theo thứ tự: phần dẫn (nếu có) rồi các Khoản."""
    pieces: list[dict] = []
    if article["intro"]:
        first_cl = article["clauses"][0]["line_start"] - 1 if article["clauses"] \
            else article["line_end"]
        intro_lines = article["intro"].split("\n")
        if wc(" ".join(intro_lines)) <= max_words:
            pieces.append(_make_piece("intro", intro_lines, [], [], list(article["footnote_refs"]),
                                      article["line_start"], max(first_cl, article["line_start"])))
        else:                                         # phần dẫn quá dài (vd Điều 218 dài)
            for part in pack_text(" ".join(intro_lines), max_words, overlap):
                pieces.append(_make_piece("window", [part], [], [], list(article["footnote_refs"]),
                                          article["line_start"], article["line_end"]))
    for cl in article["clauses"]:
        pieces += split_clause(cl, max_words, overlap)
    return pieces


def chunk_article(article: dict, ctx: dict, max_words: int = MAX_WORDS,
                  overlap: int = OVERLAP_WORDS) -> list[dict]:
    """Chia 1 Điều thành các chunk."""
    pieces = article_pieces(article, max_words, overlap)
    if not pieces:
        return []
    total_words = sum(p["words"] for p in pieces)
    groups = [pieces] if total_words <= max_words else pack_pieces(pieces, max_words)

    chunks = []
    for i, g in enumerate(groups, start=1):
        if len(groups) == 1:
            level = "article"
        elif any(p["kind"] == "window" for p in g):
            level = "window"
        elif any(p["kind"] == "points" for p in g):
            level = "points"
        else:
            level = "clauses"
        chunks.append(build_chunk(g, level, ctx, i))
    for c in chunks:
        c["part_total"] = len(chunks)
    return chunks


def chunk_document(doc: dict, max_words: int = MAX_WORDS,
                   overlap: int = OVERLAP_WORDS) -> list[dict]:
    """Chia toàn bộ văn bản, theo thứ tự Chương > Điều."""
    docinfo = {"doc_id": doc["doc_id"], "law_name": doc["law_name"],
               "slug": doc["doc_id"].replace("/", "-")}
    chunks: list[dict] = []
    for ch in doc["chapters"]:
        for art in ch["articles"]:
            quoted = any(RE_QUOTE_START.match(ln) for ln in leaf_lines(art))
            ctx = {"doc": docinfo, "chapter": ch, "article": art, "quoted": quoted}
            chunks += chunk_article(art, ctx, max_words, overlap)
    return chunks


# ==========================================================================
# BƯỚC 5 — KIỂM TRA
# ==========================================================================
def _covered(unit: str, texts: list[str], grams_cache: dict) -> bool:
    """Câu `unit` có nằm trọn trong ít nhất 1 chunk không? (câu rất dài: kiểm tra theo n-gram)"""
    u = " ".join(unit.split())
    if any(u in t for t in texts):
        return True
    words = u.split()
    if len(words) <= N_GRAM:
        return False
    if "set" not in grams_cache:
        s = set()
        for t in texts:
            w = t.split()
            s.update(tuple(w[i:i + N_GRAM]) for i in range(len(w) - N_GRAM + 1))
        grams_cache["set"] = s
    return all(tuple(words[i:i + N_GRAM]) in grams_cache["set"]
               for i in range(len(words) - N_GRAM + 1))


def _alnum(texts) -> int:
    return sum(ch.isalnum() for t in texts for ch in t)


def validate_chunks(doc: dict, chunks: list[dict], max_words: int = MAX_WORDS) -> dict:
    """Chạy các kiểm tra, trả về báo cáo {counts, stats, errors, warnings, ok}."""
    errors: list[str] = []
    warnings: list[str] = []
    articles = {a["article_no"]: a for c in doc["chapters"] for a in c["articles"]}
    by_art: dict[int, list[dict]] = {}
    for c in chunks:
        by_art.setdefault(c["article_no"], []).append(c)

    # 1. chunk_id không trùng; text không rỗng; không vượt giới hạn
    ids = [c["chunk_id"] for c in chunks]
    if len(ids) != len(set(ids)):
        errors.append("chunk_id bị trùng")
    for c in chunks:
        if not c["text"].strip():
            errors.append(f'{c["chunk_id"]} rỗng')
        if c["word_count"] > max_words:
            errors.append(f'{c["chunk_id"]} dài {c["word_count"]} từ > {max_words}')

    # 2. Mỗi Điều có chunk; part_index liên tục; thứ tự Điều đúng
    missing = sorted(set(articles) - set(by_art))
    if missing:
        errors.append(f"Điều không có chunk: {missing}")
    order = [c["article_no"] for c in chunks]
    if order != sorted(order):
        errors.append("Chunk không theo thứ tự Điều")
    for no, cs in by_art.items():
        if [c["part_index"] for c in cs] != list(range(1, len(cs) + 1)) or \
                any(c["part_total"] != len(cs) for c in cs):
            errors.append(f"Điều {no}: part_index/part_total sai")

    # 3. Phủ nội dung: mọi câu của mọi Điều phải nằm trọn trong ít nhất 1 chunk
    uncovered = 0
    for no, art in articles.items():
        texts = [" ".join(c["text"].split()) for c in by_art.get(no, [])]
        cache: dict = {}
        for line in leaf_lines(art):
            for sent in split_sentences(line):
                if not _covered(sent, texts, cache):
                    uncovered += 1
                    if uncovered <= 10:
                        errors.append(f"Điều {no}: mất nội dung: {sent[:60]}")
    if uncovered > 10:
        errors.append(f"... tổng {uncovered} câu bị mất")

    # 4. Dựng dòng từ cây không làm rơi chữ (so với tính trực tiếp từ JSON)
    for no, art in articles.items():
        direct = [art["intro"]]
        for cl in art["clauses"]:
            direct.append((f'{cl["clause_no"]}. ' if cl["clause_no"] is not None else "") + cl["text"])
            direct += [f'{p["point_no"]}) {p["text"]}' for p in cl["points"]]
        if _alnum(direct) != _alnum(leaf_lines(art)):
            errors.append(f"Điều {no}: dựng dòng làm mất/thừa chữ")

    # 5. Tham chiếu chéo phải trỏ tới Điều có thật
    for c in chunks:
        bad = [r for r in c["references"] if r not in articles]
        if bad:
            errors.append(f'{c["chunk_id"]}: tham chiếu tới Điều không tồn tại {bad}')

    words = sorted(c["word_count"] for c in chunks) or [0]
    levels: dict[str, int] = {}
    for c in chunks:
        levels[c["level"]] = levels.get(c["level"], 0) + 1
    split_articles = sum(1 for cs in by_art.values() if len(cs) > 1)

    return {
        "counts": {
            "chunks": len(chunks),
            "articles": len(articles),
            "articles_split": split_articles,
            "by_level": levels,
            "with_references": sum(1 for c in chunks if c["references"]),
            "quoted_other_law": sum(1 for c in chunks if c["quoted_other_law"]),
        },
        "words": {
            "min": words[0],
            "median": statistics.median(words),
            "p90": words[int(len(words) * 0.9)],
            "max": words[-1],
            "limit": max_words,
        },
        "errors": errors,
        "warnings": warnings,
        "ok": not errors,
    }


# ==========================================================================
# BƯỚC 6 — XUẤT JSONL
# ==========================================================================
def export_outputs(chunks: list[dict], report: dict, out_dir: Path) -> dict:
    """Ghi chunks.jsonl (mỗi dòng 1 chunk) và chunk_report.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = out_dir / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    report_path = out_dir / "chunk_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"chunks": chunks_path, "report": report_path}


def run(input_path: str | Path = INPUT_FILE, out_dir: Path = OUTPUT_DIR,
        max_words: int = MAX_WORDS, overlap: int = OVERLAP_WORDS) -> dict:
    doc = load_structured_law(input_path)                       # Bước 1
    chunks = chunk_document(doc, max_words, overlap)            # Bước 2-4
    report = validate_chunks(doc, chunks, max_words)            # Bước 5
    paths = export_outputs(chunks, report, out_dir)             # Bước 6
    return {"doc": doc, "chunks": chunks, "report": report, "paths": paths}


def main() -> int:
    ap = argparse.ArgumentParser(description="Giai đoạn 2: chia chunk")
    ap.add_argument("--input", default=str(INPUT_FILE))
    ap.add_argument("--max-words", type=int, default=MAX_WORDS)
    ap.add_argument("--overlap", type=int, default=OVERLAP_WORDS)
    args = ap.parse_args()
    try:
        result = run(args.input, OUTPUT_DIR, args.max_words, args.overlap)
    except (FileNotFoundError, ValueError) as e:
        print(f"LỖI ĐẦU VÀO: {e}")
        return 1
    rep = result["report"]
    print("Số lượng:", json.dumps(rep["counts"], ensure_ascii=False))
    print("Số từ/chunk:", json.dumps(rep["words"], ensure_ascii=False))
    for e in rep["errors"]:
        print("LỖI:", e)
    print("Kết quả:", "ĐẠT" if rep["ok"] else "KHÔNG ĐẠT")
    for k, v in result["paths"].items():
        print(f"  đã ghi {k}: {v}")
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())