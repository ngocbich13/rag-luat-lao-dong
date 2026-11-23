"""
Test cho Giai đoạn 2 (chunker). Chạy từ thư mục gốc dự án:
    python -m pytest tests -v
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import chunker as ck  # noqa: E402  (file src/chunker.py)

LAW = ROOT / "data" / "processed" / "structured_law.json"
needs_law = pytest.mark.skipif(not LAW.exists(), reason="chưa chạy Giai đoạn 1")


# ---------- Công cụ dựng dữ liệu giả để test từng tình huống ----------
def sent(n: int, tag: str) -> str:
    """Câu gồm đúng n từ, kết thúc bằng dấu chấm."""
    return " ".join([f"{tag}{i}" for i in range(n - 1)] + [f"{tag}end."])


def point(no, text, ln=1):
    return {"point_no": no, "text": text, "footnote_refs": [], "line_start": ln, "line_end": ln}


def clause(no, text, points=(), ln=1):
    return {"clause_no": no, "text": text, "points": list(points), "footnote_refs": [],
            "line_start": ln, "line_end": ln + len(points), "full_path": f"Chương I > Điều 5 > Khoản {no}"}


def article(no=5, intro="", clauses=()):
    return {"article_no": no, "article_title": "Tiêu đề thử", "section_no": None,
            "section_title": None, "intro": intro, "clauses": list(clauses),
            "footnote_refs": [], "line_start": 1, "line_end": 50,
            "full_path": f"Chương I > Điều {no}"}


def make_doc(*articles):
    return {"doc_id": "X/2020/QH1", "law_name": "Luật thử 2020",
            "chapters": [{"chapter_no": "I", "chapter_title": "CHUNG", "sections": [],
                          "articles": list(articles)}]}


def run_chunks(doc, max_words, overlap=5):
    chunks = ck.chunk_document(doc, max_words, overlap)
    return chunks, ck.validate_chunks(doc, chunks, max_words)


# ---------- Điều ngắn: 1 chunk ----------
def test_dieu_ngan_la_mot_chunk():
    doc = make_doc(article(5, clauses=[clause("1", sent(10, "a")), clause("2", sent(10, "b"))]))
    chunks, rep = run_chunks(doc, 100)
    assert len(chunks) == 1
    c = chunks[0]
    assert c["chunk_id"] == "X-2020-QH1_D005_P01"
    assert c["level"] == "article" and c["part_total"] == 1
    assert c["text"].startswith("1. a0") and "\n2. b0" in c["text"]   # nhãn Khoản được gắn lại
    assert c["citation"] == "Điều 5 Luật thử 2020"
    assert rep["ok"], rep["errors"]


def test_header_embedding_co_chuong_va_dieu():
    chunks, _ = run_chunks(make_doc(article(5, intro=sent(5, "x"))), 100)
    first = chunks[0]["text_for_embedding"].split("\n")[0:2]
    assert first == ["Luật thử 2020 | Chương I. CHUNG", "Điều 5. Tiêu đề thử"]


# ---------- Điều dài: cắt theo Khoản ----------
def test_dieu_dai_cat_theo_khoan_khong_cat_ngang_khoan():
    cls = [clause(str(i), sent(20, f"k{i}x")) for i in range(1, 7)]       # 6 Khoản x 20 từ
    doc = make_doc(article(5, clauses=cls))
    chunks, rep = run_chunks(doc, 50)                                      # tối đa 50 từ
    assert len(chunks) == 3
    assert all(c["word_count"] <= 50 for c in chunks)
    assert [c["part_index"] for c in chunks] == [1, 2, 3]
    assert all(c["part_total"] == 3 for c in chunks)
    # mỗi Khoản nằm trọn trong đúng 1 chunk
    for i in range(1, 7):
        assert sum(f"k{i}x0 " in c["text"] for c in chunks) == 1
    assert chunks[0]["clause_nos"] == ["1", "2"]
    assert chunks[0]["citation"] == "các khoản 1, 2 Điều 5 Luật thử 2020"
    assert rep["ok"], rep["errors"]


def test_chunk_dau_giu_phan_dan_cua_dieu():
    doc = make_doc(article(5, intro=sent(10, "dan"),
                           clauses=[clause("1", sent(30, "a")), clause("2", sent(30, "b"))]))
    chunks, rep = run_chunks(doc, 45)
    assert chunks[0]["text"].startswith("dan0")
    assert chunks[0]["citation"] == "Điều 5 Luật thử 2020"      # có phần dẫn -> trích cả Điều
    assert rep["ok"], rep["errors"]


# ---------- Khoản dài: cắt theo Điểm, lặp câu dẫn ----------
def test_khoan_dai_cat_theo_diem_lap_cau_dan():
    lead = "Quyền gồm các điểm sau đây:"                         # 5 từ
    pts = [point(n, sent(20, f"p{n}x")) for n in "abcd"]
    doc = make_doc(article(5, clauses=[clause("1", lead, pts)]))
    chunks, rep = run_chunks(doc, 50)
    assert len(chunks) == 2 and all(c["level"] == "points" for c in chunks)
    for c in chunks:
        assert c["text"].startswith("1. Quyền gồm các điểm")     # câu dẫn lặp ở mọi chunk
        assert c["word_count"] <= 50
    assert chunks[0]["point_nos"] == ["a", "b"] and chunks[1]["point_nos"] == ["c", "d"]
    assert chunks[0]["citation"] == "các điểm a, b khoản 1 Điều 5 Luật thử 2020"
    assert chunks[0]["full_path"] == "Chương I > Điều 5 > Khoản 1 > Điểm a, b"
    assert rep["ok"], rep["errors"]


# ---------- Điểm quá dài: cắt theo câu ----------
def test_diem_qua_dai_cat_theo_cau():
    long_point = " ".join(sent(20, f"s{i}x") for i in range(4))   # 4 câu x 20 từ
    doc = make_doc(article(5, clauses=[clause("1", "Dẫn ngắn:", [point("a", long_point)])]))
    chunks, rep = run_chunks(doc, 30, overlap=5)
    assert len(chunks) >= 3 and all(c["level"] == "window" for c in chunks)
    assert all(c["word_count"] <= 30 for c in chunks)
    assert all(c["point_nos"] == ["a"] for c in chunks)
    assert chunks[0]["citation"] == "điểm a khoản 1 Điều 5 Luật thử 2020"
    assert rep["ok"], rep["errors"]


# ---------- Một câu dài hơn giới hạn: cắt cửa sổ có gối đầu ----------
def test_cau_qua_dai_cat_cua_so_co_overlap():
    doc = make_doc(article(5, clauses=[clause("1", " ".join(f"t{i}" for i in range(100)) + ".")]))
    chunks, rep = run_chunks(doc, 30, overlap=10)
    assert len(chunks) >= 4 and all(c["word_count"] <= 30 for c in chunks)
    # 10 từ cuối chunk trước = 10 từ đầu chunk sau
    a, b = chunks[0]["text"].split(), chunks[1]["text"].split()
    assert a[-10:] == b[:10]
    assert rep["ok"], rep["errors"]


def test_pack_text_gac_dau_cau_ngan():
    parts = ck.pack_text(f"{sent(20, 'a')} {sent(3, 'b')} {sent(20, 'c')}", 25, 5)
    assert len(parts) == 2
    assert parts[1].startswith("bend.") is False and "b0" in parts[1]   # câu ngắn b được lặp lại


# ---------- Kiểm tra phát hiện lỗi ----------
def test_validate_bat_chunk_qua_dai_va_mat_noi_dung():
    doc = make_doc(article(5, clauses=[clause("1", sent(20, "a")), clause("2", sent(20, "b"))]))
    chunks = ck.chunk_document(doc, 100)
    chunks[0]["text"] = chunks[0]["text"].split("\n")[0]                # cố ý làm mất Khoản 2
    chunks[0]["word_count"] = 500                                       # cố ý báo quá dài
    rep = ck.validate_chunks(doc, chunks, 100)
    assert not rep["ok"]
    assert any("dài" in e for e in rep["errors"])
    assert any("mất nội dung" in e for e in rep["errors"])


# ---------- Trích dẫn ----------
@pytest.mark.parametrize("level, intro, cnos, pnos, expected", [
    ("article", False, ["1", "2"], [], "Điều 3 Luật X"),
    ("clauses", True, ["1"], [], "Điều 3 Luật X"),
    ("clauses", False, ["1"], [], "khoản 1 Điều 3 Luật X"),
    ("clauses", False, ["2", "3", "4"], [], "các khoản 2, 3, 4 Điều 3 Luật X"),
    ("points", False, ["1"], ["a"], "điểm a khoản 1 Điều 3 Luật X"),
    ("points", False, ["1"], ["a", "đ"], "các điểm a, đ khoản 1 Điều 3 Luật X"),
])
def test_make_citation(level, intro, cnos, pnos, expected):
    assert ck.make_citation("Luật X", 3, level, intro, cnos, pnos) == expected


# ---------- Tham chiếu chéo ----------
@pytest.mark.parametrize("text, own, expected", [
    ("theo quy định tại khoản 2 Điều 35 của Bộ luật này", 10, [35]),
    ("quy định tại Điều 16 và Điều 31", 10, [16, 31]),
    ("theo Điều 54 của Luật Bảo hiểm xã hội", 10, []),              # luật khác
    ("theo khoản 2 Điều 169 của Bộ luật Lao động", 10, [169]),       # chính luật này
    ("theo Điều 5 của Bộ luật Lao động số 10/2012/QH13", 10, []),    # bộ luật cũ
    ("Điều 10 này và Điều 10", 10, []),                              # bỏ chính nó
])
def test_extract_references(text, own, expected):
    assert ck.extract_references(text, own) == expected


def test_co_dau_trich_dan_thi_danh_dau_va_bo_tham_chieu():
    doc = make_doc(article(5, clauses=[clause("1", "Sửa đổi như sau:", [
        point("a", "“Điều 54. Mô tả\ntheo Điều 7 của Luật này.”.")])]))
    chunks, _ = run_chunks(doc, 100)
    assert chunks[0]["quoted_other_law"] is True and chunks[0]["references"] == []


# ---------- Toàn bộ pipeline trên dữ liệu thật ----------
@pytest.fixture(scope="module")
def real(tmp_path_factory):
    return ck.run(LAW, out_dir=tmp_path_factory.mktemp("chunks"))


@needs_law
def test_thuc_te_bao_cao_dat(real):
    assert real["report"]["errors"] == []


@needs_law
def test_thuc_te_du_220_dieu_va_khong_vuot_gioi_han(real):
    assert real["report"]["counts"]["articles"] == 220
    assert len({c["article_no"] for c in real["chunks"]}) == 220
    assert max(c["word_count"] for c in real["chunks"]) <= ck.MAX_WORDS


@needs_law
def test_thuc_te_dieu_13_la_mot_chunk(real):
    cs = [c for c in real["chunks"] if c["article_no"] == 13]
    assert len(cs) == 1 and cs[0]["level"] == "article"
    assert cs[0]["text_for_embedding"].startswith(
        "Bộ luật Lao động 2019 | Chương III. HỢP ĐỒNG LAO ĐỘNG | Mục 1.")


@needs_law
def test_thuc_te_dieu_3_cat_theo_khoan_chunk_dau_co_phan_dan(real):
    cs = [c for c in real["chunks"] if c["article_no"] == 3]
    assert len(cs) >= 2
    assert cs[0]["text"].startswith("Trong Bộ luật này, các từ ngữ dưới đây được hiểu như sau:")


@needs_law
def test_thuc_te_dieu_219_trich_dan(real):
    cs = [c for c in real["chunks"] if c["article_no"] == 219]
    assert len(cs) >= 5
    assert all(c["quoted_other_law"] and c["references"] == [] for c in cs)
    # chunk nào của Khoản 1 cũng mở đầu bằng câu dẫn của Khoản 1
    assert all(c["text"].startswith("1. Sửa đổi, bổ sung một số điều của Luật Bảo hiểm xã hội")
               for c in cs if c["clause_nos"] == ["1"])


@needs_law
def test_thuc_te_xuat_jsonl(real):
    lines = real["paths"]["chunks"].read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(real["chunks"])
    assert json.loads(lines[0])["chunk_id"] == "45-2019-QH14_D001_P01"