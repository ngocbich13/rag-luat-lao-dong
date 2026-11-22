"""
Test cho Giai đoạn 1. Chạy từ thư mục gốc dự án:
    python -m pytest tests -v
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))   # để import được parser.py

import parser as p  # noqa: E402  (file src/parser.py)

RAW = ROOT / "data" / "raw" / "luatLaoDong2019.docx"
needs_raw = pytest.mark.skipif(not RAW.exists(), reason="chưa có file luật gốc")


# ---------- Bước 1: kiểm tra file ----------
def test_file_khong_ton_tai():
    with pytest.raises(FileNotFoundError):
        p.check_input_file(ROOT / "data" / "raw" / "khong_co.docx")


def test_tu_choi_file_tam_cua_word(tmp_path):
    f = tmp_path / "~$luat.docx"
    f.write_bytes(b"x")
    with pytest.raises(ValueError, match="file tạm"):
        p.check_input_file(f)


def test_sai_duoi_file(tmp_path):
    f = tmp_path / "luat.pdf"
    f.write_bytes(b"x")
    with pytest.raises(ValueError, match=".docx"):
        p.check_input_file(f)


# ---------- Bước 3: làm sạch ----------
def test_clean_nbsp_va_khoang_trang():
    s, _ = p.clean_line("Điều\xa01.   Phạm   vi  ")
    assert s == "Điều 1. Phạm vi"


@pytest.mark.parametrize("raw, expected, marks", [
    ("3.3[3] Người sử dụng", "3. Người sử dụng", [3]),
    ("a)2[2]Thành lập", "a)Thành lập", [2]),
    ("8a.5[5] Nhân lực", "8a. Nhân lực", [5]),
    ("Bộ luật Lao động[1].", "Bộ luật Lao động.", [1]),
    ("ĐIỀU KHOẢN THI HÀNH6[6]", "ĐIỀU KHOẢN THI HÀNH", [6]),
])
def test_clean_chu_thich(raw, expected, marks):
    assert p.clean_line(raw) == (expected, marks)


# ---------- Bước 4: nhận diện cấu trúc ----------
@pytest.mark.parametrize("text, kind", [
    ("Chương III", "chapter"),
    ("Mục 1. GIAO KẾT HỢP ĐỒNG LAO ĐỘNG", "section"),
    ("Điều 3. Giải thích từ ngữ", "article"),
    ("1. Người lao động là", "clause"),
    ("8a. Nhân lực công nghiệp", "clause"),
    ("a) Đủ tuổi", "point"),
    ("đ) Về bồi thường", "point"),          # phải nhận được điểm đ)
    ("Độ tuổi lao động tối thiểu", "text"),
    ("Chương này quy định", "text"),         # không phải tiêu đề Chương
    ("“Điều 54. Điều kiện", "text"),         # đoạn trích dẫn, không phải cấu trúc
])
def test_classify(text, kind):
    assert p.classify_line(text)["type"] == kind


# ---------- Toàn bộ pipeline trên file thật ----------
@pytest.fixture(scope="module")
def result(tmp_path_factory):
    return p.run(RAW, out_dir=tmp_path_factory.mktemp("processed"))


@needs_raw
def test_bao_cao_dat(result):
    assert result["report"]["errors"] == []
    assert result["report"]["ok"]


@needs_raw
def test_so_luong_cap(result):
    c = result["report"]["counts"]
    assert c["chapters"] == 17
    assert c["sections"] == 24
    assert c["articles"] == 220


@needs_raw
def test_khong_mat_du_lieu(result):
    c = result["report"]["counts"]
    assert c["alnum_chars_clean_text"] == c["alnum_chars_json"]


@needs_raw
def test_metadata_van_ban(result):
    d = result["doc"]
    assert d["doc_id"] == "45/2019/QH14"
    assert d["title"] == "Bộ luật Lao động"
    assert d["effective_date"] == "2021-01-01"


def _articles(doc):
    return {a["article_no"]: a for c in doc["chapters"] for a in c["articles"]}


@needs_raw
def test_dieu_3_giai_thich_tu_ngu(result):
    a = _articles(result["doc"])[3]
    assert a["article_title"] == "Giải thích từ ngữ"
    assert len(a["clauses"]) == 9
    # dòng nối tiếp "Độ tuổi lao động tối thiểu..." thuộc Khoản 1, không tách thành Khoản
    assert "Độ tuổi lao động tối thiểu" in a["clauses"][0]["text"]


@needs_raw
def test_dieu_218_chi_co_noi_dung_khong_khoan(result):
    a = _articles(result["doc"])[218]
    assert a["clauses"] == [] and a["intro"]


@needs_raw
def test_dieu_219_trich_dan_khong_tach_cau_truc(result):
    a = _articles(result["doc"])[219]
    assert [c["clause_no"] for c in a["clauses"]] == ["1", "2"]
    assert "“Điều 54." in a["clauses"][0]["points"][0]["text"]


@needs_raw
def test_khoan_8a_va_chu_thich(result):
    a = _articles(result["doc"])[154]
    k = next(c for c in a["clauses"] if c["clause_no"] == "8a")
    assert 5 in k["footnote_refs"]
    assert k["citation"] == "khoản 8a Điều 154 Bộ luật Lao động 2019"


@needs_raw
def test_muc_va_duong_dan(result):
    a = _articles(result["doc"])[13]
    assert a["section_no"] == "1"
    assert a["full_path"] == "Chương III > Mục 1 > Điều 13"


@needs_raw
def test_xuat_du_file(result):
    for path in result["paths"].values():
        assert path.exists() and path.stat().st_size > 0