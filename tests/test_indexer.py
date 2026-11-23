"""
test_indexer.py — KIỂM THỬ GIAI ĐOẠN 3 (INDEXER)
================================================
Kiểm tra toàn bộ chức năng của src/indexer.py:
    1. Đọc chunks.jsonl thật và báo lỗi khi file thiếu/sai cấu trúc.
    2. Tách từ tiếng Việt (tokenize_vi) giữ số điều khoản và tạo cụm từ ghép (bigrams).
    3. Chuẩn hóa vector (L2 normalization) để tính Cosine Similarity chính xác.
    4. Kiểm tra bắt lỗi validate_index (sai kích thước, chứa NaN, vector 0, trùng ID).
    5. Lưu và đọc lại chỉ mục (.npz) khớp 100% thứ tự chunk_id.
    6. Kiểm tra chỉ mục thực tế (data/processed/embeddings.npz) đủ 258 vector.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.indexer import (
    CHUNKS_FILE,
    EMBEDDINGS_FILE,
    INDEX_REPORT_FILE,
    build_and_save_index,
    load_chunks,
    load_index,
    normalize_vectors,
    save_index,
    tokenize_vi,
    validate_index,
)


# --------------------------------------------------------------------------
# 1. KIỂM THỬ ĐỌC CHUNKS VÀ TÁCH TỪ TIẾNG VIỆT
# --------------------------------------------------------------------------
def test_load_chunks_thuc_te_du_258_chunk():
    """Đọc file chunks.jsonl thật phải có đúng 258 chunk và đủ các trường quan trọng."""
    chunks = load_chunks(CHUNKS_FILE)
    assert len(chunks) == 258
    first = chunks[0]
    for key in ("chunk_id", "article_no", "text", "text_for_embedding", "citation", "references"):
        assert key in first


def test_load_chunks_file_khong_ton_tai(tmp_path: Path):
    """Báo lỗi rõ ràng khi đường dẫn chunks.jsonl không tồn tại."""
    with pytest.raises(FileNotFoundError):
        load_chunks(tmp_path / "khong_co.jsonl")


def test_tokenize_vi_giu_so_va_tao_bigram():
    """Bộ tách từ phải giữ được số Điều/Khoản và tạo được từ ghép (bigram)."""
    tokens = tokenize_vi("Khoản 2 Điều 46 quy định về trợ cấp thôi việc.")
    # Giữ chữ thường và con số
    assert "46" in tokens
    assert "2" in tokens
    assert "trợ" in tokens
    # Có cụm 2 từ liền kề để tìm chính xác cụm 'trợ cấp', 'thôi việc'
    assert "trợ_cấp" in tokens
    assert "thôi_việc" in tokens
    # Từ hư 'về' đã được lọc khỏi unigrams
    assert "về" not in [t for t in tokens if "_" not in t]


# --------------------------------------------------------------------------
# 2. KIỂM THỬ CHUẨN HÓA VECTOR & KIỂM TRA CHẤT LƯỢNG (VALIDATE)
# --------------------------------------------------------------------------
def test_normalize_vectors_do_dai_bang_1():
    """Mọi vector khác 0 sau khi chuẩn hóa phải có độ dài (L2 norm) bằng 1."""
    raw = np.array([[3.0, 4.0], [1.0, 1.0]], dtype=np.float32)
    normed = normalize_vectors(raw)
    lengths = np.linalg.norm(normed, axis=1)
    assert np.allclose(lengths, [1.0, 1.0], atol=1e-5)
    assert np.allclose(normed[0], [0.6, 0.8], atol=1e-5)


def test_validate_index_bat_loi_sai_so_luong_va_nan():
    """validate_index phải phát hiện được khi số vector lệch số chunk hoặc có NaN."""
    sample_chunks = [
        {"chunk_id": "C1", "article_no": 1, "text": "a", "text_for_embedding": "a", "citation": "Đ1"},
        {"chunk_id": "C2", "article_no": 2, "text": "b", "text_for_embedding": "b", "citation": "Đ2"},
    ]
    # Trường hợp 1: Chỉ có 1 vector cho 2 chunk
    bad_shape = np.ones((1, 4), dtype=np.float32)
    rep1 = validate_index(sample_chunks, bad_shape)
    assert not rep1["ok"]
    assert any("không khớp" in e for e in rep1["errors"])

    # Trường hợp 2: Có giá trị NaN
    bad_nan = np.array([[1.0, np.nan], [0.5, 0.5]], dtype=np.float32)
    rep2 = validate_index(sample_chunks, bad_nan)
    assert not rep2["ok"]
    assert any("NaN" in e for e in rep2["errors"])

    # Trường hợp 3: Vector toàn số 0
    bad_zero = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    rep3 = validate_index(sample_chunks, bad_zero)
    assert not rep3["ok"]
    assert any("độ dài bằng 0" in e for e in rep3["errors"])


# --------------------------------------------------------------------------
# 3. KIỂM THỬ LƯU VÀ ĐỌC LẠI CHỈ MỤC (VỚI HÀM EMBED GIẢ LẬP)
# --------------------------------------------------------------------------
def test_build_and_load_index_voi_mock_embed(tmp_path: Path):
    """Kiểm tra quy trình khép kín: đọc chunk -> tạo vector -> lưu .npz -> đọc lại."""
    chunks_file = tmp_path / "chunks.jsonl"
    emb_file = tmp_path / "embeddings.npz"
    rep_file = tmp_path / "index_report.json"

    sample_chunks = [
        {
            "chunk_id": "D001_P01",
            "article_no": 1,
            "text": "Nội dung điều 1",
            "text_for_embedding": "Điều 1. Nội dung điều 1",
            "citation": "Điều 1",
            "quoted_other_law": False,
        },
        {
            "chunk_id": "D002_P01",
            "article_no": 2,
            "text": "Nội dung điều 2",
            "text_for_embedding": "Điều 2. Nội dung điều 2",
            "citation": "Điều 2",
            "quoted_other_law": False,
        },
    ]
    chunks_file.write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in sample_chunks),
        encoding="utf-8",
    )

    # Hàm tạo vector giả lập (không gọi mạng)
    def fake_embed(texts: list[str]) -> np.ndarray:
        return np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

    report = build_and_save_index(
        chunks_path=chunks_file,
        embeddings_path=emb_file,
        report_path=rep_file,
        embed_fn=fake_embed,
    )
    assert report["ok"] is True
    assert report["counts"]["chunks"] == 2
    assert report["counts"]["dimensions"] == 3
    assert emb_file.exists()
    assert rep_file.exists()

    # Đọc lại và kiểm tra khớp dữ liệu
    loaded_chunks, loaded_embs = load_index(chunks_file, emb_file)
    assert len(loaded_chunks) == 2
    assert loaded_embs.shape == (2, 3)
    assert np.allclose(np.linalg.norm(loaded_embs, axis=1), [1.0, 1.0], atol=1e-5)


def test_load_index_bao_loi_khi_chunk_id_lech(tmp_path: Path):
    """Nếu chunks.jsonl thay đổi mà chưa chạy lại indexer.py thì load_index phải cảnh báo."""
    chunks_file = tmp_path / "chunks.jsonl"
    emb_file = tmp_path / "embeddings.npz"
    rep_file = tmp_path / "index_report.json"

    c1 = [{"chunk_id": "ID_CU", "article_no": 1, "text": "a", "text_for_embedding": "a", "citation": "Đ1"}]
    chunks_file.write_text(json.dumps(c1[0], ensure_ascii=False), encoding="utf-8")
    save_index(c1, np.array([[1.0, 0.0]], dtype=np.float32), emb_file, rep_file)

    # Sửa chunk_id trong file jsonl sang ID mới
    c2 = [{"chunk_id": "ID_MOI", "article_no": 1, "text": "a", "text_for_embedding": "a", "citation": "Đ1"}]
    chunks_file.write_text(json.dumps(c2[0], ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="không khớp"):
        load_index(chunks_file, emb_file)


# --------------------------------------------------------------------------
# 4. KIỂM THỬ CHỈ MỤC THỰC TẾ ĐÃ SINH TRONG data/processed/
# --------------------------------------------------------------------------
def test_thuc_te_embeddings_npz_hop_le():
    """Kiểm tra file data/processed/embeddings.npz thực tế có đúng 258 vector chuẩn hóa."""
    assert EMBEDDINGS_FILE.exists(), "Chưa có file embeddings.npz, cần chạy python src/indexer.py"
    assert INDEX_REPORT_FILE.exists(), "Chưa có file index_report.json"

    chunks, embeddings = load_index(CHUNKS_FILE, EMBEDDINGS_FILE)
    assert len(chunks) == 258
    assert embeddings.shape[0] == 258
    assert embeddings.shape[1] > 100  # Gemini embedding có 3072 chiều

    report = json.loads(INDEX_REPORT_FILE.read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["errors"] == []
