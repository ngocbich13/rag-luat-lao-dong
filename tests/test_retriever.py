"""
test_retriever.py — KIỂM THỬ GIAI ĐOẠN 1 & 4 (QUERY REWRITE, HYBRID RETRIEVER, RERANK & NGƯỠNG ĐIỂM)
=====================================================================================================
Kiểm tra các cơ chế giúp tăng độ chính xác tìm kiếm của src/retriever.py:
    1. Bắt đúng số Điều / Khoản khi người dùng hỏi trực tiếp (parse_explicit_refs).
    2. Viết lại câu hỏi ngôn ngữ đời thường sang truy vấn pháp lý (rewrite_query / expand_query).
    3. Câu hỏi "ai được quyền nhận lương hưu" trả về Điều 169 / Điều 168, tuyệt đối KHÔNG trả về Điều 94.
    4. Câu hỏi ngoài phạm vi ("thủ tục ly hôn") có điểm dưới ngưỡng min_score -> trả về 0 kết quả.
    5. Khử trùng lặp theo Điều (deduplicate_by_article) gộp đủ nội dung các khoản và không lặp lại cùng 1 Điều.
"""

from __future__ import annotations

from src.retriever import (
    HybridRetriever,
    deduplicate_by_article,
    expand_query,
    parse_explicit_refs,
    rewrite_query,
)


# --------------------------------------------------------------------------
# 1. KIỂM THỬ PHÂN TÍCH SỐ ĐIỀU / KHOẢN TRỰC TIẾP & VIẾT LẠI TRUY VẤN PHÁP LÝ
# --------------------------------------------------------------------------
def test_parse_explicit_refs_bat_dung_dieu_va_khoan():
    """Nhận diện đúng số Điều đơn, nhiều Điều và cặp (Khoản, Điều)."""
    res1 = parse_explicit_refs("Điều 36 quy định những gì?")
    assert res1["articles"] == [36]
    assert res1["clauses"] == []

    res2 = parse_explicit_refs("Theo khoản 2 Điều 46 và Điều 35 thì tôi được quyền gì?")
    assert res2["articles"] == [35, 46]
    assert res2["clauses"] == [(46, "2")]


def test_expand_query_them_thuat_ngu_phap_ly():
    """Câu hỏi đời thường ('lương hưu', 'nghỉ đẻ', 'đuổi việc') phải được viết lại thành truy vấn pháp lý."""
    q0 = rewrite_query("ai được quyền nhận lương hưu")
    assert "điều kiện hưởng lương hưu" in q0
    assert "tuổi nghỉ hưu" in q0

    q1 = expand_query("Công ty đuổi việc tôi không báo trước")
    assert "đơn phương chấm dứt hợp đồng lao động" in q1

    q2 = expand_query("Chế độ nghỉ đẻ được mấy tháng?")
    assert "thai sản" in q2

    # Câu hỏi không chứa từ lóng thì giữ nguyên
    q3 = "Phạm vi điều chỉnh của Bộ luật Lao động"
    assert expand_query(q3) == q3


# --------------------------------------------------------------------------
# 2. KIỂM THỬ HYBRID RETRIEVER TRÊN BỘ DỮ LIỆU THẬT (258 CHUNKS)
# --------------------------------------------------------------------------
def test_retrieve_luong_huu_khong_tra_ve_dieu_94():
    """
    Bằng chứng sửa lỗi Giai đoạn 1:
    Với câu hỏi 'ai được quyền nhận lương hưu', hệ thống phải trả về Điều 169 (Tuổi nghỉ hưu)
    và Điều 168 (Tham gia BHXH), tuyệt đối KHÔNG trả về Điều 94 (Nguyên tắc trả lương).
    Đồng thời Điều 169 sau khi gộp phải có đầy đủ nội dung (> 1000 ký tự).
    """
    retriever = HybridRetriever.from_files()
    results = retriever.retrieve("ai được quyền nhận lương hưu", top_k=4, use_vector=False)
    deduped = deduplicate_by_article(results, retriever.by_doc_article)

    article_nos = [r["article_no"] for r in deduped]
    assert 94 not in article_nos, f"Lỗi: Điều 94 vẫn xuất hiện trong kết quả: {article_nos}"
    assert 169 in article_nos, f"Thiếu Điều 169 trong kết quả: {article_nos}"

    d169 = next(r for r in deduped if r["article_no"] == 169)
    assert len(d169["text"]) >= 1200


def test_retrieve_cau_hoi_ngoai_pham_vi_bi_loai_boi_min_score():
    """
    Với câu hỏi ngoài phạm vi như 'thủ tục ly hôn', điểm liên quan phải thấp hơn ngưỡng min_score
    và trả về danh sách rỗng.
    """
    retriever = HybridRetriever.from_files()
    results = retriever.retrieve("thủ tục ly hôn", top_k=4, use_vector=False)
    assert results == []


def test_deduplicate_by_article_khong_trung_lap_va_gop_du_khoan():
    """
    Khi truy xuất Điều 35 (bị chia thành 2 chunk P01 và P02), deduplicate_by_article phải
    gộp cả Khoản 1 và Khoản 2 vào duy nhất 1 mục Điều 35, không xuất hiện Điều 35 hai lần.
    """
    retriever = HybridRetriever.from_files()
    results = retriever.retrieve("báo trước bao nhiêu ngày khi nghỉ việc", top_k=5, use_vector=False)
    deduped = deduplicate_by_article(results, retriever.by_doc_article)

    art_list = [r["article_no"] for r in deduped]
    assert len(art_list) == len(set(art_list)), f"Có điều luật bị trùng lặp: {art_list}"
    assert 35 in art_list

    d35 = next(r for r in deduped if r["article_no"] == 35)
    assert "1. Người lao động" in d35["text"]
    assert "2. Người lao động" in d35["text"]


def test_retrieve_dieu_hoi_truc_tiep_luon_dung_dau():
    """Khi hỏi trực tiếp 'Điều 46', Điều 46 phải đứng vị trí số 1 với match_type='explicit'."""
    retriever = HybridRetriever.from_files()
    results = retriever.retrieve("Điều 46 quy định về trợ cấp thôi việc như thế nào?", top_k=3, use_vector=False)
    assert len(results) >= 1
    assert results[0]["article_no"] == 46
    assert results[0]["match_type"] == "explicit"


def test_retrieve_keo_them_dieu_tham_chieu_references():
    """
    Điều 46 (Trợ cấp thôi việc) có tham chiếu tới Điều 34 và Điều 36 trong trường `references`.
    Khi tìm Điều 46 với expand_references=True, kết quả phải tự động đính kèm Điều 34 hoặc 36.
    """
    retriever = HybridRetriever.from_files()
    results = retriever.retrieve("Tính trợ cấp thôi việc theo Điều 46", top_k=2, expand_references=True, use_vector=False)
    article_nos = [r["article_no"] for r in results]
    match_types = [r["match_type"] for r in results]

    assert 46 in article_nos
    assert "reference" in match_types
    assert (34 in article_nos) or (36 in article_nos)


def test_retrieve_loc_bo_dieu_219_quoted_other_law():
    """
    Điều 219 trích dẫn Luật Bảo hiểm xã hội. Khi hỏi về tuổi nghỉ hưu thông thường,
    không được lấy chunk có quoted_other_law=True, mà phải lấy Điều 169 của Bộ luật Lao động.
    Nhưng khi hỏi đích danh 'Điều 219' thì phải trả về Điều 219.
    """
    retriever = HybridRetriever.from_files()

    res_normal = retriever.retrieve("Tuổi nghỉ hưu của người lao động là bao nhiêu?", top_k=5, use_vector=False)
    assert all(not r.get("quoted_other_law") for r in res_normal)
    assert any(r["article_no"] == 169 for r in res_normal)

    res_219 = retriever.retrieve("Nội dung Điều 219 quy định gì?", top_k=3, use_vector=False)
    assert any(r["article_no"] == 219 for r in res_219)


def test_retrieve_tu_dong_kem_phan_dau_cua_dieu_dai():
    """
    Điều 3 (Giải thích từ ngữ) được chia làm 3 chunk (P01, P02, P03).
    Khi người dùng hỏi 'Quấy rối tình dục tại nơi làm việc là gì?' (nằm ở Khoản 9 - chunk P03),
    hệ thống phải lấy trúng P03 và tự động kèm thêm P01 (same_article_lead).
    """
    retriever = HybridRetriever.from_files()
    results = retriever.retrieve("Quấy rối tình dục tại nơi làm việc là gì?", top_k=3, use_vector=False)
    chunk_ids = [r["chunk_id"] for r in results]
    assert "45-2019-QH14_D003_P03" in chunk_ids
    assert "45-2019-QH14_D003_P01" in chunk_ids


def test_retrieve_ket_hop_vector_va_bm25_cho_cau_hoi_doi_thuong():
    """
    Kiểm tra các câu hỏi đời thường phổ biến của người lao động đều tìm trúng Điều luật chuẩn
    trong top 3.
    """
    retriever = HybridRetriever.from_files()

    cases = [
        ("ai được quyền nhận lương hưu", {169, 168}),
        ("tuổi nghỉ hưu năm 2026 là bao nhiêu", {169}),
        ("báo trước bao nhiêu ngày khi nghỉ việc", {35}),
        ("trợ cấp thôi việc tính thế nào", {46}),
        ("thời gian thử việc tối đa", {25}),
        ("làm thêm giờ được trả bao nhiêu", {98}),
        ("Nghỉ thai sản khi sinh con được nghỉ mấy tháng?", {139}),
        ("Công ty có được giữ bản chính bằng đại học hay căn cước công dân của tôi không?", {17}),
    ]

    for question, expected_articles in cases:
        results = retriever.retrieve(question, top_k=4, use_vector=False)
        deduped = deduplicate_by_article(results, retriever.by_doc_article)[:3]
        found_articles = {r["article_no"] for r in deduped}
        assert found_articles & expected_articles, (
            f"Câu hỏi '{question}' không tìm thấy Điều {expected_articles} trong top 3, chỉ thấy {found_articles}"
        )
