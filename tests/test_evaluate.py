"""
test_evaluate.py — KIỂM THỬ GIAI ĐOẠN 6 (ĐÁNH GIÁ ĐỘ CHÍNH XÁC)
===============================================================
Kiểm tra bộ câu hỏi chuẩn eval/test_questions.json và kết quả đánh giá từ src/evaluate.py:
    1. Bộ câu hỏi chuẩn có ít nhất 25 câu, chia đủ các nhóm (tra cứu trực tiếp,
       đời thường, cần nghị định hướng dẫn, ngoài phạm vi).
    2. Hệ thống đạt và vượt mọi chỉ số mục tiêu trong docs/00_problem_definition.md:
       - Recall@5 > 85%
       - Citation Accuracy > 90%
       - Refusal Accuracy > 90%
"""

from __future__ import annotations

from pathlib import Path

from src.evaluate import (
    QUESTIONS_FILE,
    load_test_questions,
    run_evaluation,
)


def test_load_test_questions_du_nhom_cau_hoi():
    """Bộ câu hỏi eval/test_questions.json phải có >= 25 câu và đủ 4 nhóm."""
    questions = load_test_questions(QUESTIONS_FILE)
    assert len(questions) >= 25

    categories = {q["category"] for q in questions}
    assert "direct_lookup" in categories
    assert "colloquial" in categories
    assert "multi_article_or_decree" in categories
    assert "out_of_scope" in categories


def test_run_evaluation_dat_tat_ca_chi_so_muc_tieu(tmp_path: Path):
    """Chạy đánh giá trên toàn bộ 25 câu hỏi phải đạt Recall@5 > 85%, Citation > 90%, Refusal > 90%."""
    report_file = tmp_path / "eval_report.json"
    report = run_evaluation(
        top_k=5,
        use_vector=False,
        live_llm=False,
        report_path=report_file,
    )

    assert report_file.exists()
    summary = report["summary"]
    assert summary["recall_at_5"] >= 0.85, f"Recall@5 chỉ đạt {summary['recall_at_5']}"
    assert summary["citation_accuracy"] >= 0.90, f"Citation accuracy chỉ đạt {summary['citation_accuracy']}"
    assert summary["refusal_accuracy"] >= 0.90, f"Refusal accuracy chỉ đạt {summary['refusal_accuracy']}"
    assert report["ok"] is True
