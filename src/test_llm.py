"""
test_llm.py — KIỂM TRA KẾT NỐI GEMINI API VÀ THỬ NGHIỆM RAG CƠ BẢN
===================================================================
Chạy:  python src/test_llm.py
Cần:   file .env có GEMINI_API_KEY
"""

from __future__ import annotations

import os
import sys
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

# Trích nguyên văn Điều 46 (Bộ luật Lao động 2019, bản hợp nhất 18/VBHN-VPQH)
CONTEXT = """[Điều 46. Trợ cấp thôi việc]
1. Khi hợp đồng lao động chấm dứt theo quy định tại các khoản 1, 2, 3, 4, 6, 7, 9 và 10 Điều 34 của Bộ luật này thì người sử dụng lao động có trách nhiệm trả trợ cấp thôi việc cho người lao động đã làm việc thường xuyên cho mình từ đủ 12 tháng trở lên, mỗi năm làm việc được trợ cấp một nửa tháng tiền lương, trừ trường hợp đủ điều kiện hưởng lương hưu theo quy định của pháp luật về bảo hiểm xã hội và trường hợp quy định tại điểm e khoản 1 Điều 36 của Bộ luật này.
2. Thời gian làm việc để tính trợ cấp thôi việc là tổng thời gian người lao động đã làm việc thực tế cho người sử dụng lao động trừ đi thời gian người lao động đã tham gia bảo hiểm thất nghiệp theo quy định của pháp luật về bảo hiểm thất nghiệp và thời gian làm việc đã được người sử dụng lao động chi trả trợ cấp thôi việc, trợ cấp mất việc làm.
3. Tiền lương để tính trợ cấp thôi việc là tiền lương bình quân của 06 tháng liền kề theo hợp đồng lao động trước khi người lao động thôi việc.
4. Chính phủ quy định chi tiết Điều này."""

SYSTEM_RAG = (
    "Bạn là trợ lý tra cứu Bộ luật Lao động Việt Nam. "
    "Chỉ trả lời dựa trên phần TÀI LIỆU được cung cấp. "
    "Mỗi ý phải trích dẫn Điều/Khoản. "
    "Nếu tài liệu không đủ căn cứ, nói rõ 'Không đủ căn cứ trong tài liệu được cung cấp' "
    "và không suy đoán. Trả lời ngắn gọn bằng tiếng Việt."
)


def ask(client: genai.Client, user_prompt: str, system_prompt: str | None = None):
    """Gửi 1 câu hỏi lên Gemini API và đo thời gian, số token."""
    start = time.perf_counter()
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0.1,
        max_output_tokens=500,
    )
    resp = client.models.generate_content(
        model=MODEL,
        contents=user_prompt,
        config=config,
    )
    elapsed = time.perf_counter() - start
    text = (resp.text or "").strip()
    usage = getattr(resp, "usage_metadata", None)
    tin = int(getattr(usage, "prompt_token_count", 0) or 0) if usage else 0
    tout = int(getattr(usage, "candidates_token_count", 0) or 0) if usage else 0
    return text, tin, tout, elapsed


def show(title: str, result: tuple[str, int, int, float]):
    text, tin, tout, sec = result
    print(f"\n=== {title} ===")
    print(text)
    print(f"[token vào: {tin} | token ra: {tout} | {sec:.1f}s]")


def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        sys.exit("Thiếu GEMINI_API_KEY. Hãy kiểm tra file .env ở thư mục gốc dự án.")

    client = genai.Client(api_key=api_key)
    print(f"Model đang dùng: {MODEL}")

    # Test 1: Gọi API cơ bản
    show(
        "Test 1: gọi API cơ bản",
        ask(client, "Trả lời một câu: Thủ đô của Việt Nam là gì?"),
    )

    # Test 2: Có tài liệu -> phải trả lời dựa trên tài liệu và trích dẫn Điều 46
    q2 = "Tôi làm 3 năm rồi nghỉ thì trợ cấp thôi việc tính thế nào?"
    show(
        "Test 2: có căn cứ trong tài liệu",
        ask(client, f"TÀI LIỆU:\n{CONTEXT}\n\nCÂU HỎI: {q2}", system_prompt=SYSTEM_RAG),
    )

    # Test 3: Câu hỏi ngoài tài liệu -> phải từ chối
    q3 = "Thuế thu nhập cá nhân của tôi là bao nhiêu?"
    show(
        "Test 3: ngoài phạm vi, kỳ vọng từ chối",
        ask(client, f"TÀI LIỆU:\n{CONTEXT}\n\nCÂU HỎI: {q3}", system_prompt=SYSTEM_RAG),
    )


if __name__ == "__main__":
    main()
