"""Kiểm tra API key và gọi LLM thử.

Chạy:  python src/test_llm.py
Cần:   file .env có ANTHROPIC_API_KEY (xem .env.example)
"""
import os
import sys
import time

from dotenv import load_dotenv
from anthropic import Anthropic, APIError, AuthenticationError

load_dotenv()

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5")

# Trích nguyên văn Điều 46 (Bộ luật Lao động, bản hợp nhất 18/VBHN-VPQH)
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


def ask(client, user, system=None, max_tokens=400):
    start = time.perf_counter()
    kwargs = dict(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": user}],
    )
    if system:
        kwargs["system"] = system
    msg = client.messages.create(**kwargs)
    elapsed = time.perf_counter() - start
    text = "".join(b.text for b in msg.content if b.type == "text")
    return text, msg.usage.input_tokens, msg.usage.output_tokens, elapsed


def show(title, result):
    text, tin, tout, sec = result
    print(f"\n=== {title} ===")
    print(text)
    print(f"[token vào: {tin} | token ra: {tout} | {sec:.1f}s]")


def main():
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("Thiếu ANTHROPIC_API_KEY. Tạo file .env từ .env.example rồi điền key.")

    client = Anthropic()
    print(f"Model đang dùng: {MODEL}")

    try:
        # Test 1: gọi API cơ bản (không có tài liệu, chỉ để kiểm tra kết nối)
        show("Test 1: gọi API cơ bản",
             ask(client, "Trả lời một câu: Thủ đô của Việt Nam là gì?"))

        # Test 2: có tài liệu -> phải trả lời dựa trên tài liệu và trích dẫn
        q2 = "Tôi làm 3 năm rồi nghỉ thì trợ cấp thôi việc tính thế nào?"
        show("Test 2: có căn cứ trong tài liệu",
             ask(client, f"TÀI LIỆU:\n{CONTEXT}\n\nCÂU HỎI: {q2}", system=SYSTEM_RAG))

        # Test 3: câu hỏi ngoài tài liệu -> phải từ chối
        q3 = "Thuế thu nhập cá nhân của tôi là bao nhiêu?"
        show("Test 3: ngoài phạm vi, kỳ vọng từ chối",
             ask(client, f"TÀI LIỆU:\n{CONTEXT}\n\nCÂU HỎI: {q3}", system=SYSTEM_RAG))

    except AuthenticationError:
        sys.exit("API key không hợp lệ. Kiểm tra lại key trong file .env.")
    except APIError as e:
        sys.exit(f"Lỗi API: {e}")


if __name__ == "__main__":
    main()
