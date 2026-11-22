# 00. Định nghĩa bài toán – Chatbot RAG Bộ luật Lao động

> Tác giả: Huỳnh Thị Ngọc Bích · Cập nhật: 22/11/2026 · Trạng thái: V1 (bản nháp, sẽ cập nhật sau khi có số liệu baseline)

## 1. Mục tiêu

Chatbot hỏi đáp về **Bộ luật Lao động** cho người lao động. Mọi câu trả lời phải **dựa trên điều luật được truy xuất, kèm trích dẫn Điều/Khoản** và nguyên văn để người dùng tự kiểm chứng. Khi không đủ căn cứ, hệ thống **từ chối thay vì đoán**.

## 2. Người dùng

Người lao động khu vực tư nhân, không chuyên luật, hỏi bằng ngôn ngữ đời thường. Ví dụ:

- "Công ty cho tôi nghỉ việc mà không báo trước thì sao?"
- "Làm việc 3 năm rồi nghỉ thì được trợ cấp thôi việc thế nào?"
- "Thử việc có phải ký hợp đồng thử việc không?"

Hệ quả thiết kế: người dùng không dùng thuật ngữ luật ("đơn phương chấm dứt hợp đồng lao động"), nên cần bước **viết lại câu hỏi** (query rewriting) trước khi tìm kiếm.

## 3. Phạm vi V1

**Nguồn dữ liệu:** Văn bản hợp nhất số **18/VBHN-VPQH** (12/02/2026) của Văn phòng Quốc hội, hợp nhất Bộ luật Lao động số 45/2019/QH14 với các sửa đổi từ Luật Công nghiệp công nghệ số 71/2025/QH15, Luật Giáo dục nghề nghiệp 124/2025/QH15 và Luật Dân số 113/2025/QH15.

**Trong phạm vi:** tra cứu và giải thích các quy định trong Bộ luật (hợp đồng lao động, tiền lương, thời giờ làm việc, kỷ luật, chấm dứt hợp đồng, trợ cấp, v.v.).

**Có điều kiện (cờ `needs_decree`):** một số điều chỉ nêu nguyên tắc và giao Chính phủ quy định chi tiết (ví dụ Điều 46, 47 về trợ cấp). Với các điều này, hệ thống trả lời phần có trong Bộ luật và **nói rõ phần chi tiết do nghị định hướng dẫn, chưa có trong hệ thống V1**.

**Ngoài phạm vi (non-goals):**
- Tư vấn pháp lý chính thức; đánh giá khả năng thắng kiện hay kết luận về một vụ việc cụ thể.
- Soạn đơn từ, hợp đồng.
- Luật khác (thuế, bảo hiểm xã hội chi tiết...).
- Tải lên và xử lý hợp đồng cá nhân (tránh dữ liệu cá nhân trong V1).

## 4. Các loại câu hỏi và hành vi mong muốn

| Loại | Ví dụ | Hành vi mong muốn |
|---|---|---|
| Tra cứu trực tiếp | "Điều 36 quy định gì?" | Trả nguyên văn kèm trích dẫn |
| Tình huống đời thường | "Bị đuổi việc không báo trước thì sao?" | Tìm điều liên quan, giải thích, trích dẫn |
| Nhiều điều luật | "Nghỉ việc thì được những khoản gì?" | Kết hợp nhiều điều, trích dẫn từng ý |
| Cần nghị định hướng dẫn | "Lương làm thêm giờ ban đêm tính thế nào?" | Nêu phần trong Bộ luật, báo rõ phần cần nghị định |
| Ngoài phạm vi | "Thuế thu nhập cá nhân bao nhiêu?" | Từ chối lịch sự |
| Cá nhân hóa / tranh chấp | "Tôi có thắng kiện không?" | Không kết luận, khuyến nghị liên hệ cơ quan chức năng hoặc luật sư |

## 5. Rủi ro và nguyên tắc thiết kế

Trả lời sai về pháp luật có thể khiến người lao động ra quyết định bất lợi (ký giấy, bỏ lỡ quyền lợi, nghỉ việc sai cách). Vì vậy:

1. **Có căn cứ mới trả lời.** Từ chối hoặc nói "chưa đủ căn cứ" tốt hơn trả lời liều.
2. **Luôn trích dẫn** Điều/Khoản và hiển thị nguyên văn.
3. **Luôn có cảnh báo** "Thông tin chỉ mang tính tham khảo, không phải tư vấn pháp lý".
4. **Ghi rõ phiên bản luật** và ngày hiệu lực của dữ liệu đang dùng.
5. **Không đưa kết luận cuối** về vụ việc cụ thể của người dùng.

## 6. Ràng buộc

- **Phần cứng:** laptop 8GB RAM, không GPU → embedding model nhỏ chạy CPU; LLM gọi qua API; Docker để cuối.
- **Chi phí và độ trễ:** theo dõi số token và thời gian mỗi câu hỏi; ngưỡng cụ thể sẽ đặt sau khi có baseline.
- **Dữ liệu cá nhân:** câu hỏi được gửi tới API LLM bên ngoài; giao diện nhắc người dùng không nhập thông tin cá nhân.
- **Ngôn ngữ:** tiếng Việt.

## 7. Chỉ số thành công

Mục tiêu ban đầu, sẽ điều chỉnh sau khi có số liệu baseline. Đánh giá trên bộ câu hỏi `eval/golden_set.jsonl`.

| Tầng | Chỉ số | Mục tiêu ban đầu |
|---|---|---|
| Tìm kiếm | Điều luật đúng nằm trong top 5 (Recall@5) | > 85% |
| Trả lời | Đúng và đủ ý (chấm tay, thêm LLM-as-judge) | > 80% |
| Trích dẫn | Điều/Khoản trích dẫn đúng | > 90% |
| An toàn | Từ chối đúng với câu ngoài phạm vi | > 90% |
| Trung thực | Câu trả lời chứa nội dung không có trong luật | gần 0% |
| Vận hành | Độ trễ, chi phí mỗi câu hỏi | đặt ngưỡng sau baseline |

## 8. Kế hoạch dữ liệu

- File gốc: `data/raw/luatLaoDong2019.doc` (văn bản hợp nhất). Cần đổi sang `.docx` trước khi xử lý.
- Cắt riêng thân luật và phần chú thích cuối file; chú thích dùng làm metadata `amended_by`.
- Một Điều là một chunk; Điều dài (> khoảng 1.800 ký tự) tách theo Khoản; loại Điều 219 (sửa đổi các luật khác) khỏi chỉ mục.
- Lưu `refs` (các Điều được nhắc tới) để kéo thêm ngữ cảnh khi cần.
- Kiểm tra định kỳ xem có bản hợp nhất mới hơn trên nguồn chính thống.

## 9. Hướng phát triển (sau V1)

- V2: thêm Nghị định 145/2020/NĐ-CP (hướng dẫn Bộ luật Lao động) và các nghị định liên quan; kiểm tra lại hiệu lực trước khi nạp.
- Hội thoại nhiều lượt (viết lại câu hỏi nối tiếp thành câu độc lập).
- Theo dõi văn bản sửa đổi và cập nhật chỉ mục có đánh phiên bản.
- Giám sát: ghi log câu hỏi, chunk truy xuất, câu trả lời để bổ sung vào bộ đánh giá.

## 10. Nhật ký quyết định

| Ngày | Quyết định | Lý do |
|---|---|---|
| 22/11/2026 | Chọn Bộ luật Lao động thay vì Luật Đất đai | Luật Đất đai có nhiều nghị định hướng dẫn và thay đổi nhiều, dễ trả lời lỗi thời |
| 22/11/2026 | Dùng bản hợp nhất 18/VBHN-VPQH | Đã gộp các sửa đổi mới nhất, nguồn chính thống |
| 22/11/2026 | V1 chỉ có Bộ luật, gắn cờ `needs_decree` | Giữ phạm vi vừa sức, hệ thống nêu rõ giới hạn thay vì trả lời thiếu |
| 22/11/2026 | LLM gọi qua API, embedding chạy CPU | Máy 8GB RAM không GPU |