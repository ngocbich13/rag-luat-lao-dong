# ⚖️ Chatbot RAG Tra cứu Bộ luật Lao động 2019

Hệ thống hỏi đáp và đối chiếu nguyên văn **Bộ luật Lao động 2019** (Văn bản hợp nhất số `18/VBHN-VPQH` gồm **17 Chương, 220 Điều**) ứng dụng kiến trúc **Hybrid RAG (Retrieval-Augmented Generation)** kết hợp **FastAPI Streaming** và **Google Gemini**.

---

## ✨ Tính năng nổi bật

- **Tìm kiếm lai chuyên biệt cho văn bản luật tiếng Việt (Hybrid Search + Legal Rerank)**:
  - **Viết lại truy vấn pháp lý (`Query Rewriting`)**: Tự động ánh xạ ngôn ngữ đời thường của người lao động (*"nghỉ ngang"*, *"lương tháng 13"*, *"nghỉ đẻ"*, *"đuổi việc"*) sang thuật ngữ pháp lý chuẩn và Điều luật tương ứng.
  - **BM25 Từ ghép + Vector 3072 chiều (`gemini-embedding-001`)**: Bảo toàn các cụm từ ghép pháp lý (*`lương_hưu`*, *`nghỉ_hưu`*, *`thử_việc`*, *`báo_trước`*) để tránh nhầm lẫn ngữ nghĩa (ví dụ phân biệt rõ *"lương hưu"* tại Điều 169 với *"trả lương"* tại Điều 94).
  - **Gộp & Khử trùng lặp theo Điều (`deduplicate_by_article`)**: Tự động hợp nhất đầy đủ các khoản của Điều luật dài, đảm bảo mỗi Điều chỉ xuất hiện một lần với trọn vẹn nội dung.
- **Kiểm soát an toàn & Chống ảo giác (Zero-Hallucination Guardrails)**:
  - **Ngưỡng điểm tin cậy (`min_score = 0.25`)**: Tự động từ chối trả lời và không hiển thị điều luật nhiễu đối với các câu hỏi ngoài phạm vi (*thuế TNCN, đất đai, ly hôn, hình sự, giao thông...*).
  - **Câu trả lời súc tích, đúng trọng tâm**: Dòng đầu tiên luôn là **Kết luận in đậm** có ngay con số/đáp án kèm chip trích dẫn `[khoản X Điều Y]`, theo sau tối đa 3 gạch đầu dòng ngắn gọn và dòng `Cách tính:` khi áp dụng công thức trong luật (như tính tuổi nghỉ hưu theo năm).
  - **Chế độ dự phòng trung thực (`Fallback`)**: Khi kết nối AI gián đoạn, hệ thống thông báo rõ trạng thái và vẫn hiển thị nguyên văn các Điều luật đạt ngưỡng điểm liên quan để người dùng tự tra cứu.
- **Giao diện Web 2 cột hiện đại (Split-Panel Legal Workspace)**:
  - **Phản hồi Streaming (SSE)**: Hiển thị tiến trình tìm kiếm và stream câu trả lời theo thời gian thực với thời gian tới chữ đầu tiên (**TTFT trung vị ~1.33 giây**, câu hỏi lặp lại qua LRU Cache đạt **0.02 ms**).
  - **Panel Nguyên văn Điều luật tương tác**: Nhấn vào bất kỳ chip `khoản X Điều Y` nào trong câu trả lời sẽ mở ngay nguyên văn Điều luật ở khung bên phải, **tự động tô sáng (highlight) toàn bộ các khoản được trích dẫn**, thu gọn các khoản không trích dẫn và tự cuộn khoản đầu tiên lên đầu.
  - **Mục lục 17 Chương · 220 Điều & Lịch sử tra cứu**: Cho phép duyệt nhanh toàn bộ Bộ luật theo Chương/Điều, tìm theo số Điều (ví dụ gõ `"Điều 35"`), quản lý/xóa lịch sử hỏi đáp và gửi phản hồi chất lượng (`Hữu ích` / `Báo sai`).

---

## 🏗️ Kiến trúc Pipeline (7 Giai đoạn)

```text
data/raw/luatLaoDong2019.docx
   │
   ▼  [Giai đoạn 1] src/parser.py
data/processed/structured_law.json  (Cây phân cấp: Chương > Mục > Điều > Khoản > Điểm)
   │
   ▼  [Giai đoạn 2] src/chunker.py & src/ingest.py
data/processed/chunks.jsonl         (258 chunks giữ trọn vẹn ranh giới Điều/Khoản + metadata)
   │
   ▼  [Giai đoạn 3] src/indexer.py
data/processed/embeddings.npz       (Ma trận Vector 258 x 3072 đã chuẩn hóa L2 + BM25 Corpus)
   │
   ▼  [Giai đoạn 4] src/retriever.py
HybridRetriever                     (Query Rewrite + BM25 + Cosine Vector + Rerank + Min-Score)
   │
   ▼  [Giai đoạn 5] src/generator.py
RAGGenerator                        (Gemini Streaming + Công thức tính pháp lý + LRU Cache + Fallback)
   │
   ├──► [Giai đoạn 6] src/evaluate.py & src/test_stage3.py (Đánh giá tự động & Đo độ trễ)
   └──► [Giai đoạn 7] src/app.py                           (FastAPI Web Server + Split-Panel UI)
```

---

## 📂 Cấu trúc thư mục

```text
rag-luat-lao-dong/
├── data/
│   ├── raw/
│   │   └── luatLaoDong2019.docx          # File Word văn bản luật gốc (18/VBHN-VPQH)
│   └── processed/
│       ├── structured_law.json           # Dữ liệu luật cấu trúc cây (17 Chương, 220 Điều)
│       ├── chunks.jsonl                  # 258 đoạn luật chuẩn hóa kèm metadata & citation
│       ├── embeddings.npz                # Ma trận vector 258x3072 nén NumPy
│       ├── eval_report.json              # Báo cáo đánh giá 25 câu hỏi chuẩn
│       └── stage3_eval_report.json       # Báo cáo kiểm thử thực tế 22 câu hỏi & độ trễ
├── eval/
│   └── test_questions.json               # Bộ câu hỏi kiểm thử chuẩn (trong & ngoài phạm vi)
├── src/
│   ├── parser.py                         # Giai đoạn 1: Phân tích cấu trúc file Word pháp luật
│   ├── chunker.py                        # Giai đoạn 2: Chia nhỏ văn bản theo Điều/Khoản
│   ├── check_chunks.py                   # Script kiểm tra chất lượng & độ dài toàn bộ chunks
│   ├── ingest.py                         # Script nạp mở rộng nhiều văn bản luật từ data/raw/
│   ├── indexer.py                        # Giai đoạn 3: Tạo vector embedding & tách từ tiếng Việt
│   ├── retriever.py                      # Giai đoạn 4: Tìm kiếm lai BM25 + Vector & Rerank
│   ├── generator.py                      # Giai đoạn 5: Sinh câu trả lời Streaming, Cache & Fallback
│   ├── evaluate.py                       # Giai đoạn 6: Đánh giá Recall@5, MRR, Citation, Refusal
│   ├── test_stage3.py                    # Bộ kiểm thử tự động 22 câu hỏi mẫu (chạy bằng 1 lệnh)
│   └── app.py                            # Giai đoạn 7: FastAPI Web Chatbot & Giao diện người dùng
├── tests/                                # 96 bài kiểm thử đơn vị (Pytest) cho toàn bộ 7 giai đoạn
├── requirements.txt                      # Danh sách thư viện Python
└── README.md
```

---

## 📊 Kết quả Kiểm thử & Đánh giá

| Chỉ số đánh giá | Kết quả đạt được | Ghi chú |
| :--- | :---: | :--- |
| **Kiểm thử đơn vị (`pytest tests -v`)** | **`96 / 96 PASSED (100%)`** | Bao gồm 60 test xử lý văn bản gốc + 36 test RAG & Web API |
| **Độ chính xác Điều luật đúng trong Top 3** | **`18 / 18 (100%)`** | Trên bộ 22 câu hỏi thực tế (`src/test_stage3.py`) |
| **Tìm kiếm `Recall@5` & `MRR`** | **`100%` / `95.2%`** | Trên bộ 25 câu hỏi đánh giá (`eval/test_questions.json`) |
| **Từ chối câu hỏi ngoài phạm vi (`Refusal`)** | **`100%`** | Trả lời rõ chưa có căn cứ, không hiển thị điều luật lạc đề |
| **Thời gian tới chữ đầu tiên (`TTFT`)** | **Trung vị `1.33 giây`** | Giảm từ `21.7 giây` xuống `~1.33 giây` nhờ Singleton & Streaming |
| **Phản hồi câu hỏi lặp lại (`Cache Hit`)** | **`0.02 ms`** | Bộ nhớ đệm LRU Cache trong RAM |

---

## 🚀 Hướng dẫn Cài đặt & Chạy trên máy (Local)

### 1. Cài đặt môi trường
Yêu cầu **Python 3.10+**. Mở terminal tại thư mục gốc dự án:

```powershell
# Tạo và kích hoạt môi trường ảo (Windows PowerShell)
python -m venv venv
.\venv\Scripts\activate

# Cài đặt các thư viện cần thiết
python -m pip install -r requirements.txt
```

### 2. Cấu hình khóa API (`.env`)
Tạo file `.env` ở thư mục gốc dự án với nội dung:

```env
GEMINI_API_KEY=AIzaSy_KHOA_API_CUA_BAN
```

### 3. Khởi động Web Chatbot
```powershell
python src/app.py
```
Mở trình duyệt truy cập: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 🧪 Các lệnh Kiểm thử & Vận hành dữ liệu

- **Chạy toàn bộ 96 bài kiểm thử đơn vị (`pytest`)**:
  ```powershell
  python -m pytest tests -v
  ```
- **Chạy bộ kiểm thử tự động 22 câu hỏi thực tế (in điều luật, điểm, câu trả lời và độ trễ)**:
  ```powershell
  python src/test_stage3.py
  ```
- **Kiểm tra độ toàn vẹn của dữ liệu `chunks.jsonl`**:
  ```powershell
  python src/check_chunks.py
  ```
- **Nạp thêm văn bản luật mới** (đặt thêm file `.docx` như `luatBaoHiemXaHoi2024.docx` vào `data/raw/` rồi chạy):
  ```powershell
  python src/ingest.py
  python src/indexer.py
  ```

---

## ☁️ Triển khai lên Render (Deploy Cloud)

1. Đẩy mã nguồn lên GitHub (`git push origin main`).
2. Tạo một **Web Service** mới trên **[Render.com](https://render.com)** kết nối với kho GitHub của dự án.
3. Cấu hình thông số:
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn src.app:app --host 0.0.0.0 --port $PORT`
4. Trong mục **Environment Variables**, thêm biến `GEMINI_API_KEY` với giá trị khóa Gemini API của bạn rồi bấm **Deploy**.

---

## ⚠️ Lưu ý pháp lý
Thông tin do hệ thống cung cấp chỉ mang tính chất tham khảo dựa trên văn bản **Bộ luật Lao động 2019 (18/VBHN-VPQH)** hiện có trong cơ sở dữ liệu, không thay thế cho văn bản tư vấn pháp lý chính thức của cơ quan nhà nước có thẩm quyền hoặc luật sư.
