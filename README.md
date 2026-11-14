# Day 06 · Data Ingestion Engineering

## Bài cá nhân — Lê Tuấn Đạt · 2A202603949 · L3B

Bản thực hiện dùng **46 trang nội dung NIST**, MiniLM ONNX local và Qdrant local lưu trên đĩa; không cần API key hoặc Docker. Cùng corpus/model/cosine metric, A chia cố định 640 ký tự, B theo đoạn tối đa 640 ký tự, C giữ nguyên chunk của B và chỉ thêm filter năm khi câu hỏi cung cấp năm. Bộ 20 câu đã tách 15 dev / 5 test trước chạy; chọn cấu hình trên dev rồi mới đánh giá test.

Lượt gốc: **B chọn trên dev**, B/C đạt **4/5 test Evidence Hit@3**, A **3/5**; chi phí API **0 USD**. Toàn bộ nhãn coherence, metadata và grounding đã được học viên đối chiếu trực tiếp từ nguồn gốc và kết quả truy xuất (có sự hỗ trợ rà soát của AI theo đúng [RULES.md](RULES.md)); xem phương pháp và giới hạn trong báo cáo.

**Bắt đầu:** đọc [cách setup/chạy và phạm vi kiểm chứng](docs/RUNBOOK.md). Lệnh chính từ root repo: `python -m starter.src.run --output results/reproduction`. Với repo đã có kết quả, dùng thư mục output mới để không ghi đè lượt gốc.

| File | Để làm gì |
|---|---|
| [STUDENT.md](STUDENT.md) | Thông tin bài cá nhân |
| [data/manifest.json](data/manifest.json) | Nguồn, quyền, phiên bản và từng trang được tính |
| [benchmark/README.md](benchmark/README.md) | Cách chia câu hỏi, gold, filters và chống rò rỉ test |
| [configs/experiment.json](configs/experiment.json) | Ba cấu hình và model cố định |
| [starter/src/](starter/src/) | Code đọc tài liệu, chunking, embedding, lưu vector, chấm và chạy |
| [report/individual/2A202603949_LeTuanDat.md](report/individual/2A202603949_LeTuanDat.md) | Kết quả, phân tích năm khía cạnh và giới hạn |

Phần dưới là hướng dẫn đề gốc. PostgreSQL/pgvector trong starter là lựa chọn khác, không phải backend của bản thực hiện này.

**Lab cá nhân** · **4 giờ** · **Dữ liệu tự chọn**

Biến tài liệu thành kho tri thức có thể truy xuất cho AI agent, rồi đo xem cách chuẩn bị dữ liệu nào hiệu quả nhất.

| 📄 Dữ liệu | 🧩 Chiến lược | 🎯 Benchmark |
|---|---|---|
| Tối thiểu **30 trang nội dung** | Tối thiểu **3 cấu hình** | Tối thiểu **20 câu hỏi** |

## Bắt đầu nhanh

Dùng [starter PostgreSQL + pgvector](starter/README.md) để dựng database và hoàn thiện các hàm TODO, hoặc tự chọn vector store tương đương. Starter chỉ là khung: **bạn tự hiện thực pipeline và benchmark**.

## 1. Xây pipeline

![Đọc tài liệu, chuẩn hóa, chia chunk, tạo vector, lưu và kiểm tra](assets/ingestion-pipeline.svg)

Chọn tài liệu được phép sử dụng; giữ **ID tài liệu, nguồn, trang/vị trí** và metadata phù hợp. Dùng embedding model thật qua API hoặc chạy local; lưu và truy xuất bằng pgvector, Qdrant, Chroma hoặc một store tương đương.

## 2. So sánh và đánh giá

![Ba chiến lược được đánh giá trên cùng corpus và benchmark dev/test](assets/benchmark-workflow.svg)

Dùng cùng dữ liệu, model và câu hỏi cho cả ba cấu hình. Tách **15 câu dev / 5 câu test** trước khi chạy; chọn cấu hình trên dev, rồi đánh giá test sau khi cố định. Mỗi câu hỏi cần có đoạn bằng chứng và trang nguồn; thiết kế bộ câu hỏi và đánh giá theo [BENCHMARK.md](BENCHMARK.md).

Đánh giá theo năm khía cạnh: **độ chính xác truy xuất**, **tính mạch lạc của chunk**, **tính hữu dụng của metadata**, **chất lượng grounding** và **tác động của chiến lược dữ liệu**. Giải thích lựa chọn bằng kết quả, ví dụ lỗi và các đánh đổi.

## 3. Chuẩn bị bài nộp

1. Chạy pipeline và lưu cấu hình, kết quả top-3, trace cùng nhãn đánh giá thủ công.
2. Điền [thông tin học viên](STUDENT.md), [nhật ký thí nghiệm](artifacts/experiment_log.csv) và bản sao [mẫu báo cáo cá nhân](report/individual/MSSV_HoTen.md).
3. Kiểm tra mã nguồn, cách chạy, manifest dữ liệu và checklist trong [SUBMISSION.md](SUBMISSION.md).

## Tài liệu hướng dẫn

| Bạn cần tìm | Tài liệu |
|---|---|
| Setup baseline PostgreSQL + pgvector | [starter/README.md](starter/README.md) |
| Các bước làm và cách tính chỉ số | [GUIDE.md](GUIDE.md) |
| Tiến độ trong 4 giờ | [CHECKPOINTS.md](CHECKPOINTS.md) |
| Năm tiêu chí benchmark và bộ câu hỏi | [BENCHMARK.md](BENCHMARK.md) |
| Tiêu chí chấm điểm | [RUBRIC.md](RUBRIC.md) |
| Quy tắc sử dụng dữ liệu, AI và API | [RULES.md](RULES.md) |
| Cách nộp bài | [SUBMISSION.md](SUBMISSION.md) |

[data/format-example.json](data/format-example.json) chỉ minh họa định dạng, không phải corpus đủ điều kiện. Phần xử lý và benchmark do bạn tự hiện thực; không bắt buộc xây chatbot.
