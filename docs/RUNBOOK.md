# Chạy và kiểm chứng bài cá nhân

## 1. Môi trường

Đã kiểm tra trên Windows, Python 3.12.10. Dùng Python 3.12 và chạy các lệnh ở root repo. Không cần key, model trả phí hoặc Docker. Các phiên bản sử dụng nằm trong `requirements.txt` và `results/benchmark/run_info.json`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe -m starter.src.run --output results/reproduction
```

Trên Linux/macOS thay đường dẫn Python môi trường bằng `.venv/bin/python`. Chưa xác minh thực chạy trên hai hệ điều hành này; hash JSON/JSONL đã chuẩn hóa để không phụ thuộc CRLF/LF. Môi trường làm bài ban đầu dùng `--system-site-packages` để tái sử dụng thư viện có sẵn; không thay package global.

Lần đầu cần mạng để tải package, 2 PDF và model. Model ONNX int8 khoảng 23 MB, tokenizer 0,71 MB, hai PDF khoảng 3,12 MB. Các lần chạy sau có thể dùng asset đã cache. Nếu nguồn không còn đúng checksum, chương trình dừng thay vì âm thầm thay snapshot/model.

## 2. Luồng và các lệnh

```powershell
# Chỉ đọc/chuẩn hóa và ghi log trang, không truy xuất
.\.venv\Scripts\python.exe -m starter.src.run --prepare-only

# Chỉ dành cho bộ câu hỏi mới CHƯA chạy: không dùng để ghi đè freeze đã nộp
.\.venv\Scripts\python.exe -m starter.src.run --freeze-benchmark

# Chạy A/B/C; lựa chọn trên dev được ghi trước truy vấn test đầu tiên
.\.venv\Scripts\python.exe -m starter.src.run --output results/reproduction
```

Bản nộp đã chứa `benchmark/freeze.json`, không chạy lại lệnh freeze. Muốn tái lập, giữ nguyên dữ liệu/cấu hình/câu hỏi và chỉ chọn thư mục output mới. Nếu output đã có trace, chương trình từ chối ghi đè. Chỉ chạy một tiến trình dùng cùng thư mục Qdrant vì backend local khóa dữ liệu đang mở.

## 3. Các file đầu ra

| File | Minh chứng |
|---|---|
| `data/normalized_pages.jsonl` | Văn bản từng trang, nguồn và metadata |
| `results/preparation.json` | Đọc cả 112 trang vật lý; trang được chọn, rỗng, lỗi, warning |
| `benchmark/freeze.json` | Hash corpus, gold, mẫu coherence và cấu hình trước benchmark |
| `<output>/chunks_A.jsonl`, `chunks_B.jsonl`, `chunks_C.jsonl` | Chunk và offset truy nguồn; B/C có cùng ID/text |
| `<output>/traces.jsonl` | 60 trace: 45 dev trước, 15 test sau, đầy đủ top-3 |
| `<output>/selection.json` | Cấu hình chọn từ dev, thời điểm và số truy vấn test lúc chọn = 0 |
| `<output>/metrics.json` | Hit@1/Hit@3 với tử/mẫu riêng dev/test |
| `<output>/persistence.json` | Số điểm trước/sau nạp lại và sau đóng/mở backend |
| `<output>/manual_audit_inputs.json` | 10 vị trí nguồn × 3 chiến lược và 5 test × 3 để đọc/chấm |
| `results/benchmark/audit_labels.json` | Nhãn đối chiếu coherence, metadata, filter, grounding của lượt gốc và lý do chi tiết |
| `results/benchmark/audit_summary.json` | Tổng hợp coherence, metadata, filter và grounding từ nhãn |
| `artifacts/experiment_log.csv` | Cấu hình, giả thuyết và kết quả A/B/C |
| `<output>/run_info.json` | Model/revision/hash, thư viện, thời gian, cache, truncation và chi phí |
| `results/tests.xml` | Kết quả unit/integration tests |
| `results/verification.json` | Đối chiếu artifacts, hash, metric, thứ tự dev/test và mở lại Qdrant; không chạy thêm benchmark |

`.cache/`, `storage/` và `data/raw/` không đưa vào Git: được tái tạo bằng lệnh chạy và manifest. Qdrant sử dụng official local client để lưu bền vững, tìm cosine và filter payload. Đây là exact search trong local backend; không đo ANN/HNSW hoặc dịch vụ Qdrant server. Unit tests có vector fixture 3 chiều chỉ để kiểm tra storage; toàn bộ benchmark dùng vector MiniLM thật 384 chiều.

Đường dẫn backend: `storage/qdrant-751bff37`; tên collection thực tế là `day06_a`, `day06_b`, `day06_c`.

`upsert` nhận toàn bộ snapshot corpus của một strategy: kiểm tra vector trước, ghi ID ổn định rồi xóa ID cũ không còn trong snapshot. Test đổi nội dung xác minh ID cũ bị bỏ; file PDF gốc và bộ benchmark không được sửa để lấy điểm tốt hơn.

## 4. Đọc số đo đúng phạm vi

- Hit chỉ được tính khi đúng nguồn/trang hợp lệ và chứa toàn bộ quote đã khai báo; không suy từ đúng trang rằng đủ bằng chứng. Có thể false miss khi bằng chứng cần ghép nhiều chunk; grounding cần đọc riêng top-3.
- C chỉ filter khi câu hỏi có năm; các câu khác chạy không filter. Không lấy gold source/page/answer để lọc. Báo N/A nếu tập con không có filter.
- Model/corpus/câu hỏi/cosine metric giữ nguyên. B/C tái sử dụng chunk và cache embedding; thời gian phụ thuộc thứ tự/cache, không phải benchmark latency công bằng giữa nhiều hệ thống.
- Thời gian ranking loại phần query embedding; wall time lượt chạy loại bước cài/tải model ban đầu. Không đo RAM hay điện năng.
- Audit phải ghi rõ AI hỗ trợ và chưa được học viên xác nhận nếu đúng như vậy. Không gọi Hit@3 là chất lượng câu trả lời LLM; bài không sinh câu trả lời bằng LLM.
- 46 trang là phạm vi chọn từ hai snapshot 2023/2024, không đại diện mọi tài liệu hoặc hướng dẫn hiện hành. Dữ liệu/đoạn trích được ghi nguồn và chính sách sử dụng NIST trong manifest; không dùng logo/hình bên thứ ba làm tài sản sản phẩm.

## 5. Đối chiếu nhãn audit thủ công

Mở `manual_audit_inputs.json` để đọc chunk ở mỗi anchor (`S01..S10`) và top-3 của 5 câu test (`Q04, Q08, Q13, Q17, Q20`); dùng `source`/trang PDF vật lý để đối chiếu theo thang điểm GUIDE. Kết quả đánh giá chi tiết cùng lý do được lưu trong `results/benchmark/audit_labels.json` (`student_reviewed: true`, có sự hỗ trợ rà soát của AI và đã được học viên kiểm tra lại toàn bộ).

Repo cá nhân được lưu tại URL trong `STUDENT.md` (nhánh `main`) để nộp theo hướng dẫn lớp.
