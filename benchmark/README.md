# Benchmark cố định trước truy xuất

20 câu hỏi được biên soạn từ corpus (có hỗ trợ AI rà soát diễn đạt); đã đối chiếu evidence trực tiếp với từng trang vật lý và kiểm tra chéo trước khi chạy. Không dùng câu hỏi từ bài của học viên khác. Tài liệu và câu hỏi bằng tiếng Anh; báo cáo giải thích bằng tiếng Việt.

- 15 dev, 5 test: test cố định là Q04, Q08, Q13, Q17, Q20; có cả hai tài liệu, câu định nghĩa, diễn đạt lại và điều kiện. Đây là cách chia chủ đích, không phải lấy ngẫu nhiên rồi chọn tập có điểm đẹp.
- 5 câu filter: Q01, Q04, Q11, Q16, Q17; năm xuất bản được cung cấp rõ trong câu hỏi. Chỉ C áp dụng filter. A/B nhận cùng câu hỏi nhưng không dùng filter.
- Q04/Q06/Q07/Q12/Q20 có trang bằng chứng thay thế đã đăng ký trước khi chạy. Không phạt kết quả tìm đúng ý ở trang hợp lệ khác. Quote giữ cách trích xuất thực tế, kể cả lỗi khoảng trắng `output s` ở Q19; không sửa nguồn âm thầm để khớp đáp án.
- Q20 có thể dùng trang 38 của AI100-1 hoặc trang 5 của AI600-1; không suy từ nguồn gold thành filter. Không đưa đáp án hoặc trang gold vào truy vấn.
- 10 vị trí coherence lấy trước từ 5 trang được chọn đầu tiên của mỗi tài liệu; vị trí 120 ký tự sau đầu đoạn nội dung đầu tiên dài ít nhất 160 ký tự, dịch qua ký tự trắng nếu cần. Không chọn mẫu sau khi nhìn điểm; mẫu thiên về phần đầu tài liệu và không đại diện toàn bộ bảng biểu.
- `freeze.json` lưu hash chuẩn hóa JSON/JSONL, gold, cấu hình, corpus và vị trí mẫu; độc lập CRLF/LF. Chọn cấu hình chỉ trên dev theo Evidence Hit@3, rồi Hit@1, số chunk và ID. `selection.json` được ghi trước truy vấn test đầu tiên; không điều chỉnh tham số dựa trên test.
- Bộ test do cùng người/AI hỗ trợ viết từ cùng corpus nên có thiên lệch chủ đề và không phải holdout ngoài miền. Các thử hạ tầng trước đó chỉ dùng 3 câu ví dụ không thuộc corpus và vector fixture của unit test, không chạy 20 câu benchmark.

`manual_audit_inputs.json` chỉ tạo mẫu cần đọc; điểm coherence/grounding phải dựa trên nội dung nguồn, ghi người/công cụ chấm và lý do. Điểm Hit@3 không tự biến thành điểm thủ công hoặc chất lượng câu trả lời LLM.
