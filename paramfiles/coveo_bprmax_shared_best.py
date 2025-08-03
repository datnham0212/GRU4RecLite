from collections import OrderedDict  # Import thư viện OrderedDict để tạo dictionary có thứ tự
gru4rec_params = OrderedDict([  # Khởi tạo một OrderedDict để lưu các tham số cho GRU4Rec
('loss', 'bpr-max'),  # Hàm mất mát sử dụng là bpr-max
('constrained_embedding', True),  # Sử dụng embedding bị ràng buộc
('embedding', 0),  # Kích thước embedding (0 có thể là giá trị mặc định)
('elu_param', 1),  # Tham số cho hàm kích hoạt ELU
('layers', [512]),  # Cấu trúc mạng với một lớp ẩn có 512 đơn vị
('n_epochs', 10),  # Số lượng epoch để huấn luyện
('batch_size', 144),  # Kích thước batch trong quá trình huấn luyện
('dropout_p_embed', 0.35),  # Xác suất dropout cho embedding
('dropout_p_hidden', 0.0),  # Xác suất dropout cho các lớp ẩn
('learning_rate', 0.05),  # Tốc độ học (learning rate)
('momentum', 0.4),  # Hệ số momentum cho thuật toán tối ưu
('n_sample', 2048),  # Số lượng mẫu âm được lấy trong mỗi batch
('sample_alpha', 0.2),  # Tham số alpha cho sampling
('bpreg', 1.85),  # Hệ số điều chỉnh cho bpr-max
('logq', 0.0)  # Giá trị logq (có thể liên quan đến sampling)
])
