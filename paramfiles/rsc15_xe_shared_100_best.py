from collections import OrderedDict  # Import thư viện OrderedDict để tạo dictionary có thứ tự
gru4rec_params = OrderedDict([  # Khởi tạo một OrderedDict để lưu các tham số cho GRU4Rec
('loss', 'cross-entropy'),  # Hàm mất mát sử dụng là cross-entropy
('constrained_embedding', True),  # Sử dụng embedding bị ràng buộc
('embedding', 0),  # Kích thước embedding (0 có thể là giá trị mặc định)
('elu_param', 0),  # Tham số cho hàm kích hoạt ELU
('layers', [100]),  # Cấu trúc mạng với một lớp ẩn có 100 đơn vị
('n_epochs', 10),  # Số lượng epoch để huấn luyện
('batch_size', 32),  # Kích thước batch trong quá trình huấn luyện
('dropout_p_embed', 0.0),  # Xác suất dropout cho embedding
('dropout_p_hidden', 0.4),  # Xác suất dropout cho các lớp ẩn
('learning_rate', 0.2),  # Tốc độ học (learning rate)
('momentum', 0.2),  # Hệ số momentum cho thuật toán tối ưu
('n_sample', 2048),  # Số lượng mẫu âm được lấy trong mỗi batch
('sample_alpha', 0.5),  # Tham số alpha cho sampling
('bpreg', 0.0),  # Hệ số điều chỉnh cho bpr-max
('logq', 1.0)  # Giá trị logq (có thể liên quan đến sampling
])
