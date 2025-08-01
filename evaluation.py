from gru4rec_pytorch import SessionDataIterator
import torch

@torch.no_grad()  # Vô hiệu hóa tính toán gradient cho quá trình đánh giá
def batch_eval(gru, test_data, cutoff=[20], batch_size=512, mode='conservative', item_key='ItemId', session_key='SessionId', time_key='Time'):
    # Kiểm tra xem mô hình có gặp lỗi trong quá trình huấn luyện hay không
    if gru.error_during_train: 
        raise Exception('Đang cố gắng đánh giá một mô hình không được huấn luyện đúng cách (error_during_train=True)')
    
    # Khởi tạo các dictionary để lưu trữ các chỉ số recall và MRR cho từng giá trị cutoff
    recall = dict()
    mrr = dict()
    for c in cutoff:
        recall[c] = 0
        mrr[c] = 0
    
    # Khởi tạo trạng thái ẩn cho từng lớp GRU
    H = []
    for i in range(len(gru.layers)):
        H.append(torch.zeros((batch_size, gru.layers[i]), requires_grad=False, device=gru.device, dtype=torch.float32))
    
    n = 0  # Bộ đếm số lượng mẫu đã xử lý

    # Định nghĩa một hook reset để điều chỉnh trạng thái ẩn khi các phiên kết thúc
    reset_hook = lambda n_valid, finished_mask, valid_mask: gru._adjust_hidden(n_valid, finished_mask, valid_mask, H)
    
    # Tạo một bộ lặp dữ liệu cho dữ liệu kiểm tra
    data_iterator = SessionDataIterator(
        test_data, batch_size, 0, 0, 0, item_key, session_key, time_key, 
        device=gru.device, itemidmap=gru.data_iterator.itemidmap
    )
    
    # Lặp qua dữ liệu kiểm tra theo từng batch
    for in_idxs, out_idxs in data_iterator(enable_neg_samples=False, reset_hook=reset_hook):
        # Tách trạng thái ẩn để ngăn tích lũy gradient
        for h in H: 
            h.detach_()
        
        # Thực hiện một lượt truyền xuôi qua mô hình GRU
        O = gru.model.forward(in_idxs, H, None, training=False)
        oscores = O.T  # Chuyển vị các điểm đầu ra
        tscores = torch.diag(oscores[out_idxs])  # Trích xuất điểm số cho các mục tiêu
        
        # Tính toán thứ hạng dựa trên chế độ đánh giá
        if mode == 'standard': 
            ranks = (oscores > tscores).sum(dim=0) + 1
        elif mode == 'conservative': 
            ranks = (oscores >= tscores).sum(dim=0)
        elif mode == 'median':  
            ranks = (oscores > tscores).sum(dim=0) + 0.5 * ((oscores == tscores).sum(dim=0) - 1) + 1
        else: 
            raise NotImplementedError
        
        # Cập nhật các chỉ số recall và MRR cho từng giá trị cutoff
        for c in cutoff:
            recall[c] += (ranks <= c).sum().cpu().numpy()
            mrr[c] += ((ranks <= c) / ranks.float()).sum().cpu().numpy()
        
        n += O.shape[0]  # Tăng bộ đếm số lượng mẫu
    
    # Chuẩn hóa các chỉ số recall và MRR theo tổng số lượng mẫu
    for c in cutoff:
        recall[c] /= n
        mrr[c] /= n
    
    # Trả về các chỉ số recall và MRR đã tính toán
    return recall, mrr
