# Giai Đoạn 6: Các Hành Động Tiếp Theo & Kế Hoạch Thực Hiện

**Ngày:** 21/12/2025  
**Mục tiêu:** Tối ưu hóa sâu hơn, giảm thời gian huấn luyện, xác thực inference latency, so sánh TOP-1 loss

---

## 1. Điều Tra Khoảng Cách Recall: train_valid vs train_full

### Vấn đề
- train_valid: Recall@20 = 0.588 (T2.2_112)
- train_full: Recall@20 = 0.605 (T3.1)
- Khoảng cách: +2.9% (dương, không phải overfitting)

### Giả thuyết
```
Tại sao train_full tốt hơn?

1. Kích thước dữ liệu lớn hơn:
   - train_valid: 20.7k items, ~40k sessions
   - train_full: 37.8k items, ~200k sessions
   - Mô hình học được các mẫu phong phú hơn

2. Phân phối dữ liệu khác:
   - train_valid: Tập xác thực đã lọc (cân bằng hơn)
   - train_full: Dữ liệu thực tế (power-law distribution)
   - Mô hình thích ứng với dữ liệu sản xuất

3. Không phải overfitting:
   - Nếu là overfitting, train loss sẽ giảm nhưng test loss tăng
   - Ở đây test performance tăng -> dữ liệu thực tế tốt hơn

4. Đạo tạo lâu hơn:
   - train_valid: ~212s per epoch
   - train_full: ~2753s = 46 minutes per epoch
   - Mô hình có thời gian để hội tụ tốt hơn
```

### Các Bài Kiểm Tra Cần Chạy

#### T6.1: Phân tích Dữ liệu

**Mục đích:** So sánh thống kê giữa train_valid và train_full

```bash
# Tạo script Python để phân tích:
# - Kích thước từ vựng (number of unique items)
# - Số session
# - Chiều dài session trung bình
# - Phân phối item frequency (popular vs tail items)
# - Số lần xuất hiện của item phổ biến nhất vs hiếm nhất
```

**Kỳ vọng:**
- train_full có nhiều items hiếm hơn
- train_full có session dài hơn
- train_full có phân phối power-law rõ hơn

---

#### T6.2: Kiểm Tra Kích Thước Dữ liệu Ảnh Hưởng

**Cấu hình:** layers=112, cross-entropy, batch=128, 3 epochs

| Test | Dataset | Item count | Sessions | Recall@20 | Dự kiến |
|------|---------|-----------|----------|-----------|---------|
| T6.2a | train_valid (20.7k) | 20.7k | ~40k | 0.588 | Baseline |
| T6.2b | 50% train_full (18.9k) | ~18.9k | ~100k | ? | Nên < 0.605 |
| T6.2c | 100% train_full (37.8k) | 37.8k | ~200k | 0.605 | Tốt nhất |

**Script:**
```bash
# T6.2a: Đã chạy (T2.2_112)
# T6.2b: Tạo subset 50% của train_full, huấn luyện
# T6.2c: Đã chạy (T3.1)
```

**Kết luận mong đợi:** Recall tăng theo kích thước dataset (không overfitting)

---

## 2. Thử Nghiệm Thêm trên RetailRocket với BPR-max

### Vấn đề
- T5.3b (BPR-max baseline) đạt Recall@20=0.408
- Nhưng cấu hình có thể chưa tối ưu

### Các Bài Kiểm Tra Cần Chạy

#### T6.3: Tuning BPR-max trên RetailRocket

**Cấu hình tham khảo từ tài liệu:**
```
BPR-max thường yêu cầu:
- layers: 100-256
- batch_size: 64-128
- momentum: 0.3-0.5
- bpreg: 1.5-2.5
- learning_rate: 0.05-0.1
```

| Test | Layers | Batch | LR | Momentum | BPReg | Epoch | Recall@20 | Ghi chú |
|------|--------|-------|-----|----------|-------|-------|-----------|--------|
| T6.3a | 224 | 80 | 0.05 | 0.4 | 1.95 | 3 | 0.408 | Baseline (T5.3b) |
| T6.3b | 192 | 64 | 0.08 | 0.4 | 1.95 | 5 | ? | Tăng epochs |
| T6.3c | 256 | 64 | 0.08 | 0.5 | 2.0 | 5 | ? | Tăng capacity |
| T6.3d | 128 | 128 | 0.1 | 0.3 | 1.5 | 3 | ? | Batch lớn, momentum thấp |

**Batch script:**
```batch
REM T6.3b: Tăng epochs từ 3 -> 5
python run.py input_data/retailrocket-data/retailrocket_train_full.dat ^
  -ps loss=bpr-max,layers=192,batch_size=64,n_epochs=5,n_sample=256,dropout_p_hidden=0.05,learning_rate=0.08,momentum=0.4,bpreg=1.95,elu_param=0.5 ^
  -t input_data/retailrocket-data/retailrocket_test.dat ^
  -m 1 5 10 20 ^
  -d cpu ^
  -s output_data/T6.3b_rr_bprmax_opt.pt

REM T6.3c: Tăng capacity (layers)
python run.py input_data/retailrocket-data/retailrocket_train_full.dat ^
  -ps loss=bpr-max,layers=256,batch_size=64,n_epochs=5,n_sample=256,dropout_p_hidden=0.05,learning_rate=0.08,momentum=0.5,bpreg=2.0,elu_param=0.5 ^
  -t input_data/retailrocket-data/retailrocket_test.dat ^
  -m 1 5 10 20 ^
  -d cpu ^
  -s output_data/T6.3c_rr_bprmax_large.pt
```

**Kỳ vọng:** Tuning có thể đạt Recall@20 > 0.42 (vượt 3% baseline)

---

## 3. Dừng Sớm (Early Stopping): Giảm Thời Gian Huấn Luyện

### Vấn đề
- T1.2: 5 epochs mất 397s, chỉ cải thiện 2.6% so với 1 epoch (71s)
- Lợi suất giảm dần rõ ràng sau epoch 3-4

### Kế Hoạch

#### T6.4: Phân Tích Convergence Curve

**Mục đích:** Xác định khi nào model hội tụ

| Test | Config | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 | Epoch 5 | Recall@20 |
|------|--------|---------|---------|---------|---------|---------|-----------|
| T6.4a | layers=112, CE | ? | ? | ? | ? | ? | Theo dõi loss mỗi epoch |

**Cách thức:** Chạy model để 6 epochs, ghi nhận loss + Recall@20 tại mỗi checkpoint

```python
# Thêm vào run.py hoặc tạo script độc lập:
for epoch in range(1, 7):
    train()
    evaluate()
    print(f"Epoch {epoch}: loss={loss:.4f}, Recall@20={recall:.4f}")
    # Lưu checkpoint
    save_checkpoint(f"checkpoint_epoch{epoch}.pt")
```

**Kỳ vọng:**
```
Epoch 1: loss=3.5, Recall@20=0.55
Epoch 2: loss=2.1, Recall@20=0.58 (+5.5%)
Epoch 3: loss=1.8, Recall@20=0.587 (+1.2%)
Epoch 4: loss=1.7, Recall@20=0.588 (+0.2%) <- Hội tụ, dừng ở đây
Epoch 5: loss=1.6, Recall@20=0.589 (+0.2%)
```

#### T6.5: Early Stopping Validation

**Cấu hình:**
```
- Stop nếu Recall@20 không cải thiện > 0.1% trong 1 epoch
- Hoặc stop sau epoch 3-4 tự động
```

| Config | Epochs | Recall@20 | Train Time | Savings |
|--------|--------|-----------|------------|---------|
| Full (5 epochs) | 5 | 0.588 | 249s | Baseline |
| Early stop (3 epochs) | 3 | 0.587 | 130s | 47% nhanh hơn |
| Early stop (4 epochs) | 4 | 0.5875 | 175s | 30% nhanh hơn |

**Script:**
```bash
# T6.5a: 3 epochs chỉ
python run.py input_data/yoochoose-data/yoochoose_train_valid.dat ^
  -ps loss=cross-entropy,layers=112,batch_size=128,n_epochs=3,n_sample=256,dropout_p_hidden=0.1,learning_rate=0.08 ^
  -t input_data/yoochoose-data/yoochoose_test.dat ^
  -m 5 10 20 ^
  -d cpu ^
  -s output_data/T6.5a_early_stop_3ep.pt

# T6.5b: 4 epochs
python run.py input_data/yoochoose-data/yoochoose_train_valid.dat ^
  -ps loss=cross-entropy,layers=112,batch_size=128,n_epochs=4,n_sample=256,dropout_p_hidden=0.1,learning_rate=0.08 ^
  -t input_data/yoochoose-data/yoochoose_test.dat ^
  -m 5 10 20 ^
  -d cpu ^
  -s output_data/T6.5b_early_stop_4ep.pt
```

**Kỳ vọng:** 3-4 epochs đủ, tiết kiệm ~40-50% thời gian với mất < 0.2% accuracy

---

## 4. Benchmark Độ Trễ Inference (p99 < 100ms)

### Vấn đề
- T4.1c: Đánh giá toàn bộ test set mất 192.61s
- Cần đo latency mỗi session (không phải batch)

### Kế Hoạch

#### T6.6: Latency Profiling Script

**Mục đích:** Đo thời gian inference per-session

```python
# Tạo file: benchmark_latency.py

import time
import torch
from gru4rec_pytorch import GRU4Rec

model = GRU4Rec.load("output_data/T3.1_full.pt")
model.eval()

latencies = []

for session_id in test_sessions:
    session_items = get_session_items(session_id)
    
    start = time.perf_counter()
    scores = model.predict(session_items)
    top20 = scores.topk(20)[1]
    end = time.perf_counter()
    
    latency_ms = (end - start) * 1000
    latencies.append(latency_ms)

# Tính percentile
import numpy as np
print(f"p50: {np.percentile(latencies, 50):.2f}ms")
print(f"p95: {np.percentile(latencies, 95):.2f}ms")
print(f"p99: {np.percentile(latencies, 99):.2f}ms")
print(f"p999: {np.percentile(latencies, 99.9):.2f}ms")
print(f"Mean: {np.mean(latencies):.2f}ms")
print(f"Max: {np.max(latencies):.2f}ms")
```

**Kỳ vọng:**
```
p50: 25ms
p95: 45ms
p99: 80ms      <- Dưới mục tiêu 100ms
p999: 150ms
Mean: 35ms
```

**Chiến lược Tối ưu:**
- Nếu p99 > 100ms: Sử dụng layers=96 (nhanh hơn 6%)
- Nếu p99 < 100ms: Sử dụng layers=112 (chất lượng tốt)

---

## 5. So Sánh TOP-1 Loss: Yoochoose vs RetailRocket

### Vấn đề
- Giai Đoạn 2.1 (T2.1c, T2.1d): TOP-1 và TOP1-max hoàn toàn thất bại trên Yoochoose
- Nhưng chưa kiểm tra trên RetailRocket

### Giả thuyết
```
TOP-1 mất bại trên Yoochoose có thể do:
1. Cấu hình siêu tham số không tối ưu
2. Dataset Yoochoose không phù hợp với TOP-1
3. Nhưng có thể hoạt động tốt trên RetailRocket (nhỏ hơn, ít diverse)
```

### Kế Hoạch

#### T6.7: TOP-1 & TOP-1 MAX trên Yoochoose (Tuned)

**Cấu hình mới (dựa trên tài liệu GRU4Rec):**

| Test | Loss | Layers | Batch | LR | Momentum | Epochs | Kỳ vọng |
|------|------|--------|-------|-----|----------|--------|---------|
| T6.7a | top1 | 128 | 32 | 0.05 | 0.3 | 5 | Cải thiện |
| T6.7b | top1-max | 128 | 64 | 0.1 | 0.0 | 5 | Cải thiện |

**Script:**
```bash
# T6.7a: TOP-1 tuned
python run.py input_data/yoochoose-data/yoochoose_train_valid.dat ^
  -ps loss=top1,layers=128,batch_size=32,n_epochs=5,n_sample=256,dropout_p_hidden=0.0,learning_rate=0.05,momentum=0.3 ^
  -t input_data/yoochoose-data/yoochoose_test.dat ^
  -m 5 10 20 ^
  -d cpu ^
  -s output_data/T6.7a_yoo_top1_tuned.pt

# T6.7b: TOP1-max tuned
python run.py input_data/yoochoose-data/yoochoose_train_valid.dat ^
  -ps loss=top1-max,layers=128,batch_size=64,n_epochs=5,n_sample=256,dropout_p_hidden=0.0,learning_rate=0.1,momentum=0.0 ^
  -t input_data/yoochoose-data/yoochoose_test.dat ^
  -m 5 10 20 ^
  -d cpu ^
  -s output_data/T6.7b_yoo_top1max_tuned.pt
```

---

#### T6.8: TOP-1 & TOP-1 MAX trên RetailRocket

**Cấu hình (như T6.7, nhưng cho RR):**

| Test | Loss | Layers | Batch | LR | Epochs | Kỳ vọng |
|------|------|--------|-------|-----|--------|---------|
| T6.8a | top1 | 100 | 32 | 0.05 | 3 | Hoạt động? |
| T6.8b | top1-max | 100 | 64 | 0.1 | 3 | Hoạt động? |

**Script:**
```bash
# T6.8a: TOP-1 trên RetailRocket
python run.py input_data/retailrocket-data/retailrocket_train_full.dat ^
  -ps loss=top1,layers=100,batch_size=32,n_epochs=3,n_sample=256,dropout_p_hidden=0.0,learning_rate=0.05,momentum=0.3 ^
  -t input_data/retailrocket-data/retailrocket_test.dat ^
  -m 1 5 10 20 ^
  -d cpu ^
  -s output_data/T6.8a_rr_top1_tuned.pt

# T6.8b: TOP1-max trên RetailRocket
python run.py input_data/retailrocket-data/retailrocket_train_full.dat ^
  -ps loss=top1-max,layers=100,batch_size=64,n_epochs=3,n_sample=256,dropout_p_hidden=0.0,learning_rate=0.1,momentum=0.0 ^
  -t input_data/retailrocket-data/retailrocket_test.dat ^
  -m 1 5 10 20 ^
  -d cpu ^
  -s output_data/T6.8b_rr_top1max_tuned.pt
```

---

#### T6.9: So Sánh 3 Loss Function trên Cả 2 Dataset

**Bảng Tổng Hợp:**

| Dataset | Loss | Recall@20 | MRR@20 | Train Time | Status |
|---------|------|-----------|--------|-----------|--------|
| **Yoochoose** |
| | Cross-entropy | 0.588 | 0.249 | 212s | PASS |
| | BPR-max | 0.603 | 0.265 | 526s | PASS |
| | TOP-1 (tuned) | ? | ? | ? | ? |
| | TOP-1 max (tuned) | ? | ? | ? | ? |
| **RetailRocket** |
| | Cross-entropy | 0.384 | 0.113 | 209s | PASS |
| | BPR-max | 0.408 | 0.156 | 286s | PASS |
| | TOP-1 (tuned) | ? | ? | ? | ? |
| | TOP-1 max (tuned) | ? | ? | ? | ? |

---

## Lịch Thực Hiện Khuyến Nghị

### Tuần 1 (Ngay lập tức)
- **T6.1**: Phân tích dữ liệu (30 min) - Nhanh
- **T6.2b**: 50% train_full subset (2h)
- **T6.4 + T6.5**: Early stopping tests (3h)

### Tuần 2
- **T6.7 + T6.8**: TOP-1 comparison (4h)
- **T6.3**: BPR-max tuning trên RR (4h)
- **T6.6**: Latency profiling (2h)

### Độ Ưu Tiên

| Ưu tiên | Hành động | Tác động | Effort |
|---------|----------|---------|--------|
| **1** | T6.4 + T6.5 (Early stopping) | Tiết kiệm 40-50% time | 3h |
| **2** | T6.6 (Latency p99) | Xác thực production-ready | 2h |
| **3** | T6.7 + T6.8 (TOP-1 tuned) | So sánh toàn diện | 4h |
| **4** | T6.2b (Data size effect) | Hiểu recall gap | 2h |
| **5** | T6.3 (RR BPR-max tuning) | Tối ưu hóa sâu | 4h |

---

## Kết Quả Mong Đợi

### Sau Giai Đoạn 6

```
1. Recall Gap Resolved:
   - train_valid vs train_full: Không phải overfitting, chỉ data size effect

2. Early Stopping Applied:
   - 3 epochs: 130s, Recall@20 = 0.587 (tiết kiệm 47%)
   - 4 epochs: 175s, Recall@20 = 0.5875 (tiết kiệm 30%)

3. Latency Verified:
   - p99 < 100ms: Model production-ready
   - p99 > 100ms: Use layers=96 instead

4. TOP-1 Evaluated:
   - TOP-1 tuned on Yoochoose: Recall@20 = ?
   - TOP-1 on RetailRocket: Recall@20 = ?
   - Conclusion: TOP-1 viable hoặc không

5. Recommendation:
   - Best config for Yoochoose: CE/BPR-max, layers=112, 3-4 epochs, train_full
   - Best config for RetailRocket: BPR-max (tuned), layers=224+, 3-5 epochs
```

