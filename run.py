import argparse
import os
import shutil

# Bộ định dạng tùy chỉnh cho argparse để điều chỉnh độ rộng văn bản trợ giúp dựa trên kích thước terminal
class MyHelpFormatter(argparse.HelpFormatter):
    def __init__(self, *args, **kwargs):
        super(MyHelpFormatter, self).__init__(*args, **kwargs)
        self._width = shutil.get_terminal_size().columns

# Định nghĩa các tham số dòng lệnh cho script
parser = argparse.ArgumentParser(formatter_class=MyHelpFormatter, description='Huấn luyện hoặc tải mô hình GRU4Rec & đo lường recall và MRR trên tập kiểm tra được chỉ định.')
# Các tham số cho dữ liệu huấn luyện, tham số, tải/lưu mô hình, kiểm tra và đánh giá
parser.add_argument('path', metavar='PATH', type=str, help='Đường dẫn đến dữ liệu huấn luyện hoặc mô hình đã được lưu.')
parser.add_argument('-ps', '--parameter_string', metavar='PARAM_STRING', type=str, help='Tham số huấn luyện dưới dạng chuỗi.')
parser.add_argument('-pf', '--parameter_file', metavar='PARAM_PATH', type=str, help='Tham số huấn luyện từ tệp cấu hình.')
parser.add_argument('-l', '--load_model', action='store_true', help='Tải một mô hình đã được huấn luyện.')
parser.add_argument('-s', '--save_model', metavar='MODEL_PATH', type=str, help='Đường dẫn để lưu mô hình đã được huấn luyện.')
parser.add_argument('-t', '--test', metavar='TEST_PATH', type=str, nargs='+', help='Đường dẫn đến tập dữ liệu kiểm tra.')
parser.add_argument('-m', '--measure', metavar='AT', type=int, nargs='+', default=[20], help='Giá trị cắt giảm cho các chỉ số (ví dụ: Recall@20).')
parser.add_argument('-e', '--eval_type', metavar='EVAL_TYPE', choices=['standard', 'conservative', 'median'], default='standard', help='Chế độ đánh giá để xử lý các trường hợp hòa.')
parser.add_argument('-ss', '--sample_store_size', metavar='SS', type=int, default=10000000, help='Kích thước bộ đệm cho các mẫu âm trong quá trình huấn luyện.')
parser.add_argument('-g', '--gru4rec_model', metavar='GRFILE', type=str, default='gru4rec_pytorch', help='Tệp chứa lớp GRU4Rec.')
parser.add_argument('-d', '--device', metavar='D', type=str, default='cuda:0', help='Thiết bị để thực hiện tính toán (ví dụ: GPU).')
parser.add_argument('-ik', '--item_key', metavar='IK', type=str, default='item_id', help='Tên cột cho ID sản phẩm.')
parser.add_argument('-sk', '--session_key', metavar='SK', type=str, default='session_id', help='Tên cột cho ID phiên.')
parser.add_argument('-tk', '--time_key', metavar='TK', type=str, default='timestamp', help='Tên cột cho dấu thời gian.')
parser.add_argument('-pm', '--primary_metric', metavar='METRIC', choices=['recall', 'mrr'], default='recall', help='Chỉ số đánh giá chính.')
parser.add_argument('-lpm', '--log_primary_metric', action='store_true', help='Ghi lại giá trị của chỉ số chính vào cuối quá trình chạy.')
parser.add_argument('--eval-metrics', type=str, default='recall_mrr,coverage,ild,aggregate_diversity,inter_user_diversity',
                    help='Các chỉ số đánh giá cần tính, phân tách bằng dấu phẩy (ví dụ: recall_mrr,coverage,ild)')
parser.add_argument('--seed', metavar='SEED', type=int, default=42, help='Random seed cho reproducibility (default: 42)')
args = parser.parse_args()

# Thay đổi thư mục làm việc thành vị trí của script
import os.path
orig_cwd = os.getcwd()
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Import các thư viện và module cần thiết
import numpy as np
import pandas as pd
import datetime as dt
import sys
import time
from collections import OrderedDict
import importlib
import torch
import random

def set_seed(seed):
    """Set random seeds for reproducibility across numpy, torch, and random"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

GRU4Rec = importlib.import_module(args.gru4rec_model).GRU4Rec
import evaluation
import importlib.util
import joblib
os.chdir(orig_cwd)

# Hàm để tải dữ liệu từ tệp (pickle hoặc tệp phân cách bằng TAB)
def load_data(fname, args):
    if fname.endswith('.pickle'):
        print('Đang tải dữ liệu từ tệp pickle: {}'.format(fname))
        data = joblib.load(fname)
        # Kiểm tra tên cột trong dữ liệu
        if args.session_key not in data.columns:
            print('LỖI. Cột được chỉ định cho ID phiên "{}" không có trong tệp dữ liệu ({})'.format(args.session_key, fname))
            sys.exit(1)
        if args.item_key not in data.columns:
            print('LỖI. Cột được chỉ định cho ID sản phẩm "{}" không có trong tệp dữ liệu ({})'.format(args.item_key, fname))
            sys.exit(1)
        if args.time_key not in data.columns:
            print('LỖI. Cột được chỉ định cho thời gian "{}" không có trong tệp dữ liệu ({})'.format(args.time_key, fname))
            sys.exit(1)
    else:
        # Đối với tệp phân cách bằng TAB, kiểm tra tiêu đề và tải dữ liệu
        with open(fname, 'rt') as f:
            header = f.readline().strip().split('\t')
        if args.session_key not in header:
            print('LỖI. Cột được chỉ định cho ID phiên "{}" không có trong tệp dữ liệu ({})'.format(args.session_key, fname))
            sys.exit(1)
        if args.item_key not in header:
            print('LỖI. Cột được chỉ định cho ID sản phẩm "{}" không có trong tệp dữ liệu ({})'.format(args.item_key, fname))
            sys.exit(1)
        if args.time_key not in header:
            print('LỖI. Cột được chỉ định cho thời gian "{}" không có trong tệp dữ liệu ({})'.format(args.time_key, fname))
            sys.exit(1)
        print('Đang tải dữ liệu từ tệp phân cách bằng TAB: {}'.format(fname))
        data = pd.read_csv(fname, sep='\t', usecols=[args.session_key, args.item_key, args.time_key], dtype={args.session_key:'int32', args.item_key:'str'})
    return data

# Đảm bảo chính xác một trong các tùy chọn tham số được cung cấp
if (args.parameter_string is not None) + (args.parameter_file is not None) + (args.load_model) != 1:
    print('LỖI. Chính xác một trong các tham số sau phải được cung cấp: --parameter_string, --parameter_file, --load_model')
    sys.exit(1)

# Tải hoặc huấn luyện mô hình GRU4Rec
if args.load_model:
    print('Đang tải mô hình đã huấn luyện từ tệp: {} (vào thiết bị "{}")'.format(args.path, args.device))
    gru = GRU4Rec.loadmodel(args.path, device=args.device)
else:
    # Tải tham số từ tệp hoặc chuỗi
    if args.parameter_file:
        param_file_path = os.path.abspath(args.parameter_file)
        param_dir, param_file = os.path.split(param_file_path)
        spec = importlib.util.spec_from_file_location(param_file.split('.py')[0], os.path.abspath(args.parameter_file))
        params = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(params)
        gru4rec_params = params.gru4rec_params
        print('Đã tải tham số từ tệp: {}'.format(param_file_path))
    if args.parameter_string:
        gru4rec_params = OrderedDict([x.split('=') for x in args.parameter_string.split(',')])
    print('Đang tạo mô hình GRU4Rec trên thiết bị "{}"'.format(args.device))
    # Set random seed for reproducibility
    set_seed(args.seed)
    print('Random seed set to: {}'.format(args.seed))
    gru = GRU4Rec(device=args.device)
    gru.set_params(**gru4rec_params)
    print('Đang tải dữ liệu huấn luyện...')
    data = load_data(args.path, args)
    print('Bắt đầu huấn luyện')
    t0 = time.time()
    # Huấn luyện mô hình
    gru.fit(data, sample_cache_max_size=args.sample_store_size, item_key=args.item_key, session_key=args.session_key, time_key=args.time_key)
    t1 = time.time()
    print('Thời gian huấn luyện tổng cộng: {:.2f}s'.format(t1 - t0))
    # Lưu mô hình đã huấn luyện nếu được chỉ định
    if args.save_model is not None:
        print('Đang lưu mô hình đã huấn luyện vào: {}'.format(args.save_model))
        gru.savemodel(args.save_model)

# Đánh giá mô hình trên các tập dữ liệu kiểm tra nếu được cung cấp
if args.test is not None:
    # Xác định chỉ số chính
    if args.primary_metric.lower() == 'recall':
        pm_index = 0
    elif args.primary_metric.lower() == 'mrr':
        pm_index = 1
    else:
        raise RuntimeError('Giá trị không hợp lệ `{}` cho tham số `primary_metric`'.format(args.primary_metric))
    for test_file in args.test:
        print('Đang tải dữ liệu kiểm tra...')
        test_data = load_data(test_file, args)
        print('Bắt đầu đánh giá (cut-off={}, sử dụng chế độ {} để xử lý hòa)'.format(args.measure, args.eval_type))
        t0 = time.time()
        # Thực hiện đánh giá theo lô
        eval_metrics = args.eval_metrics.split(',') if hasattr(args, 'eval_metrics') else ['recall_mrr', 'coverage', 'ild']
        res = evaluation.batch_eval(gru, test_data, batch_size=512, cutoff=args.measure, mode=args.eval_type, item_key=args.item_key, session_key=args.session_key, time_key=args.time_key, eval_metrics=eval_metrics)
        t1 = time.time()
        print('Đánh giá mất {:.2f}s'.format(t1 - t0))
        # In kết quả đánh giá
        if 'recall' in res and 'mrr' in res:
            for c in args.measure:
                print('Recall@{}: {:.6f} MRR@{}: {:.6f}'.format(c, res['recall'][c], c, res['mrr'][c]))
        if 'item_coverage' in res:
            print('Item coverage: {:.6f}'.format(res['item_coverage']))
        if 'catalog_coverage' in res:
            print('Catalog coverage: {:.6f}'.format(res['catalog_coverage']))
        if 'ild' in res:
            print('ILD: {:.6f}'.format(res['ild']))
        if 'aggregate_diversity' in res:
            print('Aggregate diversity: {:.6f}'.format(res['aggregate_diversity']))
        if 'inter_user_diversity' in res:
            print('Inter-user diversity: {:.6f}'.format(res['inter_user_diversity']))

        # Ghi lại chỉ số chính nếu được chỉ định
        if args.log_primary_metric: 
            print('CHỈ SỐ CHÍNH: {}'.format([x for x in res[pm_index].values()][0]))
            