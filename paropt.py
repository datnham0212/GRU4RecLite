import argparse
import os
import optuna
import json

# Bộ định dạng tùy chỉnh cho argparse để điều chỉnh độ rộng văn bản trợ giúp dựa trên kích thước terminal
class MyHelpFormatter(argparse.HelpFormatter):
    def __init__(self, *args, **kwargs):
        super(MyHelpFormatter, self).__init__(*args, **kwargs)
        try:
            columns = int(os.popen('stty size', 'r').read().split()[1])
        except:
            columns = None
        if columns is not None:
            self._width = columns

# Định nghĩa các tham số dòng lệnh cho script
parser = argparse.ArgumentParser(formatter_class=MyHelpFormatter, description='Tối ưu hóa tham số mô hình GRU4Rec bằng Optuna.')
parser.add_argument('path', metavar='PATH', type=str, help='Đường dẫn đến dữ liệu huấn luyện hoặc mô hình đã được lưu.')
parser.add_argument('test', metavar='TEST_PATH', type=str, help='Đường dẫn đến tập dữ liệu kiểm tra.')
parser.add_argument('-g', '--gru4rec_model', metavar='GRFILE', type=str, default='gru4rec_pytorch', help='Tệp chứa lớp GRU4Rec.')
parser.add_argument('-fp', '--fixed_parameters', metavar='PARAM_STRING', type=str, help='Các tham số cố định dưới dạng chuỗi.')
parser.add_argument('-opf', '--optuna_parameter_file', metavar='PATH', type=str, help='Tệp mô tả không gian tham số cho Optuna.')
parser.add_argument('-m', '--measure', metavar='AT', type=int, nargs='?', default=20, help='Đo recall & MRR tại độ dài danh sách gợi ý được chỉ định.')
parser.add_argument('-nt', '--ntrials', metavar='NT', type=int, nargs='?', default=50, help='Số lần thử nghiệm tối ưu hóa.')
parser.add_argument('-fm', '--final_measure', metavar='AT', type=int, nargs='*', default=[20], help='Đo recall & MRR sau khi tối ưu hóa.')
parser.add_argument('-pm', '--primary_metric', metavar='METRIC', choices=['recall', 'mrr'], default='recall', help='Chỉ số chính để tối ưu hóa.')
parser.add_argument('-e', '--eval_type', metavar='EVAL_TYPE', choices=['standard', 'conservative', 'median', 'tiebreaking'], default='standard', help='Chế độ đánh giá để xử lý các trường hợp hòa.')
parser.add_argument('-d', '--device', metavar='D', type=str, default='cuda:0', help='Thiết bị để thực hiện tính toán.')
parser.add_argument('-ik', '--item_key', metavar='IK', type=str, default='ItemId', help='Tên cột cho ID sản phẩm.')
parser.add_argument('-sk', '--session_key', metavar='SK', type=str, default='SessionId', help='Tên cột cho ID phiên.')
parser.add_argument('-tk', '--time_key', metavar='TK', type=str, default='Time', help='Tên cột cho dấu thời gian.')

args = parser.parse_args()

import pexpect
import numpy as np
from collections import OrderedDict
import importlib
import re

# Tạo lệnh để chạy mô hình GRU4Rec với các tham số đã được tối ưu hóa
def generate_command(optimized_param_str):
    command = 'python run.py "{}" -t "{}" -g {} -ps {},{} -m {} -pm {} -lpm -e {} -d {} -ik {} -sk {} -tk {}'.format(
        args.path, args.test, args.gru4rec_model, args.fixed_parameters, optimized_param_str, args.measure, 
        args.primary_metric, args.eval_type, args.device, args.item_key, args.session_key, args.time_key
    )
    return command

# Chạy mô hình GRU4Rec một lần với các tham số đã tối ưu hóa và trả về giá trị của chỉ số chính
def run_once(optimized_param_str):
    command = generate_command(optimized_param_str)
    cmd = pexpect.spawnu(command, timeout=None, maxread=1)
    line = cmd.readline()
    while line:
        line = line.strip()
        print(line)
        # Trích xuất giá trị của chỉ số chính từ đầu ra
        if re.match('PRIMARY METRIC: -*\\d\\.\\d+e*-*\\d*', line):
            t = line.split(':')[1].lstrip()
            val = float(t)
            break
        line = cmd.readline()
    return val

# Lớp để định nghĩa và xử lý không gian tham số cho tối ưu hóa
class Parameter:
    def __init__(self, name, dtype, values, step=None, log=False):
        assert dtype in ['int', 'float', 'categorical']
        assert type(values) == list
        assert len(values) == 2 or dtype == 'categorical'
        self.name = name
        self.dtype = dtype
        self.values = values
        self.step = step
        if self.step is None and self.dtype == 'int':
            self.step = 1
        self.log = log

    # Tạo một đối tượng Parameter từ chuỗi JSON
    @classmethod
    def fromjson(cls, json_string):
        obj = json.loads(json_string)
        return Parameter(obj['name'], obj['dtype'], obj['values'], obj['step'] if 'step' in obj else None, obj['log'] if 'log' in obj else False)

    # Gợi ý một giá trị cho tham số trong quá trình tối ưu hóa
    def __call__(self, trial):
        if self.dtype == 'int':
            return trial.suggest_int(self.name, int(self.values[0]), int(self.values[1]), step=self.step, log=self.log)
        if self.dtype == 'float':
            return trial.suggest_float(self.name, float(self.values[0]), float(self.values[1]), step=self.step, log=self.log)
        if self.dtype == 'categorical':
            return trial.suggest_categorical(self.name, self.values)

    def __str__(self):
        desc = 'THAM SỐ {} \t loại={}'.format(self.name, self.dtype)
        if self.dtype == 'int' or self.dtype == 'float':
            desc += ' \t phạm vi=[{}..{}] (bước={}) \t {} scale'.format(
                self.values[0], self.values[1], self.step if self.step is not None else 'N/A', 
                'UNIFORM' if not self.log else 'LOG'
            )
        if self.dtype == 'categorical':
            desc += ' \t tùy chọn: [{}]'.format(','.join([str(x) for x in self.values]))
        return desc

# Hàm mục tiêu cho tối ưu hóa Optuna
def objective(trial, par_space):
    optimized_param_str = []
    for par in par_space:
        val = par(trial)
        optimized_param_str.append('{}={}'.format(par.name, val))
    optimized_param_str = ','.join(optimized_param_str)
    val = run_once(optimized_param_str)
    return val

# Tải không gian tham số từ tệp tham số Optuna
par_space = []
with open(args.optuna_parameter_file, 'rt') as f:
    print('-' * 80)
    print('KHÔNG GIAN THAM SỐ')
    for line in f:
        par = Parameter.fromjson(line)
        print('\t' + str(par))
        par_space.append(par)
    print('-' * 80)

# Tạo một nghiên cứu Optuna và tối ưu hóa các tham số
study = optuna.create_study(direction='maximize')
study.optimize(lambda trial: objective(trial, par_space), n_trials=args.ntrials)

# Chạy đánh giá cuối cùng với các tham số tốt nhất
print('Chạy đánh giá cuối cùng @{}:'.format(args.final_measure))
optimized_param_str = ','.join(['{}={}'.format(k, v) for k, v in study.best_params.items()])
command = 'python run.py "{}" -t "{}" -g {} -ps {},{} -m {} -e {} -d {} -ik {} -sk {} -tk {}'.format(
    args.path, args.test, args.gru4rec_model, args.fixed_parameters, optimized_param_str, 
    ' '.join([str(x) for x in args.final_measure]), args.eval_type, args.device, args.item_key, args.session_key, args.time_key
)
cmd = pexpect.spawnu(command, timeout=None, maxread=1)
line = cmd.readline()
while line:
    line = line.strip()
    print(line)
    line = cmd.readline()
