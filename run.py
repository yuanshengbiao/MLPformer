import argparse
import os
import sys
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np
import time
import psutil
import os
import datetime


def main():
    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')
    # basic config
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--task_id', type=str, default='test', help='task id')
    parser.add_argument('--model', type=str, default='MLPformer',help='model name, options: [MLPformer, Autoformer, Informer, Transformer]')

    # supplementary config for MLPformer model
    parser.add_argument('--version', type=str, default='Fourier',help='for MLPformer, there are two versions to choose, options: [Fourier, Wavelets]')
    parser.add_argument('--mode_select', type=str, default='random',
                        help='for MLPformer, there are two mode selection method, options: [random, low]')
    parser.add_argument('--modes', type=int, default=64, help='modes to be selected random 64')
    parser.add_argument('--L', type=int, default=3, help='ignore level')
    parser.add_argument('--base', type=str, default='legendre', help='mwt base')
    parser.add_argument('--cross_activation', type=str, default='tanh',
                        help='mwt cross atention activation function tanh or softmax')
    parser.add_argument('--revin', type=int, default=1, help='whether to apply RevIN')
    parser.add_argument('--num_nodes', type=int, default=7)
    parser.add_argument('--num_experts_list', type=list, default=[4, 4, 4])
    parser.add_argument('--patch_size_list', nargs='+', type=int, default=[16, 12, 8, 32, 12, 8, 6, 4, 8, 6, 4, 2])
    parser.add_argument('--residual_connection', type=int, default=0)
    parser.add_argument('--k', type=int, default=2, help='choose the Top K patch size at the every layer ')

    # FilterNet
    parser.add_argument('--embed_size', default=128, type=int)
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--seq_len_decoder', default=480+48, type=int)
    #parser.add_argument('--dropout', type=float, default=0, help='dropout')

    # data loaders
    parser.add_argument('--data', type=str, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/ETT-small/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='pcg_data.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, '
                             'S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, '
                             'b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--detail_freq', type=str, default='h', help='like freq, but use in predict')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=480, help='prediction sequence length')
    # parser.add_argument('--cross_activation', type=str, default='tanh'

    # model defines
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', default=[24], help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.01, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
    parser.add_argument('--l1_lambda', type=float, default=0.0001, help='')
    parser.add_argument('--l2_lambda', type=float, default=0.0001, help='')

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=1, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multi gpus')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--fnet_d_ff', type=int, default=2048, help='dimension of fcn of fnet')
    parser.add_argument('--fnet_d_model', type=int, default=512, help='dimension of model of fnet')
    parser.add_argument('--complex_dropout', type=float, default=0.025, help='complex_dropout')
    parser.add_argument('--fnet_layers', type=int, default=2, help='num of fnet layers')

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_modes{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_id,
                args.model,
                args.mode_select,
                args.modes,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des,
                ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            #if args.do_predict:
            #print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            #exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.task_id,
                                                                                                      args.model,
                                                                                                      args.data,
                                                                                                      args.features,
                                                                                                      args.seq_len,
                                                                                                      args.label_len,
                                                                                                      args.pred_len,
                                                                                                      args.d_model,
                                                                                                      args.n_heads,
                                                                                                      args.e_layers,
                                                                                                      args.d_layers,
                                                                                                      args.d_ff,
                                                                                                      args.factor,
                                                                                                      args.embed,
                                                                                                      args.distil,
                                                                                                      args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()

def get_memory_info():
    """获取当前程序的内存使用信息"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    # 返回内存占用，单位为MB
    return memory_info.rss / 1024 / 1024 / 1024

def log_time_memory(func):
    """装饰器：记录函数运行时间和内存占用"""
    def wrapper(*args, **kwargs):
        # 记录开始时间和内存
        start_time = time.time()
        start_memory = get_memory_info()

        # 执行函数
        result = func(*args, **kwargs)

        # 记录结束时间和内存
        end_time = time.time()
        end_memory = get_memory_info()

        # 计算时间和内存差值
        time_used = end_time - start_time
        memory_used = end_memory - start_memory

        # 记录到日志
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = (
            f"\n[{current_time}] 函数 {func.__name__}\n"
            f"运行时间: {time_used:.2f} 秒\n"
            f"内存变化: {memory_used:.2f} GB\n"
            f"当前内存: {end_memory:.2f} GB\n"
            f"{'='*50}"
        )

        # 写入日志文件
        with open('performance_log.txt', 'a', encoding='utf-8') as f:
            f.write(log_message)

        print(log_message)
        return result

    return wrapper


if __name__ == "__main__":

# 记录程序开始时的总体信息
    start_total_time = time.time()
    start_total_memory = get_memory_info()


    #main()
    try:
        # 运行你的主要程序逻辑
        result = main()
        # 可以添加多个函数调用

    finally:
        # 记录程序结束时的总体信息
        end_total_time = time.time()
        end_total_memory = get_memory_info()

        # 计算总体运行情况
        total_time = end_total_time - start_total_time
        total_memory = end_total_memory - start_total_memory

        # 记录总体信息
        summary = (
            f"\n{'='*50}\n"
            f"程序总体运行情况：\n"
            f"总运行时间: {total_time:.2f} 秒\n"
            f"总内存变化: {total_memory:.2f} GB\n"
            f"最终内存占用: {end_total_memory:.2f} GB\n"
            f"{'='*50}\n"
        )

        # 写入日志文件
        with open('performance_log.txt', 'a', encoding='utf-8') as f:
            f.write(summary)

        print(summary)