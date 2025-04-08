import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def MAE1(pred, true, folder_path):
    import numpy as np
    import matplotlib.pyplot as plt
    # 初始化一个列表来存储差值序列
    diff_sequences = []
    # 计算第二个维度上每隔5个元素的起始索引
    indices = np.arange(0, pred.shape[1], 5)
    # 对每个起始索引进行循环，并计算差值
    for idx in indices:
        # 确保我们不会超出数组的边界
        end_idx = min(idx + 5, pred.shape[1])
        # 计算差值，并取第一个样本（例如，索引为0的样本）作为示例进行绘图
        diff = np.mean(np.abs(pred[0, idx:end_idx, :] - true[0, idx:end_idx, :]).flatten())
        # 将差值序列添加到列表中
        diff_sequences.append(diff)
    # 绘制折线图
    plt.figure()  # 设置图形大小
    plt.plot(diff_sequences)  # 绘制折线图，并添加标签
    plt.title('mae')  # 设置图形标题
    plt.xlabel('Index (every 5 elements)')  # 设置x轴标签
    plt.ylabel('mae')  # 设置y轴标签
    plt.grid(True)  # 显示网格线
    # 保存生成的折线图
    plt.savefig(folder_path + '/mae.png')  # 将图形保存为PNG文件
    # 显示图形（如果需要的话）
    plt.show()

def MSE1(pred, true, folder_path):
    import numpy as np
    import matplotlib.pyplot as plt
    # 初始化一个列表来存储差值序列
    diff_sequences = []
    # 计算第二个维度上每隔5个元素的起始索引
    indices = np.arange(0, pred.shape[1], 5)
    # 对每个起始索引进行循环，并计算差值
    for idx in indices:
        # 确保我们不会超出数组的边界
        end_idx = min(idx + 5, pred.shape[1])
        # 计算差值，并取第一个样本（例如，索引为0的样本）作为示例进行绘图
        diff = np.mean((pred[0, idx:end_idx, :] - true[0, idx:end_idx, :]) ** 2).flatten()
        # 将差值序列添加到列表中
        diff_sequences.append(diff)
    # 绘制折线图
    plt.figure()  # 设置图形大小
    plt.plot(diff_sequences)  # 绘制折线图，并添加标签
    plt.title('mse')  # 设置图形标题
    plt.xlabel('Index (every 5 elements)')  # 设置x轴标签
    plt.ylabel('mse')  # 设置y轴标签
    plt.grid(True)  # 显示网格线
    # 保存生成的折线图
    plt.savefig(folder_path + '/mse.png')  # 将图形保存为PNG文件
    # 显示图形（如果需要的话）
    plt.show()

def metric(pred, true, folder_path):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    MAE1(pred, true, folder_path)
    MSE1(pred, true, folder_path)
    return mae, mse, rmse, mape, mspe
