import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader

from net import MyAlexNet
import numpy as np
from torch.optim import lr_scheduler
import os

from torchvision import transforms
from torchvision.datasets import ImageFolder

# 解决中文乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

BOOT_TRAIN = r'E:\Coding\GitHub\AlexNet\data\train'
BOOT_VAL = r'E:\Coding\GitHub\AlexNet\data\test'

# 1、定义一个数据预处理的转换
# 特征缩放：使用归一化进行特征缩放，将图像的像素值归一化到[-1, 1]之间
normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

# 定义train_transforms这个变量，相当于定义了一个管道，对每张图片进行下面这些转换
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图片尺寸
    transforms.RandomVerticalFlip(),  # 以一定的概率随机垂直翻转图像，一种数据增强技术，增加训练数据的多样性
    transforms.ToTensor(),  # 将图像转换为张量，并将像素值缩放到[0, 1]的范围
    normalize  # 归一化
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

# 2、将训练集和测试集进行转换
# ImageFolder是PyTorch提供的一个数据集类，用于加载图像数据。它假设图像按照文件夹结构组织，文件夹名称对应于类标签。
# 因此路径下面包含子文件夹，子文件夹下存放对应类（子文件夹名）的图像
train_dataset = ImageFolder(root=BOOT_TRAIN, transform=train_transforms)
val_dataset = ImageFolder(root=BOOT_VAL, transform=val_transforms)

# DataLoader是PyTorch 提供的一个类，用于对数据集进行批处理和随机打乱（shuffle）等操作，以便在训练模型时高效加载数据。
# batch_size表示每个批次包含32张图片，即每次训练迭代中，模型将同时处理32张图片
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

# 3、定义训练参数和函数
# 定义设备，先使用GPU，失效的话就用cpu
# 最好是在一个设备中进行，因为从GPU上将数据移到CPU上很麻烦
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# .to(device) 方法将模型移动到device设备上,如果 device 是 'cuda'，模型将被加载到 GPU 上，'cpu'则加载CPU上
print("using {} device.".format(device))
model = MyAlexNet().to(device)

# 定义一个损失函数--交叉熵损失函数
loss_fn = nn.CrossEntropyLoss()

# 定义一个优化器--随机梯度下降（SGD）优化器
# model.parameters()获取模型中的所有可学习参数
# momentum动量项，帮助优化器在梯度变化较大的情况下平滑更新过程
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 创建了一个学习率调度器StepLR，用于根据预设的策略调整学习率
# optimizer 是之前定义的优化器，调度器将根据它调整学习率。
# step_size=10 表示每经过 10 个训练周期（epoch），学习率将被更新。
# gamma=0.5 指定学习率调整的因子。当达到 step_size 时，学习率将乘以 gamma，即变为原来的一半（0.5）。
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# 定义训练函数
# dataloader（用于加载数据的迭代器）
def train(dataloader, model, loss_fn, optimizer):
    # 初始化
    loss, current, n = 0.0, 0.0, 0
    for batch, (x, y) in enumerate(dataloader):  # 遍历 dataloader 中的每个批次（batch），enumerate 函数会返回批次索引 batch 和批次数据 (x, y) x 是输入图像，y 是对应的标签
        x, y = x.to(device), y.to(device)  # 将输入图像和标签移动到指定的计算设备,一次迭代有32张图片和标签
        y_hat = model(x)  # 用模型预测
        cur_loss = loss_fn(y_hat, y)  # 计算当前批次的损失值，使用指定的损失函数 loss，y的真实值和预测值作为输入
        _, pred = torch.max(y_hat, axis=1)
        cur_acc = torch.sum(pred == y) / y_hat.shape[0]  # pred == y返回一个布尔张量，表示预测是否正确  y_hat.shape[0]是当前批次的样本数量

        # 反向传播 （训练）
        # 前三行代码是用于更新模型参数的关键步骤, 训练循环中的核心部分
        optimizer.zero_grad()  # 在每次参数更新之前，需要将模型参数的梯度清零。因为 PyTorch 默认是累加梯度的
        cur_loss.backward()  # backward()方法会根据损失的计算图自动执行反向传播。它会计算每个参数的梯度，并将这些梯度存储在相应的参数的.grad 属性中。
        optimizer.step()  # 这一行执行优化器的步进操作，使用计算得到的梯度来更新模型的参数。

        loss += cur_loss.item()  # 计算总的损失
        current += cur_acc.item()  # 计算总的准确率
        n = n + 1  # 训练次数

    train_loss = loss / n  # 所有批次的损失之和取平均值，注意，一次循环训练一个批次，计算一个损失值，n表示总的批次数
    train_acc = current / n  # 所有批次的准确率之和取平均值
    print('train_loss：' + str(train_loss))
    print('train_acc：' + str(train_acc))
    return train_loss, train_acc

# 定义验证函数--这个函数就是训练函数删去  梯度清零、梯度计算、梯度更新三步
def evaluate(dataloader, model, loss_fn):
    # 将模型转化为验证模型
    model.eval()
    # 初始化
    loss, current, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            cur_loss = loss_fn(y_hat, y)
            _, pred = torch.max(y_hat, axis=1)
            cur_acc = torch.sum(pred == y) / y_hat.shape[0]
            n = n + 1

            loss += cur_loss.item()
            current += cur_acc.item()

    val_loss = loss / n
    val_acc = current / n
    print('val_loss：' + str(val_loss))
    print('val_acc：' + str(val_acc))
    return val_loss, val_acc

# 定义画图函数
def matplot_loss(train_loss, val_loss):
    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.legend(loc = 'best')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title("训练集和验证集loss值对比图")
    plt.savefig('loss_plot.png')  # 保存图像为PNG文件
    plt.show()
    plt.close()

def matplot_acc(train_acc, val_acc):
    plt.plot(train_acc, label='train_acc')
    plt.plot(val_acc, label='val_acc')
    plt.legend(loc = 'best')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.title("训练集和验证集acc值对比图")
    plt.savefig('accuracy_plot.png')  # 保存图像为PNG文件
    plt.show()
    plt.close()

# 4、开始训练
loss_train = []
acc_train = []
loss_val = []
acc_val = []

epoch = 20  # 整个训练过程将进行 20 次完整的训练

best_val_acc = 0.0  # 初始化一个变量 best_val_acc，用于跟踪最小的验证准确率。这可以在后续逻辑中用于模型保存或其他用途（例如早停）
best_loss_diff = float('inf')  # 追踪最佳误差差异

for t in range(epoch):
    lr_scheduler.step()  # 看到64行，每经过10个训练周期（epoch），学习率将被乘以gamma=0.5
    print(f"epoch {t+1}\n----------")

    # 训练和评估
    train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer)
    val_loss, val_acc = evaluate(val_dataloader, model, loss_fn)

    # 将每一次完整训练计算的损失和准确率存放在一个数组中，数组大小为epoch
    loss_train.append(train_loss)
    acc_train.append(train_acc)
    loss_val.append(val_loss)
    acc_val.append(val_acc)

    # 计算训练和验证损失的差异
    loss_diff = abs(train_loss - val_loss)

    # 保存最好的模型权重（参数）
    if val_acc > best_val_acc and loss_diff < 0.08:  # 这里的0.1是阈值，可以根据需要调整
        if not os.path.exists('save_model'):  # 如果没有该文件夹，则创建一个
            os.makedirs('save_model')

        best_val_acc = val_acc  # 更新最佳验证准确率
        best_loss_diff = loss_diff  # 更新最佳误差差异
        print(f"save best model, 第{t+1}轮, with val acc: {best_val_acc:.4f} and loss diff: {loss_diff:.4f}")

        # model.state_dict() 返回一个包含所有模型参数的字典，将其保存到指定路径中
        torch.save(model.state_dict(), 'save_model/best_model.pth')

    # 保存最后一轮的权重文件
    if t == epoch - 1:
        torch.save(model.state_dict(), 'save_model/last_model.pth')

# 绘制图形
matplot_loss(loss_train, loss_val)
matplot_acc(acc_train, acc_val)
print("Done!!!")
