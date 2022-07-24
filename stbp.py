import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import numpy as np
from spikingjelly.clock_driven import neuron, encoding, functional,surrogate
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

parser = argparse.ArgumentParser(description='spikingjelly LIF MNIST Training_STBP_accomplish')

parser.add_argument('--device', default='cpu', help='运行的设备，例如“cpu”或“cuda:0”\n Device, e.g., "cpu" or "cuda:0"')

parser.add_argument('--dataset-dir', default='./', help='保存MNIST数据集的位置，例如“./”\n Root directory for saving MNIST dataset, e.g., "./"')
parser.add_argument('--log-dir', default='./log/stbp/', help='保存tensorboard日志文件的位置，例如“./”\n Root directory for saving tensorboard logs, e.g., "./"')
parser.add_argument('--model-output-dir', default='./log/stbp', help='模型保存路径，例如“./”\n Model directory for saving, e.g., "./"')

parser.add_argument('-b', '--batch-size', default=64, type=int, help='Batch 大小，例如“64”\n Batch size, e.g., "64"')
parser.add_argument('-T', '--timesteps', default=30, type=int, dest='T', help='仿真时长，例如“100”\n Simulating timesteps, e.g., "100"')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, metavar='LR', help='学习率，例如“1e-3”\n Learning rate, e.g., "1e-3": ', dest='lr')#原文给的是0.5，我觉得有点大吧
parser.add_argument('--tau', default=2.0, type=float, help='LIF神经元的时间常数tau，例如“100.0”\n Membrane time constant, tau, for LIF neurons, e.g., "100.0"')
parser.add_argument('-N', '--epoch', default=100, type=int, help='训练epoch，例如“100”\n Training epoch, e.g., "100"')#好多啊，我多半跑不完
parser.add_argument('-channels', default=128, type=int, help='channels of CSNN')
class STCNN(nn.Module):   #Spatio-Temporal Convolution Neural Network
    def __init__(self,T: int, channels: int):
        super().__init__()
        self.T = T
        # 原文给的是  In the MNIST, our
        # network contains two convolution layers with kernel size of
        # 5 × 5 and two average pooling layers alternatively, followed
        # by one full connected layer
        self.stconv = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=5, padding=1, bias=False),#3，3不是更常见嘛
            nn.BatchNorm2d(channels),
        )
        self.conv = nn.Sequential(
            neuron.IFNode(surrogate_function=surrogate.ATan()), #这里的tau设置为2.0，与neuron中的初始一样
            nn.AvgPool2d(2, 2),  # 13 * 13

            nn.Conv2d(channels, channels, kernel_size=5, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            nn.AvgPool2d(2, 2),  # 5 * 5  [N,128,5,5]
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 5 * 5, 10, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
        )
    def forward(self,x):
        # x.shape = [N, C, H, W]
        x = self.stconv(x)
        # x = self.conv(x)
        out_count = self.fc(self.conv(x))
        for t in range(1, self.T):
            out_count += self.fc(self.conv(x))
        out_fr = out_count / self.T
        return out_fr
        # return x
class STFC(nn.Module):
    def __init__(self,T: int):
        super().__init__()
        self.T = T
        self.encoder = encoding.PoissonEncoder()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 10, bias=False),
            neuron.LIFNode(surrogate_function=surrogate.ATan())
        )
    def forward(self,x):
        out_count = self.fc(self.encoder(x).float())
        for t in range(1,self.T):
            out_count += self.fc(self.encoder(x).float())
        out_fr = out_count / self.T
        return out_fr

def main():
    args = parser.parse_args()
    print("happy de start learning!\n")
    print(args)
    print("____________________________________________________________")
    device = args.device
    dataset_dir = args.dataset_dir
    log_dir = args.log_dir
    model_output_dir = args.model_output_dir
    batch_size = args.batch_size
    lr = args.lr
    T = args.T
    tau = args.tau
    train_epoch = args.epoch
    writer = SummaryWriter(log_dir)

    # 初始化数据加载器
    train_dataset = torchvision.datasets.MNIST(
        root=dataset_dir,
        train=True,
        transform=torchvision.transforms.ToTensor(),  # 转化为tensor
        download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root=dataset_dir,
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )

    # 将自定义的Dataset根据batch size大小、是否shuffle等选项封装成一个batch size大小的Tensor，后续只需要再包装成Variable即可作为模型输入用于训练。
    train_data_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    # shuffle=True,https://blog.csdn.net/qq_35248792/article/details/109510917 打乱
    test_data_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    #这下数据就加载完成了

    net = STCNN(T=args.T,channels=args.channels) #贫穷，没有cuda可以用来运行cupy，所以就只有cpu着一个选择了
    print(net)
    net.to(device)

    #文章里面用的是adam，这里也就是adam了  而且好像根据啥论文来着，我想不起来了，好像说整体水平adam好像比sgd要好一些
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    #文章既然没有提学习率衰减我就不加了。StepLR和CosineAnnealingLR我就不管了
    #方便起见我直接泊松编码了
    encoder = encoding.PoissonEncoder()
    #这里写的是3.2.2的卷积，所以泊松都不需要
    train_times = 0
    max_test_accuracy = 0
    test_accs = []
    train_accs = []
    for epoch in range(train_epoch):
        print("Epoch {}:".format(epoch))
        print("Training...")
        train_correct_sum = 0
        train_sum = 0
        train_loss = 0
        net.train()
        for img, label in tqdm(train_data_loader):
            img = img.float().to(device)
            print(img.shape)
            label = label.to(device)
            label_one_hot = F.one_hot(label, 10).float()#毕竟是mnist一共10分类
            optimizer.zero_grad()
            # out_fr = net(encoder(img).float())
            out_fr = net(img)
            loss = F.mse_loss(out_fr, label_one_hot) #loss 是mse
            loss.backward()
            optimizer.step()
            # 优化一次参数后，需要重置网络的状态，因为SNN的神经元是有“记忆”的
            functional.reset_net(net)
            #计算正确率
            train_correct_sum += (out_fr.argmax(1) == label.to(device)).float().sum().item()
            train_sum += label.numel()
            train_batch_accuracy = (out_fr.argmax(1) == label.to(device)).float().mean().item()
            train_loss += loss.item() * label.numel()
            #
            # writer.add_scalar('train_batch_accuracy', train_batch_accuracy, train_times)
            # train_accs.append(train_batch_accuracy)
            train_times += 1
        train_accuracy = train_correct_sum / train_sum
        train_loss /= train_sum
        writer.add_scalar('train_accuracy', train_accuracy, epoch)
        train_accs.append(train_accuracy)
        print("Testing...")
        net.eval()
        with torch.no_grad():
            test_correct_sum = 0
            test_sum = 0
            for img, label in tqdm(test_data_loader):
                img = img.float().to(device)
                label = label.to(device)
                out_fr = net(img)
                test_sum += label.numel()
                test_correct_sum += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)
            test_accuracy = test_correct_sum / test_sum
            writer.add_scalar('test_accuracy', test_accuracy, epoch)
            test_accs.append(test_accuracy)
            max_test_accuracy = max(max_test_accuracy, test_accuracy)
        print("Epoch {}: train_acc = {}, test_acc={}, max_test_acc={}, train_times={}".format(epoch, train_accuracy,
                                                                                             test_accuracy,
                                                                                              max_test_accuracy,
                                                                                              train_times))
    torch.save(net, model_output_dir + "/lif_snn_mnist.ckpt")


    # 保存绘图用数据
    net.eval()
    with torch.no_grad():
        img, label = test_dataset[0]
        img = img.to(device)
        for t in range(T):
            if t == 0:
                out_spikes_counter = net(encoder(img).float())
            else:
                out_spikes_counter += net(encoder(img).float())
        out_spikes_counter_frequency = (out_spikes_counter / T).cpu().numpy()
        print(f'Firing rate: {out_spikes_counter_frequency}')
        output_layer = net[-1]  # 输出层
        v_t_array = output_layer.v.cpu().numpy().squeeze().T  # v_t_array[i][j]表示神经元i在j时刻的电压值
        np.save("v_t_array.npy", v_t_array)
        s_t_array = output_layer.spike.cpu().numpy().squeeze().T  # s_t_array[i][j]表示神经元i在j时刻释放的脉冲，为0或1
        np.save("s_t_array.npy", s_t_array)

    train_accs = np.array(train_accs)
    np.save('train_accs.npy', train_accs)
    test_accs = np.array(test_accs)
    np.save('test_accs.npy', test_accs)

if __name__ == '__main__':
    main()
