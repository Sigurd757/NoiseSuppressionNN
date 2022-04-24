import os
import sys
import torch
import time
from torch.utils.data import DataLoader
from torch import nn
import argparse
from tensorboardX import SummaryWriter

from data_preparation.data_preparation import FileDateset
from data_preparation.eval_data_preparation import EvalFileDateset
# from model.Baseline import Base_model
# from model.ops import pytorch_LSD

#from model.dccrn.dc_crn import DCCRN
from model.dccrn.DFSMNNet import DFSMNNet
from model.dccrn.tln import TLN_10ms
from model.dccrn.tln_spec import TLN_Fb
model_label = "TLN_Fb"

import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--train_data', default="/media/qcm/HardDisk1/jsonfile/denoise/train220316.json")
    parser.add_argument('--val_data', default="/media/qcm/HardDisk1/jsonfile/denoise/eval220316.json")
    parser.add_argument('--checkpoints_dir', default="./checkpoints/AEC_baseline")
    parser.add_argument('--event_dir', default="./event_file/AEC_baseline")
    parser.add_argument('--min_lr', type=float, default=1e-6)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print("GPU是否可用：", torch.cuda.is_available())  # True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型名称
    label_name = time.strftime("%Y%m%d%H%M", time.localtime())
    print(label_name)
    model_title = "Denoise_" + model_label + "_epoch" + str(args.epochs) + "_batch" + str(args.batch_size) + "_" + label_name
    
    # 实例化 Dataset
    train_set = FileDateset(dataset_path=args.train_data)  # 实例化训练数据集
    val_set = EvalFileDateset(dataset_path=args.val_data)  # 实例化验证数据集

    # 数据加载器
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=True)

    # ###########    保存检查点的地址(如果检查点不存在，则创建)   ############
    if not os.path.exists(os.path.join(args.checkpoints_dir, model_title)):
        os.makedirs(os.path.join(args.checkpoints_dir, model_title))

    ################################
    #          实例化模型          #
    ################################
    # model = Base_model().to(device)  # 实例化模型
    # model = DCCRN().to(device)  # 实例化模型
    # model = DFSMNNet().to(device)  # 实例化模型
    # model = TLN_10ms().to(device)  # 实例化模型
    model = TLN_Fb().to(device)  # 实例化模型

    # ###########    损失函数   ############
    criterion = nn.MSELoss(reduce=True, size_average=True, reduction='mean')

    ###############################
    # 创建优化器 Create optimizers #
    ###############################
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, )
    lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20], gamma=0.1)
    # lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.min_lr, last_epoch=-1)


    # ###########    TensorBoard可视化 summary  ############
    writer = SummaryWriter(args.event_dir)  # 创建事件文件

    # ###########    加载模型检查点   ############
    start_epoch = 0
    if args.model_name:
        print("加载模型：", args.checkpoints_dir + args.model_name)
        checkpoint = torch.load(args.checkpoints_dir + args.model_name)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint['epoch']
        lr_schedule.load_state_dict(checkpoint['lr_schedule'])  # 加载lr_scheduler

    for epoch in range(start_epoch, args.epochs):
        model.train()  # 训练模型
        total_loss = 0
        total_lsd = 0
        for batch_idx, (train_X, train_mask, train_clean_speech_magnitude) in enumerate(
                train_loader):
            train_X = train_X.to(device)  # 输入带噪语音 [batch_size, 322, 999] (, F, T)
            train_mask = train_mask.to(device)  # IRM [batch_size 161, 999]
            train_clean_speech_magnitude = train_clean_speech_magnitude.to(device)

            # 前向传播
            # pred_mask = model(train_X)[1]  # [batch_size, 322, 999]--> [batch_size, 161, 999]
            pred_mask = model(train_X)
            #pred_mask = pred_mask[:,:,160:]

            # train_loss = criterion(pred_mask, train_mask)
            train_loss = model.loss(pred_mask, train_clean_speech_magnitude, loss_mode='SI-SNR')

            # 预测输出信号频谱 = mask * 输入信号频谱 [batch_size, 161, 999]
            pred_out_spectrum = pred_mask * train_X
            #train_lsd = pytorch_LSD(train_clean_speech_magnitude, pred_out_spectrum)

            # 反向传播
            optimizer.zero_grad()  # 将梯度清零
            train_loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            
            total_loss += train_loss.item() * args.batch_size
            #total_lsd += train_lsd.item() * args.batch_size
            step_num = len(train_loader)
            if batch_idx % 10 == 0 and  batch_idx != 0:
                print(f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}'
                            f'batch_idx: {batch_idx:6}'
                            f'lr: {optimizer.param_groups[0]["lr"]:10}'
                            f'loss: {train_loss.item():.5}'
                            #f'lsd: {train_lsd.item():.5}'
                            f'left: {(step_num - batch_idx)/step_num*100:.4}%'
                            )

            # ###########    可视化打印   ############
        print('Train Epoch: {} Loss: {:.6f} LSD: {:.6f}'.format(epoch + 1, total_loss/len(train_set), total_lsd/len(train_set)))

        # ###########    TensorBoard可视化 summary  ############
        lr_schedule.step()  # 学习率衰减
        # writer.add_scalar(tag="lr", scalar_value=model.state_dict()['param_groups'][0]['lr'], global_step=epoch + 1)
        writer.add_scalar(tag="train_loss", scalar_value=train_loss.item(), global_step=epoch + 1)
        # writer.add_scalar(tag="train_lsd", scalar_value=train_lsd.item(), global_step=epoch + 1)
        writer.flush()
        '''
        # 神经网络在验证数据集上的表现
        model.eval()  # 测试模型
        print('start evaluation...')
        total_eval_loss = 0
        total_eval_lsd = 0
        # 测试的时候不需要梯度
        with torch.no_grad():
            for val_batch_idx, (val_X, val_mask, val_clean_speech_magnitude) in enumerate(
                    val_loader):
                val_X = val_X.to(device)  # 远端语音cat麦克风语音 [batch_size, 322, 999] (, F, T)
                val_mask = val_mask.to(device)  # IRM [batch_size 161, 999]
                val_clean_speech_magnitude = val_clean_speech_magnitude.to(device)

                # 前向传播
                val_pred_mask = model(val_X)
                val_loss = criterion(val_pred_mask, val_mask)

                # 近端语音信号频谱 = mask * 麦克风信号频谱 [batch_size, 161, 999]
                val_pred_out_spectrum = val_pred_mask * val_X
                val_lsd = pytorch_LSD(val_clean_speech_magnitude, val_pred_out_spectrum)
                
                total_eval_loss += val_loss.item() * args.batch_size
                total_eval_lsd += val_lsd.item() * args.batch_size
                eval_step_num = len(val_loader)
                if val_batch_idx % 10 == 0 and  val_batch_idx != 0:
                    print(f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}'
                                f'batch_idx: {val_batch_idx:6}'
                                f'lr: {optimizer.param_groups[0]["lr"]:10}'
                                f'loss: {val_loss.item():.5}'
                                f'lsd: {val_lsd.item():.5}'
                                f'left: {(eval_step_num - val_batch_idx)/eval_step_num*100:.4}%'
                                )
            
            # ###########    可视化打印   ############
            print('  val Epoch: {} \tLoss: {:.6f}\tlsd: {:.6f}'.format(epoch + 1,total_eval_loss/len(val_set), total_eval_lsd/len(val_set)))
            ######################
            # 更新tensorboard    #
            ######################
            writer.add_scalar(tag="val_loss", scalar_value=val_loss.item(), global_step=epoch + 1)
            writer.add_scalar(tag="val_lsd", scalar_value=val_lsd.item(), global_step=epoch + 1)
            writer.flush()
        '''
        # # ###########    保存模型   ############
        if (epoch + 1) % 2 == 0:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                'lr_schedule': lr_schedule.state_dict()
            }
            model_title_index = model_title + str(epoch + 1)
            torch.save(checkpoint, '%s/%s/epochno%d.pth' % (args.checkpoints_dir, model_title, epoch + 1))


if __name__ == "__main__":
    main()
