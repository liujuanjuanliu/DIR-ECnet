
import argparse
import os,sys,shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import math
from PIL import Image

import scipy.io as sio
import numpy as np
import pdb

from statistics import mean 
# from model.attentionnet import SharedAttentionBranch as AttentionBranch
from model.attentionnet import count_parameters
from model.attentionnet import RegionInBlock
from model.attentionnet import FusionBlock
# from model.attentionnet import ScaleNet
# from model.attentionnet import scalenet50
# from model.attentionnet import MANet
from model.attentionnet import AttentionBlock
import prettytable
import tkinter
import matplotlib

matplotlib.use('TkAgg')
# from model.attentionnet import SABlock
from model.resnet import resnet50
from rafdb_dataset import ImageList
from sampler import ImbalancedDatasetSampler
from utils import util
from model.losses import FocalLoss
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from model.attentionnet import ViT
from model.attentionnet import RegionInBlock
from model.attentionnet import F_to_F
from model.attentionnet import Transblock

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#######################################################################################################################################
# Training settings

parser = argparse.ArgumentParser(description='PyTorch RAFDB Training using novel local, global attention branch + region branch with non-overlapping patches')

parser.add_argument('--root_path', type=str, default='./rafdb_data/aligned/',   # ../data/RAFDB/Image/aligned/
                    help='path to root path of images')
parser.add_argument('--database', type=str, default='RAFDB',
                    help='Which Database for train. (RAFDB, Flatcam, FERPLUS)')
parser.add_argument('--train_list', type=str, default='./rafdb_data/EmoLabel/train_label.txt',
                    help='path to training list')
parser.add_argument('--test_list', type=str, default='./rafdb_data/EmoLabel/test_label.txt',
                    help='path to test list')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',   help='number of data loading workers (default: 4)')   # 4

parser.add_argument('--epochs', default=60, type=int, metavar='N', help='number of total epochs to run')  # 60

parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')

parser.add_argument('-b', '--batch-size', default=8, type=int, metavar='N', help='mini-batch size (default: 256)')   # 64,本机可用16

parser.add_argument('-b_t', '--batch-size_t', default=8, type=int, metavar='N', help='mini-batch size (default: 256)')  # 64

parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, metavar='LR', help='initial learning rate')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',  help='momentum')

parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,  metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--print-freq', '-p', default=200, type=int,metavar='N', help='print frequency (default: 10)')

parser.add_argument('--resume', default='checkpoints/model_bestk.pth.tar', type=str, metavar='PATH',   help='path to latest checkpoint (default: none)')

parser.add_argument('--pretrained', default='pretrainedmodels/vgg_msceleb_resnet50_ft_weight.pkl', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--model_dir', '-m', default='checkpoints', type=str)

parser.add_argument('--imagesize', type=int, default=224, help='image size (default: 224)')

parser.add_argument('--num_classes', type=int, default=7, help='number of expressions(class)')

parser.add_argument('--num_attentive_regions', type=int, default=25, help='number of non-overlapping patches(default:25)')

parser.add_argument('--num_regions', type=int, default=4, help='number of non-overlapping patches(default:4)')

parser.add_argument('--train_rule', default='Resample', type=str, help='data sampling strategy for train loader:Resample, DRW,Reweight, None')

parser.add_argument('--loss_type', default="CE", type=str, help='loss type:Focal, CE')

best_prec1 = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





def main():
    global args, best_prec1
    args = parser.parse_args()
    print('\n\t\t Aum Sri Sai Ram\n\t\tRAFDB FER using Local and global Attention along with region branch (non-overlapping patches)\n\n')
    print(args)
    print('\nimg_dir: ', args.root_path)
    print('\ntrain rule: ', args.train_rule, ' and loss type: ', args.loss_type, '\n')

    print('\n lr is : ', args.lr)

    print('img_dir:', args.root_path)

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.25, hue=0.05),
            transforms.Resize((args.imagesize, args.imagesize)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])


    valid_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.25, hue=0.05),
            transforms.Resize((args.imagesize, args.imagesize)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    train_dataset = ImageList(root=args.root_path, fileList=args.train_list,
                  transform=train_transform)


    test_data = ImageList(root=args.root_path, fileList=args.test_list,
                  transform=valid_transform)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size_t,
                         shuffle=True, num_workers=args.workers, pin_memory=True)

    cls_num_list = train_dataset.get_cls_num_list()
    cls_num_list1 = test_data.get_cls_num_list()
    print('Train split class wise is :', cls_num_list)
    print('test split class is:', cls_num_list1)

    if args.train_rule == 'None':
       train_sampler = None
       per_cls_weights = None
    elif args.train_rule == 'Resample':
       train_sampler = ImbalancedDatasetSampler(train_dataset)
       per_cls_weights = None
    elif args.train_rule == 'Reweight':
       train_sampler = None
       beta = 0.9999
       effective_num = 1.0 - np.power(beta, cls_num_list)
       per_cls_weights = (1.0 - beta) / np.array(effective_num)
       per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
       per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)

    if args.loss_type == 'CE':
       criterion = nn.CrossEntropyLoss(weight=per_cls_weights).to(device)
    elif args.loss_type == 'Focal':
       criterion = FocalLoss(weight=per_cls_weights, gamma=2).to(device)
    else:
       warnings.warn('Loss type is not listed')
       return


    train_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=(train_sampler is None),
                                                   num_workers=args.workers, pin_memory=True, sampler=train_sampler)


    print('length of RAFDB train Database: ' + str(len(train_dataset)))

    print('length of RAFDB test Database: ' + str(len(test_loader.dataset)))


    # prepare model
    basemodel = resnet50(pretrained=False)
    sub_model = RegionInBlock(inplanes_f=1024)
    fusion_model = FusionBlock(global_fer=256, local_fer=256)
    trans_model = Transblock()
    att_model = F_to_F(dim=6272, num_classes=7)
    fusion_model = torch.nn.DataParallel(fusion_model).to(device)
    basemodel = torch.nn.DataParallel(basemodel).to(device)
    sub_model = torch.nn.DataParallel(sub_model).to(device)
    att_model = torch.nn.DataParallel(att_model).to(device)
    trans_model = torch.nn.DataParallel(trans_model).to(device)


    print('\nNumber of parameters:')
    print('Base Model: {}, Attention Branch:{}  Region Branch:{} and Total: {}'.format(count_parameters(basemodel),count_parameters(sub_model),  count_parameters(att_model), count_parameters(basemodel)+count_parameters(sub_model)+count_parameters(att_model)))


    optimizer =  torch.optim.SGD([{"params": basemodel.parameters(), "lr": 0.0001, "momentum":args.momentum,
                                 "weight_decay":args.weight_decay}])

    optimizer.add_param_group({"params": sub_model.parameters(), "lr": args.lr, "momentum":args.momentum,
                                  "weight_decay":args.weight_decay})

    optimizer.add_param_group({"params": att_model.parameters(), "lr": args.lr, "momentum":args.momentum,
                                 "weight_decay":args.weight_decay})
    optimizer.add_param_group({"params": trans_model.parameters(), "lr": args.lr, "momentum": args.momentum,
                               "weight_decay": args.weight_decay})

    optimizer.add_param_group({"params": fusion_model.parameters(), "lr": args.lr, "momentum": args.momentum,
                                "weight_decay": args.weight_decay})

    #optimizer.add_param_group({"params": manet_model.parameters(), "lr": args.lr, "momentum": args.momentum,
                               #"weight_decay": args.weight_decay})
    recorder = RecorderMeter(args.epochs)

    if args.pretrained:

        util.load_state_dict(basemodel, 'pretrainedmodels/vgg_msceleb_resnet50_ft_weight.pkl')


    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            recorder = checkpoint['recorder']
            basemodel.load_state_dict(checkpoint['base_state_dict'])
            sub_model.load_state_dict(checkpoint['sub_state_dict'])
            att_model.load_state_dict(checkpoint['att_state_dict'])
            trans_model.load_state_dict(checkpoint['trans_state_dict'])
            # fusion_model.load_state_dict(checkpoint['fusion_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    print('\nTraining starting:\n')
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train_acc, train_loss = train(train_loader, basemodel, sub_model, att_model, trans_model,  criterion, optimizer, epoch)
        adjust_learning_rate(optimizer, epoch)
        prec1, val_loss = validate(test_loader, basemodel, sub_model, att_model, trans_model,  criterion, epoch)
        recorder.update(epoch, train_loss, train_acc, val_loss, prec1)
        curve_name = '0727-all-1024' + '.png'
        recorder.plot_curve(os.path.join('./log/', curve_name))

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        txt_name = './log/' + '0727-all-1024' + '-log.txt'
        with open(txt_name, 'a') as f:
            f.write('The best accuracy:' + str(best_prec1) + '////' + str(prec1) + "\n")
        # Mycode

        save_checkpoint({
            'epoch': epoch + 1,
            'base_state_dict': basemodel.state_dict(),
            'sub_state_dict': sub_model.state_dict(),
            'att_state_dict': att_model.state_dict(),
            'trans_state_dict': trans_model.state_dict(),
            # 'fusion_state_dict': fusion_model.state_dict(),
            'prec1': prec1,
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
            'recorder': recorder,
        }, prec1, is_best, epoch)

def train(train_loader,  basemodel, sub_model, att_model, trans_model, criterion,  optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()
    att_loss = AverageMeter()
    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        input = input.to(device)
        target = target.to(device)
        attention_branch_feat, _ = basemodel(input)
        #print('_:', attention_branch_feat.shape)
        # attention_branch_feat = basemodel(input)
        sub_preds = sub_model(_)
        g_input = trans_model(_)
        att_preds = att_model(sub_preds, g_input)  # region_branch_feat
        loss = criterion(att_preds, target)
        att_loss.update(loss.item(), input.size(0))
        output = att_preds
        avg_prec, _ = accuracy(output, target, topk=(1, 5))
        top1.update(avg_prec.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Training Epoch: [{0}][{1}/{2}]\t'
                  'Time  ({batch_time.avg})\t'
                  'Data ({data_time.avg})\t'
                  'att_loss ({att_loss.avg})\t' 
                  'Prec1  ({top1.avg}) \t'.format(
                  epoch, i, len(train_loader), batch_time = batch_time, data_time=data_time, att_loss=att_loss, top1=top1))
    torch.cuda.empty_cache()
    return top1.avg, att_loss.avg



def validate(val_loader,  basemodel, sub_model, att_model, trans_model, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()
    att_loss = AverageMeter()
    mode = 'Testing'
    # switch to evaluate mode
    basemodel.eval()
    sub_model.eval()
    att_model.eval()
    trans_model.eval()
    # fusion_model.eval()
    end = time.time()
    total_correct = 0
    total_num = 0
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            data_time.update(time.time() - end)
            input = input.to(device)
            target = target.to(device)
            attention_branch_feat, _ = basemodel(input)
            sub_preds = sub_model(_)
            g_input = trans_model(_)
            att_preds = att_model(sub_preds, g_input)
            loss = criterion(att_preds, target)
            att_loss.update(loss.item(), input.size(0))
            output = att_preds
            avg_prec, _ = accuracy(output, target, topk=(1, 5))
            top1.update(avg_prec.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
        print('{0} [{1}/{2}]\t'            
                  'att_loss  ({att_loss.avg})\t'
                  'Prec@1  ({top1.avg})\t'
                  .format(mode, i, len(val_loader),  att_loss=att_loss,  top1=top1))
    torch.cuda.empty_cache()
    return top1.avg, att_loss.avg



def save_checkpoint(state, prec1, is_best, epoch,  filename='checkpoint.pth.tar'):   #  filename='checkpoint.pth.tar'
    epoch_num = state['epoch']
    full_bestname = os.path.join(args.model_dir, 'model_best0426_1024.pth.tar')
    PATH = './path-0727-all/model'
    if prec1:
        torch.save(state, PATH + '_%d.pth' % (epoch+1))
    if is_best:
        torch.save(state, full_bestname)



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed"""
    for param_group in optimizer.param_groups:
           param_group['lr'] *= 0.95


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# 参考链接: https://blog.csdn.net/qq_40243750/article/details/124255865?app_version=5.6.0&code=app_1562916241&csdn_share_tail=%7B%22type%22%3A%22blog%22%2C%22rType%22%3A%22article%22%2C%22rId%22%3A%22124255865%22%2C%22source%22%3A%22weixin_46031768%22%7D&ctrtid=mnxwm&uLinkId=usr1mkqgl919blen&utm_source=app
class ConfusionMatrix(object):
    def __init__(self, num_classes: int, labels: list, normalize=True):
        """
		normalize：是否设元素为百分比形式
        """
        self.normalize = normalize
        # self.reset(total_epoch)
        self.labels = labels
        self.num_classes = num_classes
        self.matrix = np.zeros((self.num_classes, self.num_classes))

    def update(self, predicts, labels):
        """

        :param predicts: 一维预测向量，eg：array([0,5,1,6,3,...],dtype=int64)
        :param labels:   一维标签向量：eg：array([0,5,0,6,2,...],dtype=int64)
        :return:
        """
        for predict, label in zip(predicts, labels):
            self.matrix[predict, label] += 1

    def summary(self):
        # calculate accuracy,计算正确率
        sum_TP = 0
        for i in range(self.num_classes):
            # 统计混淆矩阵对角线元素的和
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)
        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)
        return str(acc)

    def getMatrix(self, normalize=True):
        """
        根据传入的normalize判断要进行percent的转换，
        如果normalize为True，则矩阵元素转换为百分比形式，
        如果normalize为False，则矩阵元素就为数量
        Returns:返回一个以百分比或者数量为元素的矩阵

        """
        if normalize:
            per_sum = self.matrix.sum(axis=1)  # 计算每行的和，用于百分比计算
            for i in range(self.num_classes):
                self.matrix[i] = (self.matrix[i] / per_sum[i])  # 百分比转换
            self.matrix = np.around(self.matrix, 2)  # 保留2位小数点
            self.matrix[np.isnan(self.matrix)] = 0  # 可能存在NaN，将其设为0
        return self.matrix

    def plot(self):
        self.matrix = self.getMatrix(self.normalize)
        plt.imshow(self.matrix, cmap=plt.cm.Blues)  # 仅画出颜色格子，没有值
        plt.title('Confusion matrix')  # title
        plt.xlabel("Predict label")
        plt.ylabel("Truth label")
        plt.yticks(range(self.num_classes), self.labels)  # y轴标签
        plt.xticks(range(self.num_classes), self.labels, rotation=45)  # x轴标签

        for x in range(self.num_classes):
            for y in range(self.num_classes):
                value = float(format('%.2f' % self.matrix[y, x]))  # 数值处理
                plt.text(x, y, value, verticalalignment='center', horizontalalignment='center')  # 写值

        plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域

        plt.colorbar()  # 色条
        plt.savefig('./figure-0727-all/ConfusionMatrix(acc='+self.summary()+').png',
                    bbox_inches='tight')  # bbox_inches='tight'可确保标签信息显示全
        # plt.show()
        plt.close()


# 画图
class RecorderMeter(object):
    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)

    def update(self, idx,  train_loss, train_acc, val_loss,  prec1):
        self.epoch_losses[idx, 0] = train_loss * 30
        self.epoch_losses[idx, 1] = val_loss * 30   # 为了在图上显示的明显，*30
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = prec1
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):
        title = 'The accuracy/loss curve of train/val'
        dpi = 80
        width, height = 1800, 1800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 5
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color='b', linestyle='-', label='valid-prec1', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle=':', label='train-loss-x30', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis, color='b', linestyle=':', label='valid-loss-x30', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print('Saved figure')
        plt.close(fig)

if __name__ == "__main__":
    
    main()
    print("Process has finished!")
   
