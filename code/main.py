from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from image_list import ImageList
from prepocess import *
from network import ResNet50Fc, Discriminator, MultiClassify
from active import random_active, uncertainty_active
import numpy as np
from tensorboardX import SummaryWriter
from utils import single_entropy, margin, get_consistency

import random


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target, path) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        feature, output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def train_sim(args, model, discrim, device, source_train_loader, target_train_loader, optimizer, epoch):
    model.eval()
    discrim.train()
    for batch_idx, ((source_data, source_label, source_path), (target_data, target_label, target_path)) in enumerate(
            zip(source_train_loader, target_train_loader)):
        source_data, source_label = source_data.to(device), source_label.to(device)
        target_data, target_label = target_data.to(device), target_label.to(device)

        optimizer.zero_grad()

        with torch.no_grad():
            source_feature, source_output = model(source_data)
            target_feature, target_output = model(target_data)

        source_sim = discrim(source_feature.detach())
        target_sim = discrim(target_feature.detach())

        sim_loss = F.binary_cross_entropy(source_sim, torch.zeros_like(source_sim)) + \
                   F.binary_cross_entropy(target_sim, torch.ones_like(target_sim))
        sim_loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Sim Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(source_data),
                min(len(source_train_loader.dataset), len(target_train_loader.dataset)),
                       100. * batch_idx / min(len(source_train_loader), len(target_train_loader)), sim_loss.item()))


def train_multi(args, model, model1, device, train_loader1, train_loader2, train_loader3, train_loader4, train_loader5,
                optimizer1, epoch):
    model.eval()
    model1.train()
    iters = zip(train_loader1, train_loader2, train_loader3, train_loader4, train_loader5)

    for batch_idx, ((data1, target1, path1), (data2, target2, path2), (data3, target3, path3),
                    (data4, target4, path4), (data5, target5, path5)) in enumerate(iters):
        data1 = data1.to(device)
        data2 = data2.to(device)
        data3 = data3.to(device)
        data4 = data4.to(device)
        data5 = data5.to(device)

        target1 = target1.to(device)
        target2 = target2.to(device)
        target3 = target3.to(device)
        target4 = target4.to(device)
        target5 = target5.to(device)

        with torch.no_grad():
            feature1, output1 = model(data1)
            feature2, output2 = model(data2)
            feature3, output3 = model(data3)
            feature4, output4 = model(data4)
            feature5, output5 = model(data5)

        optimizer1.zero_grad()

        y1_d1, y2_d1, y3_d1, y4_s1, y5_s1 = model1(feature1.detach())
        y1_d2, y2_d2, y3_d2, y4_s2, y5_s2 = model1(feature2.detach())
        y1_d3, y2_d3, y3_d3, y4_s3, y5_s3 = model1(feature3.detach())
        y1_d4, y2_d4, y3_d4, y4_s4, y5_s4 = model1(feature4.detach())
        y1_d5, y2_d5, y3_d5, y4_s5, y5_s5 = model1(feature5.detach())

        loss1 = F.cross_entropy(y1_d1, target1)
        loss2 = F.cross_entropy(y2_d2, target2)
        loss3 = F.cross_entropy(y3_d3, target3)
        loss4 = F.cross_entropy(y4_s4, target4)
        loss5 = F.cross_entropy(y5_s5, target5)
        loss = loss1 + loss2 + loss3 + loss4 + loss5

        loss.backward()
        optimizer1.step()


def test(args, model, device, test_loader, multi):
    model.eval()
    multi.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target, path in test_loader:
            data, target = data.to(device), target.to(device)
            feature, output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return correct / len(test_loader.dataset)


def find(args, model, model1, device, train_loader):
    model.eval()
    model1.eval()
    stat = list()
    with torch.no_grad():
        for batch_idx, (data, target, path) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            feature, output = model(data)
            target_sim = model1(feature.detach())

            entropy = single_entropy(output)

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct = pred.eq(target.view_as(pred))

            for i in range(len(correct)):
                stat.append([path[i], target[i].item(), batch_idx * args.batch_size + i,
                             pred[i].item(), entropy[i].item(), correct[i].item()])

    stat = sorted(stat, key=lambda x: x[0])
    np.savetxt('stat.csv', stat, delimiter=',', fmt='%s')
    return stat


def uncertainty_evaluate(args, model, multi, discrim, device, train_loader):
    model.eval()
    multi.eval()
    stat = list()
    with torch.no_grad():
        for batch_idx, (data, target, path) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            feature, output = model(data)
            y1, y2, y3, y4, y5 = multi(feature)
            target_sim = discrim(feature.detach())

            # uncertainty = margin(output) + get_consistency(y1, y2, y3, y4, y5) + target_sim
            uncertainty = margin(output) + get_consistency(y1, y2, y3, y4, y5)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct = pred.eq(target.view_as(pred))

            for i in range(len(correct)):
                stat.append([path[i], target[i].item(), batch_idx * args.batch_size + i,
                             pred[i].item(), uncertainty[i].item(), correct[i].item()])

    stat = sorted(stat, key=lambda x: x[4])
    stat = np.array(stat)
    return stat


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--gpu', default=None, type=str,
                        help='GPU id to use.')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--source', type=str, default='', help="The source dataset path list")
    parser.add_argument('--source-val', type=str, default='', help="The source validation dataset path list")
    parser.add_argument('--target', type=str, default='', help="The target dataset path list")
    parser.add_argument('--target-val', type=str, default='', help="The target validation dataset path list")
    parser.add_argument('--class-num', default=31, type=int, help='class num of dataset.')
    args = parser.parse_args()
    use_cuda = args.gpu and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    # cudnn.benchmark = True
    cudnn.deterministic = True
    device = torch.device("cuda:" + args.gpu if use_cuda else "cpu")

    writer = SummaryWriter()
    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

    source_train_ds = ImageList(args.source, transform=train_transform)
    source_train_ds1 = ImageList(args.source, transform=train_transform1)
    source_train_ds2 = ImageList(args.source, transform=train_transform2)
    source_train_ds3 = ImageList(args.source, transform=train_transform3)
    source_train_ds4 = ImageList(args.source, transform=train_transform4)
    source_train_ds5 = ImageList(args.source, transform=train_transform5)

    target_train_ds = ImageList(args.target, transform=test_transform)
    target_val_ds = ImageList(args.target_val, transform=test_transform)

    source_train_loader = DataLoader(source_train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                     **kwargs)
    train_loader1 = DataLoader(source_train_ds1, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
    train_loader2 = DataLoader(source_train_ds2, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
    train_loader3 = DataLoader(source_train_ds3, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
    train_loader4 = DataLoader(source_train_ds4, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
    train_loader5 = DataLoader(source_train_ds5, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)

    target_train_loader = DataLoader(target_train_ds, batch_size=args.batch_size, **kwargs)
    target_test_loader = DataLoader(target_val_ds, batch_size=args.test_batch_size, **kwargs)

    model = ResNet50Fc(bottleneck_dim=256, class_num=args.class_num).to(device)
    multi = MultiClassify(bottleneck_dim=256, class_num=args.class_num).to(device)
    discrim = Discriminator(bottleneck_dim=256).to(device)

    optimizer = optim.Adadelta(model.parameters_list(args.lr), lr=args.lr)
    optimizer1 = optim.Adadelta(multi.parameters(), lr=args.lr)

    totality = len(target_train_ds)
    for epoch in range(1, args.epochs + 1):

        train(args, model, device, source_train_loader, optimizer, epoch)
        train_multi(args, model, multi, device, train_loader1, train_loader2, train_loader3, train_loader4,
                    train_loader5, optimizer1, epoch)
        train_sim(args, model, discrim, device, source_train_loader, target_train_loader, optimizer, epoch)
        test_acc = test(args, model, device, target_test_loader, multi)

        # print(test_acc)
        writer.add_scalar('testacc', test_acc, epoch)

        if epoch in [10, 12, 14, 16, 18]:
            # if epoch in [14, 16, 18, 20, 22, 24, 28, 30, 32, 34]:
            # if epoch in [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46]:
            # if epoch in [2, 16]:
            # random_active(candidate_dataset=target_train_ds, aim_dataset=source_train_ds, active_ratio=0.01,
            #               totality=totality)
            uncertainty_rank = uncertainty_evaluate(args, model, multi, discrim, device, target_train_loader)
            active_samples = uncertainty_active(candidate_dataset=target_train_ds, aim_dataset=source_train_ds,
                                                uncertainty_rank=uncertainty_rank, current_acc=test_acc,
                                                active_ratio=0.01, totality=totality)
            source_train_ds1.add_item(active_samples)
            source_train_ds2.add_item(active_samples)
            source_train_ds3.add_item(active_samples)
            source_train_ds4.add_item(active_samples)
            source_train_ds5.add_item(active_samples)

    # np.savetxt(args.source[17] + '-' + args.target[17] + '.txt', source_train_ds.samples, delimiter=' ', fmt='%s')

    # wrong_list = find(args, model, model1, device, target_test_loader)
    # np.savetxt('sim.txt', wrong_list, fmt='%.5f')
    # np.savetxt('wrong_list.txt', wrong_list, fmt='%s')

    if args.save_model:
        torch.save(model.state_dict(), "resnet.pt")

    # writer.export_scalars_to_json("./all_scalars.json")
    writer.close()


if __name__ == '__main__':
    main()
