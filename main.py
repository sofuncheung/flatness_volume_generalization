# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
#import importlib.util

from torchsummary import summary

import os
import sys
import argparse

import utils
from utils import *
from model import resnet, keskar_models
from sharpness import Sharpness
from GP_prob.GP_prob_gpy import GP_prob
from empirical_kernel import empirical_K


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--load', '-l', action='store_true',
        help='load saved model')
parser.add_argument('--path', '-p', # config.py path
        default='/mnt/zfsusers/sofuncheung/shake-it/playground',
        type=str, help='config.py path')
parser.add_argument('--sample', '-s', default=1, type=int, help='sample number')
parser.add_argument('--empirical_kernel_only', '-k', action='store_true',
                    help='if set, calculate empirical_K only')
parser.add_argument('--save_checkpoint_on_train_acc', action='store_true',
        help='Save checkpoint based on training accuracy instead of testing')

args = parser.parse_args()

# Load config by adding sys path
sys.path.append(args.path)
from config import config


ONE_OFF = (config.sensitivity_one_off or
        config.sharpness_one_off)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
trainloader,testset,testloader,trainset_genuine = utils.load_data(
        config.train_batch_size,
        config.test_batch_size,
        config.num_workers,
        dataset='CIFAR10',
        attack_set_size=config.attack_set_size,
        binary=config.binary_dataset)

# classes = ('plane', 'car', 'bird', 'cat', 'deer',
#            'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
if config.binary_dataset:
    net = resnet.ResNet50(num_classes=1)
    #net = keskar_models.C1(num_classes=1)
else:
    net = resnet.ResNet50()
    #net = keskar_models.C1(num_classes=10)
# net = resnet.ResNet50()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()

# He-normal Initialization
net.apply(he_init)

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
# summary(net, (3, 32, 32))
# rescale(net, 'layer1', 1, 2)
# sys.exit()
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(
            os.path.join(args.path, 'checkpoint_%d'%args.sample)
            ), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(
            os.path.join(
                args.path, 'checkpoint_%d/ckpt.pth'%args.sample))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

if config.binary_dataset == True:
    criterion = nn.BCEWithLogitsLoss() # sigmoid cross entropy
else:
    criterion = nn.CrossEntropyLoss()
if config.optim == 'SGD+Momentum':
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
            momentum=0.9, weight_decay=5e-4)
elif config.optim == 'Adam':
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
elif config.optim == 'SGD':
    optimizer = optim.SGD(net.parameters(), lr=args.lr)

# Learning Rate Decay
decayRate = 0.96
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
# Training

if args.save_checkpoint_on_train_acc:
    def train(epoch):
        global best_acc
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            if config.binary_dataset:
                outputs.squeeze_(-1)
                targets = targets.type_as(outputs)
            loss = criterion(outputs, targets) # Here loss is genuine/attack mixed loss 
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            '''
            if config.binary_dataset:
                predicted = outputs > 0
            else:
                _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            '''
            #progress_bar(
            #    batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
            # Above is when using terminal. Now I would like to run it on computing cores.
            if (batch_idx+1) % 50 == 0:
                print('Training On Batch %03d' % (batch_idx+1))

        epoch_loss = train_loss/(batch_idx+1)
        epoch_acc, correct, total = dataset_accuracy(net, trainset_genuine,
                device, config.binary_dataset)  #acc=correct/total. acc measured on train_genuine 
        print('(Tainted) Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
            epoch_loss, 100.*epoch_acc, correct, total))

        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir(
                    os.path.join(args.path, 'checkpoint_%d'%args.sample)):
                os.mkdir(
                        os.path.join(args.path, 'checkpoint_%d'%args.sample))
            torch.save(state, os.path.join(
                args.path, 'checkpoint_%d/ckpt.pth'%args.sample))
            best_acc = acc

        return (epoch_loss, 100.*epoch_acc)


    def test(epoch):
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            if config.binary_dataset:
                outputs.squeeze_(-1)
                targets = targets.type_as(outputs)
            with torch.no_grad():
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                if config.binary_dataset:
                    predicted = (outputs > 0)
                else:
                    _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                if (batch_idx+1) % 20 == 0:
                    print('Testing On Batch %03d' % (batch_idx+1))
        epoch_loss = test_loss/(batch_idx+1)
        epoch_acc = correct/total
        print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (epoch_loss, 100.*epoch_acc, correct, total))

        return (epoch_loss, 100.*epoch_acc)
else:
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            if config.binary_dataset:
                outputs.squeeze_(-1)
                targets = targets.type_as(outputs)
            loss = criterion(outputs, targets) # Here loss is genuine/attack mixed loss 
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            '''
            if config.binary_dataset:
                predicted = outputs > 0
            else:
                _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            '''
            #progress_bar(
            #    batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
            # Above is when using terminal. Now I would like to run it on computing cores.
            if (batch_idx+1) % 50 == 0:
                print('Training On Batch %03d' % (batch_idx+1))

        epoch_loss = train_loss/(batch_idx+1)
        epoch_acc, correct, total = dataset_accuracy(net, trainset_genuine,
                device, config.binary_dataset)  #acc=correct/total. acc measured on train_genuine 
        print('(Tainted) Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
            epoch_loss, 100.*epoch_acc, correct, total))
        return (epoch_loss, 100.*epoch_acc)


    def test(epoch):
        global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            if config.binary_dataset:
                outputs.squeeze_(-1)
                targets = targets.type_as(outputs)
            with torch.no_grad():
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                if config.binary_dataset:
                    predicted = (outputs > 0)
                else:
                    _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                if (batch_idx+1) % 20 == 0:
                    print('Testing On Batch %03d' % (batch_idx+1))
        epoch_loss = test_loss/(batch_idx+1)
        epoch_acc = correct/total
        print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (epoch_loss, 100.*epoch_acc, correct, total))

        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir(
                    os.path.join(args.path, 'checkpoint_%d'%args.sample)):
                os.mkdir(
                        os.path.join(args.path, 'checkpoint_%d'%args.sample))
            torch.save(state, os.path.join(
                args.path, 'checkpoint_%d/ckpt.pth'%args.sample))
            best_acc = acc

        return (epoch_loss, 100.*epoch_acc)

if __name__ == '__main__':
    assert os.path.isdir(args.path), "Target Output Directory Doesn't Exist!"
    if not args.empirical_kernel_only:
        if not args.load: # No pre-trained model, still needs training.
            train_loss_acc_list = []
            test_loss_acc_list = []
            if config.sharpness_cons == True:
                sharpness_cons = []
            if config.sensitivity_cons == True:
                sensitivity_cons = []

            for epoch in range(start_epoch, start_epoch+config.train_epoch):
                # if (epoch + 1) == 100:
                #     rescale(net, 'all', None, None, config.alpha)
                #     adjust_learning_rate(optimizer, args.lr)
                train_returns = train(epoch)
                test_returns = test(epoch)

                train_loss_acc_list.append(train_returns)
                test_loss_acc_list.append(test_returns)

                if config.sharpness_cons == True:
                    S = Sharpness(net, criterion,
                            trainset_genuine, device,
                            config.sharpness_train_batch_size,
                            config.num_workers,
                            config.test_batch_size,
                            config.binary_dataset,
                            args.path,
                            args.sample
                            )
                    sharpness_cons.append(S.sharpness(opt_mtd=config.sharpness_method))
                if config.sensitivity_cons == True:
                    Sen_train = Sensitivity(net, trainset_genuine, device, epoch,
                            config.test_batch_size,
                            config.num_workers
                            )
                    # Sen_test = Sensitivity(net, testset, device, epoch)
                    sensitivity_cons.append(Sen_train.sensitivity())
                if config.lr_decay:
                    lr_scheduler.step()

                if config.break_when_reaching_zero_error == True:
                    if train_returns[1] == 100.:
                        break

                '''
                if ONE_OFF == True:
                    if train_returns[1] == 100.:
                        break
                '''

            np.save(os.path.join(
                args.path,'train_loss_acc_list_%d.npy'%args.sample), train_loss_acc_list)
            np.save(os.path.join(
                args.path, 'test_loss_acc_list_%d.npy'%args.sample), test_loss_acc_list)
            if config.sharpness_cons == True:
                np.save(os.path.join(
                    args.path, 'sharpness_list_%d.npy'%args.sample), sharpness_cons)
            if config.sensitivity_cons == True:
                np.save(os.path.join(
                     args.path, 'sensitivity_list_%d.npy'%args.sample), sensitivity_cons)

        else: # There is pre-trained model, just load it and calculate stuff.
            print('==> Loading from checkpoint..')
            assert os.path.isdir(
                    os.path.join(args.path, 'checkpoint_%d'%args.sample)
                    ), 'Error: no checkpoint directory found!'
            checkpoint = torch.load(
                    os.path.join(
                        args.path, 'checkpoint_%d/ckpt.pth'%args.sample))
            net.load_state_dict(checkpoint['net'])
            epoch = checkpoint['epoch']
            epoch_acc, correct, total = dataset_accuracy(net, trainset_genuine,
                    device, config.binary_dataset)  #acc=correct/total. acc measured on train_genuine
            print('Loaded Model Acc: %.3f%% (%d/%d)' % (
                100.*epoch_acc, correct, total))
            train_loss_acc_list = np.load('train_loss_acc_list_%d.npy'%args.sample)
            test_loss_acc_list = np.load('test_loss_acc_list_%d.npy'%args.sample)


        if config.sensitivity_one_off == True:
            Sen_train = Sensitivity(net, trainset_genuine, device, epoch,
                    config.test_batch_size,
                    config.num_workers
                    )
            sensitivity_one_off = Sen_train.sensitivity()
            sensitivity_sigmoid_one_off = Sen_train.sensitivity_sigmoid()
            print('The sensitivity one_off (log10):', np.log10(sensitivity_one_off))
            print('The sensitivity sigmoid one_off (log10):',
                    np.log10(sensitivity_sigmoid_one_off))

            Sen_test = Sensitivity(net, testset, device, epoch,
                      config.test_batch_size,
                      config.num_workers)
            sen_test_logits = Sen_test.sensitivity()
            sen_test_sigmoid = Sen_test.sensitivity_sigmoid()

        if config.sharpness_one_off == True:
            S = Sharpness(net, criterion,
                    trainset_genuine, device,
                    config.sharpness_train_batch_size,
                    config.num_workers,
                    config.test_batch_size,
                    config.binary_dataset,
                    args.path,
                    args.sample
                    )
            sharpness_one_off = S.sharpness(opt_mtd=config.sharpness_method)
            print('The sharpness one_off (log10):', np.log10(sharpness_one_off))
            # np.save('sharpness_%d.npy'%args.sample, np.log10(sharpness_one_off))

        if config.robustness_one_off == True:
            R = Robustness(net, trainset_genuine, device)
            robustness_logits = R.robustness_logits()
            robustness_sigmoid = R.robustness_sigmoid()
            print('Robustness logits (log10):', np.log10(robustness_logits))
            print('Robustness sigmoid (log10):', np.log10(robustness_sigmoid))

            R_test = Robustness(net, testset, device)
            robustness_test_logits = R_test.robustness_logits()
            robustness_test_sigmoid = R_test.robustness_sigmoid()

        if config.volume_one_off == True:
            data_train_plus_test = torch.utils.data.ConcatDataset((trainset_genuine, testset))
            if os.path.isfile(os.path.join(
                args.path, 'empirical_K.npy')
                    ):
                print('Using Existing Kernel:')
                K = np.load(os.path.join(
                    args.path, 'empirical_K.npy'))
            else:
                model = resnet.ResNet_pop_fc_50(num_classes=1)
                print('Calculating Empirical Kernel:')
                K = empirical_K(model, data_train_plus_test,
                        #1,device,
                        0.1*len(data_train_plus_test), device, # Use fc-poped model
                        sigmaw=np.sqrt(2), sigmab=1.0, n_gpus=1,
                        empirical_kernel_batch_size=256,
                        truncated_init_dist=False,
                        store_partial_kernel=False,
                        partial_kernel_n_proc=1,
                        partial_kernel_index=0)
                K = np.array(K.cpu())
                np.save(os.path.join(
                    args.path, 'empirical_K.npy'), K)
            (xs, _) = get_xs_ys_from_dataset(data_train_plus_test, 256, config.num_workers)
            ys = (model_predict(
                    net, data_train_plus_test, 256, config.num_workers, device
                    ) > 0)
            logPU = GP_prob(K, np.array(xs), np.array(ys.cpu()))
            # Note: you have to use np.array(xs) instead of xs!!!
            # This stupid hidden-bug has wasted my whole night!!!
            # Also a funny bug: if K is a tensor on GPU and xs,ys are np.array on cpu,
            # there would be problem as well!
            log_10PU = logPU * np.log10(np.e)

        if config.record == True:
            generalization = test_loss_acc_list[-1][1] # last test accuracy
            record = [generalization,
                    np.log10(sharpness_one_off),
                    log_10PU
                    ]
            np.save(os.path.join(
                args.path, 'record_%d.npy'%args.sample), record)
    else: # Calculate empirical kernel only
        if os.path.isfile(os.path.join(
            args.path, 'empirical_K.npy')
                ):
            print('Empirical Kernel Already Exist!!!:')
        else:
            data_train_plus_test = torch.utils.data.ConcatDataset((trainset_genuine, testset))
            model = resnet.ResNet_pop_fc_50(num_classes=1)
            print('Calculating Empirical Kernel:')
            K = empirical_K(model, data_train_plus_test,
                    #1,device,
                    0.1*len(data_train_plus_test), device, # Use fc-poped model
                    sigmaw=np.sqrt(2), sigmab=1.0, n_gpus=1,
                    empirical_kernel_batch_size=256,
                    truncated_init_dist=False,
                    store_partial_kernel=False,
                    partial_kernel_n_proc=1,
                    partial_kernel_index=0)
            K = np.array(K.cpu())
            np.save(os.path.join(
                args.path, 'empirical_K.npy'), K)
