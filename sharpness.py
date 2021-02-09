# -*- coding: utf-8 -*-

import torch
import torchvision
import torch.optim as optim
import sys
import gc
import os
import numpy as np
import copy
import scipy.optimize as optimize

from utils import ScipyOptimizeWrapper


class Sharpness(object):

    def __init__(self, net, loss, dataset,
            device, sharpness_train_batch_size,
            num_workers, test_batch_size,
            binary_dataset,
            output_file_pth,
            sample
            ):
        self.net = copy.deepcopy(net)
        self.loss = loss
        self.trainloader = torch.utils.data.DataLoader(
                dataset, batch_size=sharpness_train_batch_size,
                shuffle=True, num_workers=num_workers,
                drop_last=True)
        self.testloader = torch.utils.data.DataLoader(
                dataset, batch_size=test_batch_size,
                shuffle=False, num_workers=num_workers,
                drop_last=True)  # Note this is not on the test set.
        self.full_batch_loader = torch.utils.data.DataLoader(
                dataset, batch_size=len(dataset),shuffle=False,
                num_workers=0
                )
        # self.optimizer = optim.SGD(net.parameters(), lr=1e-3) # Have to use vanilla SGD.
        self.device = device
        self.binary_dataset = binary_dataset
        self.output_file_pth = output_file_pth
        self.sample = sample

    def clip_params(self, eps, params, new_params):
        for i in new_params:
            diff = new_params[i] - params[i]
            eps_mtx = eps * (torch.abs(params[i]) + 1) # mtx for matrix...
                                                # (I forget it myself after a while)
            is_out_of_bound = False
            outer_up = torch.nonzero(diff>eps_mtx, as_tuple=True)
            if len(outer_up[0]) != 0:
                is_out_of_bound = True
                diff[outer_up] = eps_mtx[outer_up]

            outer_low = torch.nonzero(diff<-eps_mtx, as_tuple=True)
            if len(outer_low[0]) != 0:
                is_out_of_bound = True
                diff[outer_low] = -eps_mtx[outer_low]
            new_params[i] = params[i] + diff
        return new_params

    def sharpness(self, clip_eps=1e-4, max_iter_epochs=100, opt_mtd='SGD'):
        net = self.net
        net.eval()
        L_w = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = net(inputs)
                if self.binary_dataset:
                    outputs.squeeze_(-1)
                    targets = targets.type_as(outputs)
                loss = self.loss(outputs, targets)
                L_w += loss.item()
            L_w = L_w/len(dataset)
        #print('L_w: ', L_w)
        w = copy.deepcopy(net.state_dict())
        w = self.del_key_from_dic(w, 'num_batches_tracked')
        self.stop_tracking(w)
        max_value = 0
        max_value_list = []
        if opt_mtd == 'SGD':
            optimizer = optim.SGD(net.parameters(), lr=1e-3)
            net.train()
            for sharpness_epoch in range(max_iter_epochs):
                print('Sharpness epoch: %d'%(sharpness_epoch+1))
                epoch_loss = 0
                for batch_idx, (inputs, targets) in enumerate(self.trainloader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    optimizer.zero_grad()
                    outputs = net(inputs)
                    if self.binary_dataset:
                        outputs.squeeze_(-1)
                        targets = targets.type_as(outputs)
                    new_loss = -1. * self.loss(outputs, targets)
                    new_loss.backward()
                    optimizer.step()

                    new_w = copy.deepcopy(net.state_dict())
                    self.stop_tracking(new_w)
                    new_w = self.del_key_from_dic(new_w, 'num_batches_tracked')

                    new_w = self.clip_params(clip_eps, w, new_w)

                    net.load_state_dict(new_w, strict=False)

                    new_outputs = net(inputs)
                    if self.binary_dataset:
                        new_outputs.squeeze_(-1)
                        targets = targets.type_as(new_outputs)
                    epoch_loss += self.loss(new_outputs, targets).item()
                epoch_loss = epoch_loss / len(dataset)
                max_value = max(max_value, epoch_loss)
                max_value_list.append(max_value)
            np.save(os.path.join(
                self.output_file_pth, 'max_value_list_%d.npy'%self.sample), max_value_list)
            sharpness = 100 * (max_value - L_w) / (1 + L_w)
        elif opt_mtd == 'L-BFGS-B':
            scipy_obj = ScipyOptimizeWrapper(net, self.loss, self.full_batch_loader)
            scipy_result = optimize.minimize(scipy_obj.f, scipy_obj.x0, method='L-BFGS-B',
                   jac=scipy_obj.jac, bounds=scipy_obj.bounds(eps=clip_eps),
                   options={'maxiter':10, 'iprint':1}
                   )
            print('f0: ', scipy_obj.f0)
            print('L-BFGS-B results:\n', scipy_result)
            print(type(scipy_result))
            max_value = - scipy_result.fun
            sharpness = 100 * (scipy_obj.f0 - scipy_result.fun) / 1 - (scipy_obj.f0)
        return sharpness


    @staticmethod
    def stop_tracking(w):
        for i in w:
            w[i].requires_grad_(False)
        return w


    @staticmethod
    def del_key_from_dic(dic, keyword):
        for i in dic.copy():
            if keyword in i:
                del dic[i]
        return dic

    @staticmethod
    def _test_clip_is_effective(eps, params, new_params, num_eps=1e-3):
        # Here the num_eps is a pretty vacuous bound for
        # normal small weights, but is necessitated by
        # the 6-significant-digits problem of float32 datatype.
        # This is not the best way of seeing whether the clip_parm
        # is effective or not. We can simply see the param before
        # and after clip_param.
        for i in new_params:
            if torch.max(
                    (torch.abs(new_params[i]-params[i])-
                    (eps*(torch.abs(params[i])+1))) > num_eps
                    ) > 0:
                where_out = torch.nonzero(
                        (torch.abs(new_params[i]-params[i])-
                    (eps*(torch.abs(params[i])+1))) > num_eps, as_tuple=True
                )
                print('Where difference go beyond box:')
                print(torch.abs(new_params[i]-params[i])[where_out]
                        )
                print(torch.abs(new_params[i]-params[i])[where_out][0].dtype)
                print('Box limits:')
                print((eps*(torch.abs(params[i])+1))[where_out])

                print('Difference minus Box:')
                print((torch.abs(new_params[i]-params[i])-
                    (eps*(torch.abs(params[i])+1)))[where_out])

                print('Whether it is large than num_eps:')
                print((torch.abs(new_params[i]-params[i])-
                    (eps*(torch.abs(params[i])+1)))[where_out] > num_eps)
                return False
        return True

    @staticmethod
    def _max_diff_minus_eps(eps, params, new_params):
        for i in new_params:
            print(
                torch.max(
                    torch.abs(new_params[i] - params[i])-eps*(torch.abs(params[i])+1)))
        return None

    @staticmethod
    def _median_diff(params, new_params):
        l = []
        for i in new_params:
            l.append(float(torch.median(new_params[i] - params[i])))
        return l

    @staticmethod
    def _print_first_w(params):
        for i in params:
            print(params[i])
            break

    @staticmethod
    def _print_different_w(params, new_params):
        for i in new_params:
            if not torch.equal(new_params[i], params[i]):
                print('\n', i, params[i])
                print('*'*88)
                print(i, new_params[i], '\n')

    @staticmethod
    def _print_w_shape(params):
        for i in params:
            print(params[i].shape)


    @staticmethod
    def _arrayify(X):
        return X.cpu().detach().contiguous().double().clone().numpy()


if __name__ == '__main__':
    # print('CPUs:', os.cpu_count())
    from main import net, criterion, trainset, device
    S = Sharpness(net, criterion, trainset, device)
    print('Sharpness:', S.sharpness())

