#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# author：fmy
import torch

def nothing(**kwargs):
    pass


def get_drop_hook(prob, device):
    def drop_hook_func(layer, input, output):  # 0.3 prob means 30% value to be 0
        return output * (torch.rand(output.size()[1], device=device) >= prob).view(1, -1, 1, 1)

    return drop_hook_func


def add_drop_func(model, model_name, prob, device):
    def func(model):
        drop = prob
        if model_name == "wideresnet":
            for name, layer in model.named_modules():
                if "block3" in name and "conv" in name and "Shortcut" not in name:
                    layer.register_forward_hook(get_drop_hook(drop, device))
        elif model_name == "efficientnet":
            layer_ls = ["layers.{}.conv{}".format(layer_num, conv_num) for layer_num in [13, 14, 15] for conv_num in
                        [1, 2, 3]]
            for name, layer in model.named_modules():
                if name in layer_ls:
                    layer.register_forward_hook(get_drop_hook(drop, device))

    return func


class BaseAttack:
    name: "base attack"

    def __init__(self, epsilon: float, step: float = 0.05,
                 iter_num: int = 40, is_targeted: bool = False,
                 ord: str = "Linf", device=torch.device("cuda:0"),
                 model_preprocessing=nothing, data_preprocessing=nothing):
        self.epsilon = epsilon
        self.step = step
        self.iter_num = iter_num
        self.is_targeted = is_targeted
        self.ord = ord
        self.model_preprocessing = model_preprocessing
        self.data_preprocessing = data_preprocessing
        self.device = device
        self.restart = 1

    def attack(self, black_model, substitute_model, loss_func, loader, target: int = None):
        if self.is_targeted and target == None:
            raise ValueError("targeted attack must have the value of attack class")

        num, mis_num = 0, 0
        self.model_preprocessing(model = substitute_model)
        substitute_model = substitute_model.to(self.device)
        substitute_model.eval()
        black_model = black_model.to(self.device)
        black_model.eval()

        for data, label in loader:
            data, label = data.to(self.device), label.to(self.device)
            restart_result = torch.zeros_like(label, device=self.device, dtype=torch.float32)
            for i in range(self.restart):
                distortion = self.distortion_generation(data)
                for _ in range(self.iter_num):
                    output = substitute_model(data + distortion)
                    if self.is_targeted:
                        loss = loss_func(output, torch.full_like(label, target, device=self.device, dtype=torch.long))
                    else:
                        loss = -loss_func(output, label)
                        # default loss function is nn.CrossEntropy, so use gradient descent algorithm
                    loss.backward()
                    self.grad_transform(distortion)
                    distortion = self.distortion_update(distortion)
                    distortion = self.clip(distortion)
                with torch.no_grad():
                    if self.is_targeted:
                        restart_result += (black_model(data + distortion).max(dim=1)[1] == target)
                    else:
                        restart_result += (black_model(data + distortion).max(dim=1)[1] != label)
            with torch.no_grad():
                mis_num += (restart_result != 0).sum().item()
                num += label.size()[0]
        return mis_num / num * 100.

    @torch.no_grad()
    def distortion_generation(self, data):
        return torch.zeros_like(data, device=self.device).requires_grad_(True)

    @torch.no_grad()
    def clip(self, distortion):
        if self.ord == "Linf":
            mask = torch.sign(distortion)
            distortion = mask * torch.min(distortion.abs_(),
                                          torch.full_like(distortion, self.epsilon, device=self.device))
        elif self.ord == "L2":
            l2_norm = distortion.pow(2).view(distortion.size()[0], -1).sum(dim=1).pow(0.5)
            mask = l2_norm <= self.epsilon  # if norm of tensor bigger than constraint, then scale it into the range
            l2_norm = torch.where(mask, torch.ones_like(l2_norm, device=self.device), l2_norm)
            distortion = distortion / (l2_norm).view(-1, 1, 1, 1)
        elif self.ord == "L1":
            l1_norm = distortion.abs().view(distortion.size()[0], -1).sum(dim=1)
            mask = l1_norm <= self.epsilon
            l2_norm = torch.where(mask, torch.ones_like(l1_norm, device=self.device), l1_norm)
            distortion = distortion / (l2_norm).view(-1, 1, 1, 1)
        else:
            raise ValueError("The norm not exists.")
        distortion.requires_grad_(True)
        return distortion

    @torch.no_grad()
    def grad_transform(self, distortion):
        # scale grad to same level for fair comparison
        # distortion.grad = distortion.grad / distortion.grad.abs().max()  # divided by maximum value
        distortion.grad.sign_()

    @torch.no_grad()
    def distortion_update(self, distortion):
        distortion = distortion - self.step * distortion.grad
        return distortion


class FGSM(BaseAttack):
    name: "fgsm attack"

    def __init__(self, epsilon: float, step: float = 0.01,
                 iter_num: int = 40, targeted: bool = False,
                 ord: str = "Linf", device=torch.device("cuda:0"),
                 model_preprocessing=nothing, data_preprocessing=nothing):
        super(FGSM, self).__init__(epsilon, 1., 1, targeted,
                                   ord, device, model_preprocessing, data_preprocessing)

    @torch.no_grad()
    def grad_transform(self, distortion):
        distortion.grad.sign_()


class BIM(BaseAttack):
    name: "bim attack"

    def __init__(self, epsilon: float, step: float = 0.05,
                 iter_num: int = 40, targeted: bool = False,
                 ord: str = "Linf", device=torch.device("cuda:0"),
                 model_preprocessing=nothing, data_preprocessing=nothing):
        super(BIM, self).__init__(epsilon, step, iter_num, targeted,
                                   ord, device, model_preprocessing, data_preprocessing)


class PGD(BaseAttack):
    name: "pgd attack"

    def __init__(self, epsilon: float, step: float = 0.05,
                 iter_num: int = 40, targeted: bool = False,
                 ord: str = "Linf", device=torch.device("cuda:0"),
                 model_preprocessing=nothing, data_preprocessing=nothing, restart: int = 10):
        super(PGD, self).__init__(epsilon, step, iter_num, targeted,
                                   ord, device, model_preprocessing, data_preprocessing)
        self.restart = restart

    @torch.no_grad()
    def distortion_generation(self, data):
        # the epsilon approximately equals to 0.015, so we suppose that the noise is 1/10 to it.
        return torch.rand_like(data, device=self.device).div_(300.).requires_grad_(True)


class Momentum(PGD):
    name: "momentum attack"

    def __init__(self, epsilon: float, step: float = 0.05,
                 iter_num: int = 40, targeted: bool = False,
                 ord: str = "Linf", device=torch.device("cuda:0"),
                 model_preprocessing=nothing, data_preprocessing=nothing, rand_init: bool = True, restart: int = 10):
        super(Momentum, self).__init__(epsilon, step, iter_num, targeted,
                                   ord, device, model_preprocessing, data_preprocessing)
        self.rand_init = rand_init
        self.restart = restart
        self.grad_accumulation = None
        self.factor = 0.9  # accumlation factor
        self.update_value = None
        self.cur = None

    @torch.no_grad()
    def distortion_generation(self, data):
        self.cur = 1
        self.grad_accumulation = torch.zeros_like(data, device=self.device)
        if self.rand_init:
            return PGD.distortion_generation(self, data)
        else:
            return BaseAttack.distortion_generation(self, data)

    @torch.no_grad()
    def grad_transform(self, distortion):
        super().grad_transform(distortion)
        self.grad_accumulation = self.factor * self.grad_accumulation + (1 - self.factor) * distortion.grad
        scale_factor = 1 / (1 - self.factor ** self.cur)
        self.update_value = self.grad_accumulation / scale_factor

    @torch.no_grad()
    def distortion_update(self, distortion):
        distortion = distortion - (self.step * self.update_value)
        self.cur += 1
        return distortion
