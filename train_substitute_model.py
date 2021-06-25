import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from utils import mkdir_p
import time

def create_model(param_dict, ema=False):

    if param_dict["dataset_name"] in ["cifar10", "cifar100"]:
        import models.wideresnet as m1
        import models.efficientnet as m2
        nets = {"wideresnet":m1.WideResNet, "efficientnet":m2.EfficientNetB0}

    elif param_dict["dataset_name"] in ["mnist","fashionmnist"]:
        import models.wideresnet_mnist as m1
        import models.efficientnet_mnist as m2
        nets = {"wideresnet":m1.WideResNet, "efficientnet":m2.EfficientNetB0}

    if param_dict["dataset_name"] == "cifar100":
        model = nets[param_dict["substitute_model"]](num_classes = 100).to(param_dict["device"])
    else:
        model = nets[param_dict["substitute_model"]]().to(param_dict["device"])

    if ema:
        for param in model.parameters():
            param.detach_()

    return model

def main(param_dict, black_model, labeled_num):

    best_acc = 0.

    param_dict["output_dir"] = "./substitute_model_res/{}_{}_{}_{}".format(param_dict["dataset_name"], param_dict["substitute_model"],
                                                param_dict["black_model"], labeled_num)

    if not os.path.isdir(param_dict["output_dir"]):
        mkdir_p(param_dict["output_dir"])

    # Data
    if param_dict["dataset_name"] == "cifar10":
        from dataset import cifar10 as dataset
        transform_train = transforms.Compose([
            dataset.RandomPadandCrop(32),
            dataset.RandomFlip(),
            dataset.ToTensor()])
        transform_val = transforms.Compose([
            dataset.ToTensor()])
        train_labeled_set, train_unlabeled_set, val_set, test_set = dataset.get_cifar10('../data', labeled_num, transform_train=transform_train, transform_val=transform_val)
    elif param_dict["dataset_name"] == "cifar100":
        from dataset import cifar100 as dataset
        transform_train = transforms.Compose([
            dataset.RandomPadandCrop(32),
            dataset.RandomFlip(),
            dataset.ToTensor()])
        transform_val = transforms.Compose([
            dataset.ToTensor()])
        train_labeled_set, train_unlabeled_set, val_set, test_set = dataset.get_cifar100('../data', labeled_num, transform_train=transform_train, transform_val=transform_val)
    elif param_dict["dataset_name"] == "mnist":
        from dataset import mnist as dataset
        transform_train = transforms.Compose([
            dataset.ToTensor()])
        transform_val = transforms.Compose([
            dataset.ToTensor()])
        train_labeled_set, train_unlabeled_set, val_set, test_set = dataset.get_mnist('../data', labeled_num, transform_train=transform_train, transform_val=transform_val)
    elif param_dict["dataset_name"] == "fashionmnist":
        from dataset import fashionmnist as dataset
        transform_train = transforms.Compose([
            dataset.ToTensor()])
        transform_val = transforms.Compose([
            dataset.ToTensor()])
        train_labeled_set, train_unlabeled_set, val_set, test_set = dataset.get_fashionmnist('../data', labeled_num, transform_train=transform_train, transform_val=transform_val)

    labeled_trainloader = torch.utils.data.DataLoader(train_labeled_set, batch_size=param_dict['batch_size'], shuffle=True, num_workers=4, drop_last=True)
    unlabeled_trainloader = torch.utils.data.DataLoader(train_unlabeled_set, batch_size=param_dict['batch_size'], shuffle=True, num_workers=4, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=param_dict['batch_size'], shuffle=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=param_dict['batch_size'], shuffle=False, num_workers=4)

    # Model
    model = create_model(param_dict)
    ema_model = create_model(param_dict, ema=True)

    cudnn.benchmark = True

    # loss and optimizer
    train_criterion = SemiLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=param_dict['lr'])
    ema_optimizer= WeightEMA(model, ema_model, alpha=param_dict['ema_decay'])
    
    black_model.eval()
    # Train and val
    with open(os.path.join(param_dict["output_dir"], "log.out"), "a+") as f:
        f.write(time.strftime("%Y/%m/%d %H:%M:%S") + "\n")

    for epoch in range(0, param_dict["epoch"]):

        train(param_dict, labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, train_criterion, epoch, black_model)
        # _, train_acc = validate(param_dict, labeled_trainloader, ema_model, criterion, epoch, mode='Train Stats')
        val_acc = validate(param_dict, val_loader, ema_model, criterion, epoch, mode='Valid Stats')
        test_acc = validate(param_dict, test_loader, ema_model, criterion, epoch, mode='Test Stats ')

        # append logger file
        with open(os.path.join(param_dict["output_dir"], "log.out"), "a+") as f:
            f.write("{epoch}\t{val_acc:.2f}\t{test_acc:.2f}\n".format(epoch = epoch, val_acc = val_acc, test_acc = test_acc))

        # save model
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        if is_best:
            save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'ema_state_dict': ema_model.state_dict(),
                    'acc': val_acc,
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict(),
                }, param_dict["output_dir"])

    with open(os.path.join(param_dict["output_dir"], "log.out"), "a+") as f:
        f.write(time.strftime("%Y/%m/%d %H:%M:%S") + "\n")

    # num = 0
    # test_for_correct = torchvision.datasets.MNIST("../data", transforms = [torchvision.transforms.ToTensor()], train = False)
    # loader = torch.utils.data.DataLoader(test_for_correct, batch_size = 64)
    # for data, label in loader:
    #     data, label = data.to(param_dict["device"]), label.to(param_dict["device"])
    #     num += (model(data).max(dim = 1)[1] == label).sum()
    # print(num)


def train(param_dict, labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, criterion, epoch, black_model):

    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    model.train()
    for batch_idx in range(param_dict["train_iteration"]):
        try:
            inputs_x, targets_x = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x = labeled_train_iter.next()

        try:
            (inputs_u, inputs_u2), _ = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2), _ = unlabeled_train_iter.next()
        if param_dict["dataset_name"] in ["mnist", "fashionmnist"]:
            inputs_x, inputs_u, inputs_u2 = inputs_x.transpose(2,3).transpose(1,2).float(), inputs_u.transpose(2,3).transpose(1,2).float(), inputs_u2.transpose(2,3).transpose(1,2).float()

        batch_size = inputs_x.size(0)
        inputs_x, targets_x = inputs_x.to(param_dict["device"]), targets_x.to(param_dict["device"])
        targets_x = black_model(inputs_x).max(dim=1)[1].cpu()
        # Transform label to one-hot
        if param_dict["dataset_name"] == "cifar100":
            targets_x = torch.zeros(batch_size, 100).scatter_(1, targets_x.view(-1,1).long(), 1)
        else:
            targets_x = torch.zeros(batch_size, 10).scatter_(1, targets_x.view(-1, 1).long(), 1)
        inputs_u = inputs_u.to(param_dict["device"])
        inputs_u2 = inputs_u2.to(param_dict["device"])

        with torch.no_grad():
            # compute guessed labels of unlabel samples
            outputs_u = model(inputs_u)
            outputs_u2 = model(inputs_u2)
            p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
            pt = p**(1/param_dict["T"])
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

        # mixup
        all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x.to(param_dict["device"]), targets_u.to(param_dict["device"]), targets_u.to(param_dict["device"])], dim=0)

        l = np.random.beta(param_dict["alpha"], param_dict["alpha"])

        l = max(l, 1-l)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        # interleave labeled and unlabed samples between batches to get correct batchnorm calculation 
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = interleave(mixed_input, batch_size)

        logits = [model(mixed_input[0])]
        for input in mixed_input[1:]:
            logits.append(model(input))

        # put interleaved samples back
        logits = interleave(logits, batch_size)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)

        Lx, Lu, w = criterion(logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:], epoch+batch_idx/param_dict["train_iteration"], param_dict)

        loss = Lx + w * Lu

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

def validate(param_dict, valloader, model, criterion, epoch, mode):

    # switch to evaluate mode
    model.eval()
    num, correct_num = 0, 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            if param_dict["dataset_name"] in ["mnist", "fashionmnist"]:
                inputs = inputs.transpose(2, 3).transpose(1, 2).float()

            inputs, targets = inputs.to(param_dict["device"]), targets.to(param_dict["device"])
            # compute output
            outputs = model(inputs)
            correct_num += (outputs.max(dim = 1)[1] == targets).sum().item()
            num += targets.size()[0]

    return correct_num / num * 100.

def save_checkpoint(state, checkpoint, filename='checkpoint.pth'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, params):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, params["lambda_u"] * linear_rampup(epoch, params["epoch"])

class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * 0.002 # learning rate 0.002

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype==torch.float32:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
                # customized weight decay
                param.mul_(1 - self.wd)

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


if __name__ == '__main__':

    # dataset : mnist, fashionmnist, cifar10, cifar100
    # substitute model: wideresnet, efficientnet
    # batch_size need to less than labeled data number
    # epoch, batch_size,
#######################################################################################################
    # 需要更改device, dataset_name, substitute_model
    # param_dict = {"batch_size":256, "lr":0.004,
    #                "device":torch.device('cuda:1'), "train_iteration":256,
    #                "alpha":0.75, "lambda_u":75, "T":0.5, "ema_decay":0.999,
    #                "dataset_name":"cifar100", "substitute_model" : "efficientnet", # efficientnet wideresnet
    #               "black_model":"mobilenet"}
    # param_dict["epoch"] = 100 if param_dict["dataset_name"] in ["cifar10", "cifar100"] else 30
    #
    # if param_dict["dataset_name"] in ["cifar10", "cifar100"]:
    #     import models.mobilenet as BlackModel
    # else:
    #     import models.mobilenet_mnist as BlackModel
    # if param_dict["dataset_name"] == "cifar100":
    #     black_model = BlackModel.MobileNet(num_classes=100).to(param_dict["device"])
    # else:
    #     black_model = BlackModel.MobileNet(num_classes=10).to(param_dict["device"])
    # black_model.load_state_dict(torch.load("black_model/mobilenet_{}.p".format(param_dict["dataset_name"]))["net"])
    # black_model.eval()
    #
    # for labeled_data_num in [100, 200, 400, 800, 1600]: # 100, 200, 400, 800, 1600
    #     if labeled_data_num <= 100:
    #         param_dict["batch_size"] = 64
    #     else:
    #         param_dict["batch_size"] = 128
    #     main(param_dict, black_model, labeled_data_num)

# #######################################################################################################

    # param_dict = {"lr":0.002, "device":torch.device('cuda:0'), "train_iteration":256,
    #                "alpha":0.75, "lambda_u":75, "T":0.5, "ema_decay":0.999,
    #                "dataset_name":"cifar10", "substitute_model" : "efficientnet", # wideresnet,efficientnet
    #               "black_model":"googlenet"} # cifar10 cifar100 mnist fashionmnist
    # param_dict["epoch"] = 100 if param_dict["dataset_name"] in ["cifar10", "cifar100"] else 30
    # if param_dict["dataset_name"] in ["cifar10", "cifar100"]:
    #     import models.googlenet as BlackModel
    # else:
    #     import models.googlenet_mnist as BlackModel
    # if param_dict["dataset_name"] == "cifar100":
    #     black_model = BlackModel.GoogLeNet(num_classes=100).to(param_dict["device"])
    # else:
    #     black_model = BlackModel.GoogLeNet(num_classes=10).to(param_dict["device"])
    # black_model.load_state_dict(torch.load("black_model/googlenet_{}.p".format(param_dict["dataset_name"]))["net"])
    # black_model.eval()
    #
    # for labeled_data_num in [100, 200, 400, 800, 1600]:
    #     if labeled_data_num <= 100:
    #         param_dict["batch_size"] = 64
    #     else:
    #         param_dict["batch_size"] = 128
    #     main(param_dict, black_model, labeled_data_num)

# #######################################################################################################

    param_dict = {"lr":0.002,
                   "device":torch.device('cuda:1'), "train_iteration":256,
                   "alpha":0.75, "lambda_u":75, "T":0.5, "ema_decay":0.999,
                   "dataset_name":"cifar10", "substitute_model" : "efficientnet", # wideresnet,efficientnet
                  "black_model":"preactresnet"}
    param_dict["epoch"] = 100 if param_dict["dataset_name"] in ["cifar10", "cifar100"] else 30

    if param_dict["dataset_name"] in ["cifar10", "cifar100"]:
        import models.preact_resnet as BlackModel
    else:
        import models.preact_resnet_mnist as BlackModel
    if param_dict["dataset_name"] == "cifar100":
        black_model = BlackModel.PreActResNet18(num_classes=100).to(param_dict["device"])
    else:
        black_model = BlackModel.PreActResNet18(num_classes=10).to(param_dict["device"])
    black_model.load_state_dict(torch.load("black_model/preactresnet_{}.p".format(param_dict["dataset_name"]))["net"])
    black_model.eval()

    for labeled_data_num in [100, 200, 400, 800, 1600]:
        if labeled_data_num <= 100:
            param_dict["batch_size"] = 64
        else:
            param_dict["batch_size"] = 128
        main(param_dict, black_model, labeled_data_num)

