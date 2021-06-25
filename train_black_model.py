import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.cuda.amp as amp

torch.backends.cudnn.benchmark = True

def train_black_model(dataset_name, model_name, device, epoch):

    if dataset_name in ["cifar10","cifar100"]:
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        if dataset_name == "cifar10":
            train_dataset = datasets.CIFAR10('..//data', train=True, download=True,
                                transform=transform)
            test_dataset = datasets.CIFAR10('..//data', train=False,
                                transform=transform)
        elif dataset_name == "cifar100":
            train_dataset = datasets.CIFAR100('..//data', train=True, download=True,
                                transform=transform)
            test_dataset = datasets.CIFAR100('..//data', train=False,
                                transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 256)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 256)

        import models.mobilenet as mobilenet
        import models.googlenet as googlenet
        import models.preact_resnet as preact_resnet
        nets = {"mobilenet":mobilenet.MobileNet, "googlenet":googlenet.GoogLeNet, "preactresnet":preact_resnet.PreActResNet18}

 
    elif dataset_name in ["mnist","fashionmnist"]:
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        if dataset_name == "mnist":
            train_dataset = datasets.MNIST('..//data', train=True, download=True,
                                transform=transform)
            test_dataset = datasets.MNIST('..//data', train=False,
                                transform=transform)
        if dataset_name == "fashionmnist":
            train_dataset = datasets.FashionMNIST('..//data', train=True, download=True,
                                transform=transform)
            test_dataset = datasets.FashionMNIST('..//data', train=False,
                                transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 256, num_workers = 4)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 256, num_workers = 4)

        import models.mobilenet_mnist as mobilenet_mnist
        import models.googlenet_mnist as googlenet_mnist
        import models.preact_resnet_mnist as preact_resnet_mnist
        nets = {"mobilenet":mobilenet_mnist.MobileNet, "googlenet":googlenet_mnist.GoogLeNet, "preactresnet":preact_resnet_mnist.PreActResNet18}

    if dataset_name == "cifar100":
        net = nets[model_name](num_classes=100).to(device)
    else:
        net = nets[model_name]().to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr = 0.01)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    loss_func = torch.nn.CrossEntropyLoss()
    scaler = amp.GradScaler()
    best_acc = 0.

    for epoch in range(1,epoch + 1):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            # with amp.autocast():
            predict = net(data)
            loss = loss_func(predict, target)
            loss.backward()
            optimizer.step()
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

        scheduler.step()

        if epoch % 5 == 0:
            correct, num = 0., 0.
            net.eval()
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                with torch.no_grad():
                    predict = net(data).max(dim = 1)[1]
                    correct += (predict == target).sum().item()
                    num += target.size()[0]
            net.train()
            print("model = {}, dataset = {}, best acc = {:.2f}, cur acc = {:.2f}".format(model_name, dataset_name, best_acc, correct / num * 100))
            if correct / num * 100 > best_acc:
                best_acc = correct / num * 100
                state = {
                    'net': net.state_dict(),
                    'acc': best_acc,
                    'epoch': epoch,
                }
            torch.save(state, './black_model/{}_{}.p'.format(model_name, dataset_name))



if __name__ == '__main__':
    # train_black_model("fashionmnist", "mobilenet", torch.device('cuda:1'), 20)
    # train_black_model("fashionmnist", "googlenet", torch.device('cuda:1'), 20)
    # train_black_model("fashionmnist", "preactresnet", torch.device('cuda:1'), 20)
    # train_black_model("cifar100", "mobilenet", torch.device('cuda:0'), 100)
    # train_black_model("cifar100", "googlenet", torch.device('cuda:1'), 100)
    # train_black_model("cifar100", "preactresnet", torch.device('cuda:0'), 100)
