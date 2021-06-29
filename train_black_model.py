import torch
from utils import utils

torch.backends.cudnn.benchmark = True

def train_black_model(dataset_name, model_name, device, epoch):

    if dataset_name in ["cifar10","cifar100"]:
        import models.mobilenet as mobilenet
        import models.googlenet as googlenet
        import models.preact_resnet as preact_resnet
        nets = {"mobilenet":mobilenet.MobileNet, "googlenet":googlenet.GoogLeNet, "preactresnet":preact_resnet.PreActResNet18}

    elif dataset_name in ["mnist","fashionmnist"]:
        import models.mobilenet_mnist as mobilenet_mnist
        import models.googlenet_mnist as googlenet_mnist
        import models.preact_resnet_mnist as preact_resnet_mnist
        nets = {"mobilenet":mobilenet_mnist.MobileNet, "googlenet":googlenet_mnist.GoogLeNet, "preactresnet":preact_resnet_mnist.PreActResNet18}

    if dataset_name == "cifar100":
        net = nets[model_name](num_classes=100).to(device)
    else:
        net = nets[model_name]().to(device)

    optimizer = torch.optim.SGD(net.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    loss_func = torch.nn.CrossEntropyLoss()
    best_acc = 0.

    train_loader = utils.get_dataloader(dataset_name, "train")
    test_loader = utils.get_dataloader(dataset_name, "test")

    for epoch in range(1,epoch + 1):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            predict = net(data)
            loss = loss_func(predict, target)
            loss.backward()
            optimizer.step()
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

    train_black_model("cifar10", "mobilenet", torch.device('cuda:0'), 100)
