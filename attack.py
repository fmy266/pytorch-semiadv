import torch
from advertorch import attacks
from torchvision import transforms, datasets
from train_substitute_model import create_model
import os

def get_accuracy(model, dataloader, device):
    model.to(device)
    correct_num, num = 0, 0
    for data, label in dataloader:
        data, label = data.to(device), label.to(device)
        correct_num += (model(data).max(dim = 1)[1] == label).sum().item()
        num += label.size()[0]
    return correct_num / num * 100.


def get_drop_hook(prob):
    def drop_hook_func(layer, input, output):  # 0.3 prob means 30% value to be 0
        return output * (torch.rand(output.size()[1]) >= prob).view(1, -1, 1, 1)
    return drop_hook_func

def load_black_model(dataset_name, model_name):
    if dataset_name in ["cifar10", "cifar100"]:
        import models.mobilenet as mobilenet
        import models.googlenet as googlenet
        import models.preact_resnet as preact_resnet
        nets = {"mobilenet": mobilenet.MobileNet, "googlenet": googlenet.GoogLeNet,
                "preactresnet": preact_resnet.PreActResNet18}

    elif dataset_name in ["mnist", "fashionmnist"]:
        import models.mobilenet_mnist as mobilenet_mnist
        import models.googlenet_mnist as googlenet_mnist
        import models.preact_resnet_mnist as preact_resnet_mnist
        nets = {"mobilenet": mobilenet_mnist.MobileNet, "googlenet": googlenet_mnist.GoogLeNet,
                "preactresnet": preact_resnet_mnist.PreActResNet18}

    if dataset_name == "cifar100":
        net = nets[model_name](num_classes=100)
    else:
        net = nets[model_name]()

    return net

def get_adversary(attack_method, model, epsilon, params, is_targeted=False):
    loss_function = torch.nn.CrossEntropyLoss(reduction="sum")
    if attack_method == "FGSM":
        return attacks.GradientSignAttack(predict=model, loss_fn=loss_function,
                                          clip_min=params["clip_min"], clip_max=params["clip_max"],
                                          targeted=is_targeted, eps=0.15)
    elif attack_method == "BIM_L2":
        return attacks.L2BasicIterativeAttack(predict=model, loss_fn=loss_function,
                                              clip_min=params["clip_min"], clip_max=params["clip_max"],
                                              targeted=is_targeted, eps=epsilon * params["epsilon"],
                                              nb_iter=50)
    elif attack_method == "BIM_Linf":
        return attacks.LinfBasicIterativeAttack(predict=model, loss_fn=loss_function,
                                                clip_min=params["clip_min"], clip_max=params["clip_max"],
                                                targeted=is_targeted, eps=epsilon * params["epsilon"],
                                                nb_iter=50)
    elif attack_method == "PGD_L2":
        return attacks.L2PGDAttack(predict=model, loss_fn=loss_function,
                                   clip_min=params["clip_min"], clip_max=params["clip_max"],
                                   targeted=is_targeted, eps=epsilon * params["epsilon"],
                                   nb_iter=50)
    elif attack_method == "PGD_Linf":
        return attacks.LinfPGDAttack(predict=model, loss_fn=loss_function,
                                     clip_min=params["clip_min"], clip_max=params["clip_max"],
                                     targeted=is_targeted, eps=epsilon * params["epsilon"],
                                     nb_iter=50)
    elif attack_method == "CW":
        return attacks.CarliniWagnerL2Attack(predict=model, num_classes=10,
                                             clip_min=params["clip_min"], clip_max=params["clip_max"],
                                             targeted=is_targeted)
    elif attack_method == "Momen_L2":
        return attacks.L2MomentumIterativeAttack(predict=model, loss_fn=loss_function,
                                                 clip_min=params["clip_min"], clip_max=params["clip_max"],
                                                 targeted=is_targeted, eps=epsilon * params["epsilon"],
                                                 nb_iter=50)
    elif attack_method == "Momen_Linf":
        return attacks.LinfMomentumIterativeAttack(predict=model, loss_fn=loss_function,
                                                   clip_min=params["clip_min"], clip_max=params["clip_max"],
                                                   targeted=is_targeted, eps=epsilon * params["epsilon"],
                                                   nb_iter=50)

def get_dataloader(dataset_name):
    if dataset_name in ["cifar10", "cifar100"]:
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        if dataset_name == "cifar10":
            dataset = datasets.CIFAR10('..//data', train=False,
                                       transform=transform)
        elif dataset_name == "cifar100":
            dataset = datasets.CIFAR100('..//data', train=False,
                                        transform=transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=256)

    elif dataset_name in ["mnist", "fashionmnist"]:
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ])
        if dataset_name == "mnist":
            dataset = datasets.MNIST('..//data', train=False,
                                     transform=transform)
        if dataset_name == "fashionmnist":
            dataset = datasets.FashionMNIST('..//data', train=False,
                                            transform=transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=256, num_workers=4)

    return loader

def attack(black_model, substitute_model, save_params, attack_params):
    # without drop
    black_model = black_model.to(attack_params["device"])
    substitute_model = substitute_model.to(attack_params["device"])
    for attack_method in ["FGSM", "BIM_L2", "BIM_Linf", "PGD_L2", "PGD_Linf", "Momen_L2", "Monen_Linf"]: # "CW",
        # untargeted attack
        adversary = get_adversary(attack_method, substitute_model, save_params["epsilon"], attack_params)
        dataloader = get_dataloader(save_params["dataset_name"])
        mis_num, num = 0, 0
        for data, label in dataloader:
            data, label = data.to(attack_params["device"]), label.to(attack_params["device"])
            adv_examples = adversary.perturb(data, y=None)
            mis_num += (label != black_model(adv_examples).max(dim=1)[1]).sum().item()
            num += label.size()[0]
        constraint = attack_method.split("_")[1] if len(attack_method.split("_")) == 2 else -1
        with open("result.out", "a+") as f:
            f.write("{black_model}\t{substitute_model}\t{dataset_name}\t{labeled_data_num}\t\
                    {epsilon}\t{targeted}\t{target_num}\t{attack_method}\t{constraint}\t{is_drop}\t{ASR:.2f}\n".format(
                black_model=save_params["black_model_name"], substitute_model=save_params["substitute_model_name"],
                dataset_name=save_params["dataset_name"], labeled_data_num=save_params["labeled_data_num"],
                epsilon=save_params["epsilon"], targeted=-1, target_num=-1,
                attack_method=attack_method, constraint=constraint, is_drop=-1, ASR=mis_num / num * 100
            ))
        # targeted attack
        adversary = get_adversary(attack_method, substitute_model, save_params["epsilon"], attack_params, is_targeted = True)
        dataloader = get_dataloader(save_params["dataset_name"])
        for target in list(range(10)):
            mis_num, num = 0, 0
            for data, label in dataloader:
                data, label = data.to(attack_params["device"]), label.to(attack_params["device"])
                adv_examples = adversary.perturb(data, y= torch.full((data.size()[0],), target, device = attack_params["device"], dtype = torch.long))
                mis_num += (target == black_model(adv_examples).max(dim=1)[1]).sum().item()
                num += label.size()[0]
            constraint = attack_method.split("_")[1] if len(attack_method.split("_")) == 2 else -1
            with open("result.out", "a+") as f:
                f.write("{black_model}\t{substitute_model}\t{dataset_name}\t{labeled_data_num}\t\
                        {epsilon}\t{targeted}\t{target_num}\t{attack_method}\t{constraint}\t{is_drop}\t{ASR:.2f}\n".format(
                    black_model=save_params["black_model_name"], substitute_model=save_params["substitute_model_name"],
                    dataset_name=save_params["dataset_name"], labeled_data_num=save_params["labeled_data_num"],
                    epsilon=save_params["epsilon"], targeted=1, target_num=target,
                    attack_method=attack_method, constraint=constraint, is_drop=-1, ASR=mis_num / num * 100
                ))

    # only adding it into convolutional layers
    if save_params["substitute_model_name"] == "wideresnet":
        for name, layer in substitute_model.named_modules():
            if "block3" in name and "conv" in name and "Shortcut" not in name:
                layer.register_forward_hook(get_drop_hook(0.5))
    elif save_params["substitute_model_name"] == "efficientnet":
        layer_ls = ["layers.{}.conv{}".format(layer_num, conv_num) for layer_num in [13, 14, 15] for conv_num in [1, 2, 3]]
        for name, layer in substitute_model.named_modules():
            if name in layer_ls:
                layer.register_forward_hook(get_drop_hook(0.5))

    for attack_method in ["FGSM", "BIM_L2", "BIM_Linf", "PGD_L2", "PGD_Linf", "Momen_L2", "Monen_Linf"]: # "CW",
        adversary = get_adversary(attack_method, substitute_model, save_params["epsilon"], attack_params)
        dataloader = get_dataloader(save_params["dataset_name"])
        mis_num, num = 0, 0
        for data, label in dataloader:
            data, label = data.to(attack_params["device"]), label.to(attack_params["device"])
            adv_examples = adversary.perturb(data, y=None)
            mis_num += (label != black_model(adv_examples).max(dim=1)[1]).sum().item()
            num += label.size()[0]
        constraint = attack_method.split("_")[1] if len(attack_method.split("_")) == 2 else -1
        with open("result.out", "a+") as f:
            f.write("{black_model}\t{substitute_model}\t{dataset_name}\t{labeled_data_num}\t\
                    {epsilon}\t{targeted}\t{target_num}\t{attack_method}\t{constraint}\t{is_drop}\t{ASR:.2f}\n".format(
                black_model=save_params["black_model_name"], substitute_model=save_params["substitute_model_name"],
                dataset_name=save_params["dataset_name"], labeled_data_num=save_params["labeled_data_num"],
                epsilon=save_params["epsilon"], targeted=-1, target_num=-1,
                attack_method=attack_method, constraint=constraint, is_drop=1, ASR=mis_num / num * 100
            ))

        # targeted attack
        adversary = get_adversary(attack_method, substitute_model, save_params["epsilon"], attack_params, is_targeted = True)
        dataloader = get_dataloader(save_params["dataset_name"])
        for target in list(range(10)):
            mis_num, num = 0, 0
            for data, label in dataloader:
                data, label = data.to(attack_params["device"]), label.to(attack_params["device"])
                adv_examples = adversary.perturb(data, y= torch.full((data.size()[0],), target, device = attack_params["device"], dtype = torch.long))
                mis_num += (target == black_model(adv_examples).max(dim=1)[1]).sum().item()
                num += label.size()[0]
            constraint = attack_method.split("_")[1] if len(attack_method.split("_")) == 2 else -1
            with open("result.out", "a+") as f:
                f.write("{black_model}\t{substitute_model}\t{dataset_name}\t{labeled_data_num}\t\
                        {epsilon}\t{targeted}\t{target_num}\t{attack_method}\t{constraint}\t{is_drop}\t{ASR:.2f}\n".format(
                    black_model=save_params["black_model_name"], substitute_model=save_params["substitute_model_name"],
                    dataset_name=save_params["dataset_name"], labeled_data_num=save_params["labeled_data_num"],
                    epsilon=save_params["epsilon"], targeted=1, target_num=target,
                    attack_method=attack_method, constraint=constraint, is_drop=1, ASR=mis_num / num * 100
                ))

def main():

    #     "cifar10": {"clip_max": (1 - 0.45) / 0.23, "clip_min": (0 - 0.45) / 0.23, "epsilon": 0.017,
    #                 "device": torch.device("cuda:1")},
    #     "cifar100": {"clip_max": (1 - 0.45) / 0.23, "clip_min": (0 - 0.45) / 0.23, "epsilon": 0.017,
    #                  "device": torch.device("cuda:1")},

    attack_params = {
        "cifar10": {"clip_max": 0., "clip_min": 1., "epsilon": 0.004,
                    "device": torch.device("cuda:0")},
        "cifar100": {"clip_max": 0., "clip_min": 1., "epsilon": 0.004,
                     "device": torch.device("cuda:1")},
        "mnist": {"clip_max": (1 - 0.13) / 0.31, "clip_min": (0 - 0.13) / 0.31, "epsilon": 0.013,
                      "device": torch.device("cuda:1")},
        "fashionmnist": {"clip_max": (1 - 0.13) / 0.31, "clip_min": (0 - 0.13) / 0.31, "epsilon": 0.013,
                             "device": torch.device("cuda:1")}
    }

    black_model_name_ls = ["mobilenet", "googlenet", "preactresnet"] # , "googlenet"
    substitute_model_name_ls = ["wideresnet", "efficientnet"] #
    dataset_name_ls = ["mnist", "fashionmnist", "cifar10", "cifar100"] #
    labeled_data_num_ls = [1600, 800, 400, 200, 100] # 100, 200, 400, 800,
    epsilon_ls = [1., 2., 4., 8.] #

    for black_model_name in black_model_name_ls:
        for substitute_model_name in substitute_model_name_ls:
            for dataset_name in dataset_name_ls:
                for labeled_data_num in labeled_data_num_ls:
                    for epsilon in epsilon_ls:
                        save_params = {"black_model_name": black_model_name,
                                       "substitute_model_name": substitute_model_name,
                                       "dataset_name": dataset_name, "labeled_data_num": labeled_data_num,
                                       "epsilon": epsilon}

                        # loading substitute model
                        substitute_model = create_model(
                            {"dataset_name": dataset_name, "substitute_model": substitute_model_name,
                             "device": attack_params[save_params["dataset_name"]]["device"]})
                        substitute_model_path = "{}_{}_{}_{}/checkpoint.pth".format(dataset_name, substitute_model_name,
                                                                     black_model_name, labeled_data_num)
                        substitute_model_path = os.path.join("substitute_model_res", substitute_model_path)
                        substitute_model.load_state_dict(torch.load(substitute_model_path), False)

                        # loading black model
                        black_model = load_black_model(dataset_name, black_model_name)
                        black_model_path = "{}_{}.p".format(black_model_name, dataset_name)
                        black_model_path = os.path.join("black_model", black_model_path)
                        black_model.load_state_dict(torch.load(black_model_path), False)

                        # attack
                        attack(black_model, substitute_model, save_params, attack_params[save_params["dataset_name"]])

                        # test accuracy
                        # acc = get_accuracy(substitute_model, get_dataloader(dataset_name),
                        #              attack_params[save_params["dataset_name"]]["device"])
                        # print("{model_name}\t{black_model_name}\t{dataset_name}\t{label_num}\t{acc:.2f}".format(
                        #     model_name = substitute_model_name,
                        #     black_model_name = black_model_name, dataset_name = dataset_name,
                        #     label_num = labeled_data_num, acc = acc
                        # ))


if __name__ == '__main__':
    main()
