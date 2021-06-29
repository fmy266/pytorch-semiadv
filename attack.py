import torch
from train_substitute_model import create_model
import os
from utils import utils
import attack_method


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


def get_adversary(attack_name, epsilon, device, iter_num=40, restart=10, ord="Linf",
                  is_targeted=False, model_preprocessing=attack_method.nothing, rand_init=True):
    '''
    :param attack_name: option : FGSM, BIM, PGD, Momentum
    :param epsilon: the magnitude of distortion
    :param iter_num: the number of iteration
    :param restart: for pgd
    :param ord: option : L1, L2, Linf
    :param is_targeted: option : True or False
    :param model_preprocessing: for adding drop into the model to enhance transferability of adv
    :return:
    '''
    if attack_name == "FGSM":
        return attack_method.FGSM(epsilon=epsilon, targeted=is_targeted, device=device, ord=ord,
                                  model_preprocessing=model_preprocessing)
    elif attack_name == "BIM":
        return attack_method.BIM(epsilon=epsilon, targeted=is_targeted, device=device, iter_num=iter_num,
                                 ord=ord, model_preprocessing=model_preprocessing)
    elif attack_name == "PGD":
        return attack_method.PGD(epsilon=epsilon, targeted=is_targeted, device=device, iter_num=iter_num,
                                 ord=ord, restart=restart, model_preprocessing=model_preprocessing)
    elif attack_name == "Momentum":
        return attack_method.Momentum(epsilon=epsilon, targeted=is_targeted, device=device, iter_num=iter_num,
                                      ord=ord, model_preprocessing=model_preprocessing, rand_init=rand_init,
                                      restart=restart)


def log(save_params, epsilon, targeted, target_num, attack_method, ord, is_drop, drop_prab, iter_num, ASR):
    info = "{black_model}\t{substitute_model}\t{dataset_name}\t{labeled_data_num}\t\
                {epsilon}\t{targeted}\t{target_num}\t{attack_method}\t{ord}\t{is_drop}\t{drop_prab}\t{iter_num}\t{ASR:.2f}\n".format(
            black_model=save_params["black_model_name"], substitute_model=save_params["substitute_model_name"],
            dataset_name=save_params["dataset_name"], labeled_data_num=save_params["labeled_data_num"],
            epsilon=epsilon, targeted=targeted, target_num=target_num,
            attack_method=attack_method, ord=ord,
            is_drop=is_drop, drop_prab=drop_prab, iter_num=iter_num, ASR=ASR
        )
    print(info)
    with open("result.out", "a+") as f:
        f.write(info)


def attack(black_model, substitute_model, save_params, attack_params, is_drop, drop_prab, device):

    epsilon = 0.15 # give your distortion
    ord = "Linf"
    iter_num = 50
    dataloader = utils.get_dataloader(save_params["dataset_name"], "test")
    loss_func = torch.nn.CrossEntropyLoss(reduction="sum")

    adversary = get_adversary("PGD", epsilon=epsilon, ord=ord,
                              iter_num=iter_num, device=device)

    asr = adversary.attack(black_model, substitute_model, loss_func, loader=dataloader)

    log(save_params, targeted=False, target_num=-1, attack_method="PGD", ord=ord, is_drop=is_drop,
        drop_prab=drop_prab, iter_num=iter_num, ASR=asr, epsilon=epsilon)


def main():

    save_params = {"black_model_name": "mobilenet",
                   "substitute_model_name": "wideresnet",
                   "dataset_name": "cifar10", "labeled_data_num": 1600}
    device = torch.device("cuda:0")
    drop_prob = 0.3

    # loading substitute model
    substitute_model = create_model(
        {"dataset_name": dataset_name, "substitute_model": substitute_model_name,
         "device": device})
    substitute_model_path = "{}_{}_{}_{}/checkpoint.pth".format(dataset_name,
                                                                substitute_model_name,
                                                                black_model_name,
                                                                labeled_data_num)
    substitute_model_path = os.path.join("substitute_model_res", substitute_model_path)
    substitute_model.load_state_dict(torch.load(substitute_model_path)["state_dict"])
    substitute_model.eval()

    # loading black model
    black_model = load_black_model(dataset_name, black_model_name)
    black_model_path = "{}_{}.p".format(black_model_name, dataset_name)
    black_model_path = os.path.join("black_model", black_model_path)
    black_model.load_state_dict(torch.load(black_model_path)["net"])
    black_model.to(device)
    black_model.eval()

    # attack after adding drop func
    attack_method.add_drop_func(substitute_model, substitute_model_name, drop_prob, device)(substitute_model)
    attack(black_model, substitute_model, save_params, attack_params[dataset_name], 1, drop_prob, device)


if __name__ == '__main__':
    main()
