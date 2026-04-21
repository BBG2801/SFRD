### SFRD base ###
DIM_IN = {
    "CIFAR10_32": {1: 2, 10: 2, 50: 2},
    "CIFAR100_32": {1: 2, 10: 2, 50: 2},
    "ImageNet_128": {1: 2, 10: 2, 50: 2},
}

NUM_LAYERS = {
    "CIFAR10_32": {1: 2, 10: 2, 50: 2},
    "CIFAR100_32": {1: 2, 10: 2, 50: 2},
    "ImageNet_128": {1: 3, 10: 3, 50: 3},
}

LAYER_SIZE = {
    "CIFAR10_32": {1: 6, 10: 6, 50: 20},
    "CIFAR100_32": {1: 10, 10: 15, 50: 30},
    "ImageNet_128": {1: 20, 10: 20, 50: 40},
}

DIM_OUT = {
    "CIFAR10_32": {1: 3, 10: 3, 50: 3},
    "CIFAR100_32": {1: 3, 10: 3, 50: 3},
    "ImageNet_128": {1: 3, 10: 3, 50: 3},
}

W0_INITIAL = {
    "CIFAR10_32": {1: 30, 10: 30, 50: 30},
    "CIFAR100_32": {1: 30, 10: 30, 50: 30},
    "ImageNet_128": {1: 30, 10: 30, 50: 30},
}

W0 = {
    "CIFAR10_32": {1: 10, 10: 10, 50: 10},
    "CIFAR100_32": {1: 10, 10: 10, 50: 10},
    "ImageNet_128": {1: 40, 10: 40, 50: 40},
}


### TM optimization ###
SYN_STEPS = {
    "CIFAR10_32": {1: 60, 10: 60, 50: 60},
    "CIFAR100_32": {1: 60, 10: 60, 50: 60},
    "ImageNet_128": {1: 10, 10: 40, 50: 40},
}

EXPERT_EPOCHS = {
    "CIFAR10_32": {1: 2, 10: 2, 50: 2},
    "CIFAR100_32": {1: 2, 10: 2, 50: 2},
    "ImageNet_128": {1: 2, 10: 2, 50: 2},
}

MAX_START_EPOCH = {
    "CIFAR10_32": {1: 10, 10: 10, 50: 40},
    "CIFAR100_32": {1: 40, 10: 40, 50: 40},
    "ImageNet_128": {1: 10, 10: 20, 50: 20},
}

LR_LR = {
    "CIFAR10_32": {1: 1e-5, 10: 1e-5, 50: 1e-5},
    "CIFAR100_32": {1: 1e-5, 10: 1e-5, 50: 1e-5},
    "ImageNet_128": {1: 1e-6, 10: 1e-5, 50: 1e-5},
}

LR_TEACHER = {
    "CIFAR10_32": {1: 1e-2, 10: 1e-2, 50: 1e-2},
    "CIFAR100_32": {1: 1e-2, 10: 1e-2, 50: 1e-2},
    "ImageNet_128": {1: 1e-2, 10: 1e-2, 50: 1e-2},
}


def load_default(args):
    if args.dim_in is None:
        args.dim_in = DIM_IN[f"{args.dataset}_{args.res}"][args.ipc]

    if args.num_layers is None:
        args.num_layers = NUM_LAYERS[f"{args.dataset}_{args.res}"][args.ipc]

    if args.layer_size is None:
        args.layer_size = LAYER_SIZE[f"{args.dataset}_{args.res}"][args.ipc]

    if args.dim_out is None:
        args.dim_out = DIM_OUT[f"{args.dataset}_{args.res}"][args.ipc]

    if args.w0_initial is None:
        args.w0_initial = W0_INITIAL[f"{args.dataset}_{args.res}"][args.ipc]

    if args.w0 is None:
        args.w0 = W0[f"{args.dataset}_{args.res}"][args.ipc]

    if args.syn_steps is None:
        args.syn_steps = SYN_STEPS[f"{args.dataset}_{args.res}"][args.ipc]

    if args.expert_epochs is None:
        args.expert_epochs = EXPERT_EPOCHS[f"{args.dataset}_{args.res}"][args.ipc]

    if args.max_start_epoch is None:
        args.max_start_epoch = MAX_START_EPOCH[f"{args.dataset}_{args.res}"][args.ipc]

    if args.lr_lr is None:
        args.lr_lr = LR_LR[f"{args.dataset}_{args.res}"][args.ipc]

    if args.lr_teacher is None:
        args.lr_teacher = LR_TEACHER[f"{args.dataset}_{args.res}"][args.ipc]

    return args
