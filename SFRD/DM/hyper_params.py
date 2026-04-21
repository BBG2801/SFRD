### SFRD default neural-field hyper-parameters ###
DIM_IN = {
    "CIFAR10_32": {1: 2, 10: 2, 50: 2},
    "CIFAR100_32": {1: 2, 10: 2, 50: 2},
    "ImageNet_128": {1: 2, 10: 2, 50: 2},
    "ImageNet_256": {1: 2},
}

NUM_LAYERS = {
    "CIFAR10_32": {1: 2, 10: 2, 50: 2},
    "CIFAR100_32": {1: 2, 10: 2, 50: 2},
    "ImageNet_128": {1: 3, 10: 3, 50: 3},
    "ImageNet_256": {1: 3},
}

LAYER_SIZE = {
    "CIFAR10_32": {1: 6, 10: 6, 50: 20},
    "CIFAR100_32": {1: 10, 10: 15, 50: 30},
    "ImageNet_128": {1: 20, 10: 20, 50: 40},
    "ImageNet_256": {1: 40},
}

DIM_OUT = {
    "CIFAR10_32": {1: 3, 10: 3, 50: 3},
    "CIFAR100_32": {1: 3, 10: 3, 50: 3},
    "ImageNet_128": {1: 3, 10: 3, 50: 3},
    "ImageNet_256": {1: 3},
}

W0_INITIAL = {
    "CIFAR10_32": {1: 30, 10: 30, 50: 30},
    "CIFAR100_32": {1: 30, 10: 30, 50: 30},
    "ImageNet_128": {1: 30, 10: 30, 50: 30},
    "ImageNet_256": {1: 30},
}

W0 = {
    "CIFAR10_32": {1: 10, 10: 10, 50: 10},
    "CIFAR100_32": {1: 10, 10: 10, 50: 10},
    "ImageNet_128": {1: 40, 10: 40, 50: 40},
    "ImageNet_256": {1: 40},
}


def load_default(args):
    key = f"{args.dataset}_{args.res}"

    if args.dim_in is None:
        args.dim_in = DIM_IN[key][args.ipc]

    if args.num_layers is None:
        args.num_layers = NUM_LAYERS[key][args.ipc]

    if args.layer_size is None:
        args.layer_size = LAYER_SIZE[key][args.ipc]

    if args.dim_out is None:
        args.dim_out = DIM_OUT[key][args.ipc]

    if args.w0_initial is None:
        args.w0_initial = W0_INITIAL[key][args.ipc]

    if args.w0 is None:
        args.w0 = W0[key][args.ipc]

    return args
