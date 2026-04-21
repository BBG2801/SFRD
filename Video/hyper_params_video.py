### SFRD ###
DIM_IN = {"miniUCF101": {1: 3, 5: 3}}

NUM_LAYERS = {"miniUCF101": {1: 6, 5: 6}}

LAYER_SIZE = {"miniUCF101": {1: 60, 5: 40}}

DIM_OUT = {"miniUCF101": {1: 3, 5: 3}}

W0_INITIAL = {"miniUCF101": {1: 5, 5: 5}}

W0 = {"miniUCF101": {1: 5, 5: 5}}


def load_default(args):

    if args.dim_in == None:
        args.dim_in = DIM_IN[args.dataset][args.ipc]

    if args.num_layers == None:
        args.num_layers = NUM_LAYERS[args.dataset][args.ipc]

    if args.layer_size == None:
        args.layer_size = LAYER_SIZE[args.dataset][args.ipc]

    if args.dim_out == None:
        args.dim_out = DIM_OUT[args.dataset][args.ipc]

    if args.w0_initial == None:
        args.w0_initial = W0_INITIAL[args.dataset][args.ipc]

    if args.w0 == None:
        args.w0 = W0[args.dataset][args.ipc]

    return args