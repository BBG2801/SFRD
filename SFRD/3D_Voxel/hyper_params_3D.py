### SFRD ###
DIM_IN = {"ModelNet_32": {1: 3}, "ShapeNet_32": {1: 3}}

NUM_LAYERS = {"ModelNet_32": {1: 4}, "ShapeNet_32": {1: 3}}

LAYER_SIZE = {"ModelNet_32": {1: 80}, "ShapeNet_32": {1: 20}}

DIM_OUT = {"ModelNet_32": {1: 1}, "ShapeNet_32": {1: 1}}

W0_INITIAL = {"ModelNet_32": {1: 30}, "ShapeNet_32": {1: 30}}

W0 = {"ModelNet_32": {1: 40}, "ShapeNet_32": {1: 40}}


def load_default(args):

    if args.dim_in == None:
        args.dim_in = DIM_IN[f"{args.dataset}_{args.res}"][args.ipc]

    if args.num_layers == None:
        args.num_layers = NUM_LAYERS[f"{args.dataset}_{args.res}"][args.ipc]

    if args.layer_size == None:
        args.layer_size = LAYER_SIZE[f"{args.dataset}_{args.res}"][args.ipc]

    if args.dim_out == None:
        args.dim_out = DIM_OUT[f"{args.dataset}_{args.res}"][args.ipc]

    if args.w0_initial == None:
        args.w0_initial = W0_INITIAL[f"{args.dataset}_{args.res}"][args.ipc]

    if args.w0 == None:
        args.w0 = W0[f"{args.dataset}_{args.res}"][args.ipc]

    return args