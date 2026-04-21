import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import numpy as np
import torch
import torch.nn as nn
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug, set_seed, save_and_print, TensorDataset, get_voxels
import time

import shutil
from hyper_params import load_default
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from SFRD import SFRD
try:
    from SynSet.relation_distill_3d import (
        build_frozen_conv3d_extractor,
        InterClassRelationDistillationLoss3D,
        build_relation_stats,
    )
except ImportError:
    from relation_distill_3d import (
        build_frozen_conv3d_extractor,
        InterClassRelationDistillationLoss3D,
        build_relation_stats,
    )
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description='SFRD 3D Gradient Matching')
    parser.add_argument('--method', type=str, default='DC')
    parser.add_argument('--dataset', type=str, default='ModelNet')
    parser.add_argument('--model', type=str, default='Conv3DNet')
    parser.add_argument('--ipc', type=int, default=1)
    parser.add_argument('--eval_mode', type=str, default='S')
    parser.add_argument('--num_exp', type=int, default=1)
    parser.add_argument('--num_eval', type=int, default=5)
    parser.add_argument('--epoch_eval_train', type=int, default=300)
    parser.add_argument('--Iteration', type=int, default=1000)
    parser.add_argument('--lr_net', type=float, default=0.01)
    parser.add_argument('--batch_real', type=int, default=256)
    parser.add_argument('--batch_train', type=int, default=256)
    parser.add_argument('--dsa_strategy', type=str, default="")
    parser.add_argument('--data_path', type=str, default='../data')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sh_file', type=str, default='run_SFRD.sh')
    parser.add_argument('--FLAG', type=str, default="")
    parser.add_argument('--save_path', type=str, default="./results")

    parser.add_argument('--batch_syn', type=int, default=0)
    parser.add_argument('--dipc', type=int, default=0)
    parser.add_argument('--res', type=int)

    parser.add_argument('--dim_in', type=int)
    parser.add_argument('--num_layers', type=int)
    parser.add_argument('--layer_size', type=int)
    parser.add_argument('--dim_out', type=int)
    parser.add_argument('--w0_initial', type=float)
    parser.add_argument('--w0', type=float)
    parser.add_argument('--lr_nf', type=float)
    parser.add_argument('--epochs_init', type=int, default=5000)
    parser.add_argument('--lr_nf_init', type=float, default=5e-4)

    parser.add_argument('--use_relation_distill', type=int, default=1)
    parser.add_argument('--lambda_rel', type=float, default=0.3)
    parser.add_argument('--batch_real_rel', type=int, default=8)
    parser.add_argument('--syn_decode_chunk', type=int, default=16)

    args = parser.parse_args()
    set_seed(args.seed)
    args = load_default(args)

    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True
    args.use_relation_distill = bool(args.use_relation_distill)

    if args.dataset == "ModelNet":
        args.data_path += "/modelnet40_normal_resampled"
    elif args.dataset == "ShapeNet":
        args.data_path += "/shapenetcore_partanno_segmentation_benchmark_v0_normal"
    else:
        save_and_print(args.log_path, f"Invalid dataset: {args.dataset}")
        exit()

    sub_save_path_1 = f"{args.dataset}_{args.res}_{args.model}_{args.ipc}ipc_{args.dipc}dipc"
    sub_save_path_2 = (
        f"SFRD_DC#{args.batch_syn}_({args.dim_in},{args.num_layers},{args.layer_size},{args.dim_out})"
        f"_({args.w0_initial},{args.w0})_({args.epochs_init},{args.lr_nf_init:.0e})_{args.lr_nf:.0e}"
        f"_rel{int(args.use_relation_distill)}_lam{args.lambda_rel:.2f}_br{args.batch_real_rel}_chunk{args.syn_decode_chunk}"
    )

    args.save_path = f"{args.save_path}/{sub_save_path_1}/{sub_save_path_2}#{args.FLAG}"
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    shutil.copy(f"./scripts/{args.sh_file}", f"{args.save_path}/{args.sh_file}")
    args.log_path = f"{args.save_path}/log.txt"

    eval_it_pool = np.arange(0, args.Iteration + 1, 500).tolist() if args.eval_mode in ['S', 'SS'] else [args.Iteration]
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path, resolution=args.res)
    args.channel, args.im_size, args.num_classes, args.mean, args.std = channel, im_size, num_classes, mean, std
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    for exp in range(args.num_exp):
        save_and_print(args.log_path, f'\n================== Exp {exp} ==================\n ')
        save_and_print(args.log_path, f'Hyper-parameters: {args.__dict__}')

        voxels_all = []
        labels_all = []
        indices_class = [[] for c in range(num_classes)]
        save_and_print(args.log_path, "BUILDING DATASET")
        for i in tqdm(range(len(dst_train))):
            sample = dst_train[i]
            voxels_all.append(torch.unsqueeze(sample[0], dim=0))
            labels_all.append(sample[1])

        for i, lab in tqdm(enumerate(labels_all), total=len(labels_all)):
            indices_class[lab].append(i)
        voxels_all = torch.cat(voxels_all, dim=0).to(args.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

        synset = SFRD(args)
        synset.init(voxels_all, labels_all, indices_class)

        relation_criterion = None
        if args.use_relation_distill:
            frozen_extractor = build_frozen_conv3d_extractor(
                channel=channel,
                num_classes=num_classes,
                im_size=im_size,
            ).to(args.device)
            relation_criterion = InterClassRelationDistillationLoss3D(
                feature_extractor=frozen_extractor,
                num_classes=num_classes,
                eps=1e-8,
                ignore_diag=False,
            ).to(args.device)

        criterion = nn.CrossEntropyLoss().to(args.device)

        best_acc = {m: 0 for m in model_eval_pool}
        best_std = {m: 0 for m in model_eval_pool}

        save_and_print(args.log_path, '%s training begins' % get_time())

        for it in range(args.Iteration + 1):
            save_this_it = False

            if it in eval_it_pool and it > 0:
                for model_eval in model_eval_pool:
                    save_and_print(args.log_path, '-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d' % (args.model, model_eval, it))
                    if args.dsa:
                        args.epoch_eval_train = 1000
                        args.dc_aug_param = None
                        save_and_print(args.log_path, f'DSA augmentation strategy: {args.dsa_strategy}')
                        save_and_print(args.log_path, f'DSA augmentation parameters: {args.dsa_param.__dict__}')
                    else:
                        args.dc_aug_param = get_daparam(args.dataset, args.model, model_eval, args.ipc)
                        save_and_print(args.log_path, f'DC augmentation parameters: {args.dc_aug_param}')

                    if args.dsa or args.dc_aug_param['strategy'] != 'none':
                        args.epoch_eval_train = 1000
                    else:
                        args.epoch_eval_train = 300

                    accs_test = []
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)
                        voxel_syn_eval, label_syn_eval = synset.get(need_copy=True)
                        _, _, acc_test = evaluate_synset(it_eval, net_eval, voxel_syn_eval, label_syn_eval, testloader, args)
                        accs_test.append(acc_test)
                    accs_test = np.array(accs_test)
                    acc_test_mean = np.mean(accs_test)
                    acc_test_std = np.std(accs_test)
                    if acc_test_mean > best_acc[model_eval]:
                        best_acc[model_eval] = acc_test_mean
                        best_std[model_eval] = acc_test_std
                        save_this_it = True
                        torch.save({"best_acc": best_acc, "best_std": best_std}, f"{args.save_path}/best_performance.pt")
                    save_and_print(args.log_path, 'Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------' % (len(accs_test), model_eval, acc_test_mean, acc_test_std))
                    save_and_print(args.log_path, f"{args.save_path}")
                    save_and_print(args.log_path, f"{it:5d} | Accuracy/{model_eval}: {acc_test_mean}")
                    save_and_print(args.log_path, f"{it:5d} | Max_Accuracy/{model_eval}: {best_acc[model_eval]}")
                    save_and_print(args.log_path, f"{it:5d} | Std/{model_eval}: {acc_test_std}")
                    save_and_print(args.log_path, f"{it:5d} | Max_Std/{model_eval}: {best_std[model_eval]}")
                    del voxel_syn_eval, label_syn_eval

                if save_this_it:
                    synset.save(name=f"SFRD_DC_{args.ipc}ipc#synset_best.pt")

            net = get_network(args.model, channel, num_classes, im_size).to(args.device)
            net.train()
            net_parameters = list(net.parameters())
            optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)
            optimizer_net.zero_grad()
            args.dc_aug_param = None

            for ol in range(args.outer_loop):
                BN_flag = False
                BNSizePC = 16
                for module in net.modules():
                    if 'BatchNorm' in module._get_name():
                        BN_flag = True
                if BN_flag:
                    vox_real = torch.cat([get_voxels(voxels_all, indices_class, c, BNSizePC) for c in range(num_classes)], dim=0)
                    net.train()
                    _ = net(vox_real)
                    for module in net.modules():
                        if 'BatchNorm' in module._get_name():
                            module.eval()

                base_loss = torch.tensor(0.0).to(args.device)
                for c in range(num_classes):
                    vox_real = get_voxels(voxels_all, indices_class, c, args.batch_real)
                    lab_real = torch.ones((vox_real.shape[0],), device=args.device, dtype=torch.long) * c

                    if args.batch_syn > 0:
                        indices = np.random.permutation(range(c * synset.num_per_class, (c + 1) * synset.num_per_class))[:args.batch_syn]
                    else:
                        indices = range(c * synset.num_per_class, (c + 1) * synset.num_per_class)

                    vox_syn, lab_syn = synset.get(indices=indices)

                    if args.dsa:
                        seed = int(time.time() * 1000) % 100000
                        vox_real = DiffAugment(vox_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        vox_syn = DiffAugment(vox_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                    output_real = net(vox_real)
                    loss_real = criterion(output_real, lab_real)
                    gw_real = torch.autograd.grad(loss_real, net_parameters)
                    gw_real = list((_.detach().clone() for _ in gw_real))

                    output_syn = net(vox_syn)
                    loss_syn = criterion(output_syn, lab_syn)
                    gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

                    base_loss = base_loss + match_loss(gw_syn, gw_real, args)

                relation_loss = torch.tensor(0.0, device=args.device)
                relation_stats = None
                if args.use_relation_distill:
                    relation_loss, relation_details = relation_criterion.forward_from_sfrd(
                        synset=synset,
                        voxels_all=voxels_all,
                        indices_class=indices_class,
                        batch_real_per_class=args.batch_real_rel,
                        syn_decode_chunk=args.syn_decode_chunk if args.syn_decode_chunk > 0 else None,
                        return_details=True,
                        with_replacement=False,
                    )
                    relation_stats = build_relation_stats(relation_details, relation_loss)
                    total_loss = (1.0 - args.lambda_rel) * base_loss + args.lambda_rel * relation_loss
                else:
                    total_loss = base_loss

                synset.optim_zero_grad()
                total_loss.backward()
                synset.optim_step()

                if ol == args.outer_loop - 1:
                    break

                voxel_syn_train, label_syn_train = synset.get(need_copy=True)
                dst_syn_train = TensorDataset(voxel_syn_train, label_syn_train)
                trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=args.batch_train, shuffle=True, num_workers=0)
                for il in range(args.inner_loop):
                    epoch('train', trainloader, net, optimizer_net, criterion, args)

            if it % 10 == 0:
                if relation_stats is None:
                    save_and_print(args.log_path, '%s iter = %04d, total_loss = %.4f, base_loss = %.4f' % (get_time(), it, total_loss.item(), base_loss.item()))
                else:
                    save_and_print(
                        args.log_path,
                        '%s iter = %04d, total_loss = %.4f, base_loss = %.4f, relation_loss = %.4f, lambda_rel = %.3f, centroid_mse = %.6f'
                        % (get_time(), it, total_loss.item(), base_loss.item(), relation_loss.item(), args.lambda_rel, relation_stats.centroid_mse)
                    )

if __name__ == '__main__':
    main()
