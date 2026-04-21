import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import numpy as np
import torch
from utils import get_dataset, get_network, get_eval_pool, evaluate_synset, ParamDiffAug, set_seed, save_and_print, get_voxels
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


def save_all_visuals(synset, base_dir, tag, threshold=0.5, ncols=2, instance_id=0):
    os.makedirs(base_dir, exist_ok=True)

    overview_path = os.path.join(base_dir, f"{tag}_per_class.png")
    per_class_dir = os.path.join(base_dir, f"{tag}_each_class")

    synset.save_visualization_per_class(
        save_path=overview_path,
        instance_id=instance_id,
        ncols=ncols,
        threshold=threshold,
    )
    synset.save_visualization_each_class(
        save_dir=per_class_dir,
        instance_id=instance_id,
        threshold=threshold,
    )


def main():
    parser = argparse.ArgumentParser(description='SFRD 3D Distribution Matching')
    parser.add_argument('--method', type=str, default='DM')
    parser.add_argument('--dataset', type=str, default='ModelNet')
    parser.add_argument('--model', type=str, default='Conv3DNet')
    parser.add_argument('--ipc', type=int, default=1)
    parser.add_argument('--eval_mode', type=str, default='S')
    parser.add_argument('--num_exp', type=int, default=1)
    parser.add_argument('--num_eval', type=int, default=1)
    parser.add_argument('--epoch_eval_train', type=int, default=1000)
    parser.add_argument('--Iteration', type=int, default=20000)
    parser.add_argument('--lr_net', type=float, default=0.01)
    parser.add_argument('--batch_real', type=int, default=128)
    parser.add_argument('--batch_train', type=int, default=128)
    parser.add_argument('--dsa_strategy', type=str, default="")
    parser.add_argument('--data_path', type=str, default='../data')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sh_file', type=str, default='run_SFRD.sh')
    parser.add_argument('--FLAG', type=str, default="")
    parser.add_argument('--save_path', type=str, default="./results")

    parser.add_argument('--batch_syn', type=int, default=0)
    parser.add_argument('--dipc', type=int, default=0)
    parser.add_argument('--res', type=int)

    parser.add_argument('--dim_in', type=int)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--layer_size', type=int, default=128)
    parser.add_argument('--dim_out', type=int)
    parser.add_argument('--w0_initial', type=float, default=30.0)
    parser.add_argument('--w0', type=float, default=30.0)
    parser.add_argument('--lr_nf', type=float, default=1e-6)
    parser.add_argument('--epochs_init', type=int, default=5000)
    parser.add_argument('--lr_nf_init', type=float, default=5e-4)

    parser.add_argument('--shared_mode', type=str, default='per_class')
    parser.add_argument('--modulation_type', type=str, default='shift')
    parser.add_argument('--shift_init', type=float, default=0.0)
    parser.add_argument('--latent_std', type=float, default=0.01)

    parser.add_argument('--lr_nf_backbone', type=float, default=None)
    parser.add_argument('--lr_nf_shift', type=float, default=None)
    parser.add_argument('--lr_nf_init_backbone', type=float, default=None)
    parser.add_argument('--lr_nf_init_shift', type=float, default=None)

    parser.add_argument('--train_backbone', type=int, default=1)
    parser.add_argument('--train_latent', type=int, default=1)
    parser.add_argument('--init_batch_per_step', type=int, default=8)
    parser.add_argument('--init_instances_per_epoch', type=int, default=-1)

    parser.add_argument('--vis_it', type=int, default=500)
    parser.add_argument('--vis_threshold', type=float, default=0.5)
    parser.add_argument('--vis_instance_id', type=int, default=0)

    parser.add_argument('--use_relation_distill', type=int, default=1)
    parser.add_argument('--lambda_rel', type=float, default=0.3)
    parser.add_argument('--batch_real_rel', type=int, default=8)
    parser.add_argument('--syn_decode_chunk', type=int, default=16)

    args = parser.parse_args()
    set_seed(args.seed)
    args = load_default(args)

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy in ['none', 'None', ''] else True
    args.use_relation_distill = bool(args.use_relation_distill)

    if args.dataset == "ModelNet":
        args.data_path += "/modelnet40_normal_resampled"
    elif args.dataset == "ShapeNet":
        args.data_path += "/shapenetcore_partanno_segmentation_benchmark_v0_normal"
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")

    sub_save_path_1 = f"{args.dataset}_{args.res}_{args.model}_{args.ipc}ipc_{args.dipc}dipc"
    sub_save_path_2 = (
        f"SFRD_DM#{args.batch_syn}_({args.dim_in},{args.num_layers},{args.layer_size},{args.dim_out})"
        f"_({args.w0_initial},{args.w0})_({args.epochs_init},{args.lr_nf_init:.0e})_{args.lr_nf:.0e}"
        f"_rel{int(args.use_relation_distill)}_lam{args.lambda_rel:.2f}_br{args.batch_real_rel}_chunk{args.syn_decode_chunk}"
    )
    args.save_path = f"{args.save_path}/{sub_save_path_1}/{sub_save_path_2}#{args.FLAG}"

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(f"{args.save_path}/imgs", exist_ok=True)

    if args.sh_file is not None:
        src_sh = f"./scripts/{args.sh_file}"
        if os.path.isfile(src_sh):
            shutil.copy(src_sh, f"{args.save_path}/{args.sh_file}")

    args.log_path = f"{args.save_path}/log.txt"

    eval_it_pool = np.arange(0, args.Iteration + 1, 500).tolist() if args.eval_mode in ['S', 'SS'] else [args.Iteration]
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(
        args.dataset, args.data_path, resolution=args.res
    )
    args.channel, args.im_size, args.num_classes, args.mean, args.std = channel, im_size, num_classes, mean, std

    args.dim_in = 3
    args.dim_out = channel
    args.train_backbone = bool(args.train_backbone)
    args.train_latent = bool(args.train_latent)

    if args.lr_nf_backbone is None:
        args.lr_nf_backbone = args.lr_nf
    if args.lr_nf_shift is None:
        args.lr_nf_shift = args.lr_nf * 5
    if args.lr_nf_init_backbone is None:
        args.lr_nf_init_backbone = args.lr_nf_init
    if args.lr_nf_init_shift is None:
        args.lr_nf_init_shift = args.lr_nf_init * 5

    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    for exp in range(args.num_exp):
        save_and_print(args.log_path, f'\n================== Exp {exp} ==================\n')
        save_and_print(args.log_path, f'Hyper-parameters: {args.__dict__}')

        voxels_all = []
        labels_all = []
        indices_class = [[] for _ in range(num_classes)]

        save_and_print(args.log_path, "BUILDING DATASET")
        for i in tqdm(range(len(dst_train))):
            sample = dst_train[i]
            voxels_all.append(torch.unsqueeze(sample[0], dim=0))
            labels_all.append(sample[1])

        for i, lab in enumerate(labels_all):
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

        try:
            save_all_visuals(
                synset=synset,
                base_dir=f"{args.save_path}/imgs",
                tag="init",
                threshold=args.vis_threshold,
                ncols=2,
                instance_id=args.vis_instance_id,
            )
        except Exception as e:
            save_and_print(args.log_path, f"[Visualization Warning] Failed to save init voxel plot from main: {e}")

        best_acc = {m: 0 for m in model_eval_pool}
        best_std = {m: 0 for m in model_eval_pool}

        save_and_print(args.log_path, 'training begins')

        for it in range(args.Iteration + 1):
            save_this_it = False

            if it in eval_it_pool:
                for model_eval in model_eval_pool:
                    save_and_print(
                        args.log_path,
                        '-------------------------\n'
                        f'Evaluation\nmodel_train = {args.model}, model_eval = {model_eval}, iteration = {it}'
                    )

                    accs_test = []
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)
                        voxel_syn_eval, label_syn_eval = synset.get(need_copy=True)
                        _, _, acc_test = evaluate_synset(
                            it_eval, net_eval, voxel_syn_eval, label_syn_eval, testloader, args
                        )
                        accs_test.append(acc_test)

                    accs_test = np.array(accs_test)
                    acc_test_mean = np.mean(accs_test)
                    acc_test_std = np.std(accs_test)

                    if acc_test_mean > best_acc[model_eval]:
                        best_acc[model_eval] = acc_test_mean
                        best_std[model_eval] = acc_test_std
                        save_this_it = True
                        torch.save(
                            {"best_acc": best_acc, "best_std": best_std},
                            f"{args.save_path}/best_performance.pt"
                        )

                    save_and_print(
                        args.log_path,
                        f'Evaluate {len(accs_test)} random {model_eval}, '
                        f'mean = {acc_test_mean:.4f} std = {acc_test_std:.4f}\n-------------------------'
                    )
                    save_and_print(args.log_path, f"{it:5d} | Accuracy_frac/{model_eval}: {acc_test_mean:.4f}")
                    save_and_print(args.log_path, f"{it:5d} | Accuracy_pct/{model_eval}: {acc_test_mean * 100:.2f}")
                    save_and_print(args.log_path, f"{it:5d} | Max_Accuracy_frac/{model_eval}: {best_acc[model_eval]:.4f}")
                    save_and_print(args.log_path, f"{it:5d} | Max_Accuracy_pct/{model_eval}: {best_acc[model_eval] * 100:.2f}")
                    save_and_print(args.log_path, f"{it:5d} | Std/{model_eval}: {acc_test_std}")
                    save_and_print(args.log_path, f"{it:5d} | Max_Std/{model_eval}: {best_std[model_eval]}")

                    del voxel_syn_eval, label_syn_eval

                if save_this_it:
                    synset.save(name=f"SFRD_DM_{args.ipc}ipc#synset_best.pt")
                    try:
                        save_all_visuals(
                            synset=synset,
                            base_dir=f"{args.save_path}/imgs",
                            tag=f"best_iter_{it:05d}",
                            threshold=args.vis_threshold,
                            ncols=2,
                            instance_id=args.vis_instance_id,
                        )
                    except Exception as e:
                        save_and_print(args.log_path, f"[Visualization Warning] Failed to save best voxel plot: {e}")

            net = get_network(args.model, channel, num_classes, im_size).to(args.device)
            net.train()
            for param in list(net.parameters()):
                param.requires_grad = False

            embed = net.module.embed if torch.cuda.device_count() > 1 else net.embed

            base_loss = torch.tensor(0.0, device=args.device)
            for c in range(num_classes):
                vox_real = get_voxels(voxels_all, indices_class, c, args.batch_real)

                if args.batch_syn > 0:
                    indices = np.random.permutation(
                        range(c * synset.num_per_class, (c + 1) * synset.num_per_class)
                    )[:args.batch_syn]
                else:
                    indices = range(c * synset.num_per_class, (c + 1) * synset.num_per_class)

                vox_syn, lab_syn = synset.get(indices=indices)

                if args.dsa:
                    raise NotImplementedError(
                        "Current DiffAugment in utils.py is 2D-oriented. "
                        "For voxel experiments, keep dsa_strategy empty/none."
                    )

                output_real = embed(vox_real).detach()
                output_syn = embed(vox_syn)
                base_loss = base_loss + torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0)) ** 2)

            base_loss = base_loss / num_classes

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

            if it % 10 == 0:
                if relation_stats is None:
                    save_and_print(args.log_path, f'iter = {it:04d}, total_loss = {total_loss.item():.4f}, base_loss = {base_loss.item():.4f}')
                else:
                    save_and_print(
                        args.log_path,
                        f'iter = {it:04d}, total_loss = {total_loss.item():.4f}, '
                        f'base_loss = {base_loss.item():.4f}, relation_loss = {relation_loss.item():.4f}, '
                        f'lambda_rel = {args.lambda_rel:.3f}, centroid_mse = {relation_stats.centroid_mse:.6f}'
                    )

            if args.vis_it > 0 and (it % args.vis_it == 0):
                try:
                    save_all_visuals(
                        synset=synset,
                        base_dir=f"{args.save_path}/imgs",
                        tag=f"iter_{it:05d}",
                        threshold=args.vis_threshold,
                        ncols=2,
                        instance_id=args.vis_instance_id,
                    )
                except Exception as e:
                    save_and_print(args.log_path, f"[Visualization Warning] Failed to save iter voxel plot: {e}")


if __name__ == '__main__':
    main()
