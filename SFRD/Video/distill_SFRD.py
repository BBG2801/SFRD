import os
import argparse
import numpy as np
import torch
from tqdm import trange
import warnings
import shutil

warnings.filterwarnings("ignore", category=DeprecationWarning)

from hyper_params import load_default
from SFRD import SFRD
from utils import (
    get_dataset,
    get_network,
    get_eval_pool,
    evaluate_synset_nf,
    get_time,
    set_seed,
    save_and_print,
    get_videos,
)
try:
    from relation_distill_video import (
        build_frozen_video_extractor,
        InterClassRelationDistillationLossVideo,
        build_relation_stats,
    )
except ImportError:
    from SynSet.relation_distill_video import (
        build_frozen_video_extractor,
        InterClassRelationDistillationLossVideo,
        build_relation_stats,
    )


def cache_full_dataset(dst_train, log_path):
    save_and_print(log_path, "Caching full training dataset into RAM")
    video_all = []
    label_all = []

    for i in trange(len(dst_train)):
        video, label = dst_train[i]
        video_all.append(video)
        label_all.append(label)

    video_all = torch.stack(video_all)
    save_and_print(log_path, f"first training sample shape = {tuple(video_all[0].shape)}")
    label_all = torch.tensor(label_all)
    save_and_print(log_path, f"video_all shape = {tuple(video_all.shape)}, label_all shape = {tuple(label_all.shape)}")

    return video_all, label_all


def main(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.use_relation_distill = bool(args.use_relation_distill)

    eval_it_pool = np.arange(args.startIt, args.Iteration + 1, args.eval_it).tolist()

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(
        args.dataset, args.data_path
    )
    args.channel, args.im_size, args.num_classes, args.mean, args.std = channel, im_size, num_classes, mean, std

    if args.preload:
        save_and_print(args.log_path, "Preloading dataset")
    else:
        save_and_print(args.log_path, "preload is off, but video distillation needs full cached training tensors. Caching anyway.")

    video_all, label_all = cache_full_dataset(dst_train, args.log_path)

    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)
    args.distributed = torch.cuda.device_count() > 1

    save_and_print(args.log_path, "=" * 50)
    save_and_print(args.log_path, f'Hyper-parameters: {args.__dict__}')

    indices_class = [[] for _ in range(num_classes)]
    save_and_print(args.log_path, "BUILDING DATASET INDEX")
    for i, lab in enumerate(label_all.tolist()):
        indices_class[lab].append(i)

    synset = SFRD(args)
    synset.init(video_all, label_all, indices_class)

    relation_criterion = None
    if args.use_relation_distill:
        frozen_extractor = build_frozen_video_extractor(
            channel=channel,
            num_classes=num_classes,
            im_size=im_size,
            frames=args.frames,
        ).to(args.device)
        relation_criterion = InterClassRelationDistillationLossVideo(
            feature_extractor=frozen_extractor,
            num_classes=num_classes,
            eps=1e-8,
            ignore_diag=False,
        ).to(args.device)

    syn_lr = torch.tensor(0.01)

    best_acc = {m: 0 for m in model_eval_pool}
    best_std = {m: 0 for m in model_eval_pool}

    save_and_print(args.log_path, f'{get_time()} training begins')

    for it in range(0, args.Iteration + 1):
        save_this_it = False

        if it in eval_it_pool and it >= 0:
            for model_eval in model_eval_pool:
                save_and_print(
                    args.log_path,
                    '-------------------------\n'
                    f'Evaluation\nmodel_train = {args.model}, model_eval = {model_eval}, iteration = {it}'
                )

                accs_test = []
                for it_eval in range(args.num_eval):
                    net_eval = get_network(
                        model_eval, channel, num_classes, im_size, frames=args.frames
                    ).to(args.device)

                    args.lr_net = syn_lr.detach()
                    _, _, acc_test = evaluate_synset_nf(
                        it_eval, net_eval, synset, testloader, args, mode='none'
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
                    f'Evaluate {len(accs_test)} random {model_eval}, mean = {acc_test_mean:.4f} std = {acc_test_std:.4f}\n'
                    '-------------------------'
                )
                save_and_print(args.log_path, f"{args.save_path}")
                save_and_print(args.log_path, f"{it:5d} | Accuracy/{model_eval}: {acc_test_mean}")
                save_and_print(args.log_path, f"{it:5d} | Max_Accuracy/{model_eval}: {best_acc[model_eval]}")
                save_and_print(args.log_path, f"{it:5d} | Std/{model_eval}: {acc_test_std}")
                save_and_print(args.log_path, f"{it:5d} | Max_Std/{model_eval}: {best_std[model_eval]}")

            save_name = os.path.join(f"{args.save_path}/imgs", f"syn_{it}.png")
            frame_ids = synset.save_visualization(save_name)
            save_and_print(args.log_path, f"Saved synthetic video strips to {save_name}, frame_ids={frame_ids}")

            if save_this_it:
                synset.save(name=f"SFRD_DM_{args.ipc}ipc#synset_best.pt")

        net = get_network(args.model, channel, num_classes, im_size, frames=args.frames).to(args.device)
        net.train()
        for param in list(net.parameters()):
            param.requires_grad = False

        embed = net.module.embed if args.distributed else net.embed

        base_loss = torch.tensor(0.0).to(args.device)
        for c in range(0, num_classes):
            vid_real = get_videos(video_all, indices_class, c, args.batch_real).to(args.device)

            if args.batch_syn > 0:
                indices = np.random.permutation(
                    range(c * synset.num_per_class, (c + 1) * synset.num_per_class)
                )[:args.batch_syn]
            else:
                indices = range(c * synset.num_per_class, (c + 1) * synset.num_per_class)

            vid_syn, lab_syn = synset.get(indices=indices)

            output_real = embed(vid_real).detach()
            output_syn = embed(vid_syn)

            base_loss = base_loss + torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0)) ** 2)

        base_loss = base_loss / num_classes

        relation_loss = torch.tensor(0.0, device=args.device)
        relation_stats = None
        if args.use_relation_distill:
            relation_loss, relation_details = relation_criterion.forward_from_sfrd(
                synset=synset,
                video_all=video_all,
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
                save_and_print(args.log_path, f'{get_time()} iter = {it:04d}, total_loss = {total_loss.item():.4f}, base_loss = {base_loss.item():.4f}')
            else:
                save_and_print(
                    args.log_path,
                    f'{get_time()} iter = {it:04d}, total_loss = {total_loss.item():.4f}, '
                    f'base_loss = {base_loss.item():.4f}, relation_loss = {relation_loss.item():.4f}, '
                    f'lambda_rel = {args.lambda_rel:.3f}, centroid_mse = {relation_stats.centroid_mse:.6f}'
                )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SFRD Video Distillation')
    parser.add_argument('--method', type=str, default='DM')
    parser.add_argument('--dataset', type=str, default='miniUCF101', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet3D', help='model')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')

    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode, check utils.py for more info')
    parser.add_argument('--num_eval', type=int, default=1, help='how many networks to evaluate on')
    parser.add_argument('--eval_it', type=int, default=500, help='how often to evaluate')
    parser.add_argument('--epoch_eval_train', type=int, default=500, help='epochs to train a model with synthetic data')

    parser.add_argument('--Iteration', type=int, default=20000, help='how many distillation steps to perform')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--data_path', type=str, default='./data', help='dataset path')

    parser.add_argument('--preload', action='store_true', help="preload all data into RAM")
    parser.add_argument('--frames', type=int, default=16, help='number of frames')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers')
    parser.add_argument('--startIt', type=int, default=0, help='start iteration')
    parser.add_argument('--train_lr', action='store_true', help='train the learning rate')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sh_file', type=str, default='run_SFRD.sh')
    parser.add_argument('--FLAG', type=str, default="")
    parser.add_argument('--save_path', type=str, default="./results")

    parser.add_argument('--batch_syn', type=int, default=0)
    parser.add_argument('--dipc', type=int, default=0)

    parser.add_argument('--dim_in', type=int)
    parser.add_argument('--num_layers', type=int)
    parser.add_argument('--layer_size', type=int)
    parser.add_argument('--dim_out', type=int)
    parser.add_argument('--w0_initial', type=float)
    parser.add_argument('--w0', type=float)
    parser.add_argument('--lr_nf', type=float)
    parser.add_argument('--epochs_init', type=int, default=800)
    parser.add_argument('--lr_nf_init', type=float, default=5e-4)

    parser.add_argument('--shared_mode', type=str, default='per_class')
    parser.add_argument('--shared_num_layers', type=int, default=None)
    parser.add_argument('--shared_layer_size', type=int, default=None)

    parser.add_argument('--train_backbone', action='store_true')
    parser.add_argument('--train_latent', action='store_true')

    parser.add_argument('--lr_nf_backbone', type=float, default=None)
    parser.add_argument('--lr_nf_shift', type=float, default=None)
    parser.add_argument('--lr_nf_init_backbone', type=float, default=1e-5)
    parser.add_argument('--lr_nf_init_shift', type=float, default=None)

    parser.add_argument('--shift_init', type=float, default=0.0)
    parser.add_argument('--latent_std', type=float, default=0.01)
    parser.add_argument('--init_batch_per_step', type=int, default=1)
    parser.add_argument('--init_instances_per_epoch', type=int, default=-1)

    parser.add_argument('--use_relation_distill', type=int, default=1)
    parser.add_argument('--lambda_rel', type=float, default=0.3)
    parser.add_argument('--batch_real_rel', type=int, default=8)
    parser.add_argument('--syn_decode_chunk', type=int, default=16)

    args = parser.parse_args()
    set_seed(args.seed)
    args = load_default(args)

    if args.shared_num_layers is None:
        args.shared_num_layers = args.num_layers
    if args.shared_layer_size is None:
        args.shared_layer_size = args.layer_size

    if args.lr_nf_backbone is None:
        args.lr_nf_backbone = args.lr_nf
    if args.lr_nf_shift is None:
        args.lr_nf_shift = args.lr_nf * 5
    if args.lr_nf_init_backbone is None:
        args.lr_nf_init_backbone = args.lr_nf_init
    if args.lr_nf_init_shift is None:
        args.lr_nf_init_shift = args.lr_nf_init * 5

    if (not args.train_backbone) and (not args.train_latent):
        args.train_latent = True

    sub_save_path_1 = f"{args.dataset}_{args.model}_{args.ipc}ipc_{args.dipc}dipc"
    sub_save_path_2 = (
        f"{args.batch_syn}_"
        f"({args.dim_in},{args.shared_num_layers},{args.shared_layer_size},{args.dim_out})_"
        f"({args.w0_initial},{args.w0})_"
        f"(init{args.epochs_init},bkb{args.lr_nf_init_backbone:.0e},sh{args.lr_nf_init_shift:.0e})_"
        f"(train_bkb{args.lr_nf_backbone:.0e},sh{args.lr_nf_shift:.0e})_"
        f"rel{int(args.use_relation_distill)}_lam{args.lambda_rel:.2f}_br{args.batch_real_rel}_chunk{args.syn_decode_chunk}"
    )

    args.save_path = f"{args.save_path}/{sub_save_path_1}/{sub_save_path_2}#{args.FLAG}"
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        os.makedirs(f"{args.save_path}/imgs")

    if args.sh_file is not None and os.path.exists(f"./scripts/{args.sh_file}"):
        shutil.copy(f"./scripts/{args.sh_file}", f"{args.save_path}/{args.sh_file}")

    args.log_path = f"{args.save_path}/log.txt"

    main(args)
