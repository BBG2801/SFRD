import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import numpy as np
import torch
from torchvision.utils import save_image
from utils import (
    get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset,
    get_daparam, match_loss, get_time, TensorDataset, epoch,
    DiffAugment, ParamDiffAug, set_seed, save_and_print, get_images,
)
import time
import shutil
from hyper_params import load_default
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from SynSet import SFRD

try:
    from relation_distill import (
        build_frozen_resnet18_extractor,
        InterClassRelationDistillationLoss,
        build_relation_stats,
    )
except ImportError:
    from relation_distill import (
        build_frozen_resnet18_extractor,
        InterClassRelationDistillationLoss,
        build_relation_stats,
    )

from tqdm import tqdm


def str2bool(x):
    if isinstance(x, bool):
        return x
    x = str(x).strip().lower()
    if x in ["true", "1", "yes", "y", "t"]:
        return True
    if x in ["false", "0", "no", "n", "f"]:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {x}")


def maybe_empty_cache(args):
    if torch.cuda.is_available() and args.empty_cache_every > 0:
        torch.cuda.empty_cache()


def get_fixed_vis_indices(save_path, num_classes, num_per_class, vis_classes=10, vis_per_class=10):
    os.makedirs(save_path, exist_ok=True)
    vis_meta_path = os.path.join(save_path, "fixed_vis_indices.pt")

    vis_classes = min(vis_classes, num_classes)
    vis_per_class = min(vis_per_class, num_per_class)

    if os.path.exists(vis_meta_path):
        vis_meta = torch.load(vis_meta_path, map_location="cpu")
        classes_save = np.array(vis_meta["classes_save"], dtype=np.int64)
        indices_save = np.array(vis_meta["indices_save"], dtype=np.int64)

        expected_num = vis_classes * vis_per_class
        if len(indices_save) == expected_num:
            return classes_save, indices_save

    classes_save = np.arange(vis_classes, dtype=np.int64)
    indices_save = np.concatenate([
        c * num_per_class + np.arange(vis_per_class, dtype=np.int64)
        for c in classes_save
    ])

    vis_meta = {
        "classes_save": classes_save.tolist(),
        "indices_save": indices_save.tolist(),
        "vis_classes": int(vis_classes),
        "vis_per_class": int(vis_per_class),
        "num_classes": int(num_classes),
        "num_per_class": int(num_per_class),
    }
    torch.save(vis_meta, vis_meta_path)
    return classes_save, indices_save



def visualize_synset(args, synset, it, save_this_it=False):
    with torch.no_grad():
        image_save, label_save = synset.get(need_copy=True, detach_backbone=True)

        if save_this_it:
            torch.save(image_save.cpu(), os.path.join(args.save_path, "images_best.pt"))
            torch.save(label_save.cpu(), os.path.join(args.save_path, "labels_best.pt"))

        save_dir = f"{args.save_path}/imgs"
        classes_save, indices_save = get_fixed_vis_indices(
            save_path=args.save_path,
            num_classes=args.num_classes,
            num_per_class=synset.num_per_class,
            vis_classes=10,
            vis_per_class=10,
        )
        _ = classes_save

        vis_img = image_save[indices_save].detach().clone()
        for ch in range(args.channel):
            vis_img[:, ch] = vis_img[:, ch] * args.std[ch] + args.mean[ch]
        vis_img = vis_img.clamp(0.0, 1.0)
        save_image(vis_img, f"{save_dir}/Synthetic_Images#{it}.png", nrow=10)

        if args.zca:
            image_save_zca = image_save.to(args.device)
            image_save_zca = args.zca_trans.inverse_transform(image_save_zca).cpu()
            vis_img_zca = image_save_zca[indices_save].clamp(0.0, 1.0)
            save_image(vis_img_zca, f"{save_dir}/Reconstructed_Images#{it}.png", nrow=10)

        del image_save, label_save



def build_relation_criterion(args, num_classes):
    if not args.use_relation_distill or args.lambda_rel <= 0:
        return None

    frozen_feat = build_frozen_resnet18_extractor(
        mean=args.mean,
        std=args.std,
        resize_to=(224, 224),
        imagenet_weights=True,
    ).to(args.device)

    relation_criterion = InterClassRelationDistillationLoss(
        feature_extractor=frozen_feat,
        num_classes=num_classes,
        eps=1e-8,
        ignore_diag=False,
    ).to(args.device)
    return relation_criterion



def main():
    parser = argparse.ArgumentParser(description='Gradient Matching with shared-field SFRD synthetic set')
    parser.add_argument('--method', type=str, default='SFRD', help='method name')
    parser.add_argument('--objective', type=str, default='DC', help='base distillation objective')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode')
    parser.add_argument('--num_exp', type=int, default=1, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=5, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=300, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=1000, help='training iterations')
    parser.add_argument('--eval_it', type=int, default=500, help='evaluation interval')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='../data', help='dataset path')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sh_file', type=str)
    parser.add_argument('--FLAG', type=str, default="")
    parser.add_argument('--save_path', type=str, default="./results")
    parser.add_argument('--batch_syn', type=int, default=0)
    parser.add_argument('--dipc', type=int, default=0)
    parser.add_argument('--res', type=int)
    parser.add_argument('--empty_cache_every', type=int, default=1)

    parser.add_argument('--subset', type=str, default='nette', help='ImageNet subset. This only does anything when --dataset=ImageNet')
    parser.add_argument('--zca', action='store_true', help="do ZCA whitening")
    parser.add_argument('--force_save', action='store_true', help='this will save images for 50ipc')

    parser.add_argument('--dim_in', type=int)
    parser.add_argument('--num_layers', type=int)
    parser.add_argument('--layer_size', type=int)
    parser.add_argument('--dim_out', type=int)
    parser.add_argument('--w0_initial', type=float)
    parser.add_argument('--w0', type=float)
    parser.add_argument('--lr_nf', type=float)
    parser.add_argument('--epochs_init', type=int, default=5000)
    parser.add_argument('--lr_nf_init', type=float, default=5e-4)

    parser.add_argument('--shared_mode', type=str, default='global')
    parser.add_argument('--shared_num_layers', type=int, default=None)
    parser.add_argument('--shared_layer_size', type=int, default=None)
    parser.add_argument('--modulation_type', type=str, default='shift')
    parser.add_argument('--shift_init', type=float, default=0.0)
    parser.add_argument('--latent_std', type=float, default=0.01)
    parser.add_argument('--train_backbone', type=str2bool, default=True)
    parser.add_argument('--train_latent', type=str2bool, default=True)
    parser.add_argument('--lr_nf_backbone', type=float, default=None)
    parser.add_argument('--lr_nf_shift', type=float, default=None)
    parser.add_argument('--lr_nf_init_backbone', type=float, default=None)
    parser.add_argument('--lr_nf_init_shift', type=float, default=None)
    parser.add_argument('--init_instances_per_epoch', type=int, default=-1)
    parser.add_argument('--init_batch_per_step', type=int, default=32)

    parser.add_argument('--use_relation_distill', type=str2bool, default=True)
    parser.add_argument('--lambda_rel', type=float, default=0.3)
    parser.add_argument('--batch_real_rel', type=int, default=8)
    parser.add_argument('--syn_decode_chunk', type=int, default=0)

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

    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.distributed = torch.cuda.device_count() > 1
    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True

    sub_save_path_1 = f"{args.dataset}_{args.subset}_{args.res}_{args.model}_{args.ipc}ipc_{args.dipc}dipc"
    sub_save_path_2 = (
        f"SFRD-{args.objective}_{args.batch_syn}"
        f"_sharedshift"
        f"_({args.dim_in},{args.shared_num_layers},{args.shared_layer_size},{args.dim_out})"
        f"_({args.w0_initial},{args.w0})"
        f"_init({args.epochs_init},bkb{args.lr_nf_init_backbone:.0e},sh{args.lr_nf_init_shift:.0e})"
        f"_train(bkb{args.lr_nf_backbone:.0e},sh{args.lr_nf_shift:.0e})"
        f"_tb{int(args.train_backbone)}_tl{int(args.train_latent)}"
        f"_rel{int(args.use_relation_distill)}-lam{args.lambda_rel:g}-br{args.batch_real_rel}"
    )
    if args.syn_decode_chunk > 0:
        sub_save_path_2 += f"_chunk{args.syn_decode_chunk}"
    if args.zca:
        sub_save_path_2 += "_ZCA"

    args.save_path = f"{args.save_path}/{sub_save_path_1}/{sub_save_path_2}#{args.FLAG}"
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        os.makedirs(f"{args.save_path}/imgs")

    shutil.copy(f"./scripts/{args.sh_file}", f"{args.save_path}/{args.sh_file}")
    args.log_path = f"{args.save_path}/log.txt"

    eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist() if args.eval_mode in ['S', 'SS'] else [args.Iteration]
    if args.Iteration not in eval_it_pool:
        eval_it_pool.append(args.Iteration)

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(
        args.dataset, args.data_path, args.batch_real, args.subset, args=args
    )
    args.channel, args.im_size, args.num_classes, args.mean, args.std = channel, im_size, num_classes, mean, std
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    save_and_print(args.log_path, f'Hyper-parameters: {args.__dict__}')
    save_and_print(args.log_path, f'Evaluation model pool: {model_eval_pool}')

    for exp in range(args.num_exp):
        save_and_print(args.log_path, f'\n================== Exp {exp} ==================\n ')

        images_all = []
        labels_all = []
        indices_class = [[] for _ in range(num_classes)]
        save_and_print(args.log_path, 'BUILDING DATASET')
        for i in tqdm(range(len(dst_train))):
            sample = dst_train[i]
            images_all.append(torch.unsqueeze(sample[0], dim=0))
            labels_all.append(class_map[torch.tensor(sample[1]).item()])

        for i, lab in tqdm(enumerate(labels_all), total=len(labels_all)):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0).to(args.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

        synset = SFRD(args)
        synset.init(images_all, labels_all, indices_class)

        relation_criterion = build_relation_criterion(args, num_classes)
        if relation_criterion is None:
            save_and_print(args.log_path, 'Relation distillation: disabled')
        else:
            save_and_print(
                args.log_path,
                f'Relation distillation: enabled | lambda_rel={args.lambda_rel:.4f} | '
                f'batch_real_rel={args.batch_real_rel} | syn_decode_chunk={args.syn_decode_chunk}'
            )

        criterion = torch.nn.CrossEntropyLoss().to(args.device)
        best_acc = {m: 0 for m in model_eval_pool}
        best_std = {m: 0 for m in model_eval_pool}

        save_and_print(args.log_path, f'{get_time()} training begins')
        maybe_empty_cache(args)

        for it in range(args.Iteration + 1):
            save_this_it = False

            if it in eval_it_pool and it > 0:
                for model_eval in model_eval_pool:
                    save_and_print(
                        args.log_path,
                        '-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'
                        % (args.model, model_eval, it)
                    )
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
                        image_syn_eval, label_syn_eval = synset.get(need_copy=True)
                        _, _, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
                        accs_test.append(acc_test)
                        del net_eval, image_syn_eval, label_syn_eval
                        maybe_empty_cache(args)

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

                visualize_synset(args, synset, it, save_this_it=save_this_it)
                if save_this_it:
                    synset.save(name=f"SFRD_DC_{args.ipc}ipc#synset_best.pt")
                maybe_empty_cache(args)

            net = get_network(args.model, channel, num_classes, im_size).to(args.device)
            net.train()
            net_parameters = list(net.parameters())
            optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)
            optimizer_net.zero_grad()
            loss_avg = 0.0
            args.dc_aug_param = None

            for ol in range(args.outer_loop):
                bn_flag = any('BatchNorm' in module._get_name() for module in net.modules())
                if bn_flag:
                    bn_size_per_class = 16
                    img_real_bn = torch.cat([
                        get_images(images_all, indices_class, c, bn_size_per_class) for c in range(num_classes)
                    ], dim=0)
                    net.train()
                    _ = net(img_real_bn)
                    for module in net.modules():
                        if 'BatchNorm' in module._get_name():
                            module.eval()
                    del img_real_bn

                base_loss = torch.tensor(0.0, device=args.device)
                for c in range(num_classes):
                    img_real = get_images(images_all, indices_class, c, args.batch_real)
                    lab_real = torch.ones((img_real.shape[0],), device=args.device, dtype=torch.long) * c

                    if args.batch_syn > 0:
                        indices = np.random.permutation(range(c * synset.num_per_class, (c + 1) * synset.num_per_class))[:args.batch_syn]
                    else:
                        indices = range(c * synset.num_per_class, (c + 1) * synset.num_per_class)

                    img_syn, lab_syn = synset.get(indices=indices)

                    if args.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                    output_real = net(img_real)
                    loss_real = criterion(output_real, lab_real)
                    gw_real = torch.autograd.grad(loss_real, net_parameters)
                    gw_real = [_.detach().clone() for _ in gw_real]

                    output_syn = net(img_syn)
                    loss_syn = criterion(output_syn, lab_syn)
                    gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

                    base_loss += match_loss(gw_syn, gw_real, args)

                    del img_real, lab_real, img_syn, lab_syn, output_real, loss_real, gw_real, output_syn, loss_syn, gw_syn

                base_loss = base_loss / num_classes

                relation_loss = torch.tensor(0.0, device=args.device)
                relation_stats = None
                if relation_criterion is not None:
                    relation_loss, relation_details = relation_criterion.forward_from_ddif(
                        synset=synset,
                        images_all=images_all,
                        indices_class=indices_class,
                        batch_real_per_class=args.batch_real_rel,
                        syn_decode_chunk=args.syn_decode_chunk if args.syn_decode_chunk > 0 else None,
                        return_details=True,
                    )
                    relation_stats = build_relation_stats(relation_details, relation_loss)
                    del relation_details

                if relation_criterion is None or args.lambda_rel <= 0:
                    total_loss = base_loss
                else:
                    total_loss = (1.0 - args.lambda_rel) * base_loss + args.lambda_rel * relation_loss

                synset.optim_zero_grad()
                total_loss.backward()
                synset.optim_step()
                loss_avg += total_loss.item()

                if ol == args.outer_loop - 1:
                    break

                image_syn_train, label_syn_train = synset.get(need_copy=True)
                dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
                trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=args.batch_train, shuffle=True, num_workers=0)
                for _ in range(args.inner_loop):
                    epoch('train', trainloader, net, optimizer_net, criterion, args, aug=True if args.dsa else False)

                del image_syn_train, label_syn_train, dst_syn_train, trainloader
                maybe_empty_cache(args)

            loss_avg /= args.outer_loop

            if it % 10 == 0:
                if relation_stats is None:
                    save_and_print(
                        args.log_path,
                        '%s iter = %04d, base_loss = %.6f, total_loss = %.6f'
                        % (get_time(), it, base_loss.item(), loss_avg)
                    )
                else:
                    save_and_print(
                        args.log_path,
                        '%s iter = %04d, base_loss = %.6f, rel_loss = %.6f, total_loss = %.6f, centroid_mse = %.6f'
                        % (
                            get_time(),
                            it,
                            base_loss.item(),
                            relation_stats.loss_rel,
                            loss_avg,
                            relation_stats.centroid_mse,
                        )
                    )

            del net, base_loss, relation_loss, total_loss
            maybe_empty_cache(args)

        synset.save(name=f"SFRD_DC_{args.ipc}ipc#synset_final.pt")


if __name__ == '__main__':
    main()
