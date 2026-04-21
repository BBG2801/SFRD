import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils
from utils import (
    get_dataset, get_network, get_eval_pool, evaluate_synset, get_time,
    DiffAugment, ParamDiffAug, set_seed, save_and_print, TensorDataset,
    get_images, epoch
)
import random
from reparam_module import ReparamModule

import shutil
import matplotlib.pyplot as plt
from hyper_params import load_default
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from SynSet import *
from SynSet.relation_distill import (
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


def gpu_mem(tag=""):
    if not torch.cuda.is_available():
        print(f"[MEM] {tag}: CUDA not available")
        return
    alloc = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    peak_alloc = torch.cuda.max_memory_allocated() / 1024**2
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**2
    print(
        f"[MEM] {tag} | "
        f"alloc={alloc:.1f} MB | "
        f"reserved={reserved:.1f} MB | "
        f"peak_alloc={peak_alloc:.1f} MB | "
        f"peak_reserved={peak_reserved:.1f} MB"
    )


def reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def sample_balanced_indices_cpu(synset, batch_syn):
    assert batch_syn % synset.num_classes == 0, \
        f"batch_syn ({batch_syn}) must be divisible by num_classes ({synset.num_classes})"

    per_class = batch_syn // synset.num_classes
    all_indices = []
    for c in range(synset.num_classes):
        start = c * synset.num_per_class
        cls_indices = torch.randperm(synset.num_per_class)[:per_class] + start
        all_indices.append(cls_indices)

    batch_indices = torch.cat(all_indices, dim=0)
    batch_indices = batch_indices[torch.randperm(len(batch_indices))]
    return batch_indices


def chunk_tensor_cpu(x, chunk_size):
    if chunk_size is None or chunk_size <= 0 or chunk_size >= len(x):
        return [x]
    return [x[i:i + chunk_size] for i in range(0, len(x), chunk_size)]


def one_micro_update(
    student_net,
    flat_param,
    synset,
    batch_indices_cpu,
    syn_lr,
    criterion,
    args,
    lr_scale=1.0,
):
    x, this_y = synset.get(batch_indices_cpu)
    this_y = this_y.to(args.device, non_blocking=True)

    if args.dsa and (not args.no_aug):
        x = DiffAugment(x, args.dsa_strategy, param=args.dsa_param)

    if args.distributed:
        forward_params = flat_param.unsqueeze(0).expand(torch.cuda.device_count(), -1)
    else:
        forward_params = flat_param

    logits = student_net(x, flat_param=forward_params)
    ce_loss = criterion(logits, this_y)

    grad = torch.autograd.grad(
        ce_loss,
        flat_param,
        create_graph=True
    )[0]

    new_flat_param = flat_param - syn_lr * lr_scale * grad

    del x, this_y, logits, ce_loss, grad
    return new_flat_param


def unroll_student_params(
    student_net,
    synset,
    initial_flat_param,
    syn_lr,
    criterion,
    args,
):
    current_param = initial_flat_param
    total_steps = args.syn_steps
    chunk_size = max(1, args.unroll_chunk_size)

    for step in range(total_steps):
        if args.truncate_unroll and step > 0 and (step % chunk_size == 0):
            current_param = current_param.detach().requires_grad_(True)

        full_batch_indices_cpu = sample_balanced_indices_cpu(synset, args.batch_syn)

        micro_chunks = chunk_tensor_cpu(full_batch_indices_cpu, args.micro_batch_syn)
        full_bs = len(full_batch_indices_cpu)

        for micro_indices_cpu in micro_chunks:
            lr_scale = float(len(micro_indices_cpu)) / float(full_bs)
            current_param = one_micro_update(
                student_net=student_net,
                flat_param=current_param,
                synset=synset,
                batch_indices_cpu=micro_indices_cpu,
                syn_lr=syn_lr,
                criterion=criterion,
                args=args,
                lr_scale=lr_scale,
            )

        del full_batch_indices_cpu, micro_chunks

    return current_param


def maybe_empty_cache(args):
    if torch.cuda.is_available() and args.empty_cache_every > 0:
        torch.cuda.empty_cache()


def build_relation_criterion(args):
    relation_backbone = build_frozen_resnet18_extractor(
        mean=args.mean,
        std=args.std,
        resize_to=(224, 224),
        imagenet_weights=True,
    ).to(args.device)

    relation_criterion = InterClassRelationDistillationLoss(
        feature_extractor=relation_backbone,
        num_classes=args.num_classes,
        eps=1e-8,
        ignore_diag=False,
    ).to(args.device)

    return relation_criterion


def main(args):
    torch.autograd.set_detect_anomaly(False)

    if args.max_experts is not None and args.max_files is not None:
        args.total_experts = args.max_experts * args.max_files

    save_and_print(args.log_path, "CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    eval_it_pool = list(range(500, args.Iteration + 1, 500))
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(
        args.dataset, args.data_path, args.batch_real, args.subset, args=args
    )
    args.channel, args.im_size, args.num_classes, args.mean, args.std = channel, im_size, num_classes, mean, std
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    args.im_size = im_size

    if args.dsa:
        args.dc_aug_param = None

    args.dsa_param = ParamDiffAug()

    dsa_params = args.dsa_param
    zca_trans = args.zca_trans if args.zca else None
    args.dsa_param = dsa_params
    args.zca_trans = zca_trans
    args.distributed = torch.cuda.device_count() > 1

    save_and_print(args.log_path, f'Hyper-parameters: {args.__dict__}')
    save_and_print(args.log_path, f'Evaluation model pool: {model_eval_pool}')

    images_all = []
    labels_all = []
    indices_class = [[] for _ in range(num_classes)]
    save_and_print(args.log_path, "BUILDING DATASET")
    for i in tqdm(range(len(dst_train))):
        sample = dst_train[i]
        images_all.append(torch.unsqueeze(sample[0], dim=0))
        labels_all.append(class_map[torch.tensor(sample[1]).item()])

    for i, lab in tqdm(enumerate(labels_all), total=len(labels_all)):
        indices_class[lab].append(i)

    images_all = torch.cat(images_all, dim=0).to("cpu")
    labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")

    synset = SFRD(args)
    synset.init(images_all, labels_all, indices_class)

    if args.batch_syn == 0:
        args.batch_syn = num_classes * synset.num_per_class

    if args.micro_batch_syn <= 0:
        args.micro_batch_syn = args.batch_syn

    assert args.batch_syn % synset.num_classes == 0, \
        f"batch_syn ({args.batch_syn}) must be divisible by num_classes ({synset.num_classes})"

    syn_lr = torch.tensor(args.lr_teacher, device=args.device)
    syn_lr = syn_lr.detach().to(args.device).requires_grad_(True)
    optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5)

    criterion = nn.CrossEntropyLoss().to(args.device)
    relation_criterion = build_relation_criterion(args) if args.use_relation_distill else None

    save_and_print(args.log_path, '%s training begins' % get_time())

    expert_dir = os.path.join(args.buffer_path, args.dataset)
    if args.dataset == "ImageNet":
        subset_names = {
            "nette": "imagenette",
            "woof": "imagewoof",
            "fruits": "imagefruit",
            "yellow": "imageyellow",
            "cats": "imagemeow",
            "birds": "imagesquawk"
        }
        expert_dir = os.path.join(expert_dir, subset_names[args.subset])
    if not args.zca:
        expert_dir += "_NO_ZCA"
    expert_dir = os.path.join(expert_dir, args.model)
    save_and_print(args.log_path, "Expert Dir: {}".format(expert_dir))

    if args.load_all:
        buffer = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, f"replay_buffer_{n}.pt")):
            buffer = buffer + torch.load(os.path.join(expert_dir, f"replay_buffer_{n}.pt"))
            n += 1
        if n == 0:
            raise AssertionError(f"No buffers detected at {expert_dir}")
    else:
        expert_files = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, f"replay_buffer_{n}.pt")):
            expert_files.append(os.path.join(expert_dir, f"replay_buffer_{n}.pt"))
            n += 1
        if n == 0:
            raise AssertionError(f"No buffers detected at {expert_dir}")
        file_idx = 0
        expert_idx = 0
        random.shuffle(expert_files)
        if args.max_files is not None:
            expert_files = expert_files[:args.max_files]
        save_and_print(args.log_path, "loading file {}".format(expert_files[file_idx]))
        buffer = torch.load(expert_files[file_idx])
        if args.max_experts is not None:
            buffer = buffer[:args.max_experts]
        random.shuffle(buffer)

    best_acc = {m: 0 for m in model_eval_pool}
    best_std = {m: 0 for m in model_eval_pool}

    del labels_all
    if not args.use_relation_distill:
        del images_all
        indices_class = None

    maybe_empty_cache(args)

    for it in range(0, args.Iteration + 1):
        save_this_it = False
        debug_mem = (it <= 2 and args.debug_mem)

        if debug_mem:
            reset_peak()
            gpu_mem(f"iter {it} start")

        if it in eval_it_pool or it == 200 or it == 100:
            for model_eval in model_eval_pool:
                save_and_print(
                    args.log_path,
                    '-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'
                    % (args.model, model_eval, it)
                )
                if args.dsa:
                    save_and_print(args.log_path, f'DSA augmentation strategy: {args.dsa_strategy}')
                    save_and_print(args.log_path, f'DSA augmentation parameters: {args.dsa_param.__dict__}')
                else:
                    save_and_print(args.log_path, f'DC augmentation parameters: {args.dc_aug_param}')

                accs_test = []
                for it_eval in range(args.num_eval):
                    net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)
                    image_syn_eval, label_syn_eval = synset.get(need_copy=True)
                    save_and_print(args.log_path, f"Evaluate dataset size: {image_syn_eval.shape} {label_syn_eval.shape}")

                    args.lr_net = syn_lr.item()
                    _, _, acc_test = evaluate_synset(
                        it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args
                    )
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

                save_and_print(
                    args.log_path,
                    'Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'
                    % (len(accs_test), model_eval, acc_test_mean, acc_test_std)
                )
                save_and_print(args.log_path, f"{args.save_path}")
                save_and_print(args.log_path, f"{it:5d} | Accuracy/{model_eval}: {acc_test_mean}")
                save_and_print(args.log_path, f"{it:5d} | Max_Accuracy/{model_eval}: {best_acc[model_eval]}")
                save_and_print(args.log_path, f"{it:5d} | Std/{model_eval}: {acc_test_std}")
                save_and_print(args.log_path, f"{it:5d} | Max_Std/{model_eval}: {best_std[model_eval]}")

            save_and_print(args.log_path, f"{it:5d} | Synthetic_LR: {syn_lr.detach().cpu()}")

            if save_this_it:
                synset.save(
                    name=f"SFRD_TM_{args.ipc}ipc#synset_best.pt",
                    auxiliary={"syn_lr": syn_lr.detach().cpu()}
                )

        if it in eval_it_pool and (save_this_it or it % 500 == 0):
            with torch.no_grad():
                image_save, label_save = synset.get(need_copy=True)

                if save_this_it:
                    torch.save(image_save.cpu(), os.path.join(args.save_path, "images_best.pt"))
                    torch.save(label_save.cpu(), os.path.join(args.save_path, "labels_best.pt"))

                save_dir = f"{args.save_path}/imgs"

                if args.ipc < 50 or args.force_save:
                    classes_save, indices_save = get_fixed_vis_indices(
                        save_path=args.save_path,
                        num_classes=num_classes,
                        num_per_class=synset.num_per_class,
                        vis_classes=10,
                        vis_per_class=10,
                    )

                    upsampled = image_save[indices_save]

                    if args.dataset != "ImageNet":
                        upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                        upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)

                    grid = torchvision.utils.make_grid(
                        upsampled, nrow=10, normalize=True, scale_each=True
                    )
                    plt.figure(figsize=(14, 14))
                    plt.imshow(np.transpose(grid.detach().cpu().numpy(), (1, 2, 0)))
                    plt.axis("off")
                    plt.tight_layout(pad=0.1)
                    plt.savefig(
                        f"{save_dir}/Synthetic_Images#{it}.png",
                        dpi=300, bbox_inches="tight", pad_inches=0.02
                    )
                    plt.close()

                    for clip_val in [2.5]:
                        stdv = torch.std(image_save)
                        meanv = torch.mean(image_save)
                        upsampled = torch.clip(
                            image_save,
                            min=meanv - clip_val * stdv,
                            max=meanv + clip_val * stdv
                        )
                        upsampled = upsampled[indices_save]

                        if args.dataset != "ImageNet":
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)

                        grid = torchvision.utils.make_grid(
                            upsampled, nrow=10, normalize=True, scale_each=True
                        )
                        plt.figure(figsize=(14, 14))
                        plt.imshow(np.transpose(grid.detach().cpu().numpy(), (1, 2, 0)))
                        plt.axis("off")
                        plt.tight_layout(pad=0.1)
                        plt.savefig(
                            f"{save_dir}/Clipped_Synthetic_Images#{it}.png",
                            dpi=300, bbox_inches="tight", pad_inches=0.02
                        )
                        plt.close()

                    if args.zca:
                        image_save = image_save.to(args.device)
                        image_save = args.zca_trans.inverse_transform(image_save)
                        image_save = image_save.cpu()

                        torch.save(image_save.cpu(), os.path.join(save_dir, f"images_zca_{it}.pt"))

                        upsampled = image_save[indices_save]
                        if args.dataset != "ImageNet":
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)

                        grid = torchvision.utils.make_grid(
                            upsampled, nrow=10, normalize=True, scale_each=True
                        )
                        plt.figure(figsize=(14, 14))
                        plt.imshow(np.transpose(grid.detach().cpu().numpy(), (1, 2, 0)))
                        plt.axis("off")
                        plt.tight_layout(pad=0.1)
                        plt.savefig(
                            f"{save_dir}/Reconstructed_Images#{it}.png",
                            dpi=300, bbox_inches="tight", pad_inches=0.02
                        )
                        plt.close()

                        for clip_val in [2.5]:
                            stdv = torch.std(image_save)
                            meanv = torch.mean(image_save)
                            upsampled = torch.clip(
                                image_save,
                                min=meanv - clip_val * stdv,
                                max=meanv + clip_val * stdv
                            )
                            upsampled = upsampled[indices_save]

                            if args.dataset != "ImageNet":
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)

                            grid = torchvision.utils.make_grid(
                                upsampled, nrow=10, normalize=True, scale_each=True
                            )
                            plt.figure(figsize=(14, 14))
                            plt.imshow(np.transpose(grid.detach().cpu().numpy(), (1, 2, 0)))
                            plt.axis("off")
                            plt.tight_layout(pad=0.1)
                            plt.savefig(
                                f"{save_dir}/Clipped_Reconstructed_Images#{it}.png",
                                dpi=300, bbox_inches="tight", pad_inches=0.02
                            )
                            plt.close()

                    del upsampled

                del image_save, label_save
                maybe_empty_cache(args)

        student_net = get_network(args.model, channel, num_classes, im_size, dist=False).to(args.device)
        student_net = ReparamModule(student_net)
        if args.distributed:
            student_net = torch.nn.DataParallel(student_net)
        student_net.train()

        if debug_mem:
            gpu_mem(f"iter {it} after student_net build")

        num_params = sum([np.prod(p.size()) for p in student_net.parameters()])

        if args.load_all:
            expert_trajectory = buffer[np.random.randint(0, len(buffer))]
        else:
            expert_trajectory = buffer[expert_idx]
            expert_idx += 1
            if expert_idx == len(buffer):
                expert_idx = 0
                file_idx += 1
                if file_idx == len(expert_files):
                    file_idx = 0
                    random.shuffle(expert_files)
                if args.max_files != 1:
                    del buffer
                    buffer = torch.load(expert_files[file_idx])
                if args.max_experts is not None:
                    buffer = buffer[:args.max_experts]
                random.shuffle(buffer)

        start_epoch = np.random.randint(0, args.max_start_epoch)
        starting_params_list = expert_trajectory[start_epoch]
        target_params_list = expert_trajectory[start_epoch + args.expert_epochs]

        target_params = torch.cat(
            [p.data.to(args.device).reshape(-1) for p in target_params_list], 0
        )
        starting_params = torch.cat(
            [p.data.to(args.device).reshape(-1) for p in starting_params_list], 0
        )
        student_param0 = starting_params.detach().clone().requires_grad_(True)

        if debug_mem:
            gpu_mem(f"iter {it} after student init params")

        if debug_mem:
            reset_peak()
            gpu_mem(f"iter {it} before unroll")

        student_param_last = unroll_student_params(
            student_net=student_net,
            synset=synset,
            initial_flat_param=student_param0,
            syn_lr=syn_lr,
            criterion=criterion,
            args=args,
        )

        if debug_mem:
            gpu_mem(f"iter {it} after unroll")

        param_loss = torch.nn.functional.mse_loss(student_param_last, target_params, reduction="sum")
        param_dist = torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")

        if debug_mem:
            gpu_mem(f"iter {it} after param loss")

        param_loss = param_loss / num_params
        param_dist = param_dist / num_params
        base_loss = param_loss / param_dist

        relation_loss = torch.tensor(0.0, device=args.device)
        relation_stats = None
        if args.use_relation_distill:
            relation_loss, relation_details = relation_criterion.forward_from_ddif(
                synset=synset,
                images_all=images_all,
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
        optimizer_lr.zero_grad(set_to_none=True)

        if debug_mem:
            reset_peak()
            gpu_mem(f"iter {it} before backward")

        total_loss.backward()

        if debug_mem:
            gpu_mem(f"iter {it} after backward")

        synset.optim_step()
        optimizer_lr.step()

        if debug_mem:
            gpu_mem(f"iter {it} after optimizer step")

        syn_lr.data = syn_lr.data.clip(min=0.001)

        if it % 10 == 0:
            if relation_stats is None:
                save_and_print(
                    args.log_path,
                    '%s iter = %04d, total_loss = %.4f, base_loss = %.4f, syn_steps = %d, batch_syn = %d, micro_batch_syn = %d, chunk = %d'
                    % (get_time(), it, total_loss.item(), base_loss.item(), args.syn_steps, args.batch_syn, args.micro_batch_syn, args.unroll_chunk_size)
                )
            else:
                save_and_print(
                    args.log_path,
                    '%s iter = %04d, total_loss = %.4f, base_loss = %.4f, relation_loss = %.4f, lambda_rel = %.3f, centroid_mse = %.6f, syn_steps = %d, batch_syn = %d, micro_batch_syn = %d, chunk = %d'
                    % (
                        get_time(), it, total_loss.item(), base_loss.item(), relation_loss.item(),
                        args.lambda_rel, relation_stats.centroid_mse,
                        args.syn_steps, args.batch_syn, args.micro_batch_syn, args.unroll_chunk_size
                    )
                )

        del student_param_last, student_param0
        del starting_params, target_params
        del starting_params_list, target_params_list
        del param_loss, param_dist, base_loss, total_loss
        if args.use_relation_distill:
            del relation_loss
        del student_net

        maybe_empty_cache(args)

        if debug_mem:
            gpu_mem(f"iter {it} end")

    synset.save(
        name=f"SFRD_TM_{args.ipc}ipc#synset_final.pt",
        auxiliary={"syn_lr": syn_lr.detach().cpu()}
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SFRD Trajectory Matching')

    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--subset', type=str, default='imagenette', help='ImageNet subset. This only does anything when --dataset=ImageNet')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode, check utils.py for more info')
    parser.add_argument('--num_eval', type=int, default=5, help='how many networks to evaluate on')
    parser.add_argument('--eval_it', type=int, default=500, help='how often to evaluate')
    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=15000, help='how many distillation steps to perform')
    parser.add_argument('--lr_init', type=float, default=0.01, help='how to init lr (alpha)')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'], help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='../data', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='../buffers', help='buffer path')
    parser.add_argument('--zca', action='store_true', help='do ZCA whitening')
    parser.add_argument('--load_all', action='store_true', help='only use if you can fit all expert trajectories into RAM')
    parser.add_argument('--no_aug', type=bool, default=False, help='this turns off diff aug during distillation')
    parser.add_argument('--max_files', type=int, default=None, help='number of expert files to read (leave as None unless doing ablations)')
    parser.add_argument('--max_experts', type=int, default=None, help='number of experts to read per file (leave as None unless doing ablations)')
    parser.add_argument('--force_save', action='store_true', help='this will save images for 50ipc')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sh_file', type=str, default='run_SFRD.sh')
    parser.add_argument('--FLAG', type=str, default="")
    parser.add_argument('--save_path', type=str, default="./results")

    parser.add_argument('--syn_steps', type=int)
    parser.add_argument('--expert_epochs', type=int)
    parser.add_argument('--max_start_epoch', type=int)
    parser.add_argument('--lr_lr', type=float)
    parser.add_argument('--lr_teacher', type=float)

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
    parser.add_argument('--lr_nf_init', type=float, default=1e-4)

    parser.add_argument('--shared_mode', type=str, default='global')
    parser.add_argument('--shared_num_layers', type=int, default=4)
    parser.add_argument('--shared_layer_size', type=int, default=160)

    parser.add_argument('--use_shift', type=str2bool, default=True)
    parser.add_argument('--use_scale', type=str2bool, default=False)

    parser.add_argument('--modulation_type', type=str, default='shift')
    parser.add_argument('--shift_init', type=float, default=0.0)
    parser.add_argument('--latent_std', type=float, default=0.01)

    parser.add_argument('--train_backbone', type=str2bool, default=False)
    parser.add_argument('--train_latent', type=str2bool, default=True)

    parser.add_argument('--lr_nf_backbone', type=float, default=None)
    parser.add_argument('--lr_nf_shift', type=float, default=None)
    parser.add_argument('--lr_nf_init_backbone', type=float, default=None)
    parser.add_argument('--lr_nf_init_shift', type=float, default=None)

    parser.add_argument('--epochs_init', type=int, default=5000)
    parser.add_argument('--init_instances_per_epoch', type=int, default=128)
    parser.add_argument('--init_batch_per_step', type=int, default=32)

    parser.add_argument('--micro_batch_syn', type=int, default=20,
                        help='micro batch size for virtual large synthetic batch. keep this small to stabilize memory.')
    parser.add_argument('--unroll_chunk_size', type=int, default=4,
                        help='truncate graph every N synthetic steps. memory mainly depends on this value.')
    parser.add_argument('--truncate_unroll', type=str2bool, default=True,
                        help='whether to detach student params between chunks.')
    parser.add_argument('--empty_cache_every', type=int, default=1,
                        help='call torch.cuda.empty_cache every iteration if > 0.')
    parser.add_argument('--debug_mem', type=str2bool, default=False)

    parser.add_argument('--use_relation_distill', type=str2bool, default=True)
    parser.add_argument('--lambda_rel', type=float, default=0.3)
    parser.add_argument('--batch_real_rel', type=int, default=8)
    parser.add_argument('--syn_decode_chunk', type=int, default=32)

    args = parser.parse_args()
    set_seed(args.seed)
    args = load_default(args)

    if args.lr_nf_backbone is None:
        args.lr_nf_backbone = args.lr_nf
    if args.lr_nf_shift is None:
        args.lr_nf_shift = args.lr_nf * 5

    if args.lr_nf_init_backbone is None:
        args.lr_nf_init_backbone = args.lr_nf_init
    if args.lr_nf_init_shift is None:
        args.lr_nf_init_shift = args.lr_nf_init * 5

    args.modulation_type = 'shift'
    args.use_shift = True
    args.use_scale = False

    sub_save_path_1 = f"{args.dataset}_{args.subset}_{args.res}_{args.model}_{args.ipc}ipc_{args.dipc}dipc"

    sub_save_path_2 = (
        f"{args.syn_steps}_{args.expert_epochs}_{args.max_start_epoch}_{args.lr_lr:.0e}_{args.lr_teacher:.0e}"
        f"#"
        f"{args.batch_syn}_mb{args.micro_batch_syn}_chunk{args.unroll_chunk_size}_tr{int(args.truncate_unroll)}"
        f"_"
        f"sharedshift"
        f"_({args.dim_in},{args.shared_num_layers},{args.shared_layer_size},{args.dim_out})"
        f"_({args.w0_initial},{args.w0})"
        f"_init({args.epochs_init},bkb{args.lr_nf_init_backbone:.0e},sh{args.lr_nf_init_shift:.0e})"
        f"_train(bkb{args.lr_nf_backbone:.0e},sh{args.lr_nf_shift:.0e})"
        f"_tb{int(args.train_backbone)}_tl{int(args.train_latent)}"
        f"_rel{int(args.use_relation_distill)}_lam{args.lambda_rel:.2f}_br{args.batch_real_rel}_chunk{args.syn_decode_chunk}"
    )

    if args.zca:
        sub_save_path_2 += "_ZCA"

    args.save_path = f"{args.save_path}/{sub_save_path_1}/{sub_save_path_2}#{args.FLAG}"

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        os.makedirs(f"{args.save_path}/imgs")

    shutil.copy(f"./scripts/{args.sh_file}", f"{args.save_path}/{args.sh_file}")
    args.log_path = f"{args.save_path}/log.txt"

    main(args)
