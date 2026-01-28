import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_input_dtype, CLIP, CustomTextCLIP
from .distributed import is_master
from .precision import get_autocast


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        val = float(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


def postprocess_clip_output(model_out):
    return model_out


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def train_one_epoch(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):

    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    if getattr(args, "accum_freq", 1) != 1:
        raise ValueError("Current priors training only supports accum_freq == 1, "
                         "please set args.accum_freq = 1 in your config.")

    model.train()

    data['train'].set_epoch(epoch)
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    stats_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i

        if not args.skip_scheduler:
            scheduler(step)

        if isinstance(batch, dict):
            img_gt = batch["gt"]
            img_lq = batch["lq"]
            deg_label = batch["deg_label"]
        else:
            raise RuntimeError("Expected batch to be a dict with keys: 'gt', 'lq', 'deg_label'.")

        img_gt = img_gt.to(device=device, dtype=input_dtype, non_blocking=True)
        img_lq = img_lq.to(device=device, dtype=input_dtype, non_blocking=True)
        deg_label = deg_label.to(device=device, non_blocking=True).long()

        batch_size = img_gt.size(0)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        with autocast():
            outputs = model(
                img_gt=img_gt,
                img_lq=img_lq,
                deg_label=deg_label,
                return_embeddings=True,
            )
            total_loss = outputs["total_loss"]
            content_loss = outputs["content_loss"]
            deg_loss = outputs["deg_loss"]

        backward(total_loss, scaler)

        if scaler is not None:
            if getattr(args, "horovod", False):
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if getattr(args, "grad_clip_norm", None) is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if getattr(args, "grad_clip_norm", None) is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if getattr(args, "grad_clip_norm", None) is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        with torch.no_grad():
            if "z_S" in outputs and "z_T" in outputs:
                cos_sim = F.cosine_similarity(outputs["z_S"], outputs["z_T"], dim=-1).mean().item()
            else:
                cos_sim = 0.0
            if "deg_logits" in outputs:
                pred = outputs["deg_logits"].argmax(dim=-1)
                deg_acc = (pred == deg_label).float().mean().item()
            else:
                deg_acc = 0.0

        stats = {
            "total_loss": total_loss.detach().item(),
            "content_loss": content_loss.detach().item(),
            "deg_loss": deg_loss.detach().item(),
            "cos_sim": cos_sim,
            "deg_acc": deg_acc,
        }

        for key, val in stats.items():
            if key not in stats_m:
                stats_m[key] = AverageMeter()
            stats_m[key].update(val, batch_size)

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1

        if is_master(args) and (i % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            samples_per_second = args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.batch_size / batch_time_m.val

            stat_log = " ".join(
                [
                    f"{name}: {meter.val:#.5g} ({meter.avg:#.5g})"
                    for name, meter in stats_m.items()
                ]
            )

            logging.info(
                f"Train Priors Epoch: {epoch} "
                f"[{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                + stat_log
            )

            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
            }
            log_data.update({name: meter.val for name, meter in stats_m.items()})

            for name, val in log_data.items():
                name_full = "train_priors/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name_full, val, step)
                if getattr(args, "wandb", False):
                    assert wandb is not None, "Please install wandb."
                    wandb.log({name_full: val, "step": step})

            batch_time_m.reset()
            data_time_m.reset()
    # end for


def evaluate(model, data, epoch, args, tb_writer=None):

    metrics = {}
    if not is_master(args):
        return metrics

    device = torch.device(args.device)
    model.eval()

    if "val" not in data or not (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        return metrics

    dataloader = data["val"].dataloader
    num_samples = 0
    samples_per_val = dataloader.num_samples

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    total_loss_sum = 0.0
    content_loss_sum = 0.0
    deg_loss_sum = 0.0
    cos_sim_sum = 0.0
    deg_acc_sum = 0.0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if isinstance(batch, dict):
                img_gt = batch["gt"]
                img_lq = batch["lq"]
                deg_label = batch["deg_label"]
            else:
                raise RuntimeError("Expected batch to be a dict with keys: 'gt', 'lq', 'deg_label'.")

            img_gt = img_gt.to(device=device, dtype=input_dtype, non_blocking=True)
            img_lq = img_lq.to(device=device, dtype=input_dtype, non_blocking=True)
            deg_label = deg_label.to(device=device, non_blocking=True).long()

            batch_size = img_gt.size(0)

            with autocast():
                outputs = model(
                    img_gt=img_gt,
                    img_lq=img_lq,
                    deg_label=deg_label,
                    return_embeddings=True,
                )
                total_loss = outputs["total_loss"]
                content_loss = outputs["content_loss"]
                deg_loss = outputs["deg_loss"]

            if "z_S" in outputs and "z_T" in outputs:
                cos_sim = F.cosine_similarity(outputs["z_S"], outputs["z_T"], dim=-1).mean().item()
            else:
                cos_sim = 0.0

            if "deg_logits" in outputs:
                pred = outputs["deg_logits"].argmax(dim=-1)
                deg_acc = (pred == deg_label).float().mean().item()
            else:
                deg_acc = 0.0

            total_loss_sum += total_loss.item() * batch_size
            content_loss_sum += content_loss.item() * batch_size
            deg_loss_sum += deg_loss.item() * batch_size
            cos_sim_sum += cos_sim * batch_size
            deg_acc_sum += deg_acc * batch_size
            num_samples += batch_size

            if (i % 50) == 0:
                logging.info(
                    f"Eval Priors Epoch: {epoch} [{num_samples}/{samples_per_val}] "
                    f"TotalLoss: {total_loss_sum / max(num_samples,1):.4f} "
                    f"CosSim: {cos_sim_sum / max(num_samples,1):.4f} "
                    f"DegAcc: {deg_acc_sum / max(num_samples,1):.4f}"
                )

    if num_samples == 0:
        return metrics

    metrics = {
        "priors_val_total_loss": total_loss_sum / num_samples,
        "priors_val_content_loss": content_loss_sum / num_samples,
        "priors_val_deg_loss": deg_loss_sum / num_samples,
        "priors_val_cos_sim": cos_sim_sum / num_samples,
        "priors_val_deg_acc": deg_acc_sum / num_samples,
        "epoch": epoch,
        "num_samples": num_samples,
    }

    logging.info(
        f"Eval Priors Epoch: {epoch} "
        + "\t".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    )


    if getattr(args, "save_logs", False):
        if tb_writer is not None:
            for name, val in metrics.items():
                tb_writer.add_scalar(f"val_priors/{name}", val, epoch)

        os.makedirs(args.checkpoint_path, exist_ok=True)
        with open(os.path.join(args.checkpoint_path, "results_priors.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if getattr(args, "wandb", False):
        assert wandb is not None, "Please install wandb."
        for name, val in metrics.items():
            wandb.log({f"val_priors/{name}": val, "epoch": epoch})

    return metrics


def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    return metrics


def maybe_compute_generative_loss(model_out):
    return None
