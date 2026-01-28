import logging
from collections import OrderedDict
import os
import numpy as np

import math
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torchvision.utils as tvutils
from tqdm import tqdm
from ema_pytorch import EMA

import models.lr_scheduler as lr_scheduler
import models.networks as networks
from models.optimizer import Lion

from models.modules.loss import MatchingLoss
from models import modules as M

from .base_model import BaseModel

logger = logging.getLogger("base")


class DenoisingModel(BaseModel):
    def __init__(self, opt):
        super(DenoisingModel, self).__init__(opt)

        if opt["dist"]:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1 
        train_opt = opt["train"]

        # define network
        self.model = networks.define_G(opt).to(self.device)
        if opt["dist"]:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[torch.cuda.current_device()],
                find_unused_parameters=True,
            )
        else:
            self.model = DataParallel(self.model)

        if "structure_prior" in opt:
            sp_opt = opt["structure_prior"]
            which_sp = sp_opt["which_model"]
            sp_setting = sp_opt["setting"]
            self.struct_prior = getattr(M, which_sp)(**sp_setting).to(self.device)

            if opt["dist"]:
                self.struct_prior = DistributedDataParallel(
                    self.struct_prior,
                    device_ids=[torch.cuda.current_device()],
                    find_unused_parameters=True,
                )
            else:
                self.struct_prior = DataParallel(self.struct_prior)

            logger.info("Use structure prior module [{}]".format(which_sp))
        else:
            self.struct_prior = None

        self.ema = EMA(self.model, beta=0.995, update_every=10).to(self.device)

        self.load()

        if self.is_train:
            self.model.train()

            is_weighted = opt["train"]["is_weighted"]
            loss_type = opt["train"]["loss_type"]
            self.loss_fn = MatchingLoss(loss_type, is_weighted).to(self.device)
            self.weight = opt["train"]["weight"]

            # optimizers
            wd_G = float(train_opt.get("weight_decay_G", 0) or 0)

            optim_params = []

            for k, v in self.model.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning("Params [{:s}] will not optimize.".format(k))

            if self.struct_prior is not None:
                for k, v in self.struct_prior.named_parameters():
                    if v.requires_grad:
                        optim_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning("Struct prior params [{:s}] will not optimize.".format(k))

            if train_opt["optimizer"] == "Adam":
                self.optimizer = torch.optim.Adam(
                    optim_params,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            elif train_opt["optimizer"] == "AdamW":
                self.optimizer = torch.optim.AdamW(
                    optim_params,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            elif train_opt["optimizer"] == "Lion":
                self.optimizer = Lion(
                    optim_params,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            else:
                print("Not implemented optimizer, default using Adam!")

            self.optimizers.append(self.optimizer)

            # schedulers
            if train_opt["lr_scheme"] == "MultiStepLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(
                            optimizer,
                            train_opt["lr_steps"],
                            restarts=train_opt["restarts"],
                            weights=train_opt["restart_weights"],
                            gamma=train_opt["lr_gamma"],
                            clear_state=train_opt["clear_state"],
                        )
                    )
            elif train_opt["lr_scheme"] == "TrueCosineAnnealingLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimizer,
                            T_max=train_opt["niter"],
                            eta_min=train_opt["eta_min"],
                        )
                    )
            else:
                raise NotImplementedError("MultiStepLR learning rate scheme is enough.")

            self.log_dict = OrderedDict()

    def feed_data(self, state, LQ, GT=None, deg_context=None, content_context=None):
        self.state = state.to(self.device)  # noisy_state
        self.condition = LQ.to(self.device)  # LQ
        if GT is not None:
            self.state_0 = GT.to(self.device)  # GT
        self.deg_context = deg_context

        if content_context is not None:
            self.content_context = content_context.detach().to(self.device, non_blocking=True)
            assert self.content_context.dim() == 2, "content_context must be [B, D]"
            try:
                expected = self.opt["network_G"]["setting"].get("context_dim", None)
            except Exception:
                expected = None
            if expected is not None:
                assert self.content_context.shape[-1] == expected, (
                    f"content_context dim {self.content_context.shape[-1]} != expected {expected}"
                )
        else:
            self.content_context = None

    def optimize_parameters(self, step, timesteps, sde=None):
        sde.set_mu(self.condition)

        self.optimizer.zero_grad()
        timesteps = timesteps.to(self.device)

        struct_tokens = self.struct_prior(self.condition) if self.struct_prior is not None else None

        noise = sde.noise_fn(
            self.state,
            timesteps.squeeze(),
            deg_context=self.deg_context,
            content_context=self.content_context,
            struct_tokens=struct_tokens,
        )
        score = sde.get_score_from_noise(noise, timesteps)

        xt_1_expection = sde.reverse_sde_step_mean(self.state, score, timesteps)
        xt_1_optimum = sde.reverse_optimum_step(self.state, self.state_0, timesteps)
        loss = self.weight * self.loss_fn(xt_1_expection, xt_1_optimum)

        loss.backward()
        self.optimizer.step()

        self.ema.update()

        self.log_dict["loss"] = loss.item()

    def test(self, sde=None, mode="posterior", save_states=False):
        sde.set_mu(self.condition)

        struct_tokens = self.struct_prior(self.condition) if self.struct_prior is not None else None

        self.model.eval()
        with torch.no_grad():
            if mode == "sde":
                self.output = sde.reverse_sde(
                    self.state,
                    save_states=save_states,
                    deg_context=self.deg_context,
                    content_context=self.content_context,
                    struct_tokens=struct_tokens,
                )
            else:
                self.output = sde.reverse_posterior(
                    self.state,
                    save_states=save_states,
                    deg_context=self.deg_context,
                    content_context=self.content_context,
                    struct_tokens=struct_tokens,
                )
        self.model.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict["Input"] = self.condition.detach()[0].float().cpu()
        out_dict["Output"] = self.output.detach()[0].float().cpu()
        if need_GT:
            out_dict["GT"] = self.state_0.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.model)
        if isinstance(self.model, (nn.DataParallel, DistributedDataParallel)):
            net_struc_str = "{} - {}".format(self.model.__class__.__name__, self.model.module.__class__.__name__)
        else:
            net_struc_str = "{}".format(self.model.__class__.__name__)
        if self.rank <= 0:
            logger.info("Network G structure: {}, with parameters: {:,d}".format(net_struc_str, n))
            logger.info(s)

    @staticmethod
    def _extract_state_dict(obj):

        if isinstance(obj, OrderedDict):
            return obj
        if isinstance(obj, dict):
            for k in ("params", "state_dict", "ema", "model", "net"):
                if k in obj and isinstance(obj[k], (dict, OrderedDict)):
                    return obj[k]
            return obj
        raise TypeError(f"Unsupported checkpoint type: {type(obj)}")

    @staticmethod
    def _match_module_prefix(src_sd, ref_sd):
        """
        Make src_sd key prefix ('module.') match ref_sd.
        """
        if not src_sd or not ref_sd:
            return src_sd

        src_keys = list(src_sd.keys())
        ref_keys = list(ref_sd.keys())
        if not src_keys or not ref_keys:
            return src_sd

        src_has = src_keys[0].startswith("module.")
        ref_has = ref_keys[0].startswith("module.")

        if src_has == ref_has:
            return src_sd
        if ref_has and not src_has:
            return OrderedDict((("module." + k), v) for k, v in src_sd.items())
        if (not ref_has) and src_has:
            return OrderedDict(((k[len("module."):]), v) for k, v in src_sd.items())
        return src_sd

    def load(self):

        load_path_G = self.opt["path"].get("pretrain_model_G", None)
        if load_path_G is None:
            return

        logger.info("Loading checkpoint from [{:s}] ...".format(load_path_G))
        strict_load = self.opt["path"].get("strict_load", True)
        if strict_load is None:
            strict_load = True
        strict_load_sp = self.opt["path"].get("strict_load_struct", True)
        if strict_load_sp is None:
            strict_load_sp = True

        loaded = torch.load(load_path_G, map_location="cpu")

        target_g = self.model.module if isinstance(
            self.model, (nn.DataParallel, DistributedDataParallel)
        ) else self.model

        if isinstance(loaded, dict) and "G" in loaded:
            target_g.load_state_dict(loaded["G"], strict=strict_load)

            if self.struct_prior is not None:
                if "SP" not in loaded or loaded["SP"] is None:
                    raise RuntimeError("Structure prior enabled but checkpoint has no 'SP'.")
                target_sp = self.struct_prior.module if isinstance(
                    self.struct_prior, (nn.DataParallel, DistributedDataParallel)
                ) else self.struct_prior
                target_sp.load_state_dict(loaded["SP"], strict=strict_load_sp)
            return

        g_sd = self._extract_state_dict(loaded)
        g_sd = self._match_module_prefix(g_sd, target_g.state_dict())

        logger.warning(
            "Loaded checkpoint has no key 'G'. Treating it as legacy EMA/plain state_dict for G ONLY. "
            "This file format does NOT include SP. For EMA+SP, please use the merged EMA_lastest.pth "
            "saved by this code (a dict with keys {'G','SP'})."
        )
        target_g.load_state_dict(g_sd, strict=strict_load)

    def save(self, iter_label):

        target_g = self.model.module if isinstance(self.model, (nn.DataParallel, DistributedDataParallel)) else self.model
        target_sp = self.struct_prior.module if isinstance(self.struct_prior, (nn.DataParallel, DistributedDataParallel)) else self.struct_prior

        merged = {
            "G": target_g.state_dict(),
            "SP": target_sp.state_dict() if self.struct_prior is not None else None,
        }
        save_filename = "{}_G.pth".format(iter_label)
        save_path = os.path.join(self.opt["path"]["models"], save_filename)
        torch.save(merged, save_path)

        if hasattr(self, "ema") and self.ema is not None:
            ema_model_for_save = self.ema.ema_model.module if isinstance(
                self.ema.ema_model, (nn.DataParallel, DistributedDataParallel)
            ) else self.ema.ema_model

            ema_merged = {
                "G": ema_model_for_save.state_dict(),  
                "SP": target_sp.state_dict() if self.struct_prior is not None else None,
            }
            ema_path = os.path.join(self.opt["path"]["models"], "EMA_latest.pth")
            torch.save(ema_merged, ema_path)
