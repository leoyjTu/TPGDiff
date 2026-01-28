# src/open_clip/prior_stage_model.py

from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F

from .loss import ContentDistillLoss, DegradationCELoss


class GEGLU(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=-1)
        return a * F.gelu(b)

class ResidualGLUBlock(nn.Module):
    def __init__(self, dim, ffn_mult=4, dropout=0.1):
        super().__init__()
        ffn_dim = int(dim * ffn_mult)
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, 2 * ffn_dim)   # for GEGLU
        self.act = GEGLU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(ffn_dim, dim)

    def forward(self, x):
        h = x
        x = self.ln(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x + h

class DegradationPredictor(nn.Module):
    def __init__(self, dim, num_labels, num_blocks=2, ffn_mult=4, dropout=0.1):
        super().__init__()
        self.trunk = nn.Sequential(*[
            ResidualGLUBlock(dim, ffn_mult=ffn_mult, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.cls = nn.Linear(dim, num_labels)

    def forward(self, x):
        x = self.trunk(x)
        return self.cls(x)  # (B, N)


class PriorStageModel(nn.Module):

    def __init__(
        self,
        teacher_encoder: nn.Module,
        student_encoder: nn.Module,
        deg_backbone: nn.Module,
        embed_dim: int,
        num_degradations: int,
        content_loss_weight: float = 1.0,
        deg_loss_weight: float = 1.0,
        use_cosine_distill: bool = True,
        normalize_embedding: bool = True,
        freeze_teacher: bool = True,
        freeze_deg_backbone: bool = False,
        deg_num_blocks: int = 2,
        deg_ffn_mult: int = 4,
        deg_dropout: float = 0.1,
        label_smoothing: float = 0.05,   
    ):

        super().__init__()

        self.teacher = teacher_encoder
        self.student = student_encoder
        self.deg_backbone = deg_backbone
        self.deg_head = DegradationPredictor(
            dim=embed_dim,
            num_labels=num_degradations,
            num_blocks=deg_num_blocks,
            ffn_mult=deg_ffn_mult,
            dropout=deg_dropout,
        )

        self.content_loss_weight = float(content_loss_weight)
        self.deg_loss_weight = float(deg_loss_weight)
        self.normalize_embedding = normalize_embedding

        self.content_criterion = ContentDistillLoss(use_cosine=use_cosine_distill)
        self.deg_criterion = DegradationCELoss(reduction="mean", label_smoothing=label_smoothing)
        if freeze_teacher:
            self._freeze_module(self.teacher)

        if freeze_deg_backbone:
            self._freeze_module(self.deg_backbone)

        self._teacher_frozen = bool(freeze_teacher)
        self._deg_backbone_frozen = bool(freeze_deg_backbone)

    @staticmethod
    def _freeze_module(module: nn.Module) -> None:
        for p in module.parameters():
            p.requires_grad = False
        module.eval()

    def train(self, mode: bool = True):

        super().train(mode)
        if getattr(self, "_teacher_frozen", False):
            try:
                self.teacher.eval()
            except Exception:
                pass
        if getattr(self, "_deg_backbone_frozen", False):
            try:
                self.deg_backbone.eval()
            except Exception:
                pass
        return self

    def encode_teacher(self, img: torch.Tensor) -> torch.Tensor:
        z = self.teacher(img)
        if self.normalize_embedding:
            z = F.normalize(z, dim=-1)
        return z

    def encode_student(self, img: torch.Tensor) -> torch.Tensor:
        z = self.student(img)
        if self.normalize_embedding:
            z = F.normalize(z, dim=-1)
        return z

    def encode_for_degradation(self, img: torch.Tensor) -> torch.Tensor:
        requires_grad = any(p.requires_grad for p in self.deg_backbone.parameters())
        if requires_grad:
            feat = self.deg_backbone(img)  # [B, D]
        else:
            with torch.no_grad():
                feat = self.deg_backbone(img)  # [B, D]
        return feat

    @torch.no_grad()
    def get_content_prior(self, img_lq: torch.Tensor) -> torch.Tensor:
        was_training = self.training
        self.eval()
        z = self.encode_student(img_lq)
        if was_training:
            self.train()
        return z

    @torch.no_grad()
    def get_degradation_prior(self, img_lq: torch.Tensor, as_prob: bool = True) -> torch.Tensor:
        self.eval()
        feat = self.encode_for_degradation(img_lq)
        logits = self.deg_head(feat)
        if as_prob:
            return logits.softmax(dim=-1)
        return logits

    def forward(
        self,
        img_gt: torch.Tensor,        # [B,3,H,W]
        img_lq: torch.Tensor,        # [B,3,H,W]
        deg_label: torch.Tensor,     # [B]
        return_embeddings: bool = True,
    ) -> Dict[str, torch.Tensor]:

        device = img_lq.device

        outputs: Dict[str, torch.Tensor] = {}

        with torch.no_grad():
            z_T = self.encode_teacher(img_gt)   # [B, D]
        z_S = self.encode_student(img_lq)       # [B, D]

        content_loss = self.content_criterion(z_S, z_T)
        content_loss = content_loss * self.content_loss_weight
        outputs["content_loss"] = content_loss

        deg_feat = self.encode_for_degradation(img_lq)  # [B, D]
        deg_logits = self.deg_head(deg_feat)            # [B, N]
        deg_label = deg_label.to(device).long()
        deg_loss = self.deg_criterion(deg_logits, deg_label)
        deg_loss = deg_loss * self.deg_loss_weight
        outputs["deg_loss"] = deg_loss

        total_loss = content_loss + deg_loss
        outputs["total_loss"] = total_loss

        if return_embeddings:
            outputs["z_T"] = z_T.detach()
            outputs["z_S"] = z_S
        outputs["deg_logits"] = deg_logits

        return outputs
