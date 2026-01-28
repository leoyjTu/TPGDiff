import os
from typing import List

import torch
from PIL import Image
from tqdm import tqdm

import open_clip
from open_clip.prior_stage_model import PriorStageModel
import torch.nn as nn



CHECKPOINT_PATH = "/pretrained/tpgd_ViT-B-32.pt"

VAL_ROOT = "/datasets/val"

DEGRADATION_TYPES: List[str] = ['blurry', 'hazy', 'low-light', 'noisy', 'rainy']


IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG",
    ".png", ".PNG", ".ppm", ".PPM",
    ".bmp", ".BMP", ".tif"
]


def is_image_file(filename: str) -> bool:
    return any(filename.endswith(ext) for ext in IMG_EXTENSIONS)


def get_paths_from_images(path: str):
    assert os.path.isdir(path), f"{path} is not a valid directory."
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                images.append(os.path.join(dirpath, fname))
    assert images, f"{path} has no valid image file."
    return images


def build_prior_model(device: torch.device):

    base_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        model_name="ViT-B-32",              
        pretrained="laion2b_s34b_b79k",     
        precision="fp32",
        device=device,
    )

    visual = base_model.visual      
    embed_dim = visual.output_dim    

    import copy
    teacher_encoder = copy.deepcopy(visual)
    student_encoder = copy.deepcopy(visual)
    deg_backbone = copy.deepcopy(visual)

    prior_model = PriorStageModel(
        teacher_encoder=teacher_encoder,
        student_encoder=student_encoder,
        deg_backbone=deg_backbone,
        embed_dim=embed_dim,
        num_degradations=len(DEGRADATION_TYPES),
        content_loss_weight=1.0,
        deg_loss_weight=1.0,
        use_cosine_distill=True,
        normalize_embedding=True,
        freeze_teacher=True,
        freeze_deg_backbone=True,
    )

    prior_model.to(device)
    return prior_model, preprocess_val


def load_prior_checkpoint(model: nn.Module, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state[k[7:]] = v
        else:
            new_state[k] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unique_labels = sorted(DEGRADATION_TYPES)
    deg2id = {lab: i for i, lab in enumerate(unique_labels)}

    model, preprocess = build_prior_model(device)
    load_prior_checkpoint(model, CHECKPOINT_PATH, device)
    model.eval()

    total_correct = 0
    total_samples = 0

    for degra in unique_labels:   
        gt_id = deg2id[degra]

        root_path = os.path.join(VAL_ROOT, degra, "LQ")
        if not os.path.isdir(root_path):
            continue

        image_paths = get_paths_from_images(root_path)
        correct = 0

        for im_path in tqdm(image_paths, desc=f"Eval {degra}"):
            img = Image.open(im_path).convert("RGB")
            img_tensor = preprocess(img).unsqueeze(0).to(device)

            with torch.no_grad():
                probs = model.get_degradation_prior(img_tensor, as_prob=True)  # [1, N]
                pred_id = int(probs.argmax(dim=-1).item())

            if pred_id == gt_id:
                correct += 1

        acc = correct / len(image_paths)
        total_correct += correct
        total_samples += len(image_paths)

        print(f"Degradation: {degra:16s} | Accuracy: {acc:.6f}  "
              f"({correct}/{len(image_paths)})")

    if total_samples > 0:
        overall_acc = total_correct / total_samples
        print("=" * 60)
        print(f"Overall accuracy: {overall_acc:.6f}  ({total_correct}/{total_samples})")
    else:
        print("No samples found. Please check VAL_ROOT and folder structure.")


if __name__ == "__main__":
    main()
