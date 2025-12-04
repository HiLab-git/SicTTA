import logging
import time
import os
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import SimpleITK as sitk
import csv
from PIL import Image

from copy import deepcopy
from robustbench.data import get_dataset, convert_2d
from robustbench.utils import load_model, setup_source
from robustbench.losses import DiceLoss
from robustbench.tta import (
    setup_norm, setup_tent, setup_sictta, setup_sar, setup_meant
)
from utils.evaluate import get_multi_class_evaluation_score
from utils.conf import cfg, load_cfg_fom_args

logger = logging.getLogger(__name__)


def get_adaptation_model(base_model):
    method = cfg.MODEL.METHOD.lower()
    logger.info(f"Test-time adaptation method: {method.upper()}")

    setup_map = {
        "source_test": setup_source,
        "norm": setup_norm,
        "tent": lambda m: setup_tent(m)[1],
        "sar": setup_sar,
        "sictta": setup_sictta,
        "meant": setup_meant
    }

    if method in setup_map:
        return setup_map[method](base_model)
    else:
        raise ValueError(f"Unknown adaptation method: {cfg.MODEL.METHOD}")


def run_adaptation():
    load_cfg_fom_args("Adaptation evaluation")
    base_model = load_model(cfg.MODEL.NETWORK, cfg.MODEL.CKPT_DIR, cfg.MODEL.DATASET, cfg.MODEL.METHOD).cuda()
    model = get_adaptation_model(base_model)

    dice_loss = DiceLoss(cfg.MODEL.NUMBER_CLASS).cuda()
    metric = ['dice', 'dice']
    save_model_dir = os.path.join('save_model', f"{cfg.MODEL.DATASET}_{cfg.MODEL.NETWORK}")
    os.makedirs(save_model_dir, exist_ok=True)

    for epoch in tqdm(range(cfg.ADAPTATION.EPOCH), ncols=70):
        for target_domain in cfg.ADAPTATION.TARGET_DOMAIN:
            all_scores_dice = []
            all_scores_dice2 = []

            db_all, _, _ = get_dataset(dataset=cfg.MODEL.DATASET, domain=target_domain, online=True)
            loader = torch.utils.data.DataLoader(db_all, batch_size=cfg.ADAPTATION.BATCH_SIZE, shuffle=False, num_workers=10)

            result_dir = os.path.join('results', cfg.MODEL.DATASET, f"{cfg.MODEL.METHOD}-{cfg.MODEL.DATASET}-I-{target_domain}-M-{cfg.SOURCE.SOURCE_DOMAIN}")
            os.makedirs(os.path.join(result_dir, 'mask'), exist_ok=True)

            name_scores = []
            name_scores2 = []

            for batch in loader:
                volume, label, names = batch['image'].cuda(), batch['label'].cuda(), batch['names']
                volume, label = convert_2d(volume, label)

                if cfg.MODEL.METHOD in ["sictta"]:
                    output = model(volume, names)
                else:
                    output = model(volume)

                prediction = output.argmax(1).cpu().numpy()
                label_np = label.cpu().numpy().squeeze(1)

                for i, name in enumerate(names):
                    img_name = os.path.basename(name)
                    out_path = os.path.join(result_dir, 'mask', img_name)

                    if 'Fundus' in cfg.MODEL.DATASET:
                        arr = deepcopy(prediction[i])
                        arr[arr == 0] = 255
                        arr[arr == 2] = 0
                        arr[arr == 1] = 128
                        img = Image.fromarray(arr.astype(np.uint8)).resize((512, 512), Image.NEAREST)
                        img.save(out_path)
                    else:
                        sitk.WriteImage(sitk.GetImageFromArray(prediction[i] / 1.0), out_path)

                    score = get_multi_class_evaluation_score(prediction[i], label_np[i], cfg.MODEL.NUMBER_CLASS, metric[0])
                    score2 = get_multi_class_evaluation_score(prediction[i], label_np[i], cfg.MODEL.NUMBER_CLASS, metric[1])
                    if cfg.MODEL.NUMBER_CLASS > 2:
                        score.append(np.mean(score))
                        score2.append(np.mean(score2))

                    name_scores.append([img_name] + score)
                    name_scores2.append([img_name] + score2)
                    all_scores_dice.append(score)
                    all_scores_dice2.append(score2)

            for scores, metric_name, all_name_scores in zip([all_scores_dice, all_scores_dice2], metric, [name_scores, name_scores2]):
                scores_np = np.array(scores)
                mean_score = scores_np.mean(axis=0)
                std_score = scores_np.std(axis=0)
                all_name_scores.append(['mean'] + list(mean_score))
                all_name_scores.append(['std'] + list(std_score))

                csv_path = os.path.join(result_dir, f"test_{metric_name}_all.csv")
                with open(csv_path, mode='w', newline='') as f:
                    writer = csv.writer(f)
                    head = ['image'] + [f"class_{i}" for i in range(1, cfg.MODEL.NUMBER_CLASS)]
                    if cfg.MODEL.NUMBER_CLASS > 2:
                        head.append("average")
                    writer.writerow(head)
                    for row in all_name_scores:
                        writer.writerow(row)

                print(f"[Epoch {epoch}] [{target_domain}] {metric_name} Mean: {mean_score}, Std: {std_score}")

        model_path = os.path.join(save_model_dir, f"{cfg.MODEL.METHOD}-{cfg.SOURCE.SOURCE_DOMAIN}-{cfg.MODEL.EXPNAME}-model-latest.pth")
        torch.save(model.state_dict(), model_path)
        logger.info(f"Adapted model saved to {model_path}")


if __name__ == '__main__':
    run_adaptation()
