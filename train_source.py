import logging
import time
import os
import numpy as np
import torch
from tqdm import tqdm
import torch.optim as optim
import csv
import torch.nn as nn
import SimpleITK as sitk

from robustbench.data import get_dataset, convert_2d
from robustbench.utils import load_model, setup_source
from robustbench.losses import DiceLoss
from utils.evaluate import get_multi_class_evaluation_score
from utils.conf import cfg, load_cfg_fom_args

logger = logging.getLogger(__name__)


def train_source_model():
    """Train the segmentation model on the source domain."""
    logger.info(f"[Config] Max Epochs: {cfg.SOURCE.MAX_EPOCHES}")

    # Load dataset
    db_train, db_valid, db_test = get_dataset(
        dataset=cfg.MODEL.DATASET,
        domain=cfg.SOURCE.SOURCE_DOMAIN,
        online=True
    )
    train_loader = torch.utils.data.DataLoader(db_train, batch_size=cfg.SOURCE.BATCH_SIZE, shuffle=False, num_workers=16)

    # Load model
    base_model = load_model(cfg.MODEL.NETWORK, cfg.MODEL.CKPT_DIR, cfg.MODEL.DATASET, cfg.MODEL.METHOD).cuda()
    model = setup_source(base_model)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = DiceLoss(cfg.MODEL.NUMBER_CLASS).cuda()

    # Setup output dir
    save_model_dir = os.path.join('save_model', f"{cfg.MODEL.DATASET}_{cfg.MODEL.NETWORK}")
    os.makedirs(save_model_dir, exist_ok=True)

    # Training loop
    model.train()
    for epoch in tqdm(range(cfg.SOURCE.MAX_EPOCHES), desc="Training Epochs"):
        for batch in train_loader:
            volume_batch = batch['image'].cuda()
            label_batch = batch['label'].cuda()
            volume_batch, label_batch = convert_2d(volume_batch, label_batch)

            if cfg.MODEL.NETWORK == 'PraNet':
                model.train_source(volume_batch, label_batch, optimizer)
            else:
                model.train_source(volume_batch, label_batch)

    # Save final model
    model_path = os.path.join(save_model_dir, f"{cfg.MODEL.METHOD}-{cfg.SOURCE.SOURCE_DOMAIN}-{cfg.MODEL.EXPNAME}-model-latest.pth")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")


def evaluate_all_domains(model, save_output=True):
    """Evaluate the trained model on all target domains."""
    for test_domain in cfg.SOURCE.ALL_DOMAIN:
        model.eval()
        db_test, _, _ = get_dataset(dataset=cfg.MODEL.DATASET, domain=test_domain, online=True)
        test_loader = torch.utils.data.DataLoader(db_test, batch_size=1, shuffle=False, num_workers=10)

        results_dir = os.path.join('results', cfg.MODEL.DATASET, f"{cfg.MODEL.METHOD}-{cfg.MODEL.DATASET}-I-{test_domain}-M-{cfg.SOURCE.SOURCE_DOMAIN}")
        os.makedirs(os.path.join(results_dir, 'mask'), exist_ok=True)

        all_scores_dice = []
        all_scores_dice2 = []
        name_score_list_dice = []
        name_score_list_dice2 = []

        with torch.no_grad():
            for batch in test_loader:
                volume, label, names = batch['image'], batch['label'], batch['names']
                volume, label = convert_2d(volume, label)
                output_soft = model(volume.cuda()).softmax(1)
                output = output_soft.argmax(1).cpu().numpy()
                label = label.cpu().numpy().squeeze(1)
                name = os.path.basename(names[0])

                # Save prediction
                if save_output:
                    sitk.WriteImage(sitk.GetImageFromArray(output / 1.0), os.path.join(results_dir, 'mask', name))

                # Evaluate
                score_dice = get_multi_class_evaluation_score(output, label, cfg.MODEL.NUMBER_CLASS, 'dice')
                score_dice2 = get_multi_class_evaluation_score(output, label, cfg.MODEL.NUMBER_CLASS, 'dice')

                if cfg.MODEL.NUMBER_CLASS > 2:
                    score_dice.append(np.mean(score_dice))
                    score_dice2.append(np.mean(score_dice2))

                name_score_list_dice.append([name] + score_dice)
                name_score_list_dice2.append([name] + score_dice2)
                all_scores_dice.append(score_dice)
                all_scores_dice2.append(score_dice2)

        # Save CSV
        for metric, scores, name_list in zip(['dice', 'dice2'], [all_scores_dice, all_scores_dice2], [name_score_list_dice, name_score_list_dice2]):
            scores = np.array(scores)
            mean_scores = scores.mean(axis=0)
            std_scores = scores.std(axis=0)
            name_list.append(['mean'] + list(mean_scores))
            name_list.append(['std'] + list(std_scores))

            csv_path = os.path.join(results_dir, f"test_{metric}_all.csv")
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                head = ['image'] + [f"class_{i}" for i in range(1, cfg.MODEL.NUMBER_CLASS)]
                if cfg.MODEL.NUMBER_CLASS > 2:
                    head.append('average')
                writer.writerow(head)
                writer.writerows(name_list)

            print(f"[{test_domain}] {metric.upper()} Mean: {mean_scores}, Std: {std_scores}")


if __name__ == '__main__':
    load_cfg_fom_args("Train source model")
    train_source_model()

    # Load model again for evaluation
    model_path = os.path.join('save_model', f"{cfg.MODEL.DATASET}_{cfg.MODEL.NETWORK}", f"{cfg.MODEL.METHOD}-{cfg.SOURCE.SOURCE_DOMAIN}-{cfg.MODEL.EXPNAME}-model-latest.pth")
    base_model = load_model(cfg.MODEL.NETWORK, cfg.MODEL.CKPT_DIR, cfg.MODEL.DATASET, cfg.MODEL.METHOD).cuda()
    model = setup_source(base_model)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    evaluate_all_domains(model, save_output=True)
