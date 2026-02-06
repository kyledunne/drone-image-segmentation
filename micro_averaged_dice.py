# Legacy file with micro-averaged dice class and helpers
# (before I realized the competition uses macro-averaged dice)
from torch import nn
import torch.nn.functional as F

class CrossEntropyPlusDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, preds, y):
        cross_entropy_loss = self.cross_entropy_loss(preds, y)
        intersections, pred_areas, true_areas = calculate_intersection_pred_area_and_true_area(preds, y)
        dice_score = compute_dice_score(intersections, pred_areas, true_areas)
        dice_loss = 1 - dice_score
        return cross_entropy_loss + dice_loss, dice_score

def calculate_intersection_pred_area_and_true_area(pred, true):
    """
    preds: (B, C, H, W) logits
    y: (B, H, W) masks
    """
    pred_classes = pred.argmax(dim=1)
    pred_one_hot = F.one_hot(pred_classes, config.num_classes).permute(0, 3, 1, 2).float()
    true_one_hot = F.one_hot(true, config.num_classes).permute(0, 3, 1, 2).float()
    intersection = (pred_one_hot * true_one_hot).sum(dim=(2, 3))
    pred_area = pred_one_hot.sum(dim=(2, 3))
    true_area = true_one_hot.sum(dim=(2, 3))
    return intersection, pred_area, true_area

def compute_dice_score(intersections, pred_areas, true_areas, smooth=1e-8):
    """
    intersection: (B, config.num_classes)
    pred_areas: (B, config.num_classes)
    true_areas: (B, config.num_classes)

    Computes micro-averaged Dice per image and then returns average
    of per-image scores across all images in the batch.

    This is actually identical to performing micro-averaged Dice score
    across all pixels BECAUSE all images in this dataset have the exact same dimensions (1280x720).
    If there were smaller or larger images, this approach would ensure that each image
    gets equal weight regardless of size.
    """
    intersection_per_image = intersections.sum(dim=1)
    pred_areas_per_image = pred_areas.sum(dim=1)
    true_areas_per_image = true_areas.sum(dim=1)
    dice_per_image = (2 * intersection_per_image + smooth) / (pred_areas_per_image + true_areas_per_image + smooth)
    return dice_per_image.mean()