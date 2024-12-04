import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torchvision.transforms import Compose, ToTensor
from pycocotools.coco import COCO
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json  
from pycocotools.cocoeval import COCOeval

# Dataset Loader Class
class CategoryBasedFolderDataset(torch.utils.data.Dataset):
    def __init__(self, root, ann_file, transforms=None):
        super().__init__()
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.root = root
        self.transforms = transforms

    def __getitem__(self, idx):
        # Get image info from COCO annotations
        image_info = self.coco.loadImgs(self.ids[idx])[0]
        image_id = image_info["id"]

        # Get annotation details for the image
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id))

        # Filter out invalid bounding boxes
        valid_annotations = [
            ann for ann in annotations if ann["bbox"][2] > 0 and ann["bbox"][3] > 0
        ]

        # Determine the category ID (default to 1: "Normal" if no valid annotations)
        category_id = valid_annotations[0]["category_id"] if valid_annotations else 1

        # Determine the folder dynamically based on category_id
        subfolder = "Normal" if category_id == 1 else "Threat"

        # Construct the image path
        img_path = os.path.join(self.root, subfolder, image_info["file_name"])

        # Check if image file exists
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image {img_path} not found in dataset directory.")

        # Load the image
        img = Image.open(img_path).convert("RGB")

        # Apply transforms, if any
        if self.transforms:
            img = self.transforms(img)

        # Convert annotations to match PyTorch's expected format
        if valid_annotations:
            target = {
                "boxes": torch.tensor([ann["bbox"] for ann in valid_annotations], dtype=torch.float32),
                "labels": torch.tensor([ann["category_id"] for ann in valid_annotations], dtype=torch.int64),
                "image_id": torch.tensor([image_id], dtype=torch.int64),
                "area": torch.tensor([ann["area"] for ann in valid_annotations], dtype=torch.float32),
                "iscrowd": torch.tensor([ann["iscrowd"] for ann in valid_annotations], dtype=torch.int64),
            }
        else:
            target = {
                "boxes": torch.empty((0, 4), dtype=torch.float32),
                "labels": torch.tensor([], dtype=torch.int64),
                "image_id": torch.tensor([image_id], dtype=torch.int64),
                "area": torch.tensor([], dtype=torch.float32),
                "iscrowd": torch.tensor([], dtype=torch.int64),
            }

        return img, target

    def __len__(self):
        return len(self.ids)

# Transformations
def get_transform(train):
    transforms = []
    transforms.append(ToTensor())  # Convert image to tensor
    return Compose(transforms)

# Training Function
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0
    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Skip targets without valid bounding boxes
        targets = [t for t in targets if t["boxes"].size(0) > 0]

        if len(targets) == 0:
            continue

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

# Evaluation Function
def evaluate(model, data_loader, device, ann_file):
    model.eval()
    results = []
    coco = COCO(ann_file)  # Load ground truth annotations

    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            outputs = model(images)

            for target, output in zip(targets, outputs):
                image_id = int(target["image_id"].item())
                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()

                for box, score, label in zip(boxes, scores, labels):
                    x_min, y_min, x_max, y_max = box
                    width = x_max - x_min
                    height = y_max - y_min
                    results.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [float(x_min), float(y_min), float(width), float(height)],  # Convert to float
                        "score": float(score),  # Convert to float
                    })

    # Save predictions in COCO format
    results_file = "coco_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f)

    # Load results into COCO evaluation API
    coco_pred = coco.loadRes(results_file)
    coco_eval = COCOeval(coco, coco_pred, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    metrics = {
        "bbox_mAP": coco_eval.stats[0],
        "bbox_mAP50": coco_eval.stats[1],
        "bbox_mAP75": coco_eval.stats[2],
        "bbox_mAP_small": coco_eval.stats[3],
        "bbox_mAP_medium": coco_eval.stats[4],
        "bbox_mAP_large": coco_eval.stats[5],
    }

    print("Metrics:", metrics)
    return metrics



# Visualization Function
def visualize_predictions(model, dataset, device, num_images=5, output_dir="visualizations"):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)  # Create directory to save visualizations if it doesn't exist

    for i in range(num_images):
        img, target = dataset[i]
        img_tensor = img.to(device).unsqueeze(0)
        with torch.no_grad():
            prediction = model(img_tensor)[0]

        # Plot original image
        img_np = img.permute(1, 2, 0).numpy()
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img_np)
        ax.axis("off")

        # Plot predicted boxes
        for box, label in zip(prediction["boxes"], prediction["labels"]):
            x1, y1, x2, y2 = box.cpu().numpy()
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1, str(label.item()), color="white", fontsize=12, bbox=dict(facecolor='red', alpha=0.5))

        # Save the figure
        output_path = os.path.join(output_dir, f"prediction_{i}.png")
        plt.savefig(output_path)
        plt.close(fig)  # Close the figure to free memory
        print(f"Saved visualization to {output_path}")

# Main Script
if __name__ == "__main__":
    # Paths
    train_root = "GDINO_Project_BILAL_data_T1/Training"
    train_json = "/home/ai-13/Desktop/Project/GDINO_Project_BILAL_data_T1/annotations/train1_new.json"
    test_root = "GDINO_Project_BILAL_data_T1/Testing"
    test_json = "/home/ai-13/Desktop/Project/GDINO_Project_BILAL_data_T1/annotations/test_new.json"

    # Hyperparameters
    num_epochs = 5
    learning_rate = 0.005

    # Device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Datasets and DataLoaders
    train_dataset = CategoryBasedFolderDataset(root=train_root, ann_file=train_json, transforms=get_transform(train=True))
    test_dataset = CategoryBasedFolderDataset(root=test_root, ann_file=test_json, transforms=get_transform(train=False))
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    # Model
    model = fasterrcnn_resnet50_fpn(pretrained=True)

    # Freeze all layers and unfreeze the last few
    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.backbone.body.named_parameters():
        if "layer4" in name:  # Unfreeze ResNet's layer4
            param.requires_grad = True

    for param in model.rpn.head.parameters():
        param.requires_grad = True

    for param in model.roi_heads.parameters():
        param.requires_grad = True

    # Move model to device
    model.to(device)

    # Optimizer
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=learning_rate, momentum=0.9)

    # Training Loop
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, train_loader, device, epoch)

    # Evaluation
    # Evaluation with COCO metrics
    metrics = evaluate(model, test_loader, device, test_json)
    print(f"COCO Metrics: {metrics}")


    # Visualization
    visualize_predictions(model, test_dataset, device)
