import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageDraw
import os
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
from swin_transformer import SwinTransformer  # Ensure this is correctly imported

# Helper function to load annotations
def load_annotations(annotation_path):
    with open(annotation_path, 'r') as f:
        annotations = json.load(f)
    return annotations

annotations = load_annotations('/home/ai-13/Desktop/Project/BYOL_MINE/Data_WithANNotation/Annotation/train1.json')

class BYOL(nn.Module):
    def __init__(self, base_encoder, hidden_dim=256, m=0.996):
        super(BYOL, self).__init__()
        self.online_encoder = base_encoder
        self.target_encoder = copy.deepcopy(self.online_encoder)

        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.m = m

    def forward(self, x1, x2):
        z1_online = self.online_encoder(x1)
        z2_online = self.online_encoder(x2)

        p1 = self.predictor(z1_online)
        p2 = self.predictor(z2_online)

        with torch.no_grad():
            z1_target = self.target_encoder(x1)
            z2_target = self.target_encoder(x2)

        return p1, p2, z1_target.detach(), z2_target.detach()

    def update_target_encoder(self):
        for online_params, target_params in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_params.data = self.m * target_params.data + (1 - self.m) * online_params.data

class AugmentedDatasetWithMasking(Dataset):
    def __init__(self, root, annotations, transform=None):
        self.root = root
        self.annotations = annotations
        self.transform = transform
        self.image_paths = [os.path.join(root, img['file_name']) for img in annotations['images']]
        self.image_id_to_path = {img['id']: os.path.join(root, img['file_name']) for img in annotations['images']}
        self.image_annotations = {img['id']: [] for img in annotations['images']}
        for ann in annotations['annotations']:
            self.image_annotations[ann['image_id']].append(ann)

    def mask_object(self, image, annotation):
        draw = ImageDraw.Draw(image)
        for ann in annotation:
            bbox = ann['bbox']
            x, y, w, h = bbox
            mask_x = random.randint(int(x), int(x + w))
            mask_y = random.randint(int(y), int(y + h))
            mask_w = random.randint(1, int(w // 2))
            mask_h = random.randint(1, int(h // 2))
            draw.rectangle([mask_x, mask_y, mask_x + mask_w, mask_y + mask_h], fill=(0, 0, 0))
        return image

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_id = list(self.image_id_to_path.keys())[index]
        image_path = self.image_id_to_path[image_id]
        sample = Image.open(image_path).convert("RGB")
        
        annotation = self.image_annotations[image_id]
        if annotation:
            sample = self.mask_object(sample, annotation)
        
        if self.transform is not None:
            x1 = self.transform(sample)
            x2 = self.transform(sample)
        return (x1, x2), 0  # Returning 0 as target since it's not used in self-supervised learning

def loss_fn(p, z):
    p = nn.functional.normalize(p, dim=1)
    z = nn.functional.normalize(z, dim=1)
    return 2 - 2 * (p * z).sum(dim=1).mean()

def train_byol(byol, dataloader, optimizer, epochs=10):
    byol.train()
    for epoch in range(epochs):
        total_loss = 0
        with tqdm(dataloader, unit="batch") as tepoch:
            for (x1, x2), _ in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}")
                x1, x2 = x1.cuda(), x2.cuda()
                p1, p2, z1, z2 = byol(x1, x2)
                loss = loss_fn(p1, z1) + loss_fn(p2, z2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                byol.update_target_encoder()
                total_loss += loss.item()
                tepoch.set_postfix(loss=total_loss / len(dataloader))
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}')

def show_sample_masked_data(dataset, num_samples=3):
    plt.figure(figsize=(10, 10))
    for i in range(num_samples):
        (x1, x2), _ = dataset[i]
        x1 = x1.permute(1, 2, 0).numpy()
        x2 = x2.permute(1, 2, 0).numpy()
        plt.subplot(2, num_samples, i+1)
        plt.imshow(x1)
        plt.axis('off')
        plt.subplot(2, num_samples, num_samples+i+1)
        plt.imshow(x2)
        plt.axis('off')
    plt.show()

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to a fixed size
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

dataset = AugmentedDatasetWithMasking(
    root='/home/ai-13/Desktop/Project/BYOL_MINE/Data_WithANNotation/Images', 
    annotations=annotations, 
    transform=data_transforms
)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Show sample masked data before training
show_sample_masked_data(dataset, num_samples=3)

# Initialize the Swin Transformer backbone
swin_transformer = SwinTransformer(
    pretrain_img_size=224,
    in_channels=3,
    embed_dims=96,
    patch_size=4,
    window_size=7,
    mlp_ratio=4,
    depths=(2, 2, 6, 2),
    num_heads=(3, 6, 12, 24),
    strides=(4, 2, 2, 2),
    out_indices=(0, 1, 2, 3),
    qkv_bias=True,
    qk_scale=None,
    patch_norm=True,
    drop_rate=0.,
    attn_drop_rate=0.,
    drop_path_rate=0.1,
    use_abs_pos_embed=False,
    act_cfg=dict(type='GELU'),
    norm_cfg=dict(type='LN'),
    with_cp=False,
    pretrained=None,
    convert_weights=True,
    frozen_stages=-1
)

# Path to the GroundingDINO checkpoint
checkpoint_path = '/home/ai-08/Downloads/BYOL_MINE/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth'

try:
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
except FileNotFoundError:
    print(f"Checkpoint file not found at {checkpoint_path}")
    exit(1)
except Exception as e:
    print(f"An error occurred while loading the checkpoint: {e}")
    exit(1)

try:
    # Load the Swin Transformer weights from the checkpoint
    state_dict = checkpoint['state_dict']
    backbone_state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items() if k.startswith('backbone')}
    swin_transformer.load_state_dict(backbone_state_dict, strict=False)
except KeyError:
    print("State dictionary does not contain the expected keys.")
    exit(1)
except Exception as e:
    print(f"An error occurred while loading the Swin Transformer weights: {e}")
    exit(1)

# Integrate Swin Transformer into BYOL
byol = BYOL(swin_transformer).cuda()
optimizer = optim.Adam(byol.parameters(), lr=0.003)

train_byol(byol, dataloader, optimizer, epochs=10)

# Save the trained BYOL model
torch.save(byol.state_dict(), 'byol_model_with_swin_transformer.pth')

