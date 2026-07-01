import torch
import torch.nn as nn
import timm
import os
import io
from PIL import Image
from torchvision import transforms


class DeepfakeSentinel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('vit_small_patch16_224_dino', pretrained=False, num_classes=0)
        self.head = nn.Sequential(
            nn.Linear(384, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1)
        )
    def forward(self, x):
        return self.head(self.backbone(x))

#
def model_fn(model_dir):
    model = DeepfakeSentinel()
    with open(os.path.join(model_dir, 'dpo_aligned_model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f, map_location='cpu'))
    model.eval()
    return model


def input_fn(request_body, request_content_type):
    if request_content_type == 'application/x-image':
        img = Image.open(io.BytesIO(request_body)).convert('RGB')
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return preprocess(img).unsqueeze(0)
    raise ValueError(f'Unsupported content type: {request_content_type}')


def predict_fn(input_data, model):
    with torch.no_grad():
        output = model(input_data)
        prob = torch.sigmoid(output).item()
    return {'fakeness_score': prob}