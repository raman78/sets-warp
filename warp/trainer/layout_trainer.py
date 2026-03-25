# warp/trainer/layout_trainer.py
#
# P4: CNN Layout Regression training pipeline for STO UI slots.
# Maps grayscale 224x224 screenshots to slot coordinates.

import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path
import logging

log = logging.getLogger(__name__)

# Fixed list of slots we train the regressor on (SPACE order)
REGRESSOR_SLOTS = [
    'Fore Weapons', 'Deflector', 'Sec-Def', 'Engines', 'Warp Core', 
    'Shield', 'Aft Weapons', 'Experimental', 'Devices', 'Universal Consoles', 
    'Engineering Consoles', 'Science Consoles', 'Tactical Consoles', 'Hangars'
]
NUM_SLOTS = len(REGRESSOR_SLOTS)
OUTPUT_SIZE = NUM_SLOTS * 4 # [x, y, w, h] for each slot

class LayoutRegressor(nn.Module):
    """
    MobileNetV3-Small based regressor for slot positions.
    Input: Grayscale (1-channel) image resized to 224x224.
    Output: Coordinates normalized to 0.0-1.0.
    """
    def __init__(self, output_size: int = OUTPUT_SIZE):
        super().__init__()
        # Using MobileNetV3 small for extreme speed & efficiency
        self.backbone = models.mobilenet_v3_small(weights=None)
        
        # Modify first layer for 1-channel grayscale input
        self.backbone.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Replace classifier with regression head
        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(512, output_size),
            nn.Sigmoid() # Force output to 0.0-1.0 range
        )

    def forward(self, x):
        return self.backbone(x)

def train_layout_model(dataset: list[dict], epochs: int = 50):
    """
    Skeleton for training the layout model on the built dataset.
    This would run inside the LocalTrainWorker.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LayoutRegressor(OUTPUT_SIZE).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    log.info(f"Starting layout training on {len(dataset)} samples for {epochs} epochs")
    
    # Training implementation details (transformation, loading, export to ONNX)
    # would go here, very similar to local_trainer.py _train loop.
    
    return model

if __name__ == '__main__':
    # Test model shape
    model = LayoutRegressor()
    test_input = torch.randn(1, 1, 224, 224)
    output = model(test_input)
    print(f"Model output shape: {output.shape}") # Should be [1, 56]
    assert output.shape == (1, 56)
