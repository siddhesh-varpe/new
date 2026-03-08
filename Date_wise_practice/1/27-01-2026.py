# !pip install torch torchvision opencv-python matplotlib hdbscan scikit-learn

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import hdbscan

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 1. Preprocessing (unchanged)
# ==========================================
def preprocess_image(path, size=512):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image at {path}")
    img = cv2.resize(img, (size, size))
    
    # CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    
    # Gaussian blur to remove grain
    img = cv2.GaussianBlur(img, (3,3), 0)
    
    img = img.astype(np.float32) / 255.0
    # Add batch and channel dimensions: (1, 1, H, W)
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    return img.to(device)

# ==========================================
# 2. DexiNed Components (The "Stronger" Model)
# ==========================================

class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class DenseBlock(nn.Module):
    """
    The Core of DexiNed: 
    Every layer's output is added to the input of the next.
    Preserves extremely fine details (cracks).
    """
    def __init__(self, in_c, out_c_per_layer=16):
        super(DenseBlock, self).__init__()
        self.layer1 = DenseLayer(in_c, out_c_per_layer)
        self.layer2 = DenseLayer(in_c + out_c_per_layer, out_c_per_layer)
        self.layer3 = DenseLayer(in_c + 2 * out_c_per_layer, out_c_per_layer)
        self.out_conv = nn.Conv2d(in_c + 3 * out_c_per_layer, in_c, 1) # Reduce back to original dim

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(torch.cat([x, x1], dim=1))
        x3 = self.layer3(torch.cat([x, x1, x2], dim=1))
        out = self.out_conv(torch.cat([x, x1, x2, x3], dim=1))
        return out + x  # Residual connection

class SingleScaleBlock(nn.Module):
    """Produces an edge map at a specific zoom level"""
    def __init__(self, in_c):
        super(SingleScaleBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, 1, 1)
        
    def forward(self, x):
        return self.conv(x)

class DexiNed(nn.Module):
    def __init__(self):
        super(DexiNed, self).__init__()
        
        # --- Encoder (Feature Extraction) ---
        # Block 1 (High Res, Fine Texture)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            DenseBlock(32)
        )
        
        # Block 2 (Medium Res)
        self.pool1 = nn.MaxPool2d(2) # 256
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            DenseBlock(64)
        )

        # Block 3 (Low Res, Abstract Shape)
        self.pool2 = nn.MaxPool2d(2) # 128
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            DenseBlock(128)
        )
        
        # --- Side Outputs (Multi-scale edges) ---
        self.side1 = SingleScaleBlock(32)
        self.side2 = SingleScaleBlock(64)
        self.side3 = SingleScaleBlock(128)
        
        # --- Fuse Block (Combine all scales) ---
        self.fuse = nn.Conv2d(3, 1, 1) # Inputs are the 3 side outputs

    def forward(self, x):
        # Image size: H x W
        f1 = self.block1(x)
        
        # Image size: H/2 x W/2
        f2 = self.block2(self.pool1(f1))
        
        # Image size: H/4 x W/4
        f3 = self.block3(self.pool2(f2))
        
        # --- Generate Edge Maps ---
        out1 = self.side1(f1)
        
        # Upsample deeper layers to match input size
        out2 = F.interpolate(self.side2(f2), size=x.shape[2:], mode='bilinear', align_corners=False)
        out3 = F.interpolate(self.side3(f3), size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # Concatenate maps for final fusion
        fused = self.fuse(torch.cat([out1, out2, out3], dim=1))
        edge_map = torch.sigmoid(fused)
        
        # Return fused edge map AND deep features (f3) for your Autoencoder logic
        # Note: We upsample f3 back to original size so your seam extraction works
        f3_upsampled = F.interpolate(f3, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return edge_map, f3_upsampled
    
    # ==========================================
# 3. Post-Processing & Anomaly Logic
# ==========================================

# Autoencoder (Adapted to handle DexiNed's 128-dim features)
class SeamAutoEncoder(nn.Module):
    def __init__(self, in_dim=128, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 64), 
            nn.ReLU(), 
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), 
            nn.ReLU(), 
            nn.Linear(64, in_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

def extract_seam_points(edge_map, feature_map):
    edge_np = edge_map.squeeze().detach().cpu().numpy()
    feat_np = feature_map.squeeze().detach().cpu().numpy()
    
    # Adaptive thresholding to find the weld seam
    # We only want the STRONGEST edges (the weld beads), not noise
    threshold = np.percentile(edge_np, 90) # Top 10% strongest pixels
    ys, xs = np.where(edge_np > threshold)
    
    if len(xs) == 0:
        return np.array([]), np.array([])
    
    # Extract feature vectors at these seam coordinates
    features = feat_np[:, ys, xs].T
    coords = np.stack([xs, ys], axis=1)
    
    return coords, features

def discover_gaps(z, coords):
    if len(z) == 0:
        return np.array([])
    
    z_np = z.detach().cpu().numpy()
    
    # HDBSCAN is great for this: it finds the "main cluster" (normal weld)
    # and labels outliers as -1 (anomalies/gaps)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=25, min_samples=5)
    labels = clusterer.fit_predict(z_np)
    
    # Return coordinates where label is -1 (Noise/Gap)
    return coords[labels == -1]

def visualize_results(image_path, gap_coords, edge_map):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (512, 512)) # Match processing size
    img_vis = img.copy()
    
    # Draw gaps
    for (x, y) in gap_coords:
        cv2.circle(img_vis, (int(x), int(y)), 3, (0, 0, 255), -1)
        
    edge_vis = (edge_map.squeeze().cpu().detach().numpy() * 255).astype(np.uint8)

    plt.figure(figsize=(12,6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(edge_vis, cmap='gray')
    plt.title("DexiNed Edge Map")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
    plt.title(f"Detected Gaps: {len(gap_coords)} points")
    plt.axis('off')
    
    plt.show()
    
# ==========================================
# 4. Main Execution
# ==========================================
def detect_gaps_dexined(image_path):
    # Initialize models
    dexined = DexiNed().to(device)
    ae = SeamAutoEncoder().to(device)
    
    # NOTE: In a real scenario, you MUST load weights here.
    # Without training, DexiNed initializes with random weights 
    # and will produce garbage edge maps.
    # dexined.load_state_dict(torch.load('dexined_weights.pth'))
    # ae.load_state_dict(torch.load('ae_weights.pth'))
    
    # 1. Load & Preprocess
    img_tensor = preprocess_image(image_path)
    
    # 2. Get Edges & Features
    with torch.no_grad():
        edge_map, deep_features = dexined(img_tensor)
    
    # 3. Extract Seam Data
    coords, features = extract_seam_points(edge_map, deep_features)
    
    if len(features) == 0:
        print("No seam found.")
        return

    # 4. Detect Anomalies (Gaps)
    features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
    
    # (Optional: If AE is not trained, we can't use reconstruction error effectively yet,
    # so we might rely purely on the features clustering for now)
    _, z = ae(features_tensor)
    
    gap_coords = discover_gaps(z, coords)
    
    # 5. Visualize
    visualize_results(image_path, gap_coords, edge_map)
    
# Example usage:
detect_gaps_dexined("/kaggle/input/datasets/mindingpotter/test-weld-3/1772205083976166400_0_1.png")  #Path to your test image