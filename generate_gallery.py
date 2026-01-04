import os
import cv2
import torch
import numpy as np
import pickle
import sys
from pathlib import Path

# Add strong_sort to path so we can import the model
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / 'strong_sort'))


from strong_sort.reid_multibackend import ReIDDetectMultiBackend

def get_device(device):
    if device is None:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)

def generate_gallery(dataset_path, weights_path, device='cpu', output_file='gallery.pkl'):
    """
    Scans dataset_path for folders (person names).
    Reads images, extracts features, and saves to output_file.
    """
    if not os.path.exists(dataset_path):
        print(f"Dataset path {dataset_path} does not exist!")
        return

    # Convert device string to torch.device object
    device_obj = get_device(device)
    
    # Initialize ReID model
    print(f"Loading ReID model from {weights_path}...")
    model = ReIDDetectMultiBackend(weights=weights_path, device=device_obj)
    model.warmup()

    gallery = {}
    
    # Iterate over person folders
    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_dir):
            continue
            
        print(f"Processing {person_name}...")
        features_list = []
        
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            if not img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue
                
            # Read and preprocess image
            img = cv2.imread(img_path)
            if img is None:
                print(f"  Warning: Could not read {img_path}")
                continue
                
            # ReID model expects specific logic, usually passed as list of crops
            # We treat the whole image as the "crop" here
            # model.forward() handles normalization
            
            # The model expects BGR image (cv2 format)
            # ReIDDetectMultiBackend.forward takes a list of numpy images
            feat = model([img]) # returns list of tensors
            
            # feat is a list of tensors, we expect 1 image so 1 feature
            if isinstance(feat, list) and len(feat) > 0:
                feat = feat[0]
            
            # feat shape is usually (1, feature_dim)
            if isinstance(feat, torch.Tensor):
                feat = feat.cpu().detach().numpy()
            
            features_list.append(feat)
            
        if features_list:
            # Save ALL feature vectors instead of averaging
            # This allows "Multi-Look" matching (different angles, clothes)
            # features_list is a list of (Dim,) numpy arrays
            
            # Normalize each feature independently
            normalized_features = []
            for feat in features_list:
                norm_feat = feat / np.linalg.norm(feat)
                normalized_features.append(norm_feat)
            
            gallery[person_name] = normalized_features
            print(f"  Saved {len(normalized_features)} unique looks for {person_name}")
        else:
            print(f"  No valid images found for {person_name}")

    # Save gallery
    with open(output_file, 'wb') as f:
        pickle.dump(gallery, f)
    print(f"Gallery saved to {output_file} with {len(gallery)} identities.")

if __name__ == "__main__":
    # Default paths - adjusted for the repo structure
    # Assuming people put folders in 'dataset/'
    # Weights assumed to be at root or passed
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dataset', help='Path to dataset folder')
    parser.add_argument('--weights', type=str, default='osnet_x1_0_market1501.pt', help='Path to ReID weights')
    parser.add_argument('--output', type=str, default='gallery.pkl', help='Output pickle file')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu, 0, etc.)')
    
    opt = parser.parse_args()
    
    generate_gallery(opt.dataset, opt.weights, opt.device, opt.output)
