
import os
import cv2
import pickle
import numpy as np
import argparse
from insightface.app import FaceAnalysis

# Constants
DATASET_DIR = 'dataset'
GALLERY_PATH = 'face_gallery.pkl'

def generate_face_gallery(dataset_dir, output_path):
    print("Initializing InsightFace (ArcFace)...")
    # Initialize InsightFace
    # 'buffalo_l' is a good default model pack containing RetinaFace and ArcFace
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']) 
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    gallery = {}
    
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory '{dataset_dir}' not found.")
        return

    print(f"Scanning dataset at {dataset_dir}...")
    
    total_images = 0
    total_faces = 0
    
    for person_name in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
            
        print(f"Processing {person_name}...")
        embeddings = []
        
        for image_name in os.listdir(person_dir):
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue
                
            image_path = os.path.join(person_dir, image_name)
            img = cv2.imread(image_path)
            
            if img is None:
                print(f"  Warning: Could not read {image_path}")
                continue
            
            # Detect Faces
            faces = app.get(img)
            
            if len(faces) == 0:
                print(f"  No face detected in {image_name}. Skipping.")
                continue
            
            # Strategy: Pick the largest face (assume subject is main focus)
            # Sort by bounding box area (w * h)
            best_face = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)[0]
            
            embedding = best_face.embedding
            # Normalize embedding (ArcFace embeddings are usually normalized, but good practice)
            # embedding /= np.linalg.norm(embedding) 
            
            embeddings.append(embedding)
            total_images += 1
        
        if embeddings:
            gallery[person_name] = embeddings
            total_faces += len(embeddings)
            print(f"  Saved {len(embeddings)} face embeddings for {person_name}")
        else:
            print(f"  No valid faces found for {person_name}")

    # Save Gallery
    if gallery:
        try:
             with open(output_path, 'wb') as f:
                pickle.dump(gallery, f)
             print(f"\nSuccess! Face Gallery saved to {output_path}")
             print(f"Total Identities: {len(gallery)}")
             print(f"Total Face Embeddings: {total_faces}")
        except Exception as e:
            print(f"Error saving gallery: {e}")
    else:
        print("\nStructure empty. No gallery created.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=DATASET_DIR, help='path to dataset folder')
    parser.add_argument('--output', default=GALLERY_PATH, help='path to output pickle file')
    args = parser.parse_args()
    
    generate_face_gallery(args.dataset, args.output)
