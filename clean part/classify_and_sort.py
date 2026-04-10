import os
import shutil
import torch
import numpy as np
from PIL import Image
from transformers import pipeline
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────
SOURCE_DIR = r"D:\J\data of the cars"

# Stage 1: Main View Classification
MAIN_VIEWS = {
    "Front": [
        "the front view of a car", 
        "looking squarely at the car headlights", 
        "view of the front car grille"
    ],
    "Back": [
        "the rear view of a car", 
        "the back side of a car showing taillights", 
        "view of the car trunk and license plate"
    ],
    "Side": [
        "the side profile view of a car", 
        "view showing the car doors from the side", 
        "side view of a car showing two wheels"
    ]
}

# Stage 2: Side Differentiation (Only run if Side is chosen)
SIDE_VIEWS = {
    "Left_Side": [
        "the left side view of a vehicle", 
        "profile view of the driver's side of a car", 
        "a car facing right showing its left side"
    ],
    "Right_Side": [
        "the right side view of a vehicle", 
        "profile view of the passenger's side of a car", 
        "a car facing left showing its right side"
    ]
}

def main():
    if not os.path.exists(SOURCE_DIR):
        print(f"ERROR: Source directory not found: {SOURCE_DIR}")
        return

    # 1. Initialize Pipeline
    print(">>> Loading CLIP model (Zero-Shot Classifier)...")
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline(
        "zero-shot-image-classification",
        model="openai/clip-vit-base-patch32",
        device=device
    )

    # 2. Preparation
    target_folders = ["Front", "Back", "Left_Side", "Right_Side", "Unclassified"]
    for folder in target_folders:
        os.makedirs(os.path.join(SOURCE_DIR, folder), exist_ok=True)

    all_files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f">>> Processing {len(all_files)} images with Hierarchical Voting...")

    # 3. Processing Loop
    for filename in tqdm(all_files, desc="Refined Sorting"):
        file_path = os.path.join(SOURCE_DIR, filename)
        
        try:
            img = Image.open(file_path).convert("RGB")
            
            # --- STAGE 1: Determine Front/Back/Side ---
            main_labels = list(MAIN_VIEWS.keys())
            # Flatten prompts for first pass
            flat_main_prompts = []
            rev_map = {}
            for k, prompts in MAIN_VIEWS.items():
                for p in prompts:
                    flat_main_prompts.append(p)
                    rev_map[p] = k
            
            results1 = classifier(img, candidate_labels=flat_main_prompts)
            
            # Aggregate scores for Stage 1
            cat_scores = {k: 0.0 for k in MAIN_VIEWS.keys()}
            for r in results1:
                cat_scores[rev_map[r['label']]] += r['score']
            
            win1 = max(cat_scores, key=cat_scores.get)
            conf1 = cat_scores[win1]

            if conf1 < 0.4: # Low confidence barrier
                target = "Unclassified"
            elif win1 == "Front":
                target = "Front"
            elif win1 == "Back":
                target = "Back"
            else:
                # --- STAGE 2: If Side, determine Left/Right ---
                flat_side_prompts = []
                side_rev_map = {}
                for k, prompts in SIDE_VIEWS.items():
                    for p in prompts:
                        flat_side_prompts.append(p)
                        side_rev_map[p] = k
                
                results2 = classifier(img, candidate_labels=flat_side_prompts)
                
                side_scores = {k: 0.0 for k in SIDE_VIEWS.keys()}
                for r in results2:
                    side_scores[side_rev_map[r['label']]] += r['score']
                
                target = max(side_scores, key=side_scores.get)

            # Move file
            shutil.move(file_path, os.path.join(SOURCE_DIR, target, filename))

        except Exception as e:
            # print(f"\n[!] Error {filename}: {e}")
            try:
                shutil.move(file_path, os.path.join(SOURCE_DIR, "Unclassified", filename))
            except: pass

    # 4. Final Count
    print("\n✅ Refined Classification complete!")
    for folder in target_folders:
        count = len(os.listdir(os.path.join(SOURCE_DIR, folder)))
        print(f"  - {folder}: {count} images")

if __name__ == "__main__":
    main()
