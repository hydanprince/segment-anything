import os
import sys
import argparse
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

YOLO_MODEL_PATH = "yolov8n.pt"
SAM_CHECKPOINT_PATH = "weights/sam_vit_l_0b3195.pth"

def main():
    # ---------------------------
    # Args
    # ---------------------------
    parser = argparse.ArgumentParser(
        description="Batch cattle face segmentation using YOLO and SAM",
        prog="sa"
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Root input directory containing subdirectories with images"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Root output directory (mirrors input structure)"
    )
    
    args = parser.parse_args()
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------
    # Device
    # ---------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ---------------------------
    # Load models once
    # ---------------------------
    print("Loading YOLO model...")
    yolo_model = YOLO(YOLO_MODEL_PATH)

    print("Loading SAM model...")
    sam = sam_model_registry["vit_l"](checkpoint=SAM_CHECKPOINT_PATH)
    sam.to(device)
    predictor = SamPredictor(sam)

    # ---------------------------
    # Error log
    # ---------------------------
    error_log_path = os.path.join(output_dir, "errors.txt")
    error_log = open(error_log_path, "w")

    total = 0
    success = 0
    errors = 0

    # ---------------------------
    # Walk input directory
    # ---------------------------
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in IMAGE_EXTENSIONS:
                continue

            image_path = os.path.join(root, filename)
            rel_path = os.path.relpath(image_path, input_dir)

            # Mirror subdirectory structure in output
            output_rel = os.path.splitext(rel_path)[0] + ".png"
            output_path = os.path.join(output_dir, output_rel)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            total += 1
            print(f"Processing [{total}]: {rel_path}")

            try:
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    raise FileNotFoundError(f"Could not read image: {image_path}")

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # YOLO detection
                results = yolo_model(image)
                cow_box = None
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        class_name = yolo_model.names[cls_id]
                        if class_name.lower() in ["cattle_face", "cow"]:
                            cow_box = box.xyxy[0].cpu().numpy()
                            break
                    if cow_box is not None:
                        break

                if cow_box is None:
                    raise ValueError("No cow/cattle face detected")

                x1, y1, x2, y2 = map(int, cow_box)

                # SAM segmentation
                predictor.set_image(image_rgb)
                input_box = np.array([x1, y1, x2, y2])
                masks, scores, _ = predictor.predict(
                    box=input_box,
                    multimask_output=True
                )

                # Select largest mask
                mask_areas = [np.sum(mask) for mask in masks]
                best_mask = masks[np.argmax(mask_areas)]

                # Apply mask and remove background
                masked_image = image_rgb.copy()
                masked_image[~best_mask] = 0

                # Trim bottom 20% to remove body
                ys, xs = np.where(best_mask)
                top = ys.min()
                bottom = ys.max()
                left = xs.min()
                right = xs.max()
                height = bottom - top
                new_bottom = top + int(height * 0.80)
                cropped_face = masked_image[top:new_bottom, left:right]

                # Save output
                cv2.imwrite(output_path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
                print(f"  Saved -> {output_path}")
                success += 1

            except Exception as e:
                errors += 1
                msg = f"{rel_path} | {type(e).__name__}: {e}"
                print(f"  ERROR: {msg}")
                error_log.write(msg + "\n")

    error_log.close()

    # ---------------------------
    # Summary
    # ---------------------------
    print("\n--- Done ---")
    print(f"Total images found : {total}")
    print(f"Successfully saved : {success}")
    print(f"Errors             : {errors}")
    if errors > 0:
        print(f"Error log          : {error_log_path}")


if __name__ == "__main__":
    main()
