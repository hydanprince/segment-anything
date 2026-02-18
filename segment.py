import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

# ---------------------------
# 1️⃣ Device
# ---------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ---------------------------
# 2️⃣ Paths
# ---------------------------
IMAGE_PATH = "images/WhatsApp Image 2026-02-18 at 16.52.50.jpeg"  # Replace with your test image
OUTPUT_PATH = "outputs/cattle_face_only.png"

YOLO_MODEL_PATH = "yolov8n.pt"  # Trained YOLOv8 model weights
SAM_CHECKPOINT_PATH = "weights/sam_vit_l_0b3195.pth"  # SAM checkpoint

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# ---------------------------
# 3️⃣ Load image
# ---------------------------
image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ---------------------------
# 4️⃣ Load YOLOv8 model
# ---------------------------
yolo_model = YOLO(YOLO_MODEL_PATH)

results = yolo_model(image)

cow_box = None

for r in results:
    for box in r.boxes:
        cls_id = int(box.cls[0])
        class_name = yolo_model.names[cls_id]
        if class_name.lower() in ["cattle_face", "cow"]:  # adjust to your class
            cow_box = box.xyxy[0].cpu().numpy()
            break
    if cow_box is not None:
        break

if cow_box is None:
    raise ValueError("No cow/cattle face detected in image")

x1, y1, x2, y2 = map(int, cow_box)
print(f"Detected box: {x1, y1, x2, y2}")

# ---------------------------
# 5️⃣ Load SAM
# ---------------------------
sam = sam_model_registry["vit_l"](checkpoint=SAM_CHECKPOINT_PATH)
sam.to(device)
predictor = SamPredictor(sam)
predictor.set_image(image_rgb)

input_box = np.array([x1, y1, x2, y2])

masks, scores, _ = predictor.predict(
    box=input_box,
    multimask_output=True
)

# Select largest mask
mask_areas = [np.sum(mask) for mask in masks]
best_mask = masks[np.argmax(mask_areas)]

# ---------------------------
# 6️⃣ Apply mask and remove background
# ---------------------------
masked_image = image_rgb.copy()
masked_image[~best_mask] = 0

# ---------------------------
# 7️⃣ Remove body (trim bottom 20%)
# ---------------------------
ys, xs = np.where(best_mask)
top = ys.min()
bottom = ys.max()
left = xs.min()
right = xs.max()

height = bottom - top
new_bottom = top + int(height * 0.80)  # keep top 80%, trim bottom 20%
cropped_face = masked_image[top:new_bottom, left:right]

# ---------------------------
# 8️⃣ Display
# ---------------------------
plt.figure(figsize=(6, 6))
plt.imshow(cropped_face)
plt.title("Cattle Face (No Body)")
plt.axis("off")
plt.show()

# ---------------------------
# 9️⃣ Save Output
# ---------------------------
cv2.imwrite(OUTPUT_PATH, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
print(f"Cattle face saved to {OUTPUT_PATH}")

