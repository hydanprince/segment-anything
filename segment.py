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
IMAGE_PATH = "images/cattleimage17.jpeg"
OUTPUT_PATH = "outputs/cattle_face_only.png"
CHECKPOINT_PATH = "sam_vit_l_0b3195.pth"

# ---------------------------
# 3️⃣ Load image
# ---------------------------
image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ---------------------------
# 4️⃣ YOLO Cow Detection
# ---------------------------
yolo_model = YOLO("yolov8n.pt")
results = yolo_model(image)

cow_box = None

for r in results:
    for box in r.boxes:
        cls_id = int(box.cls[0])
        if yolo_model.names[cls_id] == "cow":
            cow_box = box.xyxy[0].cpu().numpy()
            break

if cow_box is None:
    raise ValueError("No cow detected in image")

x1, y1, x2, y2 = map(int, cow_box)

# ---------------------------
# 5️⃣ Load SAM
# ---------------------------
sam = sam_model_registry["vit_l"](checkpoint=CHECKPOINT_PATH)
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
# 6️⃣ Remove Background
# ---------------------------
masked_image = image_rgb.copy()
masked_image[~best_mask] = 0

# ---------------------------
# 7️⃣ Remove Body (Trim Bottom 20%)
# ---------------------------
ys, xs = np.where(best_mask)

top = ys.min()
bottom = ys.max()
left = xs.min()
right = xs.max()

height = bottom - top

# Remove bottom 20% (neck/body area)
new_bottom = top + int(height * 0.80)

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
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
cv2.imwrite(
    OUTPUT_PATH,
    cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)
)

print(f"Cattle face saved to {OUTPUT_PATH}")



