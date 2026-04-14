from ultralytics import YOLO
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import pandas as pd

df = {}
df[0] = ["R", "G", "B", "S"]

model = YOLO("./runs/detect/train/weights/last.pt")
root = tk.Tk()
root.withdraw()

img_path = filedialog.askopenfilename()
image = cv2.imread(img_path)

results = model(image, conf=0.02)

r = results[0]

for r in results:
    boxes = r.boxes.xyxy.cpu().numpy()  # [N, 4]
    if r.boxes is None:
        print("Not detected")
        continue

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)

        # label id
        label = f"ID: {i+1}"
        cv2.putText(
            image,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,0,0),
            2
        )

        # centre
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        half = 25
        h, w, _ = image.shape

        # frame
        x_start = max(cx - half, 0)
        x_end   = min(cx + half, w)

        y_start = max(cy - half, 0)
        y_end   = min(cy + half, h)

        crop = image[y_start:y_end, x_start:x_end]

        mean_color = crop.mean(axis=(0, 1))

        mean_color = mean_color[::-1]
        S = (max(mean_color) - min(mean_color)) / max(mean_color)
        
        S = round(S, 3)

        S_label = f"S = {S}"
        cv2.putText(
            image,
            S_label,
            (x1 + 75, y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,0,0),
            2
        )

        df[i+1] = [mean_color[0], mean_color[1], mean_color[2], S]

        print(f"Object{i}: RGB = {mean_color}, Saturation = {S}")

cv2.imshow("result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

df = pd.DataFrame(df)

df.to_csv("./result.csv")
