import os
import cv2

base_dir = "dataset/roboflow_image_v3"
output_dir = "dataset/cropped_v3"

splits = ["train", "valid", "test"]

os.makedirs(output_dir, exist_ok=True)

count = 0

for split in splits:
    image_dir = os.path.join(base_dir, split, "images")
    label_dir = os.path.join(base_dir, split, "labels")

    for filename in os.listdir(image_dir):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(image_dir, filename)
        label_path = os.path.join(
            label_dir,
            os.path.splitext(filename)[0] + ".txt"
        )

        if not os.path.exists(label_path):
            print("라벨 없음:", filename)
            continue

        img = cv2.imread(image_path)
        h, w = img.shape[:2]

        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls, x_center, y_center, box_w, box_h = map(float, parts)

            x_center *= w
            y_center *= h
            box_w *= w
            box_h *= h

            x1 = int(x_center - box_w / 2)
            y1 = int(y_center - box_h / 2)
            x2 = int(x_center + box_w / 2)
            y2 = int(y_center + box_h / 2)

            # 살짝 여유 주기
            margin = 20
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(w, x2 + margin)
            y2 = min(h, y2 + margin)

            crop = img[y1:y2, x1:x2]

            save_name = f"{split}_{os.path.splitext(filename)[0]}_{i}.jpg"
            save_path = os.path.join(output_dir, save_name)

            cv2.imwrite(save_path, crop)
            count += 1
            print("crop 저장:", save_name)

print("전체 crop 완료:", count)
