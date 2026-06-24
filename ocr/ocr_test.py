import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import csv
import easyocr

image_folder = "dataset/roboflow_image_v2"
output_csv = "ocr_result_all.csv"

reader = easyocr.Reader(['ko', 'en'])

with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "ocr_text"])

    for filename in os.listdir(image_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(image_folder, filename)

            try:
                img = cv2.imread(image_path)
                img = cv2.resize(img, (1000, 1333))

                temp_path = "temp.jpg"
                cv2.imwrite(temp_path, img)

                result = reader.readtext(temp_path, detail=0)
                text = " ".join(result)

                writer.writerow([filename, text])
                print("완료:", filename)

            except Exception as e:
                writer.writerow([filename, "ERROR"])
                print("오류:", filename, e)

print("전체 OCR 완료:", output_csv)
