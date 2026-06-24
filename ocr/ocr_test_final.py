import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import easyocr
import pandas as pd


# OCR 모델 로드
reader = easyocr.Reader(['ko', 'en'])

# crop된 이미지 폴더
image_folder = "dataset/cropped_v3"

results = []

# 모든 이미지 순회
for filename in os.listdir(image_folder):

    if filename.lower().endswith((".jpg", ".jpeg", ".png")):

        image_path = os.path.join(image_folder, filename)

        try:
            text = reader.readtext(
                image_path,
                detail=0
            )

            extracted_text = " ".join(text)

            results.append([
                filename,
                extracted_text
            ])

            print("완료:", filename)

        except Exception as e:
            print("오류:", filename)
            print(e)

# CSV 저장
df = pd.DataFrame(
    results,
    columns=["image_name", "ocr_text"]
)

df.to_csv(
    "ocr_result_crop_v3.csv",
    index=False,
    encoding="utf-8-sig"
)

print("OCR 완료")
print("저장 파일: ocr_result_crop_v3.csv")
