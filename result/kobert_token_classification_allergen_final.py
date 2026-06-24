# ===============================
# 0. 라이브러리 설치
# ===============================
!pip install transformers datasets evaluate accelerate seqeval -q

import os
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    pipeline
)
import evaluate


# ===============================
# 1. GitHub CSV 데이터 불러오기
# ===============================
github_raw_url = "https://raw.githubusercontent.com/chaey22/OSP-TEAM37/main/dataset/kobert_ner_all.csv"

try:
    df = pd.read_csv(github_raw_url, encoding="utf-8")
except Exception:
    df = pd.read_csv(github_raw_url, encoding="cp949")

print("데이터 로드 성공:", df.shape)
display(df.head())


# ===============================
# 2. 학습 가능한 데이터만 필터링
# ===============================
df_clean = df[df["quality_label"] == "readable"].dropna(subset=["tokens", "labels"])

print(f"전체 데이터 개수: {len(df)}개")
print(f"학습 가능한 데이터 개수: {len(df_clean)}개")

df_clean["token_list"] = df_clean["tokens"].apply(lambda x: x.split("|"))
df_clean["label_list"] = df_clean["labels"].apply(lambda x: x.split("|"))


# ===============================
# 3. BIO 라벨 목록 생성
# ===============================
unique_labels = set()

for labels in df_clean["label_list"]:
    unique_labels.update(labels)

unique_labels = sorted(list(unique_labels))

label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2id.items()}

print("라벨 목록:", unique_labels)
print("label2id:", label2id)


# ===============================
# 4. KLUE-BERT 모델 불러오기
# ===============================
model_checkpoint = "klue/bert-base"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(unique_labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

print("KLUE-BERT 모델 로드 완료")


# ===============================
# 5. 토큰과 라벨 정렬 함수
# ===============================
def align_labels_with_tokens(token_list, label_list):
    tokenized_inputs = tokenizer(
        token_list,
        is_split_into_words=True,
        truncation=True,
        padding=False
    )

    labels = []
    word_ids = tokenized_inputs.word_ids()

    for word_idx in word_ids:
        if word_idx is None:
            labels.append(-100)
        else:
            label_str = label_list[word_idx]
            labels.append(label2id[label_str])

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


input_ids_list = []
attention_mask_list = []
labels_list = []

for _, row in df_clean.iterrows():
    encoded = align_labels_with_tokens(row["token_list"], row["label_list"])
    input_ids_list.append(encoded["input_ids"])
    attention_mask_list.append(encoded["attention_mask"])
    labels_list.append(encoded["labels"])

dataset = Dataset.from_dict({
    "input_ids": input_ids_list,
    "attention_mask": attention_mask_list,
    "labels": labels_list
})

print("학습용 Dataset 생성 완료")
print(dataset)


# ===============================
# 6. Train / Validation 분리
# ===============================
dataset_split = dataset.train_test_split(test_size=0.2, seed=42)

train_data = dataset_split["train"]
val_data = dataset_split["test"]

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


# ===============================
# 7. F1-score 평가 함수
# ===============================
metric = evaluate.load("seqeval")
label_list = unique_labels

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=-1)

    true_predictions = [
        [label_list[pred] for pred, lab in zip(prediction, label) if lab != -100]
        for prediction, label in zip(predictions, labels)
    ]

    true_labels = [
        [label_list[lab] for pred, lab in zip(prediction, label) if lab != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(
        predictions=true_predictions,
        references=true_labels
    )

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"]
    }


# ===============================
# 8. 학습 설정
# ===============================
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=5e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch"
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)


# ===============================
# 9. 모델 학습
# ===============================
print("KoBERT NER 모델 학습 시작")
trainer.train()


# ===============================
# 10. 학습된 모델 저장
# ===============================
output_dir = "./final_kobert_allergen_model"

trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print("모델 저장 완료:", output_dir)


# ===============================
# 11. 저장된 모델 로드
# ===============================
loaded_model = AutoModelForTokenClassification.from_pretrained(
    output_dir,
    local_files_only=True
)

loaded_tokenizer = AutoTokenizer.from_pretrained(
    output_dir,
    local_files_only=True
)

nlp_ner = pipeline(
    "ner",
    model=loaded_model,
    tokenizer=loaded_tokenizer,
    aggregation_strategy="simple"
)

print("KoBERT 알레르기 분석 모델 준비 완료")


# ===============================
# 12. KoBERT 라벨명 → 한국어 알레르기명 변환
# ===============================
label_to_korean = {
    "MILK": "우유",
    "SOY": "대두",
    "WHEAT": "밀",
    "EGG": "계란",
    "PEANUT": "땅콩",

    "B-MILK": "우유",
    "B-SOY": "대두",
    "B-WHEAT": "밀",
    "B-EGG": "계란",
    "B-PEANUT": "땅콩"
}


# ===============================
# 13. 알레르기 성분 사전 기반 보완
# ===============================
allergy_dictionary = {
    "우유": [
        "우유", "분유", "전지분유", "탈지분유", "혼합분유",
        "유청", "유청분말", "농축유청단백", "카제인",
        "카제인나트륨", "유당", "버터", "버터밀크"
    ],
    "대두": [
        "대두", "대두유", "대두단백", "분리대두단백",
        "콩", "콩기름", "두유"
    ],
    "밀": [
        "밀", "밀가루", "소맥분", "글루텐", "밀단백"
    ],
    "계란": [
        "계란", "달걀", "난백", "난황", "전란", "난백분", "난황분말"
    ],
    "땅콩": [
        "땅콩", "땅콩분태", "땅콩버터", "땅콩가루"
    ]
}


# ===============================
# 14. 사용자 맞춤형 위험도 판정
# ===============================
while True:
    user_input_allergies = input(
        "조심해야 하는 알레르기 성분을 입력하세요 "
        "(예: 우유, 땅콩, 계란 / 종료하려면 q): "
    ).strip()

    if user_input_allergies.lower() == "q":
        print("프로그램을 종료합니다.")
        break

    user_allergies = [
        x.strip()
        for x in user_input_allergies.split(",")
        if x.strip()
    ]

    ocr_text = input("분석할 성분표 문장을 입력하세요: ").strip()

    if not ocr_text:
        print("성분표 문장을 입력해 주세요.\n")
        continue

    # -------------------------------
    # 1) KoBERT NER 분석
    # -------------------------------
    results = nlp_ner(ocr_text)

    detected_allergens = set()
    detected_words = []

    for entity in results:
        word = entity["word"].replace("##", "").strip()
        label = entity["entity_group"]

        korean_label = label_to_korean.get(label, label)

        if korean_label in ["우유", "대두", "밀", "계란", "땅콩"]:
            detected_allergens.add(korean_label)
            detected_words.append((word, korean_label, entity["score"], "KoBERT"))

    # -------------------------------
    # 2) 성분 사전 기반 보완 탐지
    # -------------------------------
    for allergen, keywords in allergy_dictionary.items():
        for keyword in keywords:
            if keyword in ocr_text:
                detected_allergens.add(allergen)
                detected_words.append((keyword, allergen, 1.00, "Dictionary"))

    # -------------------------------
    # 3) 사용자 알레르기와 비교
    # -------------------------------
    matched = set(user_allergies) & detected_allergens

    print("\n" + "=" * 60)
    print("알레르기 성분 분석 결과")
    print("=" * 60)

    print(f"입력 성분표: {ocr_text}")
    print(f"사용자 등록 알레르기: {', '.join(user_allergies)}")

    print("\n검출된 알레르기 성분:")
    if detected_allergens:
        for allergen in sorted(detected_allergens):
            print(f"- {allergen}")
    else:
        print("- 없음")

    print("\n상세 탐지 결과:")
    if detected_words:
        # 중복 제거용
        seen = set()
        for word, allergen, score, source in detected_words:
            key = (word, allergen, source)
            if key in seen:
                continue
            seen.add(key)

            print(f"- '{word}' → {allergen} (탐지 방식: {source}, 확신도: {score:.2f})")
    else:
        print("- 탐지된 알레르기 관련 단어 없음")

    print("\n최종 판정:")
    if matched:
        print("위험")
        print(f"이 식품에는 사용자가 조심해야 하는 성분({', '.join(sorted(matched))})이 포함되어 있습니다.")
        print("섭취 전 주의가 필요합니다.")
    else:
        print("안전")
        print("사용자 등록 알레르기 성분이 탐지되지 않았습니다.")

    print("=" * 60 + "\n")
