# OSP-TEAM37

## 프로젝트 실행 환경 및 사용 방법

본 프로젝트는 식품 성분표 이미지에서 알레르기 유발 성분을 인식하기 위한 시스템으로, YOLO 기반 성분표 탐지, EasyOCR 기반 텍스트 추출, KoBERT 기반 알레르기 성분 인식 모듈로 구성되어 있음.

프로젝트는 Google Colab 환경에서 개발 및 테스트를 수행하였으며, KoBERT 모델 학습 및 추론 과정에서 GPU 사용을 권장함.

### 개발 환경

* Google Colab
* Python 3.10 이상
* GPU(T4 이상 권장)

### 사용 라이브러리

* ultralytics
* easyocr
* pandas
* numpy
* torch
* transformers
* datasets
* evaluate
* accelerate
* seqeval
* opencv-python

### 실행 순서

#### 1. 성분표 영역 탐지

YOLO 기반 객체 탐지 모델을 이용하여 식품 이미지에서 성분표 영역을 검출함.

```bash
python crop_labels.py
```

#### 2. OCR 텍스트 추출

탐지된 성분표 영역에 EasyOCR을 적용하여 텍스트를 추출함.

```bash
python ocr_test_fianl.py
```

실행 결과는 CSV 파일 형태로 저장되며, 이후 알레르기 성분 인식 모델의 입력 데이터로 활용됨.

예시 출력 파일

```text
ocr_result_crop_v3.csv
```

#### 3. KoBERT 기반 알레르기 성분 인식

##### 1. Google Colab 환경에서 `evaluate_kobert_allergen_ner.ipynb` 파일을 실행함.

NER(Named Entity Recognition) 모델을 이용하여 성분표 내 알레르기 유발 성분을 인식함.

* **결과 정산 및 분석 (`evaluate_kobert_allergen_ner.ipynb`)**: 구글 드라이브 마운트 없이 깃허브 Raw 링크를 통해 실시간으로 최종 데이터셋을 로드하고, 에포크별 Training Loss 및 Validation F1-Score 추이 테이블과 시각화 그래프(`kobert_learning_curve.png`)를 자동으로 출력 및 저장하는 모델 성과 정산 전용 노트북 파일임.

현재 학습된 알레르기 성분은 다음과 같음.

* 우유(MILK)
* 대두(SOY)
* 밀(WHEAT)
* 계란(EGG)
* 땅콩(PEANUT)



##### 2. Google Colab 환경에서 `kobert_token_classification_allergen.py` 파일을 실행함.

NER(Named Entity Recognition) 모델을 이용하여 성분표 내 알레르기 유발 성분을 인식함.

현재 학습된 알레르기 성분은 다음과 같음.

* 우유(MILK)
* 대두(SOY)
* 밀(WHEAT)
* 계란(EGG)
* 땅콩(PEANUT)

예시 입력

```text
혼합분유, 탈지분유
```

예시 출력

```text
검출된 알레르기 성분
- 우유

최종 판정
위험
```

### 실행 시 주의사항

KoBERT 모델은 최초 실행 시 HuggingFace 서버에서 KLUE-BERT 모델을 다운로드하므로 인터넷 연결이 필요함.

또한 실제 식품 포장 이미지에서는 반사광, 저조도, 곡면 포장재 등의 영향으로 OCR 인식 오류가 발생할 수 있으며, OCR 결과 품질에 따라 알레르기 성분 인식 성능이 달라질 수 있음.

초기 개발 목표는 사용자가 성분표 이미지를 입력하면 알레르기 위험도를 자동으로 판별하는 End-to-End 시스템 구현이었음. 그러나 실제 성분표 이미지에서 OCR 인식 오류가 빈번하게 발생하여 최종 구현에서는 성분표 영역 탐지, OCR 텍스트 추출, KoBERT 기반 알레르기 성분 인식 기능 구현에 중점을 두어 개발을 진행하였음.
