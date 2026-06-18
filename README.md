# OSP-TEAM37

## 프로젝트 소개
식품 성분표 이미지를 분석하여 알레르기 유발 성분을 탐지하는 AI 시스템

## 팀원
- 김채연
- 이연우
- 박수민

## 진행 상황

### 2026-06-15

#### 데이터 수집
- 성분표 이미지 101장 직접 촬영 및 수집
- Roboflow 업로드 완료
- ingredient_label 객체 라벨링 완료

#### 1차 객체 탐지 모델 학습
- Roboflow 3.0 Object Detection 사용
- 데이터셋 버전: 성분 라벨 v1

성능 결과
- mAP@50 : 99.5%
- Precision : 96.9%
- Recall : 100%
- F1 Score : 98.5%

#### 현재 진행 중
- 추가 이미지 수집 및 라벨링
- 목표 데이터 수 : 약 400장
- 성분표 영역 검출 성능 향상 예정


### 2026-06-16

#### 데이터셋 확장

* 성분표 이미지 추가 수집
* 총 385장 라벨링 완료
* Roboflow 데이터셋 버전 업데이트

#### 2차 객체 탐지 모델 재학습

* Roboflow 3.0 Object Detection 사용
* 데이터셋 버전: ingredient_label_v2
* 학습 이미지 수 : 385장

성능 결과

* mAP@50 : 99.5%
* Precision : 95.2%
* Recall : 100%
* F1 Score : 97.5%

#### OCR 환경 구축

* EasyOCR 설치 완료
* OCR 테스트 코드 작성
* 성분표 이미지에서 텍스트 추출 성공
* OCR 결과 CSV 생성 작업 진행 중

#### 현재 진행 중

* 385장 이미지 OCR 수행
* OCR 결과 CSV 생성
* BIO 라벨링용 텍스트 데이터 구축 준비

#### 다음 단계

* OCR 결과를 CSV 파일로 정리
* 알레르기 성분 BIO 라벨링 진행
* KoBERT NER 학습 데이터셋 구축


2026-06-17/18
OCR 결과 정리 및 품질 검수

* 객체 탐지로 잘라낸 성분표 이미지의 EasyOCR 결과 CSV 정리
* 이미지 파일명과 OCR 결과 파일명 매칭
* OCR 원문과 사람이 직접 교정한 텍스트를 함께 저장
* 이미지 판독 가능 여부를 readable / unreadable로 분류
* 판독이 어려운 이미지는 학습 데이터에서 제외할 수 있도록 구분

알레르기 성분 사전 구축

* 초기 탐지 대상 알레르기 항목 5종 선정

  * 우유(MILK)
  * 계란(EGG)
  * 땅콩(PEANUT)
  * 대두(SOY)
  * 밀(WHEAT)
* 원재료명과 알레르기 분류를 연결한 allergy_dictionary.csv 작성
* 전지분유, 유청, 난백액, 대두유, 밀가루 등 파생 성분을 알레르기 유형별로 정리

BIO 라벨링 기준 수립

* KoBERT NER 학습을 위한 BIO 라벨 체계 정의
* B-MILK, B-EGG, B-PEANUT, B-SOY, B-WHEAT 및 I 라벨 구성
* 알레르기 관련 성분이 아닌 토큰은 O로 처리
* tokens와 labels의 개수가 일치하도록 데이터 검수 기준 설정
* BIO 라벨링 가이드 문서 작성

KoBERT NER 샘플 데이터셋 구축

* 데이터 컬럼 구성

  * image_name
  * ocr_text
  * corrected_text
  * tokens
  * labels
  * quality_label
* 이미지 파일명과 OCR 결과를 매칭하여 ocr_text 입력
* 이미지에서 직접 확인한 알레르기 표시 문구를 corrected_text로 교정
* 성분 단위 토큰과 BIO 라벨 작성
* 초기 샘플 데이터셋을 GitHub 브랜치에 업로드하고 Pull Request 생성

현재 진행 중

* readable 이미지의 BIO 라벨링 데이터 추가 구축
* OCR 오인식 결과 수동 교정
* 알레르기 성분 사전 확장 및 검수
* KoBERT NER 데이터 로딩 및 토큰-라벨 정렬 코드 준비

다음 단계

* 샘플 데이터로 KoBERT NER 학습 코드 동작 확인
* 데이터 형식 검증 후 학습 데이터셋 확장
* KoBERT 토크나이저의 서브워드와 BIO 라벨 정렬
* 학습·검증·테스트 데이터 분리
* 모델 학습 후 Precision, Recall, F1 Score 평가
* OCR 결과에서 탐지된 알레르기 성분과 사용자 선택 알레르기를 비교하여 위험 여부 출력
