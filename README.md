# 🌟 동국대 종합설계 프로젝트

### 📌 프로젝트명  
**딥러닝 기반 음성 데이터를 활용한 고령자의 건강 이상 상태 예측 모델 연구**

이 프로젝트는 **파킨슨병(Parkinson’s Disease, PD) 환자의 음성 데이터**를 활용하여  
음성 특징(feature)을 기반으로 **정상인(HC, Healthy Control) / 파킨슨병 환자 분류**와  
**성별 분류**를 수행하는 딥러닝 기반 예측 모델 연구입니다.  

---

### 🧑‍💻 수행 역할  
1. 파킨슨병 환자의 음성 데이터 수집  
2. 수집된 음성을 **MFCC (Mel-Frequency Cepstral Coefficients)** 처리 후  
   일차원 벡터 형식의 `.csv` 파일로 변환  
3. 전처리된 데이터를 기반으로 파킨슨병 예측 모델 설계 및 학습  

---

### 🧪 데이터 전처리 방식  
- **음성 종류**: `a`, `e`, `i`, `o`, `u` 등 모음 발음 중심 음성  
- **분류 기준**:  
  - 성별: **남성 / 여성**  
  - 건강 상태: **정상(HC) / 파킨슨병(PD)**  
- 전처리 결과: **성별 + 질병 유무 기준 4분류 그룹 구성**

---

## 🔬 분류 기준 및 실험 조건

코드에서는 다양한 실험 조건과 모델 구조를 조합하여 성능을 비교할 수 있도록 설계되어 있습니다.  

### 🏗 모델 구조 차이
- **MLP (밀집층 기반)**  
- **Transformer Encoder 포함 버전**  
- **Transformer 없이 단순 신경망**  
- **Full fine-tune** vs **일부 계층만 학습**

---

### 🎧 데이터 전처리 및 입력 형태
- **MFCC만 사용**  
- **Mel 관련 추가 특징 사용**  
- **모음 발음 중심 데이터 활용**  
- **CSV 벡터 입력 형태**

---

### 🧩 분류 대상 조합
- **이진 분류**: 파킨슨병 여부 (정상 vs PD)  
- **다중 분류(4클래스)**: 성별 + 파킨슨병 여부  
  - 정상 남성  
  - 정상 여성  
  - 파킨슨병 남성  
  - 파킨슨병 여성  

---

### ⚙️ 학습 방식 / 하이퍼파라미터 변화
- Transformer **사용 여부**  
- Fine-tune **정도** (전체 vs 일부 계층)  
- 완전 연결층(full connected) 여부  

---

### 📑 실험 / 평가 스크립트
- 다양한 **테스트 스크립트(`test.py`, `test2.py` 등)** 를 활용하여  
  - 데이터 분할 방법  
  - 검증 방식  
  - 모델 성능 비교  
  등을 실험적으로 수행

---
| 파일명                                                                                                                          | 주된 기능 / 특징                                                                                                                                                                                                                          |
| ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **pd\_data.py**, **pd\_data\_mel.py**                                                                                        | 데이터 로딩/전처리 관련 코드. MFCC 추출, CSV 읽기, 특성(feature) 준비 등을 담당할 가능성이 높음. “mel”이 붙은 것은 mel-spectrogram 또는 mel 관련 처리 추가 버전일 가능성. ([GitHub][1])                                                                                               |
| **pd\_ast.py**                                                                                                               | “ast”는 아마 “attention, transformer” 또는 “augmented spectral transformer”등의 약어일 수 있음. Transformer 계열 모델을 실험하는 코드로 예상됨.                                                                                                                 |
| **final\_mlp.py**                                                                                                            | 최종적으로 MLP (Multi-Layer Perceptron) 기반의 모델 학습/평가 진행. → 단순한 fully connected 신경망.                                                                                                                                                      |
| **final\_trans.py**                                                                                                          | Transformer 기반 또는 전이학습(transfomer) 기반 실험의 최종 버전.                                                                                                                                                                                    |
| **non\_tran\_posi\_tran\_enco.py**, **pd\_non\_tran\_enco.py**, **pd\_non\_tran\_full\_fine.py**, **pd\_non\_transfomer.py** | 각각 transformer 미사용 / transformer encoder만 사용 / 전체 fine-tuning / transformer 없는 변형 등 모델 구조 차이를 두고 실험한 것들. “non\_tran”은 transformer 미사용 버전, “tran\_enco”는 transformer encoder 사용, “full\_fine”은 전체 fine-tune 가능한 transformer나 큰 네트워크. |
| **pd\_sex\_tran\_full\_fine.py**, **pd\_sex\_tran\_enco.py**, **pd\_sex\_ful.py**                                            | 성별(sex) 구분을 포함한 버전 + transformer 구조가 포함된/포함 안된 다양한 조합.                                                                                                                                                                              |
| **pd\_non\_ful.py**, **pd\_sex\_ful.py**                                                                                     | “full”은 아마 “full connected” 또는 “full fine-tune” 의미일 것 같고, transformer 없이 전체 네트워크 또는 기존 방식으로 학습하는 버전.                                                                                                                                |
| **test.py**, **test2.py**                                                                                                    | 실험/검증용 테스트 스크립트. 아마 데이터 분리 방법, 모델 평가, 다양한 파일 실험 비교 등을 위한 코드.                                                                                                                                                                        |

--
[1]: https://github.com/RealSeyeon/voice-health-prediction "GitHub - RealSeyeon/voice-health-prediction"

### 📚 참고 논문  
[성별에 따라 파킨슨 환자의 음성이 다른 점](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART001626518)

---

### 📂 데이터셋 요약

| 항목 | 내용 |
|------|------|
| 출처 | UCI Machine Learning Repository 외 |
| 구성 | 파킨슨병 환자 및 정상인의 음성 데이터 |
| 포맷 | `.wav` → MFCC → `.csv` (1D 벡터) |
| 용도 | 파킨슨병 여부 분류 모델 학습 및 테스트 |
| 분류 | 성별 / 질병 유무(HC/PD) 기준 4분류 |

---

