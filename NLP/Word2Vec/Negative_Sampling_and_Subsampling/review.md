# Distributed Representations of Words and Phrases and their Compositionality

https://arxiv.org/pdf/1310.4546

---

### 논문의 핵심 기여

이 논문은 단어를 고품질의 벡터로 표현하는 효율적인 방법인 **Skip-gram 모델**을 소개하고, 기존 모델의 한계를 극복하기 위한 개선안을 제시

1.  **Subsampling**: "the"나 "a"처럼 매우 빈번하게 등장하는 단어들의 학습 참여율을 낮춰, 학습 속도를 크게 높이고 상대적으로 드문 단어들의 벡터 표현 품질을 향상
2.  **Negative Sampling**: 기존의 복잡한 출력층 구조(Hierarchical Softmax)를 대체하는 간단하면서도 효과적인 학습 방식으로, 계산 효율성을 높임
3.  **Learning Phrases**: 각 단어의 의미를 단순히 조합해서는 본래의 뜻을 알 수 없는 Phrase들을 식별하고, 이를 하나의 단위로 취급하여 벡터로 학습하는 방법을 제안

---

### Skip-gram 모델의 구조와 원리

Skip-gram 모델의 목표는 문장 속 **하나의 중심 단어(center word)를 가지고 주변에 어떤 단어들(context words)이 나타날지 예측**하는 것.

<img width="327" height="420" alt="image" src="https://github.com/user-attachments/assets/51146765-47d4-4a89-9df1-72d0e6723fa1" />

* **Input**: 중심 단어 `w(t)`가 One-Hot Vector 형태로 입력
* **Projection**: 입력된 중심 단어는 은닉층을 거치며 우리가 학습하고자 하는 저차원의 **단어 벡터**로 표현. 이 과정은 일반적인 신경망과 달리 복잡한 행렬 곱셈이 없어 효율적임
* **Output**: 이 단어 벡터를 사용해 주변 단어들, 즉 `w(t-2)`, `w(t-1)`, `w(t+1)`, `w(t+2)` 등을 예측

모델의 학습 목표는 아래의 로그 확률을 최대화하는 것

$$\frac{1}{T}\sum_{t=1}^{T}\sum_{-c\le j\le c,j\ne0}log~p(w_{t+j}|w_{t}) $$

여기서 `c`는 중심 단어로부터 얼마나 떨어진 단어까지를 주변 단어로 간주할지 결정하는 파라미터

---

### 계산 효율을 높이는 핵심 기법들

기본 Skip-gram 모델은 출력층에서 **Softmax 함수**를 사용하는데, 이는 엄청난 계산 비용이 발생함

-> 논문은 이 문제를 해결하기 위해 두 가지 방법을 제안

#### **1. Negative Sampling**
이 기법은 "주변 단어를 정확히 예측"하는 어려운 문제를 "**실제 주변 단어(positive sample)와 가짜 주변 단어(negative samples)를 구별**"하는 더 쉬운 이진 분류 문제로 바꿈

학습 과정은 실제 정답 단어 쌍에 대한 확률은 높이고(`log σ(...)`), 단어장에서 무작위로 뽑은 `k`개의 가짜 단어(negative samples) 쌍에 대한 확률은 낮추도록(`log σ(-...)`) 진행.

$$log~\sigma(v_{w_{O}}^{\prime}\top v_{w_{I}})+\sum_{i=1}^{k}\mathbb{E}_{w_{i}\sim P_{n}(w)}[log~\sigma(-{v_{w_{i}}^{\prime}}^{\top}v_{w_{I}})] $$

이 방식은 전체 단어장이 아닌, 정답 1개와 오답 `k`개, 즉 `k+1`개의 단어에 대해서만 계산하면 되므로 계산이 빨라짐.

#### **2. Subsampling of Frequent Words**
"in", "the", "a"와 같은 단어들은 자주 등장, 그러나 정보 가치는 낮음. 

> 예를 들어, "France"와 "the"의 동시 등장은 "France"와 "Paris"의 동시 등장보다 의미 있는 정보를 적게 제공

이러한 불균형을 해소하기 위해, 논문은 아래 수식을 이용해 특정 빈도수 이상의 단어를 확률적으로 학습에서 제외하는 방법을 제안.

$$P(w_{i})=1-\sqrt{\frac{t}{f(w_{i})}}$$

* $f(w_i)$: 단어 $w_i$의 빈도
* $t$: 기준이 되는 임계값(threshold)으로, 보통 $10^{-5}$ 정도를 사용.

이 기법은 학습 속도를 높이고, 특히 드물게 등장하는 단어들의 벡터 표현 정확도를 크게 향상

---

### 실험 결과 및 의의

#### **단어 유추 능력 평가**

<img width="744" height="243" alt="image" src="https://github.com/user-attachments/assets/03b3f52e-7a53-497d-afd5-acfc2f2ea866" />


> 제안된 기법들의 성능
* **성능**: **Negative Sampling**이 Hierarchical Softmax보다 전반적으로 더 높은 정확도를 보임.
* **속도와 정확도 향상**: **Subsampling**을 적용했을 때 학습 시간 단축, 정확도 향상


<img width="752" height="548" alt="image" src="https://github.com/user-attachments/assets/f24662b0-20ea-4ff4-bbc6-d7d00dbc9303" />


> 단어 벡터들이 언어적 패턴을 선형적으로 학습했음을 보여줌
* "Spain"에서 "Madrid"로 향하는 벡터와 "Italy"에서 "Rome"으로 향하는 벡터가 거의 평행
* 이는 `vec("Madrid") - vec("Spain") + vec("Italy")` 와 같은 벡터 연산이 `vec("Paris")`와 매우 가까워지는, 즉 **단어 유추**가 가능함을 의미

#### **Phrase 학습 및 Additive Compositionality**

<img width="709" height="115" alt="image" src="https://github.com/user-attachments/assets/315fe313-9812-4720-9ea6-a701ff7e7077" />

* **Pharase 학습**: "New York Times"와 같이 자주 함께 등장하는 단어들을 하나의 토큰("New\_York\_Times")으로 묶어 학습할 수 있음.
* **Additive Compositionality**: 벡터 간 덧셈이 의미 있는 결과를 만든다는 것을 보여줌
* > 예를 들어, `vec("Vietnam") + vec("capital")`의 결과 벡터는 `vec("Hanoi")` 벡터와 가장 가깝게 나타남.
  > -> 이는 단어 벡터가 해당 단어가 등장하는 문맥의 분포를 표현하기 때문

---

### 결론

이 논문은 대규모 텍스트 데이터로부터 고품질의 단어 및 구 벡터를 효율적으로 학습하는 Skip-gram 모델과 핵심 최적화 기법들(Negative Sampling, Subsampling)을 제안

이 연구를 통해 단어의 의미적, 문법적 관계가 벡터 공간에 선형적으로 표현될 수 있음을 보여줌.
