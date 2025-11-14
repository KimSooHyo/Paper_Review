# GloVe: Global Vectors for Word Representation

https://nlp.stanford.edu/pubs/glove.pdf

---

## GloVe(Global Vectors for Word Representation)란?

> 카운트 기반과 예측 기반을 모두 사용하는 임베딩 방법론
> 

## GloVe 등장 배경

이전에는 LSA(Latent Semantic Analysis)와 Word2Vec을 많이 사용함

1. **Global Matrix Factorization** 
    - LSA(Latent Semantic Analysis)
    - 문서에서의 각 단어의 빈도수를 카운트 한 행렬이라는 전체적인 통계 정보를 입력으로 받아 차원을 축소(Truncated SVD)하여 잠재된 의미를 끌어냄
    - **카운트 기반 방법론**
    
    → 전체적인 통계 정포를 고려, 그러나 단어 의미 유추 작업에는 성능이 떨어짐
    
2. Shallow Window-Based Methods: 
    - Word2Vec
    - Local Context Window에 집중하여 학습
    - 실제값과 예측값에 대한 오차를 손실 함수를 통해 줄여나가며 학습하는 **예측 기반의 방법론**
    - objective function은 context가 주어졌을 때 word를 예측하는 것.
    
    → 단어간 의미 유추 뛰어남, 그러나 전체적인 통계 정보 학습 X
    

> 여기서 전체적인 통계 정보 학습이란 **Global Co-occurrence Matrix** (전체 동시 등장 행렬)를 말함
Global Co-occurrence Matrix : 이 행렬의 각 칸 $X_{ij}$는 "전체 말뭉치에서 중심 단어 i가 등장했을 때, 주변 단어 j가 몇 번 등장했는가?"라는 횟수를 의미
> 

**GloVe는 두 방법론을 모두 사용 !**

## 윈도우 기반 동시 등장 행렬 (Window based Co-occurrence Matrix)

단어의 동시 등장 행렬은 행과 열을 전체 단어 집합의 단어들로 구성하고, i 단어의 윈도우 크기(Window Size) 내에서 k 단어가 등장한 횟수를 i행 k열에 기재한 행렬을 말함

- I like deep learning
- I like NLP
- I enjoy flying

<img width="663" height="451" alt="image" src="https://github.com/user-attachments/assets/9c096c06-ca05-4204-baba-66febd6b63b7" />


사진 출처: [09-05) 글로브(GloVe) - 딥 러닝을 이용한 자연어 처리 입문](https://wikidocs.net/22885)

# Glove

- **GloVe의 정의:** 이 모델은 말뭉치의 전체(Global) 통계 정보를 모델이 직접적으로 포착하여 학습하기 때문에 **GloVe(Global Vectors)**

### **동시 등장 확률(Co-occurrence Probability)**

<img width="527" height="116" alt="image" src="https://github.com/user-attachments/assets/517cfde7-067c-49da-a658-9d626bfaf7fb" />



> -  **예시 상황:** 타겟 단어인 'ice'(얼음)와 'steam'(증기)을 구별
- **단순 확률( $P_{ij}$ )의 한계**
    - 'water'(물) 같은 단어는 ice와 steam 모두와 관련 → 두 경우 모두 높은 확률
    - 'fashion'(패션) 같은 단어는 둘 다와 관련이 없음 → 둘 다 낮은 확률
    - 즉, 단순 확률만으로는 두 단어의 차이를 구별하기 어려움 (Noise).
- **해결책: 확률의 비율(Ratio)**
    - $P(k|ice) / P(k|steam)$ 같은 **비율**을 계산해야함
    - **결과:**
        - 비율이 **1보다 훨씬 크면** 'ice'의 고유한 속성과 관련 (예: solid).
        - 비율이 **1보다 훨씬 작으면** 'steam'의 고유한 속성과 관련 (예: gas).
> 

### 손실 함수

<img width="764" height="322" alt="image" src="https://github.com/user-attachments/assets/0b77ef9e-c5b2-4c3a-afc6-527582e2ca43" />




GloVe의 핵심은 **"두 단어의 확률 비율 $( \frac{P_{ik}}{P_{jk}} )$ 이 단어의 의미 차이를 잘 보여준다"**

$$
F(w_i, w_j, \tilde{w}*k) = \frac{P*{ik}}{P_{jk}}
$$

: F에 단어 벡터(w)를 넣으면 확률의 비율이 나옴



 **F를 알아내기 위한 과정**



**1. 의미 차이는 벡터의 뺄셈으로 표현 : 입력값을 벡터의 차이로 바꿈**

$$
F(w_i - w_j, \tilde{w}*k) = \frac{P*{ik}}{P_{jk}}
$$

ex) King - Man + Woman = Queen


**2. 결과는 숫자 하나(스칼라) : 내적을 통해 벡터를 숫자 하나로 바꿈**

$$
F\big((w_i - w_j)^T \tilde{w}*k\big) = \frac{P*{ik}}{P_{jk}}
$$

---

**3. 지수함수를 통해 뺄셈을 나눗셈으로 바꿈** $( F(A-B) = \frac{F(A)}{F(B)} )$

$$
e^{,w_i^T \tilde{w}*k} = P*{ik}
$$

$$
w_i^T \tilde{w}*k = \log(P*{ik}) = \log(X_{ik}) - \log(X_i)
$$

→ 두 단어 벡터를 내적하면, 둘이 만날 확률의 로그값이 됨



**4. $X_{ik} = X_{ki}$. 즉 i와 k의 위치를 바꿔도 식은 성립해야 함**

$\log(X_i)$를 단어 i가 원래 가진 고유한 성질(편향)이라고 생각 → $b_i$ 치환,
균형을 맞추기 위해 k쪽에도 $\tilde{b}_k$를 달아줌

$$
w_i^T \tilde{w}_k + b_i + \tilde{b}*k = \log(X*{ik})
$$

**해석:** (두 단어의 유사도) + (단어 i의 등장 빈도) + (단어 k의 등장 빈도) = (둘이 같이 나온 횟수의 로그값)


### 5. **최종 목표 함수 (Loss Function)** : 현실 데이터는 노이즈 O → 차이를 최소화하기 위한 손실 함수

$$
J = \sum_{i,j} f(X_{ij}) \left( w_i^T \tilde{w}_j + b_i + \tilde{b}*j - \log(X*{ij}) \right)^2
$$

* $(...)^2$ 부분은 "우리가 만든 식(예측값)과 실제 데이터(로그값)의 차이를 제곱한 것"
  → 이 값이 0에 가까울수록 좋은 모델

* 가중치 함수 $f(X_{ij})$ 도입

   <img width="338" height="194" alt="image" src="https://github.com/user-attachments/assets/58a5f2d2-9a62-4fcd-9e3f-dcf727bc9e5d" />


        
  - 등장 횟수가 너무 적으면 무시
  - 등장 횟수가 너무 많으면 적당한 상한선을 둠 → 너무 큰 영향력을 갖지 못하게 함

### Word2Vec과의 관계

GloVe는 Skip-gram이 암묵적으로 하려던 것을 **수학적으로 더 명확하고 효율적인 방식**으로 직접 푼 모델

### 속도

**속도 문제**

- GloVe는 단어 개수(V)만큼의 행과 열을 가진 거대한 행렬(X)을 다룸
- "단어가 10만 개면 행렬 크기는 100억칸인데, 계산이 터지는 거 아니야?"
    
    →대부분의 단어 쌍은 평생 한 번도 같이 안 나옴. 즉, 행렬의 대부분은 0
    
- **효율성:** GloVe는 0이 아닌 칸만 골라서 학습

### 실험

**단어 유추 평가 (Word Analogy Task)**

<img width="342" height="614" alt="image" src="https://github.com/user-attachments/assets/fa31d718-4fc0-47b4-94d0-7a4fe45c7b50" />



-  GloVe가 기존의 Word2Vec(Skip-gram, CBOW)이나 SVD 방식보다 훨씬 높은 정확도를 보임


**벡터 크기와 윈도우 크기**
<img width="731" height="272" alt="image" src="https://github.com/user-attachments/assets/618b9b9f-c9b1-4e45-b68a-98a6974796ce" />



- 200차원 정도까지는 성능이 오르다가, 그 이후로는 별 차이 없음
- 윈도우가 클수록(Larger window) 유리
    - **그래프 (b) Symmetric:** 중심 단어의 양옆(왼쪽, 오른쪽)을 다 보는 방식
    - **그래프 (c) Asymmetric:** 중심 단어의 왼쪽(이전 단어)만 보는 방식
    

**GloVe와 타 모델들의 속도와 성능 비교**

<img width="726" height="375" alt="image" src="https://github.com/user-attachments/assets/a73d807a-0e36-4b2c-b2fe-8653b1a39a20" />



- GloVe는 기존 모델보다 학습 속도가 빠르고, 충분히 학습시켰을 때 도달하는 최종 성능도 더 높음

---

### 참조

https://nlp.stanford.edu/pubs/glove.pdf

https://wikidocs.net/22885

https://sumim.tistory.com/entry/NLP-%EA%B7%BC%EB%B3%B8-%EB%85%BC%EB%AC%B8-1-GloVe-Global-Vectors-for-Word-Representation
