# Understanding the difficulty of training deep feedforward neural networks (Xavier Init)

https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf

이 논문은 심층 신경망(Deep Neural Networks)의 학습이 어려운 이유를 분석하고, 이를 해결하기 위한 새로운 가중치 초기화 기법(Xaiver Initialization)을 제안

### 1. 초록 (Abstract)

 **왜 표준적인 경사 하강법과 무작위 초기화 방식이 심층 신경망에서 잘 동작하지 않는지?**

* **문제점 1: 활성화 함수**
    * **로지스틱 시그모이드(Logistic Sigmoid)** 함수는 평균이 0이 아니기 때문에, 심층 신경망의 최상위 은닉층(top hidden layer)을 쉽게 **포화(saturation)** 상태로 만듦. 이로 인해 그래디언트가 잘 흐르지 않아 학습이 저해됨.
* **문제점 2: 신호(Signal)의 소실 또는 증폭**
    * 층을 거듭할수록 활성화 값(forward propagation)이나 기울기(back-propagation)의 분산이 불안정해지는 현상을 발견함.
    * 특히 각 층의 변환을 나타내는 **자코비안 행렬(Jacobian matrix)의 특이값(singular values)이 1에서 멀어지면** 학습이 어려워짐
* **해결책 제안**
    * 이러한 문제들을 바탕으로, **정보(활성화 값과 기울기)가 여러 층에 걸쳐 잘 흐르도록 보장하는 새로운 초기화 기법**을 제안하여 훨씬 빠른 수렴을 가능하게 함.

---

### 2. 실험 환경 및 데이터셋 (Experimental Setting and Datasets)

* **데이터셋**: 모델 성능을 검증하기 위해 다양한 데이터셋을 사용
    * **Shapeset-3x2**: 온라인 학습 시나리오를 위해 직접 제작한 합성 이미지 데이터셋
    * **MNIST**: 손글씨 숫자 이미지 데이터셋
    * **CIFAR-10**: 10개 클래스의 컬러 이미지 데이터셋
    * **Small-ImageNet**: 10개 클래스의 흑백 이미지 데이터셋
* **모델 구조 및 학습**:
    * 1~5개의 은닉층을 가진 피드포워드 신경망을 사용했으며, 각 층은 1000개의 유닛으로 구성됨.
    * 활성화 함수로는 **Sigmoid, Hyperbolic Tangent (tanh), Softsign** 세 가지를 비교함.
    * 기존 가중치 초기화 방식(Standard Initialization)은 이전 층의 크기(fan-in) $n$을 사용하여 다음과 같은 균등 분포를 따름:
        $$W \sim U\left[-\frac{1}{\sqrt{n}}, \frac{1}{\sqrt{n}}\right]$$

---

### 3. 활성화 함수와 학습 중 포화 문제 (Effect of Activation Functions and Saturation)

* **Sigmoid**: 0이 아닌 평균값 때문에 최상위 은닉층이 0으로 포화되는 치명적인 문제가 발생함. 활성화 값이 0에 가까워지면 시그모이드의 기울기는 0이 되어, 하위 층으로 의미 있는 기울기가 전달되지 않아 학습이 멈춤.
 <img width="608" height="211" alt="image" src="https://github.com/user-attachments/assets/6d6c7f6d-9fdf-4004-b4a4-bb2496cb5415" />

* **Hyperbolic Tangent (tanh)**: 평균이 0이므로 시그모이드의 문제는 피할 수 있음. 하지만 표준 초기화 방식과 함께 사용될 경우, 가장 아래 첫 번째 층부터 시작하여 순차적으로 층이 포화되는 현상이 관찰됨.
* **Softsign**: tanh와 유사하지만 점근선에 더 부드럽게 도달하여(지수적 대신 다항식), 포화 문제에 더 강건한 모습을 보이며 학습이 안정적으로 진행됨.
  
  <img width="560" height="627" alt="image" src="https://github.com/user-attachments/assets/2dff1af4-2797-43b6-8896-cab55c0cdd44" />


---

### 4. 기울기 초기화와 전파 분석 (Gradients at Initialization and Propagation)

#### 이론적 배경
안정적인 학습을 위해서는 두 가지 조건이 중요함.
1.  **순전파(Forward Propagation)**:  **활성화 값의 분산**이 유지되어야 함.
2.  **역전파(Back-propagation)**: **기울기의 분산**이 유지되어야 함.

표준 초기화 방식($Var[W] = \frac{1}{3n}$)은 이 조건을 만족시키지 못하여, 층이 깊어질수록 역전파되는 기울기의 분산이 계속 작아지는 **기울기 소실(Vanishing Gradient)** 문제가 발생



## 제안된 정규화 초기화 (Normalized Initialization)
이 두 조건을 절충하기 위해, 이전 층의 유닛 수($n_j$)와 다음 층의 유닛 수($n_{j+1}$)를 모두 고려한 새로운 초기화 방법을 제안함. 이 방식이 바로 **Xavier 초기화(Xavier Initialization)**

- 순전파를 위해서는 $Var(W) = \frac{1}{n_{in}}$
- 역전파를 위해서는 $Var(W) = \frac{1}{n_{out}}$

Xavier 초기화는 이 두 조건을 모두 적절히 만족시키기 위해 두 값의 조화 평균을 타협점으로 선택

$$Var(W) = \frac{2}{n_{in} + n_{out}}$$

이 분산 값을 갖도록 균등 분포(Uniform Distribution)의 범위를 계산하면 아래 식 탄생

$$W \sim U\left[-\frac{\sqrt{6}}{\sqrt{n_j + n_{j+1}}}, \frac{\sqrt{6}}{\sqrt{n_j + n_{j+1}}}\right]$$

결론적으로 Xavier 초기화는 입력과 출력의 뉴런 수를 모두 고려하여 가중치의 분산을 최적으로 설정함으로써, 신호(활성화 값과 기울기)가 여러 층을 안정적으로 통과하도록 보장

이 초기화를 사용하면, 역전파되는 기울기의 분산이 모든 층에서 일정하게 유지되어 정보의 흐름이 원활해짐. 결과적으로 학습이 훨씬 안정되고 빨라짐.

<img width="501" height="484" alt="image" src="https://github.com/user-attachments/assets/d97f959b-dd73-4216-bc79-7c2814ef6b56" />




---

### 5. 결론 (Conclusions)

* **Sigmoid 활성화 함수는 심층 신경망 초기화에 부적합함.** 기울기 포화 문제 발생.
* **가중치 초기화는 매우 중요함.** 제안된 정규화 초기화 기법은 **층간의 변환(자코비안)을 1에 가깝게 유지**하여 활성화 값과 기울기가 소실되거나 폭발하지 않고 잘 흐르도록 도움.
* 이러한 적절한 초기화만으로도, **비지도 사전학습(unsupervised pre-training)을 사용한 모델과의 성능 격차를 상당 부분 해소**할 수 있었음.
* 특히 tanh 네트워크는 이 새로운 초기화 방식의 효과를 크게 봤고, softsign은 초기화 방식에 비교적 덜 민감하고 전반적으로 강건한 성능을 보임.
