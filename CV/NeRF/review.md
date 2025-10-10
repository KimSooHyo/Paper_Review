# NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis

https://arxiv.org/pdf/2003.08934

---

## NeRF (Neural Radiance Fields)

여러 각도에서 촬영된 **2D 이미지**들로부터 해당 장면의 **사실적인 3D 뷰**(novel view)를 생성하는 방법을 제시 

복잡한 3D 장면 정보를, 메쉬(mesh)나 복셀(voxel) 같은 명시적인 구조 없이, 
**하나의 간단한 MLP 신경망의 가중치 안에 압축**하는 것.

<img width="735" height="188" alt="image" src="https://github.com/user-attachments/assets/4d4ba986-a82b-4d6c-9932-3d837fd2a7f2" />


> 100장의 입력 이미지를 사용하여 NeRF를 최적화하면, 새로운 뷰 렌더링 가능

## 1. 핵심 아이디어: 5D 연속 함수로서의 장면 표현

<img width="746" height="241" alt="image" src="https://github.com/user-attachments/assets/b2e7fade-c86e-4d83-9903-2b28ef5499b4" />


NeRF는 3D 장면을 하나의 연속적인 5차원 함수 $F_{\Theta}$로 정의. 
이 함수는 3D 공간상의 위치와 그 지점을 바라보는 시선 방향을 입력받아, 해당 위치의 색상과 밀도를 출력

* **입력 (5D)**: 3D 위치 좌표 $\mathbf{x} = (x, y, z)$ 와 2D 뷰 방향 $\mathbf{d} = (\theta, \phi)$.
* **출력 (4D)**: 해당 위치와 방향에서의 색상(RGB) $\mathbf{c} = (r, g, b)$ 와 볼륨 밀도(Volume Density) $\sigma$.

$$F_{\Theta} : (\mathbf{x}, \mathbf{d}) \rightarrow (\mathbf{c}, \sigma)$$

함수의 다중 시점 일관성(multiview consistency)을 높이기 위해 독창적인 네트워크 구조를 사용

* **볼륨 밀도 $\sigma$**: 물체의 형태는 보는 방향과 무관하므로, 오직 **위치** $\mathbf{x}$에만 의존
* **색상 $\mathbf{c}$**: 빛의 반사(specularity) 등은 보는 방향에 따라 달라지므로, **위치 $\mathbf{x}$와 방향 $\mathbf{d}$** 모두에 의존

<img width="646" height="263" alt="image" src="https://github.com/user-attachments/assets/8dfd3ddd-6219-47e0-9116-fc37118a99b1" />


> NeRF의 MLP 아키텍처. 위치 인코딩된 $\gamma(\mathbf{x})$가 8개의 레이어를 통과하며 $\sigma$를 출력하고, 중간 피처 벡터에 방향 인코딩된 $\gamma(\mathbf{d})$가 합쳐져 최종 RGB 색상을 출력

---

## 2. 기술적 원리: 볼륨 렌더링과 최적화

NeRF는 신경망이 예측한 $(\mathbf{c}, \sigma)$ 값들을 고전 볼륨 렌더링 원리를 통해 2D 이미지의 픽셀 색상으로 변환함

### **수학적 원리**
카메라 광선 $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$ (원점 $\mathbf{o}$, 방향 $\mathbf{d}$)가 있을 때, 픽셀의 예상 색상 $C(\mathbf{r})$은 다음 식으로 계산됨

$$C(\mathbf{r}) = \int_{t_n}^{t_f} T(t) \sigma(\mathbf{r}(t)) \mathbf{c}(\mathbf{r}(t), \mathbf{d}) \, dt$$

> 여기서 $T(t)$는 투과율로, 광선이 시작점 $t_n$부터 $t$까지 어떤 입자에도 부딪히지 않고 진행할 확률을 의미

$$T(t) = \exp\left(-\int_{t_n}^{t} \sigma(\mathbf{r}(s)) \, ds\right)$$

실제 계산에서는 이 연속 적분을 이산적인 합으로 근사함. 
광선 경로를 따라 $N$개의 포인트를 샘플링한 후, 최종 픽셀 색상 $\hat{C}(\mathbf{r})$을 다음과 같이 계산

$$\hat{C}(\mathbf{r}) = \sum_{i=1}^{N} T_i (1 - \exp(-\sigma_i \delta_i)) \mathbf{c}_i \quad \text{, where} \quad T_i = \exp\left(-\sum_{j=1}^{i-1} \sigma_j \delta_j\right)$$

미분 가능 -> 오차를 최소화하는 방향으로 신경망 $F_{\Theta}$의 가중치를 업데이트할 수 있음

---

## 3. NeRF의 성능을 높인 핵심 기술

단순한 MLP만으로는 고품질의 이미지를 생성하기 어려움 -> 두 가지 핵심 기술을 도입

### **위치 인코딩 (Positional Encoding)**

신경망은 기본적으로 저주파 함수를 학습하는 경향이 있어, 이미지의 복잡하고 미세한 디테일을 표현하기 어려움. 
이를 해결하기 위해 입력 좌표 $p$를 직접 사용하지 않고, 다음과 같은 함수 $\gamma$를 통해 고차원의 데이터로 변환하여 입력

$$\gamma(p) = (\sin(2^0 \pi p), \cos(2^0 \pi p), \dots, \sin(2^{L-1} \pi p), \cos(2^{L-1} \pi p))$$

이를 통해 신경망이 고주파의 기하학적 구조와 질감을 효과적으로 학습할 수 있게 됨 -> 더 나은 결과물

논문에서는 위치 $\mathbf{x}$에 대해 $L=10$, 방향 $\mathbf{d}$에 대해 $L=4$를 사용

<img width="721" height="212" alt="image" src="https://github.com/user-attachments/assets/8daf6571-559c-45dc-bd0a-961fe31e7f53" />


> 위치 인코딩을 제거하면 고주파 디테일 표현 능력이 낮아져서 이미지가 흐릿해진 것을 볼 수 있음

### **계층적 볼륨 샘플링 (Hierarchical Volume Sampling)**

광선을 따라 무작위로 균등하게 샘플링하는 것은 비효율적 -> 
이를 해결하기 위해 NeRF는 두 개의 네트워크(coarse, fine)를 동시에 최적화함

1.  **Coarse Pass**: 먼저 $N_c$개의 샘플을 균등하게 뽑아 coarse 네트워크로 렌더링에 중요한 영향을 미칠 만한 영역(밀도가 높은 영역)을 파악
2.  **Fine Pass**: 파악된 중요 영역에 더 많은 샘플($N_f$개)을 집중적으로 할당하여 fine 네트워크로 최종 색상 값을 계산

렌더링의 효율과 품질을 동시에 향상시키는 중요한 역할을 함.
논문에서는 $N_c=64$, $N_f=128$을 사용.

---

## 4. 최적화 및 결과

최종 손실 함수는 coarse와 fine 렌더링 결과 모두의 제곱 오차를 합산함

$$\mathcal{L} = \sum_{\mathbf{r} \in \mathcal{R}} \left( ||\hat{C}_c(\mathbf{r}) - C(\mathbf{r})||_2^2 + ||\hat{C}_f(\mathbf{r}) - C(\mathbf{r})||_2^2 \right)$$

> Coarse loss를 함께 최적화하는 이유는 fine 샘플링을 위한 가중치 분포가 유의미한 값을 갖도록 안정적으로 유도하기 위함

### **성능 비교 및 제거 연구 (Ablation Study)**

NeRF는 기존 SOTA(State-of-the-Art) 모델들과 비교하여 좋은 성능을 보임

<img width="732" height="139" alt="image" src="https://github.com/user-attachments/assets/c0f61267-c852-4f19-9430-9232222c5a8c" />

> NeRF는 PSNR 기준으로 기존 모델들 보다 성능이 좋음


<img width="741" height="248" alt="image" src="https://github.com/user-attachments/assets/3fc2bf93-3945-44a7-9e46-dfd9a3f0929c" />

> 위치 인코딩과 뷰 의존성(VD)이 성능에 가장 큰 영향을 미쳤으며, 계층적 샘플링(H) 또한 성능 향상에 크게 기여했음을 알 수 있음


## 5. 결론 및 의의

NeRF는 3D 장면을 MLP 기반의 연속적인 5D 함수로 표현하는 새로운 패러다임을 제시함. 
이 방법은 사실적인 뷰를 생성하면서도, 복잡한 장면을 단 작은 신경망 가중치로 압축할 수 있었음.
