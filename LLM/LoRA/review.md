# LoRA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS

https://arxiv.org/pdf/2106.09685

---

## 1. 문제 제기: Full Fine-Tuning의 비효율성

- **문제점:** GPT-3 175B와 같은 초거대 모델을 Downstream task에 맞게 Full Fine-Tuning하는 것은 현실적으로 어려움
- **기존 해법의 한계:**
    - **Adapter:** 모델 중간에 새로운 레이어를 삽입
        
        → 파라미터 수를 줄이지만, 추가된 레이어를 순차적으로 계산해야 하므로 Inference Latency 발생
        
    - **Prefix-Tuning:** 입력 시퀀스의 일부를 최적화
        
        → 최적화가 어렵고 모델이 처리할 수 있는 실제 작업의 시퀀스 길이를 줄이는 단점이 있음
        

---

## 2. LoRA의 핵심 원리

> "모델을 adaptation시킬 때 가중치의 변화량( $\Delta W$)은 매우 낮은 '고유 순위(intrinsic rank)'를 가질 것"
> 

### 핵심 수학 공식

Full Fine-Tuning이 기존 가중치 $W_0$를 업데이트하여 $W = W_0 + \Delta W$를 만드는 방식이라면, 

LoRA는 $W_0$를 freeze 시킴. $\Delta W$를 직접 학습하는 대신, 두 개의 작은 행렬 B와 A의 곱(Low-Rank Decomposition)으로 표현

<aside>

$$
W = W_0 + \Delta W = W_0 + BA
$$

- $W_0 \in \mathbb{R}^{d \times k}$: 사전 학습된 원래 가중치 (동결)
- $B \in \mathbb{R}^{d \times r}$: 저순위(low-rank) 행렬 (학습 대상)
- $A \in \mathbb{R}^{r \times k}$: 저순위(low-rank) 행렬 (학습 대상)
- $r$: LoRA의 '순위(rank)'이며, $r \ll \min(d, k)$

$$
h = W_0x + \Delta Wx = W_0x + BAx
$$

</aside>

### LoRA의 구조적 이해

<img width="258" height="233" alt="image" src="https://github.com/user-attachments/assets/5bc83897-e648-4cd6-a122-7a01b829a5f9" />


- 입력 x가 $W_0$ 를 통과하는 기존 경로는 그대로 유지 + 동결
- 동시에 x가 **A와 B로 구성된 새로운 경로**를 통과 → 이 경로의 파라미터(A, B)만 학습됨
- 두 경로의 출력값은 덧셈을 통해 합쳐져 최종 출력 h가 됨

### LoRA의 이점: No Additional Inference Latency

학습이 완료된 후, 배포 시점에는 $W_{merged} = W_0 + BA$를 미리 계산할 수 있음.

- $W_0$ ( $d \times k$ 행렬)와 $BA$ ( $d \times k$ 행렬)는 동일한 차원을 가짐
    
    → 둘을 합쳐서 하나의 최종 가중치로 만들 수 있음.
    

---

## 3. 실험

### GPT-3 175B 테스트

<img width="627" height="260" alt="image" src="https://github.com/user-attachments/assets/9c236420-f196-4917-9d31-8d1d52c0c7a0" />


LoRA는 여러 태스크에서 FT의 성능과 대등하거나 더 높은 성능을 달성

### 안정적인 최적화

<img width="772" height="255" alt="image" src="https://github.com/user-attachments/assets/87d5428f-2146-4115-9cbd-531a62660ec3" />


> LoRA가 다른 기법보다 얼마나 안정적인지 보여줌
> 
- LoRA는 파라미터가 증가함에 따라 성능이 안정적으로 향상되거나 유지

---

## 4. 작동 원리 분석: LoRA는 왜 작동하는가?.

### Rank는 작아도 된다

<img width="690" height="189" alt="image" src="https://github.com/user-attachments/assets/8aef348a-aedd-4db6-b157-9d00d86b8b3c" />


rank r이 64일 때와 r=1 또는 r=2일 때 성능 차이가 거의 없음

→ "가중치 업데이트는 매우 낮은 고유 순위를 갖는다"는 것을 뒷받침함

### $\Delta W$의 본질

<img width="769" height="214" alt="image" src="https://github.com/user-attachments/assets/447eeeba-71c8-4686-a6f0-f812442ad903" />


- 서로 다른 rank로 학습한 $\Delta W$의 subspace 유사도를 비교한 결과, 가장 중요한 상위 1~2개의 방향(singular vector)만 일치하고 나머지는 크게 다름.
- $\Delta W$의 핵심 정보가 **rank 1~2에 집중**되어 있다는 것.

LoRA가 학습하는 $\Delta W$ 는, 기존 $W_0$에는 존재했지만 강조되지 않았던 특정 작업에 필요한 특징들을 찾아내어 이를 증폭하는 역할

## 결론

LoRA는 $W = W_0 + BA$ 라는 re-parametrization를 통해, 사전 학습된 LLM의 가중치는 동결시키고 오직 저순위 행렬 A, B만 학습

이 구조 덕분에 10,000배 이상의 파라미터 효율을 달성하고, Full Fine-Tuning과 동일하거나 더 나은 성능을 보이며, 가장 결정적으로 배포 시 가중치를 병합하여 **추가 추론 지연도 없음!!**


---

#### 참조

https://arxiv.org/pdf/2106.09685

https://huggingface.co/

https://github.com/microsoft/LoRA
