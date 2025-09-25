# End-to-End Object Detection with Transformers

![](https://velog.velcdn.com/images/smox_i/post/50586a47-851d-469d-85ac-1472fbefa2c6/image.png)

- 객체 탐지(Object Detection)를 '직접적인 집합 예측(direct set prediction)' 문제로 재정의하는 새로운 프레임워크
- 기존의 모델들의 복잡한 부분(Non-Maximum Suppresion, Anchor Generation)을 제거하고, 객체들의 집합(set of objects)를 한 번에 예측하는 End-to-End 모델


> 
## DETR 구조
![](https://velog.velcdn.com/images/smox_i/post/0113e199-3995-494b-a650-0fef8e14ea2e/image.png)
DETR는 CNN, Transformer(Encoder, Decoder), FFN(Feed Forward Network)의 단계로 구성되어있다
### 1. CNN Backbone
- CNN 모델을 사용하여(C, H, W) 형태의 특징 맵 추출
**입력 준비**
- 특징 맵을 트랜스포머에 넣어주기 위해 1차원 시퀀스로 변환 -> (d X HW) 형태가 됨. (d는 1x1 Convolution을 통해 특징 맵의 채널(C) 크기를 줄여준 것)
- 트랜스포머는 자체적으로 순서나 위치 정보를 모르기 때문에, 위치 인코딩(Positional Encoding) 값을 만들어 특징 벡터에 더해줌
### 2. Transformer - Encoder
- 셀프 어텐션(self-attention) 메커니즘을 통해 이미지 전반의 글로벌한 컨텍스트와 객체들 간의 관계를 학습
### 3. Transformer - Decoder
- Decoder의 입력으로 들어오는 것 : Encoder의 출력, Object Queries
	- Object Queries : 학습 가능한 N개의 벡터. 이 쿼리들이 모델이 학습 하면서 각 특정 위치나 객체를 탐지할 수 있게 됨
- 디코더는 Self-Attention과 Encoder-Decoder Attention을 통해 각 Object Query를 업데이트하며, 최종적으로 N개의 객체에 대한 정보를 담은 출력 임베딩을 생성
- 모든 객체를 한 번에 병렬적으로 디코딩
### 4. FFN (Feed-Forward Network)
- 트랜스포머 디코더에서 나온 N개의 출력 임베딩을 각각 입력으로 받아서 class와 bounding box 예측
- 객체가 탐지되지 않은 슬롯은 'no object'라는 특수 클래스로 예측


> ## 이분 매칭 (Bipartite Matching)
**: 모델이 출력한 N개의 예측 결과와 실제 정답 객체 간에 최적의 일대일 매칭을 찾는 과정**
- FFN이 예측을 완료한 후에 모델을 '학습'시키기 위한 손실(Loss) 계산 단계에서 사용
<img width="743" height="151" alt="image" src="https://github.com/user-attachments/assets/11014cb5-d09b-4206-9a31-6433de9113f7" />

>> _**이분 매칭이 왜 필요한가?**_
: DETR은 N개의 예측값(클래스+박스)을 출력함.  모델을 학습시키려면 예측과 정답을 비교해서 손실(Loss)을 계산해야 하는데...
>> -	어떤 예측을 어떤 정답과 비교해야 할까? 예측 1번을 정답 A와 비교해야 할까? 아니면 예측 50번을 정답 A와 비교해야 할까?
-> 이 '**누가 누구의 짝인지**'를 결정하는 문제가 바로 이분 매칭이 해결하려는 과제

---
### 실험 결과
![](https://velog.velcdn.com/images/smox_i/post/c4259a64-231a-469e-a81a-edba385f62d9/image.png)

COCO 데이터셋에서 최적화된 Faster R-CNN 모델과 비슷한 수준의 성능을 달성. 
특히 큰 객체에 대해서는 훨씬 뛰어난 성능을 보였는데, 이는 트랜스포머가 이미지 전체의 글로벌한 정보를 활용하기 때문인 듯. 반면, 작은 객체 탐지 성능은 상대적으로 낮은 경향
