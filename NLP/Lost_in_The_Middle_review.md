# Lost in the Middle: How Language Models Use Long Contexts

https://arxiv.org/pdf/2307.03172

---

## 1. 도입: 긴 컨텍스트, 정말 잘 쓰고 있을까?

최근에는 Context Window가 긴 언어 모델들이 나오고 있음. 

> Context Windows: LLM 모델이 텍스트를 생성하거나 이해할 때 입력으로 받을 수 있는 양
> 

근데 이 논문의 저자들은 근본적인 질문을 던짐.

*"모델이 긴 컨텍스트를 입력받을 수 있다는 것이, 그 컨텍스트를 효과적으로 활용한다는 뜻일까?"*

## 2. 핵심 발견: U자형 성능 곡선

<img width="382" height="363" alt="image" src="https://github.com/user-attachments/assets/24e2cf27-bb5e-4b59-a4e6-584c49f201df" />


multi-document question answering

gpt-3.5-turbo에게 총 20개의 문서를 주고 질문에 답하게 한 실험

**모델의 성능은 관련 정보의 위치에 따라 U자형 곡선 모양**

1. 시작 : 정답 문서가 1번째에 있을 때, 정확도가 약 76%로 가장 높음. 모델은 컨텍스트의 시작 부분을 아주 잘 봄.
2. 중간 : 정답 문서가 5번째, 10번째, 15번째 등 중간으로 갈수록 정확도가 급격히 떨어짐. 10번째 근처에서는 약 54%까지 하락함.
3. 끝 : 정답 문서가 20번째, 즉 맨 마지막에 있을 때, 정확도가 다시 63% 이상으로 상승함. 모델은 컨텍스트의 끝 부분도 비교적 잘 봄.

**→ 모델은 컨텍스트의 양 끝은 잘 보지만, 중간에 있는 정보는 놓쳐버림**

- 문서를 전혀 참조하지 않고(closed-book) 답할 때보다 정답이 중간에 있을 때 정확도가 더 낮음
    
    → 관련 정보가 중간에 있으면 오히려 방해가 될 수 있음
    

저자들은 이 현상을 더 확실히 검증하기 위해 다양한 모델과 조건으로 실험을 확장

## 3. 실험 1: 다중 문서 질문 답변 (Multi-Document Question Answering)

<aside>

### 실험 설계

<img width="738" height="298" alt="image" src="https://github.com/user-attachments/assets/0815b6c8-9b37-4a8c-b871-7a18cf378400" />


기본 예시

<img width="557" height="351" alt="image" src="https://github.com/user-attachments/assets/505b92e9-7e2c-4109-9b6a-639d1ccd508a" />


문서의 순서를 바꿈

<img width="540" height="392" alt="image" src="https://github.com/user-attachments/assets/3d33a3eb-f7fe-4a52-8e65-45ba1476d7fb" />


정답과 상관없는 문서를 더 추가해서 전체 컨텍스트의 길이를 늘림

</aside>

<img width="905" height="293" alt="image" src="https://github.com/user-attachments/assets/51291ced-cde4-466a-b87c-bdf0fc54d382" />


이 테스트를 10개, 20개, 30개 문서로 확장하고, gpt-3.5-turbo, Claude, MPT, LongChat 등 다양한 모델로 실험한 결과

- **일관된 U자형 패턴:** 문서 개수가 늘어나도, **모든 모델에서 U자형 곡선이 일관되게 나타남.** 컨텍스트가 길어질수록(30개) 중간의 성능 하락 폭은 더 깊어짐.
- **긴 컨텍스트 모델의 한계:** GPT-3.5-Turbo (16K)나 Claude-1.3 (100K)처럼 context window가 더 긴 모델들도 더 나은 성능을 보이지 않았으며 동일한 U자형 곡선을 보임
    - context window가 크다고 해서 그 공간을 더 효율적으로 탐색하는 건 X

<aside>

이 모델들이 정답 문서 1개만 줬을 때는 높은 정확도를 보임

→ 능력 자체가 없는 게 아니라, 긴 컨텍스트 속에서 찾지 못하는 것.

</aside>

### 4. 주요 실험 2: 키-값 검색 (Key-Value Retrieval)

<aside>

### 실험 설계

<img width="787" height="313" alt="image" src="https://github.com/user-attachments/assets/5bbee465-32b8-45c7-b9fb-61c21cbbb595" />


특정 키(key)에 해당하는 값(value)을 찾는, 더 단순한 테스트

- 실험 방식: (Key, Value) 쌍으로 이루어진 긴 JSON 데이터를 모델에게 줌. 모든 키와 값은 랜덤 UUID(무작위 문자열)임. 그리고 특정 '키'를 주고, 그에 맞는 '값'을 반환하라고 시킴.
- 문서 이해 및 생성이 아닌 only 검색 능력만 평가하기 위한 실험
</aside>

<img width="904" height="292" alt="image" src="https://github.com/user-attachments/assets/2804295c-3e71-4575-b3c2-ce27341fa3f8" />


- 결과:
    - Claude 모델 (파란색, 주황색): 이 단순 작업은 거의 완벽하게 수행함. 키-값 쌍을 확장시킨 실험에서도 정확도가 100%에 가까움.
    - 다른 모델 (GPT, MPT 등): 또다시 U자형 곡선이 나타남!
        - 특히 키-값 쌍이 140개, 300개로 길어지면 50% 미만의 정확도를 보이기도 함
- 결론: Claude를 제외한 많은 모델들이 **의미 이해가 필요 없는 단순 문자열 찾기조차도 컨텍스트 중간에 있으면 성능 하락**

---

## 5. 원인 분석

### 가설 1: 아키텍처 문제?

우리가 평가한 오픈 모델들은 Decoder-only 모델로, 타임스텝(timestep)에서 이전 토큰에만 어텐션(attend)할 수 있음

→ 그럼 양방향으로 정보를 처리하는 Encoder-Decoder 모델은 괜찮을까?

<aside>

### Flan-T5-XXL 와 Flan-UL2

- Flan-T5-XXL은 512 토큰의 시퀀스로 훈련
- Flan-UL2는 초기에 512 토큰의 시퀀스로 훈련
    - 1024 토큰으로 10만(100K) 스텝을 추가로 사전 훈련
    - 이후 인코더에 2048 토큰, 디코더에 512 토큰을 가진 시퀀스로 instruction fine-tuning

> Instruction Fine-Tuning이란?
- LLM이 사용자의 지시(Instruction)를 더 잘 이해하고 따르도록 추가로 훈련시키는 과정
- 사용자가 원하는 특정 형식이나 작업에 맞춰 응답하는 능력을 갖추게 됨.
> 

</aside>

<img width="914" height="291" alt="image" src="https://github.com/user-attachments/assets/3872b845-a54c-4aec-b04d-db8e807e01fe" />


Encoder-Decoder 모델인 Flan-T5, Flan-UL2를 테스트한 결과

- 훈련된 시퀀스 길이(예: 2K 토큰) 안에서는 좀 더 robust해보임
- 더 긴 시퀀스를 주면 **이 모델들도 U자형 그래프를 보임**

### 가설 2: 쿼리(Query) 위치 문제?

이전의 실험은 처리할 데이터(문서/키-값) 뒤에 쿼리(질문/검색할 키)를 배치 

→ decoder-only 모델은 문서나 키-값 쌍을 컨텍스트화(contextualizing)할 때 쿼리 토큰에 어텐션(attend)할 수 없음 (디코더-전용 모델은 각 타임스텝에서 이전 토큰에만 어텐션할 수 있기 때문)

> 디코더 모델은 질문이 맨 뒤에 있으면 그 전까지 질문이 있는지도 모르잖아?
> 

**그럼 질문을 앞뒤로 두 번 주면 어떨까?** 

: [쿼리(질문)], [문서 1], [문서 2], [문서 3], ... [쿼리(질문)]

- 결과 (키-값 검색): 이 방식은 단순 키-값 검색(실험 2)에서는 문제를 완벽하게 해결함. 모든 모델이 거의 100% 성능을 냄.
    
    <img width="441" height="388" alt="image" src="https://github.com/user-attachments/assets/5ff43262-05f9-4fbc-9f38-e5f5c2ba066a" />

    
- 더 복잡한 다중 문서 QA에서는 크게 ****효과가 없었음. U자형 곡선이 그대로 유지됨

→ **단순 탐색은 개선되지만, 여러 문서 사이에서 복잡한 추론을 해야 할 땐 여전히 중간을 놓침**

### 가설 3: SFT(supervised fine-tuning) 때문?

<img width="410" height="361" alt="image" src="https://github.com/user-attachments/assets/6dd925ff-553b-40cf-9e52-08e91a7ca82f" />


SFT를 한 MPT-30B-Instruct와, SFT 안 한 MPT-30B (Base) 모델 비교

우리가 평가한 모델들은 모두 instruction fine-tuned 모델들이라서, 초기 사전 훈련 후에 지시 사항과 응답 데이터셋으로 SFT(Supervised Fine-tuning)를 거침

SFT 데이터는 보통 지시사항이 입력의 시작 부분에 위치

→ 그래서 모델이 컨텍스트 시작 부분(지시사항)에 주의를 기울이도록 훈련된 게 아닐까?

- SFT 안 한 **기본 모델(mpt-30b, 노란색 선)도 U자형 곡선을 보임!**

**결론: U자형 곡선은 SFT 때문에 생긴 게 아니다!**

### 가설 4: 모델 규모의 영향?

<img width="439" height="423" alt="image" src="https://github.com/user-attachments/assets/fb10129d-4905-402c-9fa2-ab4a6daef770" />


Llama-2 모델을 7B, 13B, 70B 크기별로 테스트

- 결과 (7B 모델): 가장 작은 7B 모델은 U자형이 아니라, 그냥 맨 끝만 잘 보는 단순한 패턴
- 결과 (13B, 70B 모델): U자형 곡선, 즉 시작 부분도 잘 보는 능력은 13B, 70B 같은 충분히 큰 모델에서만 나타나는 현상이었음.

**결론: 모델이 커지면서 컨텍스트의 끝뿐만 아니라 시작 부분도 활용하는 능력이 생겼지만, '중간'은 여전히 잘 활용하지 못함**

### 6. RAG(Retrieval-Augmented Generation)는 어떡하나?

<aside>

RAG기법은 기존의 대규모 언어 모델(LLM)을 확장하여, 주어진 컨텍스트나 질문에 대해 더욱 정확하고 풍부한 정보를 제공하는 방법입니다. 

모델이 학습 데이터에 포함되지 않은 외부 데이터를 실시간으로 검색(retrieval)하고, 이를 바탕으로 답변을 생성(generation)하는 과정을 포함합니다.

</aside>

<img width="437" height="375" alt="image" src="https://github.com/user-attachments/assets/7cffc683-d754-4f56-9ea1-5960086a4a35" />


RAG 상황을 가정한 케이스 스터디

"RAG 쓸 때, 문서를 10개 주는 게 좋을까? 50개 주는 게 좋을까? 많을수록 좋은 거 아닐까?"

- **X축:** 검색해서 모델에게 준 문서 개수 (5개 ~ 50개)
- **주황색 선 (Contriever recall):** 검색한 문서 k개 안에 정답이 포함되어 있을 확률. 문서를 많이 볼수록 정답이 포함될 확률은 계속 올라감.
- **다른 선들 (모델 정확도):** 근데 모델이 실제로 정답을 맞힌 비율(정확도)은 문서 20개 정도에서 그냥 수평선(포화)이 됨.
- **결론:** 모델에게 20개 넘게 30개, 50개 문서를 줘봤자, 모델은 그 추가 정보를 **활용하지 못하고 있음.** 리콜은 오르는데 정작 정확도는 그대로 → 시간과 비용 낭비

### 7. 최종 결론 및 시사점

1. **모델은 긴 컨텍스트의 중간에 있는 정보를 효과적으로 사용하지 못함.** 성능은 시작과 끝에서 높고 중간에서 낮은 'U자형 곡선'을 그림.
2. 이 현상은 **모델 아키텍처, 규모, SFT 여부와 관계없이** 광범위하게 나타나는 근본적인 한계임.
3. 단순히 컨텍스트 창이 긴 모델이 문제를 해결하는 것은 아님.
4. **실전(RAG)에서는** 문서를 무작정 많이 넣는 것이 능사가 아님. 어차피 모델은 20개 넘어가면 잘 보지도 못함.

<aside>

- 단순히 문서를 많이 넣을 게 아니라, 가장 관련성 높은 문서를 1~2개 뽑아서 컨텍스트의 맨 앞이나 맨 뒤에 배치하는 **재정렬 전략**이 중요함. 혹은, 모델이 한 번에 처리할 **문서의 수를 제한**하는 것이 효율적일 수 있음.

*"긴 컨텍스트를 처리할 수 있다"*는 것과 "긴 컨텍스트를 *이해할 수 있다"*는 것은 완전히 다른 문제

</aside>

---

### 이어서 공부해보면 좋을 것들

- **청킹 및 요약 (Chunking & Summarization):**
    1. 긴 문서를 여러 개의 작은 '청크(chunk)'로 나눔.
    2. 모델을 시켜 각 청크를 개별적으로 요약하게 함.
    3. 이 "요약본들의 요약본"을 만들거나, 이 짧아진 요약본들만 컨텍스트에 넣어 최종 답변을 생성하게 함.
- **"[Found in the Middle](https://arxiv.org/pdf/2406.16008)" (논문):**
'found-in-the-middle'이라는 어텐션 보정 메커니즘 → 모델의 어텐션 점수에서 이 '위치 편향' 값만 분리해 제거
- **"[Never Lost in the Middle](https://arxiv.org/pdf/2311.09198)" (논문):**
"위치에 구애받지 않는 다단계 QA (PAM QA)"라는 새로운 훈련 태스크를 제안. 모델이 정보의 위치와 상관없이 정보를 찾는 능력을 강화하도록 훈련시킴.

---

### 참조

[Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/pdf/2307.03172)

https://huggingface.co/google/flan-ul2

https://wikidocs.net/231364
