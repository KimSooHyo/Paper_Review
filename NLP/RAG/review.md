
# Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

## RAG, 왜 필요한가?

기존의 사전학습된 거대 언어 모델(LLM)은 세상의 모든 지식을 모델의 **파라미터(parameter)** 안에 저장합니다. 이러한 방식은 다음과 같은 한계가 있음

1.  **지식의 확장 및 업데이트의 어려움**: 새로운 정보가 생기거나 기존 정보가 바뀌었을 때, 모델 전체를 재학습해야 하므로 비용이 많이 들고 비효율적
2.  **환각 (Hallucination) 현상**: 모델이 답변을 생성할 때, 학습 당시의 파라미터에만 의존하기 때문에 관련된 최신 지식이나 정확한 근거를 직접 활용하지 못해 잘못된 정보를 생성할 수 있음

이러한 문제를 해결하기 위해 **검색 증강 생성(Retrieval-Augmented Generation, RAG)** 모델이 제안됨

### RAG란?
> LLM의 내부 지식인 **Parametric Memory**와 외부 지식 소스로부터 정보를 검색하는 **Non-Parametric Memory**를 결합한 하이브리드 접근 방식의 모델

---

## RAG의 핵심 구조

RAG는 크게 두 가지 핵심 컴포넌트로 구성되며, 이 두 요소는 End-to-End 방식으로 함께 학습


1.  **Retriever**
    -   **역할**: 사용자의 질문(query `x`)이 주어지면, 외부 문서 모음(Document Index)에서 질문과 가장 관련성이 높은 상위 K개의 문서(`z`)를 검색 
    -   **방식**: 질문과 문서를 각각 인코딩하여 벡터로 만들고, **MIPS (Maximum Inner Product Search)**를 통해 가장 유사한 벡터를 가진 문서를 효율적으로 찾음 

2.  **Generator**
    -   **역할**: Retriever가 찾아낸 문서(`z`)와 원본 질문(`x`)을 함께 입력받아 최종 답변(`y`)을 생성하는 **Seq2Seq 모델**
    -   **장점**: 답변 생성 시 외부의 실제 정보를 근거로 삼기 때문에 더 정확하고 사실에 기반한 결과물을 만들 수 있음

---

## RAG의 두 가지 모델
<img width="769" height="225" alt="image" src="https://github.com/user-attachments/assets/2cb90f7a-6fde-4fe9-bda7-9bdd19eb3571" />

RAG는 검색된 문서를 활용하는 방식에 따라 두 가지 세부 모델로 나뉨

### 1. RAG-Sequence Model
하나의 검색된 문서를 사용하여 **답변 시퀀스 전체**를 생성하는 모델 
1. 상위 K개의 문서를 각각 조건으로 주어 답변 후보군을 생성
2. 각 후보의 확률을 모두 더하고 주변화(marginalization)하여 가장 가능성이 높은 최종 답변 시퀀스를 선택

### 2. RAG-Token Model
답변의 **각 토큰(단어)을 생성할 때마다** 다른 문서를 참조할 수 있는 더 유연한 모델
1. 첫 번째 토큰을 생성할 때 K개의 문서를 참조하여 확률을 계산하고,
2. 다음 토큰을 생성할 때도 K개의 문서를 참조하여 확률을 계산하는 과정을 반복 
3. 이를 통해 여러 문서의 정보를 조합하여 더 풍부한 답변을 생성

---

## 주요 실험 결과

### 1. Open-Domain Question Answering

<img width="365" height="191" alt="image" src="https://github.com/user-attachments/assets/381b7d67-b52f-459b-bff1-7c65df14d44e" />

- RAG 모델은 Natural Questions, TriviaQA 등 4개의 Open-Domain QA 테스트 데이터셋에서 **높은 성능 달성** 
- 특히 파라미터만 사용하는 "Closed Book" 접근 방식(T5-11B)이나, 검색 후 답을 추출하는 "Open Book" 접근 방식(DPR)보다 더 뛰어난 성능을 기록


### 2. Jeopardy Question Generation

<img width="340" height="181" alt="image" src="https://github.com/user-attachments/assets/b3b9e233-7e17-451f-ab25-14bd86ca8713" />

- 생성된 답변의 품질을 사람이 직접 평가한 결과, RAG는 BART 모델보다 훨씬 뛰어나다는 것이 입증됨 
---

## 결론

RAG는 LLM이 가진 **내부 지식(Parametric Memory)과 외부의 방대한 정보(Non-parametric Memory)를 성공적으로 결합**한 모델

이를 통해 기존 LLM의 한계였던 환각 현상을 줄이고, 답변의 신뢰도를 크게 향상시킴
