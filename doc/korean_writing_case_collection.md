# Korean Writing Case Collection

이 문서는 `doc/paper_draft_korean.md`에 남겨진 첨삭 메모를 바탕으로, 향후 한국어 문서 작성 시 재사용할 수 있는 `bad pattern / good pattern` 사례를 수집한 문서이다.

목적은 두 가지다.

1. 번역투 문장과 어색한 표현을 반복하지 않기 위한 사례 축적
2. 한국어 논문 문체에서 어떤 정보는 강조하고, 어떤 정보는 뒤로 빼야 하는지에 대한 서술 전략 정리

---

## 1. 분류 체계

첨삭 메모는 아래 다섯 범주로 나눌 수 있었다.

1. 직역투 표현
2. 불필요한 영어 혼용
3. 용어 일관성 문제
4. 한국어 문장 구조의 어색함
5. 논문 서사 전략 문제

---

## 2. Case Collection

### Case 1. `initial guess`의 직역

- Source pattern:
  - `초기 추정`
- Comment:
  - `bad translation of initial guess`
- Bad pattern:
  - 영어 기술 용어를 한국어로 기계적으로 치환했지만, 실제 연구자 언어 습관과 맞지 않음
- Better direction:
  - `초기값`
  - 문맥에 따라 `초기 궤적`, `초기 해`, `초기 입력`
- Good pattern example:
  - `초기값의 품질은 solver의 수렴 거동에 큰 영향을 준다.`

### Case 2. `infeasible solution` 류의 부자연스러운 번역

- Source pattern:
  - `비실현 해`
- Comment:
  - `bad translation`
- Bad pattern:
  - 사전식 번역이지만 한국어 공학 문헌에서 거의 쓰지 않는 표현
- Better direction:
  - `비실행 가능 해`
  - `제약을 만족하지 않는 해`
  - `feasible하지 않은 해`
- Good pattern example:
  - `초기값이 좋지 않으면 제약을 만족하지 않는 해로 수렴할 수 있다.`

### Case 3. 직역이 만든 어색한 평가 표현

- Source pattern:
  - `좋지 않은 국소해`
  - `잘 만족하는`
- Comment:
  - `awkward wording`
- Bad pattern:
  - 영어 문장의 평가 형용사를 그대로 옮겨 한국어 리듬이 깨짐
- Better direction:
  - `품질이 낮은 국소해`
  - `제약을 만족하는`
- Good pattern example:
  - `품질이 낮은 국소해에 머물 가능성도 있다.`

### Case 4. 같은 개념을 여러 방식으로 번역

- Source pattern:
  - `부드럽고`
  - `smooth하고`
- Comment:
  - `inconsitent with previous wording`
- Bad pattern:
  - 같은 문서에서 동일 개념의 한글/영문 표기를 섞어 씀
- Better direction:
  - 문서 전체에서 하나로 고정
- Good pattern example:
  - 문서 전체에서 `smooth한 궤적`으로 통일

### Case 5. 반쯤 번역된 혼합 표현

- Source pattern:
  - `반공간`과 `half-space`를 번갈아 사용
- Comment:
  - `mixed wording`
- Bad pattern:
  - 수학 용어를 한글과 영어로 번갈아 쓰며 독자에게 불필요한 전환을 강요
- Better direction:
  - `half-space(반공간)`를 처음 한 번만 병기하고 이후 하나만 사용
- Good pattern example:
  - 첫 등장: `supporting half-space(지지 반공간)`
  - 이후: `지지 반공간`

### Case 6. 기여 문장이 효과를 말하지 못함

- Source pattern:
  - `control point 공간에서 직접 작동하는 궤적 초기화 정식화를 제시한다.`
- Comment:
  - `doesn't convey contribution well`
- Bad pattern:
  - 무엇을 했는지만 말하고 왜 의미 있는지 드러내지 못함
- Better direction:
  - 기여 문장에는 효과, 차별점, 실질적 장점을 함께 넣기
- Good pattern example:
  - `control point 공간에서 직접 작동하는 정식화를 제시하여 제약 구성과 계산 구조를 단순화하였다.`

### Case 7. 어색한 추상명사 결합

- Source pattern:
  - `보수적 연속 안전 처리 방식`
- Comment:
  - `awkward wording`
- Bad pattern:
  - 추상명사를 과도하게 붙여 한국어 문장 밀도가 지나치게 올라감
- Better direction:
  - 동작 중심으로 다시 쓰기
- Good pattern example:
  - `구형 KOZ를 연속적으로 회피하도록 하는 보수적 제약 구성 방법`

### Case 8. `outer iteration`의 직역

- Source pattern:
  - `외부 반복`
- Comment:
  - `bad translation`
- Bad pattern:
  - 번역은 가능하지만 실제 논문 문체에서는 다소 딱딱하고 직역 느낌이 강함
- Better direction:
  - `SCP 반복`
  - `반복 단계`
  - `각 반복에서`
- Good pattern example:
  - `각 SCP 반복에서 KOZ 제약을 다시 선형화한다.`

### Case 9. 불필요한 영어 유지

- Source pattern:
  - `convexification`
  - `pointwise`
  - `lossless convexification`
  - `warm start`
  - `trajectory optimization`
  - `geometric`
  - `Bernstein basis polynomial`
  - `outward normal`
- Comment:
  - `Unnecessary English word`
- Bad pattern:
  - 한국어로 충분히 자연스럽게 쓸 수 있는 단어까지 무분별하게 영어로 둠
- Better direction:
  - 독자층이 익숙한 핵심 용어만 영어 유지
  - 나머지는 자연스러운 한국어로 바꾸기
- Good pattern example:
  - `pointwise constraint` -> `점별 제약`
  - `outward normal` -> `외향 법선`
  - `Bernstein basis polynomial` -> `Bernstein 기저다항식`

### Case 10. 구현 정보가 본문을 오염시킴

- Source pattern:
  - `본 논문은 구현의 dv 모드를 유일한 논문 수준 objective로 사용한다.`
  - `현재 구현에서는 ...`
  - `코드에서는 sample_count`
- Comment:
  - 코드 내부 표현이나 구현 명칭을 본문이 과도하게 따라감
- Bad pattern:
  - 논문이 코드 설명서처럼 읽힘
- Better direction:
  - 논문 본문은 수식과 개념 중심으로 서술
  - 구현 이름은 필요할 때만 부록이나 대응표에서 최소한으로 언급
- Good pattern example:
  - `목적함수는 제어 가속도와 중력 항의 차이를 기반으로 구성하였다.`

### Case 11. 한국어 문장으로 잘 성립하지 않는 선언문

- Source pattern:
  - `안전성 논리는 단순하다.`
- Comment:
  - `not a normal korean sentence`
- Bad pattern:
  - 영어식 topic sentence를 한국어에 그대로 투사
- Better direction:
  - 바로 논리 내용을 쓰거나, 더 자연스럽게 연결
- Good pattern example:
  - `이 제약이 안전성을 보장하는 이유는 다음과 같다.`

### Case 12. 과도하게 앞에 놓인 비주장(non-claim) 서술

- Source pattern:
  - `본 논문은 ... 주장하지 않는다.`
  - `대체한다고 주장하지 않는다.`
  - `전역 최적성도 주장하지 않는다.`
- Comment:
  - `factual but weakens the paper reception`
  - `non claim disclaimer should be mentioned discreately`
- Bad pattern:
  - 논문의 초반부나 중심 문단에 비주장 문장을 길게 배치하여 인상을 약화
- Better direction:
  - 비주장은 필요하지만, 한계 절이나 논의 절에 전략적으로 배치
  - 본문 전반의 주도 문장은 기여와 결과가 담당하도록 구성
- Good pattern example:
  - 서론에서는 기여를 중심으로 쓰고, 한계는 `한계 및 해석 범위` 절에서 정리

### Case 13. 논문의 핵심 주장을 지나치게 축소하는 위치 설정

- Source pattern:
  - `본 연구의 위치는 ... warm start 생성 기법에 가깝다.`
  - `핵심 질문은 ... 대체할 수 있는가가 아니라 ...`
- Comment:
  - `this is not a key purpose ... should be demoted`
- Bad pattern:
  - 연구의 본체보다 부차적 응용을 먼저 내세움
- Better direction:
  - 방법 자체의 기여를 먼저 제시하고, downstream 활용은 응용 가능성으로 위치시킴
- Good pattern example:
  - `제안 기법은 control point 공간에서 연속 안전 제약을 구성하는 방법을 제공하며, downstream 초기화에도 활용될 수 있다.`

### Case 14. 결과가 약할 때 지나치게 부정적으로 서술

- Source pattern:
  - `강한 실증적 결론을 내리기 어렵다`
  - `제한적으로만 확인되었다`
- Comment:
  - 일부 메모에서는 이런 표현이 논문 인상을 지나치게 깎는다고 지적
- Bad pattern:
  - 정직함은 유지하되, 독자에게 “그래서 의미가 없는가?”라는 인상을 줄 정도로 표현을 약화
- Better direction:
  - 부정 대신 범위를 한정
  - “입증되지 않았다”와 “가능성이 없다”를 구분
- Good pattern example:
  - `해당 효과는 추가 검증이 필요하며, 본 논문에서는 가능성 수준에서만 논의한다.`

### Case 15. 특정 실험용 해킹성 제약을 본문 중심에 배치

- Source pattern:
  - `prograde preservation constraint도 함께 사용한다`
- Comment:
  - `temporary hack solution ... does not play a big role`
- Bad pattern:
  - 실험 편의를 위한 보조 장치를 핵심 이론처럼 크게 설명
- Better direction:
  - 실험 설정 절에 짧게 두거나 부록으로 이동
- Good pattern example:
  - `실험 재현을 위해 추가 제약 하나를 사용하였으며, 이는 본 방법의 핵심 요소는 아니다.`

### Case 16. 실험 설정에서 과도한 운영 정보

- Source pattern:
  - `실패한 경우에도 가능한 한 결과를 제외하지 않고 기록하였다.`
  - `모든 보고 결과는 dv 모드의 objective를 사용한다.`
- Comment:
  - `unnecessary info`
- Bad pattern:
  - 독자에게 직접 의미가 크지 않은 운영 규칙이 실험 설명의 밀도를 해침
- Better direction:
  - 재현성과 직접 관련된 것만 남기기
  - 결과 해석에 필요하지 않은 운영 메모는 축약
- Good pattern example:
  - 실험 재현에 직접 필요한 설정만 간결하게 정리

### Case 17. 논문 한국어 문체에서 더 자연스러운 대체어가 있음

- Source pattern:
  - `비볼록`
- Comment:
  - `say non-convex`
- Interpretation:
  - 이 경우는 순수 언어 문제라기보다, 해당 독자층이 어떤 표현에 더 익숙한지에 대한 선택 문제
- Guidance:
  - `비볼록`과 `non-convex` 중 하나를 정해서 문서 전체에서 통일

---

## 3. Reusable Writing Rules

위 사례에서 재사용할 수 있는 규칙을 정리하면 다음과 같다.

1. 영어를 유지할지 번역할지는 `익숙한가`를 기준으로 정한다.
2. 같은 개념은 문서 전체에서 한 가지 표현으로 통일한다.
3. 서론과 결론의 중심 문장은 기여와 결과를 말해야 하며, 비주장은 뒤로 뺀다.
4. 코드 구현 용어(`mode`, 내부 변수명, 함수명)는 논문 본문에 직접 가져오지 않는다.
5. 한국어에서 어색한 직역 문장(`안전성 논리는 단순하다`)은 논리 연결 문장으로 고친다.
6. 부정적 한계 서술은 필요하지만, 연구의 핵심 인상을 먼저 세운 뒤 배치한다.
7. 기여 문장에는 `무엇을 했는가`뿐 아니라 `왜 의미 있는가`도 들어가야 한다.

---

## 4. Good Pattern / Bad Pattern Seed List

향후 자동 패턴 생성용 seed를 위해 핵심 쌍만 별도로 적어 둔다.

- Bad: `초기 추정`
- Good: `초기값`, `초기 해`, `초기 궤적`

- Bad: `비실현 해`
- Good: `제약을 만족하지 않는 해`, `feasible하지 않은 해`

- Bad: `외부 반복`
- Good: `각 반복에서`, `SCP 반복`

- Bad: `안전성 논리는 단순하다`
- Good: `이 제약이 안전성을 보장하는 이유는 다음과 같다`

- Bad: `본 논문은 ... 주장하지 않는다`를 서론 중심에 반복 배치
- Good: 비주장은 `한계 및 해석 범위` 절에서 정리

- Bad: `구현의 dv 모드`
- Good: 목적함수의 수학적 정의를 직접 기술

- Bad: 한국어로 충분한 단어까지 영어 유지
- Good: 핵심 기술 용어만 영어 유지, 나머지는 자연스러운 한국어 사용

---

## 5. Notes

모든 첨삭 메모를 그대로 수용해야 하는 것은 아니다. 일부 메모는 순수한 번역 교정이 아니라 논문 전략, claim positioning, 혹은 향후 결과 전망에 대한 의견을 포함한다. 따라서 이후 문서 작성에서는 다음 세 층위를 구분해야 한다.

1. 반드시 반영할 언어 문제
2. 문서 정책에 따라 선택할 용어 문제
3. 연구 전략과 claim positioning에 관한 편집 판단 문제
