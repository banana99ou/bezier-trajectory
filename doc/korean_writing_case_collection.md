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

### Case 18. 기여 bullet에 과도한 기술적 parameter 나열

- Source pattern:
  - `동일한 downstream multi-phase LGL Pass 2 solver·dynamics·tolerance 하에서 two-pass direct collocation pipeline의 Pass 1 단계를 Bézier SCP upstream으로 대체하는 matched pipeline-variant 비교 실험을 수행하여, 정의된 작동 영역 내에서 최종 비용이 보존되고 end-to-end runtime이 일부 사례에서 감소함을 확인한다.`
- Comment:
  - `feels too long. and too much detailed`
- Bad pattern:
  - 기여 bullet 한 줄에 실험 설정, 조건, 결과, 해석 범위를 모두 욱여넣어 핵심이 흐려짐
  - 기여 목록은 독자가 한눈에 기여의 윤곽을 파악해야 하는데, 본문에나 들어갈 세부 parameter가 여기서 먼저 노출됨
- Better direction:
  - 기여 bullet은 `무엇을 했다 + 왜 의미가 있다` 두 요소에 집중
  - 구체적 parameter, 조건, 수치는 5절(실험 설정)과 6절(결과)에서 제시
- Good pattern example:
  - `downstream direct collocation pipeline의 Pass 1 단계를 제안 기법으로 대체하는 비교 실험을 수행하여, 정의된 작동 영역 안에서의 초기화 교체 가능성을 확인한다.`

### Case 19. 비주장 disclaimer를 기여 목록 바로 뒤에 배치 (Case 12 강화)

- Source pattern:
  - `위 다섯 번째 기여는 본 논문에서 주장하지 않는 범위를 명확히 하기 위해 별도로 강조한다. 이 실험은 direct collocation 대비 method-class 우월성 주장이 아니라, 동일한 DCM pipeline 내부에서 Pass 1 초기화 단계를 대체했을 때의 거동을 측정한 것이며, 결과의 해석은 Bézier upstream이 feasible하고 downstream Pass 2가 수렴하는 circular orbit 영역으로 한정된다.`
- Comment:
  - `is this line really neccessary? even if it is neccessary, shouldn't it move to section 7?`
- Bad pattern:
  - Case 12의 일반 원칙("비주장은 서론에서 빼라")이 필요할 때조차도, 비주장을 기여 목록 바로 뒤에 길게 붙이면 기여의 여운을 지움
  - 독자는 기여 목록에서 "한 일"을 기억하고 싶은데 곧바로 "하지 않은 일"의 긴 설명이 뒤따르면 인상이 반전됨
- Better direction:
  - 해당 비주장 paragraph는 §7(한계 및 해석 범위)로 이동
  - 기여 bullet에 `정의된 작동 영역 안에서`처럼 한 개의 단서어(qualifier)만 넣어 scoping은 유지
  - 비주장이 정말로 필요한지 먼저 물어본 뒤, 필요하다면 §7로 옮긴다
- Good pattern example:
  - 기여 bullet에 `정의된 작동 영역 안에서` 같은 scoping qualifier 한 개 추가
  - §7에 별도 항목으로 "본 비교가 주장하지 않는 것" 정리

### Case 20. 외래어 transliteration을 loneword로 사용

- Source pattern:
  - `문제 인스턴스는 trajectory database ...의 converged 행에서 추출하였다.`
- Comment:
  - `unnatural loneword used`
- Bad pattern:
  - `인스턴스`, `런타임`, `페이로드` 등 영어를 한글 음역해서 맨 단독으로 쓰면 한국어 공학 문헌에서 어색하게 떠 있음
  - 문맥에 따라 대체 가능한 한국어 표현(예: `사례`, `실행 시간`)이 있는데 굳이 음역어를 쓰면 번역 비용을 독자에게 떠넘김
- Better direction:
  - 한국어 대체어가 있으면 그것을 사용(`문제 사례`, `실험 사례`)
  - 한국어 대체어가 부자연스럽다면 영어 그대로 두고 병기(`instance(사례)`)
  - 가장 피해야 할 것은 음역어 단독 사용
- Good pattern example:
  - Bad: `문제 인스턴스는...`
  - Good: `문제 사례는...` 또는 `문제 instance(사례)는...`

### Case 21. 절 사이 연결 문단의 과도한 길이

- Source pattern:
  - `이상의 결과는 제안 정식화 자체의 실행 가능성과 분할 수 및 차수 변화에 따른 거동을 보여준다. 다음 절에서는 본 프레임워크의 downstream 활용 가능성에 관한 matched pipeline-variant 비교 결과를 제시한다.`
- Comment:
  - `not bad. this is connector paragraph. but feels too long for a connector paragraph. what do you think?`
- Bad pattern:
  - 연결 문단(이전 절 요약 + 다음 절 예고)이 두 개 이상의 완전한 문장으로 구성되면, 절의 리듬을 끊고 본문을 한 번 더 훑어 읽게 함
  - 요약과 예고를 둘 다 full sentence로 쓰면 독자가 이미 읽은 내용을 한 번 더 처리해야 함
- Better direction:
  - 연결 문단은 한 문장 이내로 축약
  - 요약은 생략하고 예고만 남기거나, "A를 보였고, 다음 절에서는 B를 다룬다" 형식으로 합치기
  - 절 간 연결이 자연스러우면 연결 문단 자체를 생략해도 됨
- Good pattern example:
  - `다음 절에서는 downstream 활용 가능성에 관한 비교 결과를 제시한다.`

### Case 22. `자리` 등 모호한 장소 명사 사용

- Source pattern:
  - `본 절은 direct collocation 대비 우월성 주장을 위한 자리가 아니다.`
- Comment:
  - `what '자리'? and unnessary line`
- Bad pattern:
  - `자리`, `장(場)`, `맥락(context)` 같은 추상적 장소 명사를 부정문의 주어로 쓰면 무엇을 부정하고 있는지 모호해짐
  - 대개는 해당 문장 자체가 불필요한 비주장 disclaimer인 경우가 많음 (Case 12와 결합)
- Better direction:
  - 문장 자체가 필요한지 먼저 검토 — 대부분 불필요
  - 정말 필요하다면 `자리` 대신 직접적인 술어를 사용
  - `본 절은 X를 주장하지 않는다` 같은 명료한 표현으로 대체
- Good pattern example:
  - 문장 삭제가 1순위
  - 유지해야 한다면: `본 절은 direct collocation 대비 우월성을 주장하지 않는다.`

### Case 23. 영어 `not A but B` 구문의 한국어 직역

- Source pattern:
  - `따라서 본 결과는 "Pass 1 단계의 완전한 drop-in 대체"를 지지하는 것이 아니라, "7개 사례 중 6개에서 동일 국소해에 수렴하고 1개 사례에서는 phase 구조 결정의 차이로 인해 결과가 달라진다"는 범위에서 해석되어야 한다.`
- Comment:
  - `reusing english sentence structure of 'not A but B'`
- Bad pattern:
  - 영어의 `not A but B` 구조를 `A가 아니라 B`로 그대로 옮기면 한국어 문장이 길어지고 주어-술어 거리가 멀어짐
  - 두 긴 인용구가 대구를 이루면 독자가 두 개를 모두 기억한 채 마지막 술어까지 가야 함
- Better direction:
  - 긍정 주장을 먼저 제시하고, 부정 단서는 짧은 후속 문장으로 처리
  - 또는 `B이며, A는 아니다` 순서로 뒤집기
  - 인용구가 길면 독립 문장으로 분리
- Good pattern example:
  - `이 결과는 7개 사례 중 6개에서 동일 국소해로의 수렴을 의미하며, 1개 사례에서는 phase 구조 결정 차이로 국소해가 달라진다. 따라서 Pass 1 단계의 drop-in 대체라기보다는 제한된 범위 안에서의 교체 가능성으로 해석해야 한다.`

### Case 24. `본 절의 X` 류 모호한 참조

- Source pattern:
  - `본 절의 작동 영역은 두 가지 독립된 경계로 정의된다.`
- Comment:
  - `what is '본 절'?`
- Bad pattern:
  - `본 절의 작동 영역`처럼 절(section)에 속성을 귀속시키면 무엇의 작동 영역인지 모호함
  - 작동 영역은 제안 기법이나 pipeline의 속성이지 절 자체의 속성이 아님
  - 영어 `This section presents X`를 `본 절의 X`로 옮길 때 잘 발생
- Better direction:
  - 속성의 진짜 주체(기법, pipeline, 결과)를 주어로 명시
  - `이하에서`, `아래에서`처럼 위치 부사를 쓰거나, 주어를 직접 서술
- Good pattern example:
  - Bad: `본 절의 작동 영역은 두 가지 독립된 경계로 정의된다.`
  - Good: `제안 pipeline의 작동 영역은 두 가지 독립된 경계로 정의되며, 이하에서 각각을 살펴본다.`

### Case 25. `증명 요약` — "Proof sketch" 직역

- Source pattern:
  - `**증명 요약.** ...`
- Comment:
  - `awkward direct translation`
- Bad pattern:
  - 영어 논문의 `Proof sketch.` 라벨을 기계적으로 한국어로 옮김
  - 한국어 수학/공학 논문에서는 `증명 요약`이라는 라벨을 잘 쓰지 않음
- Better direction:
  - `증명.` 한 단어로 충분
  - 내용이 짧고 비엄밀하면 독자가 이미 sketch로 인식함
  - 굳이 sketch임을 밝혀야 한다면 `증명의 개요`나 본문 서술로 풀어 쓰기
- Good pattern example:
  - `**증명.** Bézier 곡선은 제어점의 컨벡스 헐 안에 놓인다. ...`

### Case 26. 용어 선택 일관성 (저자 정책: `컨벡스`)

- Source pattern:
  - `볼록 껍질`
  - `볼록 2차 계획 문제`
- Comment:
  - `sync word choice with the rest of paper. my choice of word for convex is "컨벡스"`
- Bad pattern:
  - `볼록`과 `컨벡스`를 한 문서 안에서 섞어 씀
  - 이 저자의 정책은 `convex` 계열 용어를 `컨벡스`로 통일
- Better direction:
  - `convex hull` → `컨벡스 헐`
  - `convex QP` → `컨벡스 QP`
  - `non-convex` → `논컨벡스`
  - `convexity` → `컨벡스성`
  - `convexification` → `컨벡스화`
- Good pattern example:
  - `제어점의 컨벡스 헐 안에 놓인다.`

### Case 27. `유래한다` — unusual verb for "stem from"

- Source pattern:
  - `X는 Y가 구조적으로 어려운 수준이라는 점에서 유래한다.`
- Comment:
  - `unusual word choice`
- Bad pattern:
  - `유래한다`는 원인 서술에서 자연스럽지 않음
  - 역사·기원 서술에 가까운 단어이며 기술 서술에서는 이질감
- Better direction:
  - `이는 ... 때문이다`
  - `기인한다`
  - `~에서 비롯된다`
- Good pattern example:
  - `이는 수렴 허용오차가 구조적으로 도달하기 어려운 수준이기 때문이다.`

### Case 28. `SCP 절차`의 어색함

- Source pattern:
  - `SCP 절차에서는 ...`
- Comment:
  - `awkward`
- Bad pattern:
  - `절차(procedure)`라는 추상명사가 반복 설명과 결합할 때 어색함
  - 이미 본문에서 `SCP 반복`이라는 자연스러운 표현이 정착되어 있음
- Better direction:
  - `SCP 반복에서는 ...`
  - `SCP 루프에서는 ...`
  - 알고리즘을 지칭할 때는 `SCP 알고리즘`
- Good pattern example:
  - `SCP 반복에서는 매 반복마다 지지 반공간이 재구성된다.`

### Case 29. `노름`이라는 음역의 부자연스러움

- Source pattern:
  - `Frobenius 노름`
- Comment:
  - `da fuq is this brain dead word? I know Korean people use this word. this feels soooo brain dead Konglish. either use 한자어 or English`
- Bad pattern:
  - `norm`의 한국어 음역 `노름`은 발음과 기존 한국어 단어(도박의 `노름`)와 충돌
  - 공학 문헌에서 흔히 쓰이긴 하지만 저자 기준으로는 부자연스러움
- Better direction:
  - 영어 그대로 유지: `Frobenius norm`
  - 이미 수식이 옆에 있으면 굳이 단어로 반복하지 않아도 됨
- Good pattern example:
  - `제어점 변화의 Frobenius norm이 $10^{-12}$ 이하로 감소하기 어렵다.`

### Case 30. `변화시키는 진단 실험` — awkward noun-stacked phrase

- Source pattern:
  - `반복 한도를 100부터 10000까지 변화시키는 진단 실험을 수행하였다.`
- Comment:
  - `bad word choice. choose other words for this meaning`
- Bad pattern:
  - `변화시키는`은 영어 `varying`의 직역이며, 뒤의 `진단 실험`과 결합하면 추상명사 밀도가 과도
  - Case 7(어색한 추상명사 결합)의 연장
- Better direction:
  - 동작 중심으로 풀어 쓰기
  - `바꿔가며 ... 를 측정하였다`
  - `... 를 조절하며 ... 를 관찰하였다`
- Good pattern example:
  - `반복 한도를 100부터 10000까지 바꿔가며 수렴 거동을 측정하였다.`

### Case 31. `X는 약 Y 이내에 있었고` — English "was within" 직역

- Source pattern:
  - `안전 여유는 약 7% 이내에 있었고, ...`
- Comment:
  - `again reuse of English sentence structure`
- Bad pattern:
  - `X was within Y%`를 `X는 Y% 이내에 있었고`로 그대로 옮김
  - 한국어에서는 `~에 있다`가 공간 존재에 가깝고, 수치 범위에는 `~로 좁혀지다`, `~에 머물다`가 자연스러움
- Better direction:
  - `~이내로 좁혀졌다`
  - `~이내에 머물렀다`
  - `~이내의 차이를 보였다`
- Good pattern example:
  - `delta-v 대리 지표와 안전 여유는 각각 약 2.6%, 7% 이내로 좁혀졌고, ...`

### Case 32. `보고된` — direct translation of "reported"

- Source pattern:
  - `보고된 결과는 안정화된 해로 해석할 수 있다.`
- Comment:
  - `another brain dead direct translation of "reported"`
- Bad pattern:
  - 영어 `the reported results`를 `보고된 결과`로 옮김
  - 논문 본문에서 `보고`라는 단어는 보통 관료·보고서 맥락을 연상시킴
  - 자체 논문 안에서 자기 결과를 `보고된`이라고 부르면 어색함
- Better direction:
  - `위 결과는 ...`
  - `제시된 결과는 ...`
  - `표 X의 결과는 ...`
  - 문장에 따라 지시어로 대체 (`이 결과는`)
- Good pattern example:
  - `위 결과는 안정화된 해로 해석할 수 있다.`

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
8. 기여 bullet에는 수치·parameter·조건을 쌓지 않는다. 한 줄에 한 개의 기여와 한 개의 의의만 담는다.
9. 비주장 disclaimer는 기여 목록 직후가 아니라 §7로 이동시킨다. 기여 bullet 안의 짧은 scoping qualifier 한 개만 허용한다.
10. 외래어 음역(`인스턴스`, `런타임` 등)을 loneword로 쓰지 않는다. 한국어 대체어를 쓰거나, 영어 원문을 병기한다.
11. 절 사이 연결 문단은 한 문장 이내로 축약한다. 요약과 예고를 둘 다 full sentence로 쓰지 않는다.
12. `자리`, `본 절의 X` 같은 모호한 장소·절-귀속 명사 구문은 지우거나 진짜 주어로 고친다.
13. 영어 `not A but B` 구문을 `A가 아니라 B` 형태로 직역하지 않는다. 긍정 주장을 먼저 쓰고 부정 단서는 짧게 뒤에 붙인다.
14. 영어 수학/증명 라벨(`Proof sketch`, `Remark`)을 기계적으로 번역하지 않는다. `증명.` 등 한국어 관용 라벨로 축약하거나 본문 서술로 흡수한다.
15. `convex` 계열 용어는 저자 정책에 따라 `컨벡스 / 컨벡스 헐 / 논컨벡스 / 컨벡스화`로 통일한다. `볼록`과 혼용하지 않는다.
16. `유래한다`, `에 있었다`, `보고된` 등 영어 서술 동사의 기계적 직역을 피한다. `이기 때문이다`, `로 좁혀지다`, `위 결과는` 등 한국어 서술 습관을 따른다.
17. `norm`처럼 음역이 기존 한국어 단어(`노름`)와 충돌하는 경우 영어 그대로 둔다.
18. `변화시키는 진단 실험`처럼 동사+추상명사 스택으로 영어 원문을 직역하지 않는다. 동작 중심 문장으로 다시 쓴다.

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

- Bad: 기여 bullet에 실험 조건·parameter·수치를 모두 욱여넣음
- Good: `무엇을 했다 + 정의된 작동 영역 안에서 의의`만 남기고 세부는 실험 설정 절로 이동

- Bad: 기여 목록 직후에 "주장하지 않는 범위" paragraph를 길게 배치
- Good: 기여 bullet에 짧은 scoping qualifier 한 개만 두고 비주장 본문은 §7로 이동

- Bad: `문제 인스턴스`
- Good: `문제 사례`, `실험 사례`, 또는 `instance(사례)` 병기

- Bad: 요약 문장 + 예고 문장의 2문장 연결 문단
- Good: `다음 절에서는 ...을 다룬다` 한 문장

- Bad: `...를 위한 자리가 아니다`, `본 절의 작동 영역은...`
- Good: 문장 삭제, 또는 `제안 pipeline의 작동 영역은...`

- Bad: `A를 지지하는 것이 아니라, B는 범위에서 해석되어야 한다` (긴 not-A-but-B 직역)
- Good: `B이며, A라기보다는 제한된 범위 안에서의 교체 가능성으로 해석해야 한다`

- Bad: `**증명 요약.**`
- Good: `**증명.**`

- Bad: `볼록 껍질`, `볼록 2차 계획 문제`
- Good: `컨벡스 헐`, `컨벡스 QP`

- Bad: `유래한다`
- Good: `이기 때문이다`, `기인한다`, `에서 비롯된다`

- Bad: `SCP 절차`
- Good: `SCP 반복`, `SCP 루프`

- Bad: `Frobenius 노름`
- Good: `Frobenius norm`

- Bad: `반복 한도를 변화시키는 진단 실험을 수행하였다`
- Good: `반복 한도를 바꿔가며 수렴 거동을 측정하였다`

- Bad: `안전 여유는 약 7% 이내에 있었고`
- Good: `안전 여유는 약 7% 이내로 좁혀졌고`

- Bad: `보고된 결과는 ...`
- Good: `위 결과는 ...`, `제시된 결과는 ...`

---

## 5. Notes

모든 첨삭 메모를 그대로 수용해야 하는 것은 아니다. 일부 메모는 순수한 번역 교정이 아니라 논문 전략, claim positioning, 혹은 향후 결과 전망에 대한 의견을 포함한다. 따라서 이후 문서 작성에서는 다음 세 층위를 구분해야 한다.

1. 반드시 반영할 언어 문제
2. 문서 정책에 따라 선택할 용어 문제
3. 연구 전략과 claim positioning에 관한 편집 판단 문제
