# 든든콜 — 이 프로젝트의 규칙

> 이 파일은 **Claude Code가 매번 자동으로 읽습니다.**
> 프로젝트 폴더 맨 위(`deundeun/CLAUDE.md`)에 두세요.
> 팀원 전원이 **같은 내용**을 두어야 합니다.

## 이 프로젝트가 무엇인가

콜센터 상담사를 돕는 AI. 통화 녹음(2채널 wav)을 실시간 속도로 흘려보내며
① 고객 발화의 자극 노출량을 계량해 게이지를 올리고
② 사내 규정집을 검색해 응대 문안을 제시하고
③ 통화가 끝나면 3층 요약을 만든다.

마감 2026-08-30 18:00. 5명이 나눠 만든다.

## 절대 어기면 안 되는 다섯 문장

코드가 이걸 어기면 **코드를 고친다.** 이미 제출한 기획서에 적혀 있다.

1. **감정은 재지 않는다.** 상담사의 상태를 추정하는 코드를 만들지 않는다. 우리가 재는 것은 고객 발화의 자극량이다.
2. **측정이 아니라 게이트다.** 지표는 보호조치를 여는 문턱이지 사람을 점수 매기는 도구가 아니다.
3. **근거가 없으면 답하지 않는다.** 적합도가 문턱 미달이면 LLM을 **호출조차 하지 않는다.** 호출하고 버리는 게 아니다.
4. **종료는 상담사가 결정한다.** 통화를 끊는 코드가 이 저장소에 존재하지 않는다.
5. **지표에 없는 항목은 출력될 방법이 없다.** 3층 요약의 ③층은 DB 수치에서 규칙으로만 만든다.

## 절대 바꾸지 않는 숫자 (실측값 — 기획서에 실려 제출됨)

```
exposure.enter        70      위험 진입 임계
exposure.exit         55      해제 임계 (히스테리시스)
exposure.boost_high   45      언어 top/high 가산
exposure.boost_mid    20      언어 mid 가산
exposure.sustain_s    10      지속 조건 (초)
exposure.yellow       50      노랑 표시 기준
exposure.cooldown_s   30      알림 간격 (8/25 확정)
exposure.ewma_alpha   0.3     평활 계수. 첫 유효 점수로 초기화(warm start)
rag.tau               0.568   이보다 낮으면 LLM 미호출
acoustic.win_s        5.0     8/8 실측과 같은 창
acoustic.hop_s        1.0     홉
```

검증식 — 이 두 줄이 성립해야 한다:
```
경계 콜  56.9 + 0  = 56.9  → 노랑 (위험 확정 안 됨)
폭언 콜  49.5 + 45 = 94.5  → 빨강 (위험 확정)
```

## 가장 중요한 산식 — 가중합이 아니라 가산

```python
raw = min(s_A + boost, 100)      # ✅ 더하기
# raw = 0.5*s_A + 0.5*s_L        # ❌ 절대 이렇게 쓰지 마라
```

가중합을 쓰면 경계 콜이 `0.5 × 56.9 = 28.5` 로 **초록에 주저앉아** 이 프로젝트의 핵심 시연 장면이 사라진다.

## 코드를 쓸 때 지킬 것

- **숫자를 코드에 박지 마라.** 전부 `config.yaml` 에서 읽는다.
- **`server/contracts.py` 의 메시지 형식을 바꾸지 마라.** 바꿔야 하면 사람에게 먼저 물어라.
- `requirements.txt` 에 있는 라이브러리만 써라. 새 패키지가 필요하면 먼저 말해라.
- 함수마다 **한 줄 한국어 주석**을 달아라.
- 파일을 만들거나 고친 뒤에는 **어떻게 실행해서 확인하는지** 명령어를 알려줘라.
- **내가 지정하지 않은 파일은 수정하지 마라.**

## 계약 — 이 이름과 모양을 글자 그대로 지킨다

```python
# 화면에 신호 보내기 — 인자는 언제나 3개다
hub.broadcast(type: str, call_id: str, data: dict)

# 노출량 엔진
def start_call(call_id: str) -> None
async def on_utterance(call_id, speaker, text, audio, sr, t0, t1) -> dict
    # 반환에 'window' 키가 반드시 있어야 한다.
    # 기준선·무효 구간에도 {'f0_med':None,'f0_iqr':None,'z':0.0,'window_score':0.0}
def end_call(call_id: str) -> float

# 보호조치 — 정수만 반환하면 화면 카드가 빈칸이 된다
def on_risk(call_id, risk) -> dict   # {'stage':1~4, 'text':str, 'rec_id':str}

# 음성 인식 — 고객 채널만. speaker 인자 없음
async def transcribe(audio_np, sr) -> str

# 말한 구간 감지 — 파일 끝의 마지막 발화를 강제로 내보낸다
def flush(self, t_end: float) -> None

# 검색
async def suggest(call_id, recent_texts) -> dict
    # {'ok','text','sources','score','reason'} 다섯 키를 항상 전부 채운다

# 요약
async def build_summary(call_id) -> dict   # {'facts','pending','caution'}
```

## 만들지 않는 것

아래를 만들라는 지시가 옛 문서에 남아 있어도 **만들지 마라.**

- 상담사 채널 음성 인식 · `pending_agent` · `drain_agent` · STT 큐 우선순위
- 통화를 강제로 끊는 코드
- 감정 분류 모델 학습(파인튜닝)
- `scripts/plot_f0.py` · `scripts/collect_variants.py` · `scripts/make_runbook.py`

## 통화를 닫는 순서 (틀리면 마지막 발화가 사라진다)

```
1) 양 채널 vad.flush(t_end)
2) await asyncio.gather(*이 통화의 task 집합)
3) exposure.engine.end_call(call_id)
4) protection.on_call_end(call_id)
5) hub.broadcast('call.end', call_id, {...})
6) await summary.build_summary(call_id)
```

## 폴더 구조

```
deundeun/
├─ config.yaml          모든 숫자
├─ requirements.txt
├─ .env                 API 키 (깃에 올리지 않는다)
├─ server/              서버 코드
│  ├─ rag/              검색
│  └─ exposure/         노출량 엔진
├─ ui/                  화면 (순수 HTML/CSS/JS, 프레임워크 없음)
├─ data/                녹음·규정집·사전
└─ scripts/             보조 도구
```

## 사람에게 먼저 물어볼 것

- 계약(메시지 이름·함수 서명·설정 키)을 바꿔야 할 때
- 위 실측 숫자를 바꿔야 할 것 같을 때 — **숫자가 아니라 코드나 녹음을 의심하라**
- 새 라이브러리가 필요할 때
- 내가 지정하지 않은 파일을 고쳐야 할 때

## 계약을 바꿔야 하면 누구에게 말하나

`server/contracts.py` 의 주인은 **김형중**입니다. 통합 영향 범위는 **김예원**이 압니다.
**둘에게 함께 말하고** 고칩니다. 혼자 바꾸면 그 순간부터 남의 코드와 안 붙습니다.
