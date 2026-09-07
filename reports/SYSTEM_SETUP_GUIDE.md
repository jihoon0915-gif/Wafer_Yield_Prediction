# 실행 · 배포 · 구조 가이드

README의 빠른 시작 가이드보다 자세한 버전. 로컬 실행, 분석 파이프라인 재현, Streamlit
Community Cloud 배포, 전체 API 명세, 폴더 구조를 정리한다.

## 1. 환경 설정

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt`는 런타임 전용(버전 고정 — 저장된 모델 pkl 역직렬화 호환성 때문).
분석 파이프라인을 재현하려면 `requirements-dev.txt`(CatBoost·TabNet·Optuna·statsmodels 등
추가)를 설치한다.

## 2. 대시보드 실행

```bash
streamlit run app_ui.py            # http://localhost:8501
```

API가 떠 있지 않으면 대시보드가 같은 프로세스 안에서 백그라운드 스레드로 FastAPI를
자동 기동한다(Streamlit Community Cloud처럼 포트를 하나만 열어주는 호스팅 대응).

API 문서(Swagger UI)를 따로 확인하려면 두 프로세스로 분리 실행한다.

```bash
uvicorn app_api:app --reload       # http://127.0.0.1:8000/docs
streamlit run app_ui.py            # 다른 터미널
```

## 3. 분석 파이프라인 재현 (선택)

산출물은 이미 `models/`, `reports/`에 포함되어 있어 재실행하지 않아도 대시보드는 동작한다.

```bash
pip install -r requirements-dev.txt

python scripts/10_run_all_hypotheses_h1_h13.py   # 가설 H1~H13 전체 검증
python scripts/11_final_preprocessing_pipeline.py
python scripts/12_final_modeling_pipeline.py
python scripts/13_model_interpretation_shap.py
```

## 4. 전체 API 명세

| 엔드포인트 | 메서드 | 설명 |
|---|---|---|
| `/` | GET | 헬스체크, 로드된 모델 목록 |
| `/meta` | GET | 피처 범위/기본값, 모델별 성능, Top KPI 목록 (UI 초기화용) |
| `/predict` | POST | 단일/배치 웨이퍼 결함 수 예측 (+ 경보 판정) |
| `/simulate` | POST (async) | What-If — KPI 변경 전/후 예측치 + SHAP 기여도 변화 |
| `/wafer-shap/{group_id}` | GET (async) | 실측 웨이퍼의 실제 공정값 기준 예측 + SHAP 전체 분해 |
| `/wafer-map/{group_id}` | GET | 26×26 die 단위 실측 불량 분포 |
| `/wafer-list` | GET | 웨이퍼맵 조회 가능한 웨이퍼 목록 |
| `/alerts` | GET | 최근 경보 이력 |
| `/alerts/config` | GET | 경보 임계값 + Slack 연동 상태 |
| `/alerts/test` | POST | Slack 웹훅 설정 검증(테스트 메시지 전송) |

`/simulate`, `/wafer-shap`는 SHAP 연산이 CPU-bound라 `asyncio.to_thread`로 스레드풀에
위임해 이벤트 루프를 막지 않는다.

## 5. Streamlit Community Cloud 배포

1. [share.streamlit.io](https://share.streamlit.io) 접속 → GitHub 계정으로 로그인
2. **New app** → 이 저장소 선택 → Main file path에 `app_ui.py` 입력 → **Deploy**

저장소에 배포에 필요한 설정 파일이 모두 포함되어 있다.

| 파일 | 역할 |
|---|---|
| `requirements.txt` | 런타임 의존성만(버전 고정) |
| `requirements-dev.txt` | 분석 파이프라인 재현용(배포에는 미사용) |
| `packages.txt` | LightGBM이 요구하는 시스템 패키지(`libgomp1`) |
| `.streamlit/config.toml` | 테마 설정 |

Slack 알림을 켜려면 배포 후 앱 관리 화면의 **Settings → Secrets**에 아래를 추가한다.

```toml
SLACK_WEBHOOK_URL = "https://hooks.slack.com/services/..."
SLACK_ALERT_MIN_SEVERITY = "warning"
```

API를 별도 호스트에 배포한 경우 `WAFER_API_URL` 환경변수로 지정하면 대시보드가 그쪽을
바라본다.

## 6. 폴더 구조

```
.
├── data/
│   ├── raw/                        # 원본 공정 CSV 6개
│   └── processed/                  # 병합·전처리 산출물
├── src/wafer/
│   ├── config.py                   # 경로/스펙 상수
│   ├── io.py · integrate.py        # CSV 파싱, No_Die 기준 병합, 웨이퍼맵 복원
│   ├── validate.py                 # 병합 결과 회귀 검증(9개 assertion)
│   ├── features.py                 # 도메인 피처 엔지니어링
│   ├── modeling.py                 # GroupKFold CV 러너, 전처리 파이프라인
│   ├── final_pipeline.py           # 최종 피처 목록, 커스텀 transformer
│   ├── alerting.py                 # 경보 규칙 엔진 + 이력 버퍼
│   └── slack.py                    # Slack Incoming Webhook 전송
├── scripts/
│   ├── 02~06_*.py                  # 1차 벤치마크: 회귀/분류/XAI/최적화/시뮬레이션
│   ├── 07_domain_master_dataset.py
│   ├── 08~09_h1_*.py               # UV_type 교란 심층 재검증
│   ├── hypothesis_ml_lib.py        # 가설 검증 공용 프레임워크
│   ├── h1~h13_*.py                 # 가설별 검증 스크립트
│   ├── 10_run_all_hypotheses_h1_h13.py
│   ├── 11_final_preprocessing_pipeline.py
│   ├── 12_final_modeling_pipeline.py
│   └── 13_model_interpretation_shap.py
├── models/                         # 배포 파이프라인 + 후보 모델 3종
├── reports/                        # 단계별 리포트(md/csv/png)
│   ├── FINAL_PROJECT_REPORT.md
│   ├── PROCESS_OPTIMIZATION_STRATEGY.md
│   ├── SYSTEM_SETUP_GUIDE.md       # 이 문서
│   ├── hypothesis_H{1~13}_result.md
│   └── figures/
├── app_api.py                      # FastAPI 백엔드
├── app_ui.py                       # Streamlit 대시보드
└── .env.example                    # 환경변수 템플릿
```

## 관련 문서

- [`FINAL_PROJECT_REPORT.md`](FINAL_PROJECT_REPORT.md) — 분석 방법론·모델 성능·SHAP 해석 종합, 알려진 한계
- [`PROCESS_OPTIMIZATION_STRATEGY.md`](PROCESS_OPTIMIZATION_STRATEGY.md) — 공정 개선 전략 보고서
