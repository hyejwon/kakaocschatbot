# 🛒 쇼핑몰 CS 자동 분류 챗봇

카카오톡 고객 서비스를 위한 자동 분류 챗봇 데모입니다.  
내부 지식베이스가 부족할 경우 외부 RAG로 Fallback하는 시스템입니다.

## 주요 기능

- 🎯 **자동 분류**: 고객 문의를 7개 카테고리로 자동 분류
  - 주문/결제, 배송, 교환/환불, 상품 문의, 회원/로그인, 쿠폰/포인트, 기타
- 📚 **RAG 기반 응답**: TF-IDF + 코사인 유사도를 활용한 검색
- 🔄 **Fallback 시스템**: 내부 KB 신뢰도가 낮으면 외부 KB 활용
- 🎨 **Streamlit UI**: 직관적인 웹 인터페이스

## 설치 및 실행

### 1. uv 설치 (처음 한 번만)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# 또는 Homebrew
brew install uv
```

### 2. 가상환경 생성 및 패키지 설치

```bash
# 프로젝트 디렉토리로 이동
cd shopbot-demo

# 가상환경 생성
uv venv

# 가상환경 활성화
source .venv/bin/activate

# 패키지 설치 (초고속!)
uv pip install -r requirements.txt
```

### 3. 앱 실행

```bash
streamlit run app.py
```

브라우저에서 자동으로 열립니다 (보통 http://localhost:8501)

### 4. 종료

```bash
# 가상환경 비활성화
deactivate
```

## 다음 실행부터

```bash
cd shopbot-demo
source .venv/bin/activate
streamlit run app.py
```

## 필요 패키지

- streamlit - 웹 UI
- pandas - 데이터 처리
- scikit-learn - TF-IDF 벡터화 및 유사도 계산

## 프로젝트 구조

```
shopbot-demo/
├── app.py              # 메인 애플리케이션
├── external_kb.csv     # 외부 지식베이스
├── requirements.txt    # 패키지 의존성
├── .gitignore         # Git 제외 파일
└── .venv/             # 가상환경 (Git 제외)
```

## 데이터 구조

지식베이스 CSV는 다음 컬럼을 포함합니다:
- `id`: 문서 ID
- `category`: 카테고리
- `title`: 제목
- `content`: 내용

## 설정

앱 실행 후 UI에서 다음을 조정할 수 있습니다:
- **내부 KB 신뢰도 임계값**: 낮을수록 내부 KB를 더 선호
- **검색 Top-K**: 검색 결과 개수
- **디버그 모드**: 점수 및 근거 표시

## 개발자

데모 버전으로, 실제 운영 시 정책/문구/임계값은 커스터마이징이 필요합니다.

