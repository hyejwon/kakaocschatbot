import os
import re
import pandas as pd
import reflex as rx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Utilities
# -----------------------------
CATEGORIES = ["주문/결제", "배송", "교환/환불", "상품 문의", "회원/로그인", "쿠폰/포인트", "기타(상담필요)"]

KEYWORD_RULES = {
    "교환/환불": [r"환불", r"반품", r"교환", r"취소", r"철회"],
    "배송": [r"배송", r"출고", r"송장", r"택배", r"언제 와", r"도착", r"지연"],
    "주문/결제": [r"결제", r"카드", r"승인", r"주문", r"입금", r"결제 실패"],
    "상품 문의": [r"사이즈", r"재고", r"색상", r"스펙", r"소재", r"핏", r"길이"],
    "회원/로그인": [r"로그인", r"비밀번호", r"아이디", r"인증", r"회원"],
    "쿠폰/포인트": [r"쿠폰", r"포인트", r"적립", r"할인", r"프로모션"]
}

def rule_classify(text: str):
    t = text.strip().lower()
    hits = []
    for cat, patterns in KEYWORD_RULES.items():
        score = 0
        for p in patterns:
            if re.search(p, t, re.IGNORECASE):
                score += 1
        if score > 0:
            hits.append((cat, score))
    hits.sort(key=lambda x: x[1], reverse=True)
    if not hits:
        return "기타(상담필요)", 0.35, ["키워드 매칭 없음"]
    top_cat, top_score = hits[0]
    conf = min(0.55 + 0.1 * (top_score - 1), 0.85)
    reasons = [f"키워드 매칭: {top_cat} ({top_score}개)"]
    return top_cat, conf, reasons

def load_kb(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["id", "category", "title", "content"])
    df = pd.read_csv(path)
    for col in ["id", "category", "title", "content"]:
        if col not in df.columns:
            df[col] = ""
    df = df.fillna("")
    return df

def build_retriever(df: pd.DataFrame):
    corpus = (df["title"].astype(str) + " " + df["content"].astype(str)).tolist()
    if len(corpus) == 0:
        return None, None, corpus
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=6000)
    X = vectorizer.fit_transform(corpus)
    return vectorizer, X, corpus

def retrieve(df: pd.DataFrame, vectorizer, X, query: str, topk: int = 3):
    if df.empty or vectorizer is None or X is None:
        empty_df = pd.DataFrame(columns=list(df.columns) + ["score"])
        return empty_df, []
    qv = vectorizer.transform([query])
    sims = cosine_similarity(qv, X).flatten()
    idxs = sims.argsort()[::-1][:topk]
    rows = df.iloc[idxs].copy()
    rows["score"] = [float(sims[i]) for i in idxs]
    return rows, idxs

def should_fallback_internal(internal_hits: pd.DataFrame, threshold=0.18) -> bool:
    if internal_hits.empty:
        return True
    return float(internal_hits.iloc[0]["score"]) < threshold

def generate_template_answer(category: str, user_text: str, source_title: str, source_content: str, source_type: str):
    base = f"문의 유형: **{category}**\n\n"
    if source_type == "internal":
        base += "내부 정책/FAQ를 기준으로 안내드립니다.\n\n"
    else:
        base += "내부 정책 데이터가 부족하여, 일반적인 공개 지식/가이드 기준으로 안내드립니다.\n\n"

    base += f"**참고 항목:** {source_title}\n\n"
    base += f"- 안내: {source_content}\n\n"
    
    followups = {
        "배송": "가능하시면 **주문번호**와 **수령자 성함/연락처**를 알려주시면 배송 상태 확인에 도움이 됩니다.",
        "교환/환불": "가능하시면 **주문번호**, **수령일**, **상품 상태(미개봉/사용 여부)**를 알려주세요.",
        "주문/결제": "결제 수단(카드/계좌이체 등)과 오류 메시지가 있다면 함께 알려주시면 확인에 도움이 됩니다.",
        "상품 문의": "원하시는 **사이즈/색상**과 신체 치수(예: 키/체중)를 알려주시면 더 정확히 안내드릴 수 있습니다.",
        "회원/로그인": "사용 중인 환경(앱/웹, 기기/브라우저)과 발생 시점을 알려주시면 원인 파악에 도움이 됩니다.",
        "쿠폰/포인트": "쿠폰 코드/프로모션명과 장바구니 금액, 적용 단계(결제 전/후)를 알려주시면 확인에 도움이 됩니다.",
        "기타(상담필요)": "정확한 확인이 필요합니다. **주문번호/상황**을 남겨주시면 상담으로 빠르게 안내드리겠습니다."
    }
    base += f"**추가 확인:** {followups.get(category, followups['기타(상담필요)'])}\n\n"
    base += "_(데모 버전: 실제 운영 정책에 맞춰 문구/조건은 커스터마이징됩니다.)_"
    return base

# -----------------------------
# State
# -----------------------------
class State(rx.State):
    # Input
    user_text: str = ""
    internal_threshold: float = 0.18
    topk: int = 3
    show_debug: bool = True
    
    # Results
    pred_cat: str = ""
    confidence: float = 0.0
    reasons: list[str] = []
    source_type: str = ""
    answer: str = ""
    internal_score: float = 0.0
    
    # KB Info
    internal_count: int = 0
    external_count: int = 0
    
    # Debug info
    internal_hits_display: list[dict] = []
    external_hits_display: list[dict] = []
    
    # Loading state
    is_processing: bool = False
    
    def on_load(self):
        """페이지 로드 시 KB 정보 로드"""
        internal_df = load_kb("data/internal_kb.csv")
        external_df = load_kb("data/external_kb.csv")
        self.internal_count = len(internal_df)
        self.external_count = len(external_df)
    
    def set_user_text(self, value: str):
        self.user_text = value
    
    def set_internal_threshold(self, value: list[float]):
        self.internal_threshold = value[0]
    
    def set_topk(self, value: list[float]):
        self.topk = int(value[0])
    
    def toggle_debug(self):
        self.show_debug = not self.show_debug
    
    def process_query(self):
        """문의 처리"""
        if not self.user_text.strip():
            return
        
        self.is_processing = True
        
        # Load data
        internal_df = load_kb("data/internal_kb.csv")
        external_df = load_kb("data/external_kb.csv")
        
        # Build retrievers
        internal_vec, internal_X, _ = build_retriever(internal_df)
        external_vec, external_X, _ = build_retriever(external_df)
        
        # 1) Classification
        pred_cat, conf, reasons = rule_classify(self.user_text)
        self.pred_cat = pred_cat
        self.confidence = conf
        self.reasons = reasons
        
        # 2) Retrieve from internal first
        internal_hits, _ = retrieve(internal_df, internal_vec, internal_X, self.user_text, topk=self.topk)
        use_fallback = should_fallback_internal(internal_hits, threshold=self.internal_threshold)
        
        self.source_type = "external" if use_fallback else "internal"
        
        if self.source_type == "internal":
            best = internal_hits.iloc[0]
            self.internal_score = float(best.get("score", 0.0))
        else:
            # external retrieval with category bias
            same_cat = external_df[external_df["category"].astype(str) == pred_cat]
            vec_cat, X_cat, _ = build_retriever(same_cat)
            ext_hits, _ = retrieve(same_cat, vec_cat, X_cat, self.user_text, topk=self.topk)
            if ext_hits.empty:
                ext_hits, _ = retrieve(external_df, external_vec, external_X, self.user_text, topk=self.topk)
            best = ext_hits.iloc[0] if not ext_hits.empty else pd.Series({"title":"상담 연결 안내", "content":"정확한 확인이 필요합니다.", "score":0.0})
            self.internal_score = float(internal_hits.iloc[0]["score"]) if not internal_hits.empty else 0.0
        
        # 3) Answer
        self.answer = generate_template_answer(
            category=pred_cat,
            user_text=self.user_text,
            source_title=str(best.get("title","")),
            source_content=str(best.get("content","")),
            source_type=self.source_type
        )
        
        # 4) Debug info
        if not internal_hits.empty:
            self.internal_hits_display = internal_hits[["category","title","score"]].to_dict('records')
        else:
            self.internal_hits_display = []
        
        ext_hits_all, _ = retrieve(external_df, external_vec, external_X, self.user_text, topk=self.topk)
        if not ext_hits_all.empty:
            self.external_hits_display = ext_hits_all[["category","title","score"]].to_dict('records')
        else:
            self.external_hits_display = []
        
        self.is_processing = False

# -----------------------------
# UI Components
# -----------------------------
def header() -> rx.Component:
    return rx.box(
        rx.heading("🛒 쇼핑몰 CS 자동 분류 챗봇", size="9", weight="bold"),
        rx.text(
            "내부 KB가 부족하면 외부 RAG Fallback",
            color="gray",
            size="4"
        ),
        padding="2rem",
        background="linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        color="white",
        border_radius="0.5rem",
        margin_bottom="2rem",
    )

def input_section() -> rx.Component:
    return rx.box(
        rx.heading("1) 고객 문의 입력", size="6", margin_bottom="1rem"),
        rx.text_area(
            placeholder="예) '배송이 아직 안 와요', '환불 언제 돼요?', '쿠폰이 적용이 안돼요' 등",
            value=State.user_text,
            on_change=State.set_user_text,
            rows=6,
            width="100%",
            margin_bottom="1.5rem",
        ),
        rx.heading("2) 설정", size="6", margin_bottom="1rem"),
        rx.vstack(
            rx.hstack(
                rx.text("내부 KB 신뢰도 임계값:", width="200px"),
                rx.slider(
                    default_value=[0.18],
                    min=0.05,
                    max=0.40,
                    step=0.01,
                    on_value_commit=State.set_internal_threshold,
                    width="100%",
                ),
                rx.text(f"{State.internal_threshold:.2f}", margin_left="1rem"),
                width="100%",
                align="center",
            ),
            rx.hstack(
                rx.text("검색 Top-K:", width="200px"),
                rx.slider(
                    default_value=[3],
                    min=1,
                    max=5,
                    step=1,
                    on_value_commit=State.set_topk,
                    width="100%",
                ),
                rx.text(f"{State.topk}", margin_left="1rem"),
                width="100%",
                align="center",
            ),
            rx.checkbox(
                "디버그 정보 보기",
                checked=State.show_debug,
                on_change=State.toggle_debug,
            ),
            spacing="4",
            width="100%",
            margin_bottom="1.5rem",
        ),
        rx.button(
            "🚀 처리하기",
            on_click=State.process_query,
            size="4",
            width="100%",
            loading=State.is_processing,
            color_scheme="blue",
        ),
        padding="2rem",
        background="white",
        border_radius="0.5rem",
        box_shadow="0 2px 10px rgba(0,0,0,0.1)",
    )

def kb_status() -> rx.Component:
    return rx.box(
        rx.heading("지식베이스 상태", size="6", margin_bottom="1rem"),
        rx.vstack(
            rx.text(f"내부 KB 문서 수: {State.internal_count}개", weight="bold"),
            rx.text(f"외부 KB 문서 수: {State.external_count}개", weight="bold"),
            spacing="2",
        ),
        padding="2rem",
        background="white",
        border_radius="0.5rem",
        box_shadow="0 2px 10px rgba(0,0,0,0.1)",
    )

def result_section() -> rx.Component:
    return rx.cond(
        State.pred_cat != "",
        rx.box(
            rx.heading("결과", size="7", margin_bottom="2rem"),
            rx.grid(
                # 왼쪽: 분류 결과
                rx.box(
                    rx.heading("✅ 분류 결과", size="5", margin_bottom="1rem"),
                    rx.vstack(
                        rx.hstack(
                            rx.text("예측 카테고리:", weight="bold"),
                            rx.badge(State.pred_cat, color_scheme="green", size="3"),
                        ),
                        rx.hstack(
                            rx.text("추정 신뢰도:", weight="bold"),
                            rx.text(f"{State.confidence:.2f}"),
                        ),
                        rx.hstack(
                            rx.text("라우팅:", weight="bold"),
                            rx.badge(
                                rx.cond(
                                    State.source_type == "external",
                                    "외부 RAG 사용",
                                    "내부 KB 사용"
                                ),
                                color_scheme=rx.cond(
                                    State.source_type == "external",
                                    "orange",
                                    "blue"
                                ),
                                size="3"
                            ),
                        ),
                        rx.cond(
                            State.show_debug,
                            rx.box(
                                rx.heading("근거(디버그)", size="4", margin_top="1rem", margin_bottom="0.5rem"),
                                rx.foreach(
                                    State.reasons,
                                    lambda reason: rx.text(f"• {reason}", size="2")
                                ),
                                rx.text(f"내부 Top-1 score: {State.internal_score:.4f}", size="2", color="gray"),
                            ),
                        ),
                        spacing="3",
                        align_items="start",
                    ),
                    padding="1.5rem",
                    background="white",
                    border_radius="0.5rem",
                    box_shadow="0 2px 10px rgba(0,0,0,0.1)",
                ),
                # 오른쪽: 챗봇 응답
                rx.box(
                    rx.heading("💬 챗봇 응답", size="5", margin_bottom="1rem"),
                    rx.markdown(State.answer),
                    padding="1.5rem",
                    background="white",
                    border_radius="0.5rem",
                    box_shadow="0 2px 10px rgba(0,0,0,0.1)",
                ),
                columns="2",
                spacing="4",
                width="100%",
            ),
            # Debug 섹션
            rx.cond(
                State.show_debug,
                rx.box(
                    rx.heading("검색 결과(디버그)", size="6", margin_top="2rem", margin_bottom="1rem"),
                    rx.vstack(
                        rx.box(
                            rx.heading("내부 KB Top-K", size="4", margin_bottom="1rem"),
                            rx.cond(
                                State.internal_hits_display.length() > 0,
                                rx.table.root(
                                    rx.table.header(
                                        rx.table.row(
                                            rx.table.column_header_cell("카테고리"),
                                            rx.table.column_header_cell("제목"),
                                            rx.table.column_header_cell("점수"),
                                        ),
                                    ),
                                    rx.table.body(
                                        rx.foreach(
                                            State.internal_hits_display,
                                            lambda hit: rx.table.row(
                                                rx.table.cell(hit["category"]),
                                                rx.table.cell(hit["title"]),
                                                rx.table.cell(f"{hit['score']:.4f}"),
                                            ),
                                        )
                                    ),
                                ),
                                rx.text("내부 KB 문서가 없거나 검색 결과가 없습니다.", color="gray"),
                            ),
                        ),
                        rx.box(
                            rx.heading("외부 KB Top-K", size="4", margin_bottom="1rem", margin_top="1rem"),
                            rx.cond(
                                State.external_hits_display.length() > 0,
                                rx.table.root(
                                    rx.table.header(
                                        rx.table.row(
                                            rx.table.column_header_cell("카테고리"),
                                            rx.table.column_header_cell("제목"),
                                            rx.table.column_header_cell("점수"),
                                        ),
                                    ),
                                    rx.table.body(
                                        rx.foreach(
                                            State.external_hits_display,
                                            lambda hit: rx.table.row(
                                                rx.table.cell(hit["category"]),
                                                rx.table.cell(hit["title"]),
                                                rx.table.cell(f"{hit['score']:.4f}"),
                                            ),
                                        )
                                    ),
                                ),
                                rx.text("외부 KB 문서가 없거나 검색 결과가 없습니다.", color="gray"),
                            ),
                        ),
                        spacing="4",
                    ),
                    padding="1.5rem",
                    background="white",
                    border_radius="0.5rem",
                    box_shadow="0 2px 10px rgba(0,0,0,0.1)",
                ),
            ),
            margin_top="2rem",
        ),
    )

def footer() -> rx.Component:
    return rx.box(
        rx.text(
            "데모용: 실제 운영 정책/문구/임계값/카테고리는 쇼핑몰에 맞춰 커스터마이징합니다.",
            size="2",
            color="gray",
            text_align="center",
        ),
        margin_top="3rem",
        padding="1rem",
    )

# -----------------------------
# Main Page
# -----------------------------
def index() -> rx.Component:
    return rx.container(
        header(),
        rx.grid(
            input_section(),
            kb_status(),
            columns="2",
            spacing="4",
            width="100%",
        ),
        result_section(),
        footer(),
        max_width="1400px",
        padding="2rem",
        background="#f5f7fa",
        min_height="100vh",
    )

# -----------------------------
# App
# -----------------------------
app = rx.App(
    theme=rx.theme(
        appearance="light",
        accent_color="violet",
    )
)
app.add_page(index, on_load=State.on_load)

