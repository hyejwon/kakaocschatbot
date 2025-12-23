import os
import re
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="쇼핑몰 CS 분류 + RAG 데모", layout="wide")

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
    conf = min(0.55 + 0.1 * (top_score - 1), 0.85)  # 가벼운 신뢰도 추정
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

@st.cache_data(show_spinner=False)
def build_retriever(df: pd.DataFrame):
    # content + title 합쳐서 검색 품질 개선
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
    # LLM 없이도 데모가 그럴듯하게 보이도록 "CS 템플릿" 기반 답변
    base = f"문의 유형: **{category}**\n\n"
    if source_type == "internal":
        base += "내부 정책/FAQ를 기준으로 안내드립니다.\n\n"
    else:
        base += "내부 정책 데이터가 부족하여, 일반적인 공개 지식/가이드 기준으로 안내드립니다.\n\n"

    base += f"**참고 항목:** {source_title}\n\n"
    base += f"- 안내: {source_content}\n\n"
    # 카테고리별 추가 질문(실무 느낌)
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
# UI
# -----------------------------
st.title("🛒 쇼핑몰 CS 자동 분류 챗봇 (내부 없으면 외부 RAG Fallback)")

left, right = st.columns([1.1, 0.9])

with left:
    st.subheader("1) 고객 문의 입력")
    user_text = st.text_area(
        "예) '배송이 아직 안 와요', '환불 언제 돼요?', '쿠폰이 적용이 안돼요' 등",
        height=140
    )

    st.subheader("2) 설정")
    internal_threshold = st.slider("내부 KB 신뢰도 임계값(낮을수록 내부를 더 잘 씀)", 0.05, 0.40, 0.18, 0.01)
    topk = st.slider("검색 Top-K", 1, 5, 3, 1)
    show_debug = st.checkbox("디버그 정보 보기(점수/근거)", value=True)

    run = st.button("🚀 처리하기", type="primary", use_container_width=True)

with right:
    st.subheader("지식베이스 상태")
    internal_df = load_kb("data/internal_kb.csv")
    external_df = load_kb("data/external_kb.csv")

    st.write(f"- 내부 KB 문서 수: **{len(internal_df)}**")
    st.write(f"- 외부 KB 문서 수: **{len(external_df)}**")

    with st.expander("외부 KB 미리보기"):
        st.dataframe(external_df[["id","category","title"]], use_container_width=True, hide_index=True)

# Build retrievers
internal_vec, internal_X, _ = build_retriever(internal_df)
external_vec, external_X, _ = build_retriever(external_df)

if run:
    if not user_text.strip():
        st.warning("문의 내용을 입력해주세요.")
        st.stop()

    st.divider()
    st.subheader("결과")

    # 1) Classification
    pred_cat, conf, reasons = rule_classify(user_text)

    # 2) Retrieve from internal first
    internal_hits, _ = retrieve(internal_df, internal_vec, internal_X, user_text, topk=topk)
    use_fallback = should_fallback_internal(internal_hits, threshold=internal_threshold)

    source_type = "external" if use_fallback else "internal"
    if source_type == "internal":
        best = internal_hits.iloc[0]
    else:
        # external retrieval with category bias: filter same category first, if empty then global
        same_cat = external_df[external_df["category"].astype(str) == pred_cat]
        vec_cat, X_cat, _ = build_retriever(same_cat)
        ext_hits, _ = retrieve(same_cat, vec_cat, X_cat, user_text, topk=topk)
        if ext_hits.empty:
            ext_hits, _ = retrieve(external_df, external_vec, external_X, user_text, topk=topk)
        best = ext_hits.iloc[0] if not ext_hits.empty else pd.Series({"title":"상담 연결 안내", "content":"정확한 확인이 필요합니다.", "score":0.0})

    # 3) Answer
    answer = generate_template_answer(
        category=pred_cat,
        user_text=user_text,
        source_title=str(best.get("title","")),
        source_content=str(best.get("content","")),
        source_type=source_type
    )

    colA, colB = st.columns([1,1])

    with colA:
        st.markdown("### ✅ 분류 결과")
        st.markdown(f"- 예측 카테고리: **{pred_cat}**")
        st.markdown(f"- 추정 신뢰도: **{conf:.2f}**")
        st.markdown(f"- 라우팅: **{('외부 RAG 사용' if source_type=='external' else '내부 KB 사용')}**")

        if show_debug:
            st.markdown("#### 근거(디버그)")
            for r in reasons:
                st.write(f"• {r}")
            if not internal_hits.empty:
                st.write("내부 Top-1 score:", float(internal_hits.iloc[0]["score"]))
            else:
                st.write("내부 Top-1 score: (내부 KB 없음)")

    with colB:
        st.markdown("### 💬 챗봇 응답")
        st.markdown(answer)

    if show_debug:
        st.divider()
        st.subheader("검색 결과(디버그)")

        st.markdown("**내부 KB Top-K**")
        if internal_hits.empty:
            st.info("내부 KB 문서가 없거나 검색 결과가 없습니다.")
        else:
            st.dataframe(internal_hits[["category","title","score"]], use_container_width=True, hide_index=True)

        st.markdown("**외부 KB Top-K**")
        ext_hits_all, _ = retrieve(external_df, external_vec, external_X, user_text, topk=topk)
        st.dataframe(ext_hits_all[["category","title","score"]], use_container_width=True, hide_index=True)

st.caption("데모용: 실제 운영 정책/문구/임계값/카테고리는 쇼핑몰에 맞춰 커스터마이징합니다.")
