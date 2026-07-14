import os
import re
import random
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import plotly.express as px
from wordcloud import WordCloud
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# kiwipiepy(한국어 형태소 분석기)는 설치가 안 되어 있어도 앱이 죽지 않도록 예외 처리
try:
    from kiwipiepy import Kiwi
    KIWI_AVAILABLE = True
except Exception:
    KIWI_AVAILABLE = False


# =========================================================
# 기본 설정
# =========================================================
st.set_page_config(
    page_title="유튜브 댓글 분석기",
    page_icon="📺",
    layout="wide",
)

KOREAN_STOPWORDS = set("""
그리고 그래서 그러나 하지만 그런데 그냥 진짜 정말 너무 정도 그냥 이거 저거
그거 이게 저게 그게 이건 저건 그건 이런 저런 그런 이렇게 저렇게 그렇게
합니다 했습니다 입니다 있습니다 없습니다 하는 했던 하고 있고 없고 되는
되고 것 거 좀 다시 더 또 등 등등 때문 위해 통해 대해 처럼 같은 같아요
같습니다 있는 없는 이제 아직 벌써 오늘 내일 어제 우리 저희 당신 여러분
사람 사람들 하나 이번 저번 다음 이전 이후 모든 각각 여기 저기 거기
영상 채널 댓글 구독 좋아요 진짜로 완전 완전히 근데 이제는 그니까
때문에 그러니까 이랬는데 저랬는데 하는데 그런거 이런거 저런거
""".split())

ENGLISH_STOPWORDS = set("""
the and is this that was for with you your are but not have has will
just from they what when where who how all can get got out about into
video channel comment subscribe like really very much more most also
been were had does did doing im its it's dont don't cant can't
""".split())

POSITIVE_WORDS = set("""
좋아요 좋다 좋은 최고 감동 대박 훌륭 훌륭하다 재밌다 재미있다 재밌어요
감사 감사합니다 사랑 사랑해요 웃긴다 웃겨요 멋지다 멋있다 응원 화이팅
잘한다 잘했다 신기하다 유익하다 도움 도움됩니다 짱 최고예요 굿 명작
감탄 놀랍다 대단하다 훈훈하다 힐링 명강의 유용하다
""".split())

NEGATIVE_WORDS = set("""
별로 싫다 싫어요 최악 실망 지루하다 재미없다 아쉽다 안좋다 짜증 화난다
화나요 별로다 유치하다 이상하다 불편하다 실패 후회 아쉬운 별로예요
불만 나쁘다 나쁜 문제 오류 거짓말 사기 광고 낚시 클릭베이트
""".split())


# =========================================================
# 유틸 함수
# =========================================================
def get_api_key():
    """secrets.toml 또는 환경변수에서 YOUTUBE_API_KEY를 가져옵니다."""
    try:
        if "YOUTUBE_API_KEY" in st.secrets:
            return st.secrets["YOUTUBE_API_KEY"]
    except Exception:
        pass
    return os.environ.get("YOUTUBE_API_KEY")


def extract_video_id(url_or_id: str):
    """유튜브 URL(각종 형식) 또는 순수 video ID를 받아 video ID를 반환합니다."""
    text = url_or_id.strip()
    if re.fullmatch(r"[0-9A-Za-z_-]{11}", text):
        return text

    patterns = [
        r"(?:v=|/videos/|embed/|youtu\.be/|/shorts/)([0-9A-Za-z_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            return m.group(1)
    return None


@st.cache_resource(show_spinner=False)
def get_kiwi():
    if KIWI_AVAILABLE:
        return Kiwi()
    return None


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_video_info(api_key: str, video_id: str):
    youtube = build("youtube", "v3", developerKey=api_key)
    resp = youtube.videos().list(part="snippet,statistics", id=video_id).execute()
    if not resp.get("items"):
        return None
    item = resp["items"][0]
    return {
        "title": item["snippet"]["title"],
        "channel": item["snippet"]["channelTitle"],
        "thumbnail": item["snippet"]["thumbnails"]["high"]["url"],
        "view_count": int(item["statistics"].get("viewCount", 0)),
        "like_count": int(item["statistics"].get("likeCount", 0)),
        "comment_count": int(item["statistics"].get("commentCount", 0)),
        "published_at": item["snippet"]["publishedAt"],
    }


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_comments(api_key: str, video_id: str, max_results: int, order: str, include_replies: bool):
    """유튜브 댓글을 수집합니다. commentThreads.list를 페이지네이션하며 호출합니다."""
    youtube = build("youtube", "v3", developerKey=api_key)
    comments = []
    next_page_token = None

    while len(comments) < max_results:
        remain = max_results - len(comments)
        try:
            resp = youtube.commentThreads().list(
                part="snippet,replies",
                videoId=video_id,
                maxResults=min(100, max(1, remain)),
                order=order,
                pageToken=next_page_token,
                textFormat="plainText",
            ).execute()
        except HttpError as e:
            return comments, str(e)

        for item in resp.get("items", []):
            top = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "author": top.get("authorDisplayName", ""),
                "text": top.get("textDisplay", ""),
                "like_count": int(top.get("likeCount", 0)),
                "published_at": top.get("publishedAt", ""),
                "is_reply": False,
            })

            if include_replies and item.get("replies"):
                for r in item["replies"].get("comments", []):
                    rs = r["snippet"]
                    comments.append({
                        "author": rs.get("authorDisplayName", ""),
                        "text": rs.get("textDisplay", ""),
                        "like_count": int(rs.get("likeCount", 0)),
                        "published_at": rs.get("publishedAt", ""),
                        "is_reply": True,
                    })

        next_page_token = resp.get("nextPageToken")
        if not next_page_token:
            break

    return comments[:max_results], None


def clean_text(text: str) -> str:
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@[\w가-힣]+", " ", text)
    # 이모지 및 기호 범위 제거
    text = re.sub(
        r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U00002600-\U000026FF\U0001F1E6-\U0001F1FF]+",
        " ", text
    )
    text = re.sub(r"[^0-9A-Za-z가-힣\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_repeated_char(token: str) -> bool:
    return bool(re.fullmatch(r"(.)\1+", token))


def extract_words(texts, use_kiwi: bool, kiwi_instance):
    """댓글 리스트에서 명사/단어를 추출합니다."""
    all_words = []

    if use_kiwi and kiwi_instance is not None:
        for t in texts:
            cleaned = clean_text(t)
            if not cleaned:
                continue
            try:
                for token in kiwi_instance.tokenize(cleaned):
                    if token.tag.startswith("NN") and len(token.form) >= 2:
                        all_words.append(token.form)
                    elif token.tag == "SL" and len(token.form) >= 2:  # 영어 단어
                        all_words.append(token.form.lower())
            except Exception:
                continue
    else:
        for t in texts:
            cleaned = clean_text(t)
            if not cleaned:
                continue
            for tok in cleaned.split():
                if is_repeated_char(tok):
                    continue
                if re.fullmatch(r"[가-힣]{2,}", tok):
                    all_words.append(tok)
                elif re.fullmatch(r"[A-Za-z]{2,}", tok):
                    all_words.append(tok.lower())

    filtered = [
        w for w in all_words
        if w not in KOREAN_STOPWORDS and w not in ENGLISH_STOPWORDS
    ]
    return filtered


def find_korean_font():
    """시스템에 설치된 한글 폰트를 찾습니다 (packages.txt로 fonts-nanum 설치 필요)."""
    candidates = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
        "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    for f in fm.fontManager.ttflist:
        name = f.name.lower()
        if "nanum" in name or "malgun" in name or "cjk" in name or "gothic" in name:
            return f.fname

    return None


def create_circle_mask(size=900):
    """워드클라우드를 원형으로 렌더링하기 위한 마스크를 생성합니다."""
    x, y = np.ogrid[:size, :size]
    center = size // 2
    radius = size // 2 - 15
    mask = 255 * np.ones((size, size), dtype=np.uint8)
    circle = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2
    mask[circle] = 0
    return mask


def youtube_red_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    """유튜브 브랜드 컬러(레드) 톤의 그라데이션 색상 함수."""
    hue = random.choice([355, 358, 0, 3, 6, 350])
    saturation = random.randint(70, 92)
    lightness = random.randint(35, 58)
    return f"hsl({hue}, {saturation}%, {lightness}%)"


def make_wordcloud(word_freq: dict, font_path: str):
    mask = create_circle_mask(900)
    wc = WordCloud(
        font_path=font_path,
        width=900,
        height=900,
        background_color=None,
        mode="RGBA",
        mask=mask,
        max_words=150,
        relative_scaling=0.45,
        color_func=youtube_red_color_func,
        prefer_horizontal=0.9,
        collocations=False,
        min_font_size=10,
        random_state=42,
    ).generate_from_frequencies(word_freq)
    return wc


# =========================================================
# 사이드바 UI
# =========================================================
st.title("📺 유튜브 댓글 분석기")
st.caption("영상 URL을 입력하면 댓글을 수집해 워드클라우드와 다양한 통계를 보여줍니다.")

api_key = get_api_key()
if not api_key:
    st.error(
        "YOUTUBE_API_KEY를 찾을 수 없습니다.\n\n"
        "Streamlit Cloud의 **Settings → Secrets** 에 아래처럼 등록해주세요:\n\n"
        '```\nYOUTUBE_API_KEY = "여기에_API_키_입력"\n```'
    )
    st.stop()

with st.sidebar:
    st.header("⚙️ 설정")
    video_url = st.text_input("유튜브 영상 URL 또는 ID", placeholder="https://www.youtube.com/watch?v=...")
    max_comments = st.slider("가져올 댓글 수", min_value=50, max_value=2000, value=300, step=50)
    order = st.selectbox("정렬 기준", options=["relevance", "time"], format_func=lambda x: "인기순" if x == "relevance" else "최신순")
    include_replies = st.checkbox("대댓글(답글) 포함", value=False)
    fetch_btn = st.button("🔍 댓글 분석 시작", type="primary", use_container_width=True)

    if KIWI_AVAILABLE:
        st.success("한국어 형태소 분석기(kiwipiepy) 사용 가능 ✅")
    else:
        st.warning("kiwipiepy 미설치 - 간이 단어 추출 방식으로 동작합니다.")


# =========================================================
# 메인 로직
# =========================================================
if "comments_df" not in st.session_state:
    st.session_state.comments_df = None
    st.session_state.video_info = None

if fetch_btn:
    if not video_url:
        st.warning("영상 URL을 입력해주세요.")
        st.stop()

    video_id = extract_video_id(video_url)
    if not video_id:
        st.error("영상 URL에서 video ID를 추출하지 못했습니다. URL을 다시 확인해주세요.")
        st.stop()

    with st.spinner("영상 정보를 불러오는 중..."):
        try:
            info = fetch_video_info(api_key, video_id)
        except HttpError as e:
            st.error(f"영상 정보를 불러오지 못했습니다: {e}")
            st.stop()

    if info is None:
        st.error("영상을 찾을 수 없습니다. video ID를 확인해주세요.")
        st.stop()

    with st.spinner(f"댓글을 최대 {max_comments}개 수집하는 중... (시간이 걸릴 수 있습니다)"):
        comments, err = fetch_comments(api_key, video_id, max_comments, order, include_replies)

    if err:
        st.error(f"댓글 수집 중 오류가 발생했습니다. 댓글이 비활성화된 영상이거나 API 할당량을 초과했을 수 있습니다.\n\n상세: {err}")
        st.stop()

    if not comments:
        st.warning("수집된 댓글이 없습니다.")
        st.stop()

    df = pd.DataFrame(comments)
    df["published_at"] = pd.to_datetime(df["published_at"])
    df["length"] = df["text"].str.len()

    st.session_state.comments_df = df
    st.session_state.video_info = info

# =========================================================
# 결과 표시
# =========================================================
if st.session_state.comments_df is not None:
    df = st.session_state.comments_df
    info = st.session_state.video_info

    # ---- 영상 정보 헤더 ----
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image(info["thumbnail"], use_container_width=True)
    with col2:
        st.subheader(info["title"])
        st.write(f"채널: **{info['channel']}**")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("조회수", f"{info['view_count']:,}")
        m2.metric("좋아요", f"{info['like_count']:,}")
        m3.metric("전체 댓글 수", f"{info['comment_count']:,}")
        m4.metric("수집된 댓글", f"{len(df):,}")

    st.divider()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 개요", "☁️ 워드클라우드", "🏆 인기 댓글", "😊 감정 키워드", "📥 데이터"]
    )

    # ---- 탭1: 개요 ----
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**댓글 좋아요 분포**")
            fig = px.histogram(df, x="like_count", nbins=30, labels={"like_count": "좋아요 수"})
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("**댓글 길이 분포**")
            fig = px.histogram(df, x="length", nbins=30, labels={"length": "댓글 길이(자)"})
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**시간대별 댓글 작성 추이**")
        ts = df.set_index("published_at").resample("D").size().reset_index(name="count")
        fig = px.line(ts, x="published_at", y="count", markers=True,
                       labels={"published_at": "날짜", "count": "댓글 수"})
        st.plotly_chart(fig, use_container_width=True)

    # ---- 탭2: 워드클라우드 ----
    with tab2:
        font_path = find_korean_font()
        if font_path is None:
            st.warning(
                "한글 폰트를 찾을 수 없습니다. 저장소 루트에 `packages.txt` 파일을 추가하고 "
                "`fonts-nanum` 을 한 줄 적어주세요. (Streamlit Cloud 재배포 필요)"
            )
        else:
            with st.spinner("워드클라우드 생성 중..."):
                kiwi = get_kiwi()
                words = extract_words(df["text"].tolist(), KIWI_AVAILABLE, kiwi)

                if not words:
                    st.info("분석할 단어가 충분하지 않습니다.")
                else:
                    freq = pd.Series(words).value_counts()
                    freq_dict = freq.to_dict()

                    wc = make_wordcloud(freq_dict, font_path)

                    fig, ax = plt.subplots(figsize=(9, 9))
                    ax.imshow(wc, interpolation="bilinear")
                    ax.axis("off")
                    fig.patch.set_alpha(0)
                    st.pyplot(fig, use_container_width=True)

                    st.markdown("**상위 20개 단어 빈도**")
                    top20 = freq.head(20).reset_index()
                    top20.columns = ["단어", "빈도"]
                    fig2 = px.bar(top20, x="빈도", y="단어", orientation="h",
                                  color="빈도", color_continuous_scale="Reds")
                    fig2.update_layout(yaxis={"categoryorder": "total ascending"})
                    st.plotly_chart(fig2, use_container_width=True)

    # ---- 탭3: 인기 댓글 ----
    with tab3:
        st.markdown("**좋아요 many 순 상위 댓글**")
        top_liked = df.sort_values("like_count", ascending=False).head(20)
        for _, row in top_liked.iterrows():
            with st.container(border=True):
                st.markdown(f"**{row['author']}** · 👍 {row['like_count']:,} · {row['published_at'].strftime('%Y-%m-%d')}")
                st.write(row["text"])

    # ---- 탭4: 감정 키워드 (간단한 규칙 기반) ----
    with tab4:
        st.caption("※ 머신러닝 기반 감정분석이 아닌, 사전에 정의된 단어 매칭에 의한 간단한 참고용 지표입니다.")

        def count_matches(text, word_set):
            return sum(1 for w in word_set if w in text)

        df["pos_hits"] = df["text"].apply(lambda t: count_matches(t, POSITIVE_WORDS))
        df["neg_hits"] = df["text"].apply(lambda t: count_matches(t, NEGATIVE_WORDS))

        pos_count = (df["pos_hits"] > df["neg_hits"]).sum()
        neg_count = (df["neg_hits"] > df["pos_hits"]).sum()
        neutral_count = len(df) - pos_count - neg_count

        pie_df = pd.DataFrame({
            "구분": ["긍정 추정", "부정 추정", "중립/판단불가"],
            "댓글 수": [pos_count, neg_count, neutral_count],
        })
        c1, c2 = st.columns([1, 1])
        with c1:
            fig = px.pie(pie_df, names="구분", values="댓글 수", hole=0.4,
                         color="구분",
                         color_discrete_map={"긍정 추정": "#2ecc71", "부정 추정": "#e74c3c", "중립/판단불가": "#bdc3c7"})
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.metric("긍정 추정 댓글", f"{pos_count:,}")
            st.metric("부정 추정 댓글", f"{neg_count:,}")
            st.metric("중립/판단불가", f"{neutral_count:,}")

    # ---- 탭5: 데이터 다운로드 ----
    with tab5:
        st.dataframe(df[["author", "text", "like_count", "published_at", "is_reply"]], use_container_width=True)
        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📥 CSV로 다운로드",
            data=csv,
            file_name=f"comments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

else:
    st.info("왼쪽 사이드바에 유튜브 영상 URL을 입력하고 '댓글 분석 시작' 버튼을 눌러주세요.")
