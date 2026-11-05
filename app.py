# app.py — 코어+위성 분산투자 대시보드 (최종본)
# 실행: streamlit run app.py
# requirements.txt: streamlit, yfinance>=0.2.60, plotly, pandas, numpy (한 줄에 하나씩)
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="코어+위성 투자 대시보드", page_icon="🧭", layout="wide")

# ============================================================ 종목 목록 (섹터별, 분산 한도 계산용)
UNIVERSE = {
    "반도체": "NVDA AVGO AMD TSM ASML MU AMAT LRCX KLAC QCOM TXN ARM MRVL MPWR",
    "소프트웨어·AI": "MSFT ORCL CRM NOW ADBE PLTR SNOW DDOG CRWD PANW NET INTU WDAY",
    "인터넷·플랫폼": "GOOGL AMZN META NFLX UBER SHOP DASH ABNB BKNG SPOT",
    "하드웨어·인프라": "AAPL DELL SMCI VRT CSCO APH ETN PWR GLW ANET",
    "금융": "JPM V MA BAC GS MS AXP BLK SCHW COF PGR",
    "헬스케어": "LLY UNH JNJ ABBV MRK TMO ISRG VRTX REGN AMGN ABT DHR BSX",
    "소비·유통": "COST WMT PG KO PEP MCD HD LOW TJX NKE SBUX CMG",
    "에너지·전력": "XOM CVX COP SLB CEG VST NEE GEV OXY EOG",
    "산업·방산": "CAT DE GE HON RTX LMT NOC UNP BA URI HWM AXON",
    "소재·부동산·통신": "LIN FCX NEM PLD AMT EQIX T VZ TMUS",
    "성장·크립토": "COIN HOOD MSTR APP RKLB",
}
SECTOR_OF = {}
for _s, _v in UNIVERSE.items():
    for _t in _v.split():
        SECTOR_OF.setdefault(_t, _s)
TICKERS = tuple(SECTOR_OF)

# 성향별 비중: 코어 / 추세 슬리브 / 위성(개별 분산) / 고위험
PROFILES = {
    "안정형": (70, 20, 10, 0),
    "균형형": (60, 20, 15, 5),
    "공격형": (50, 15, 25, 10),
}


# ============================================================ 데이터
@st.cache_data(ttl=1800, show_spinner=False)
def fetch(tickers: tuple, period: str = "2y") -> dict:
    """실패하면 예외를 내서 '빈 결과'가 캐시에 저장되지 않게 한다."""
    out = {}
    for i in range(0, len(tickers), 40):
        chunk = list(tickers[i:i + 40])
        try:
            raw = yf.download(chunk, period=period, auto_adjust=True, group_by="ticker",
                              progress=False, threads=True)
        except Exception:
            continue
        if raw is None or raw.empty:
            continue
        for t in chunk:
            try:
                d = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                d = d[["High", "Low", "Close", "Volume"]].dropna()
                if getattr(d.index, "tz", None) is not None:
                    d.index = d.index.tz_localize(None)
                if len(d) > 260:
                    out[t] = d
            except Exception:
                pass
    if not out:
        raise RuntimeError("시세를 받지 못했습니다. 야후 서버가 일시적으로 막혔을 수 있습니다.")
    return out


def above200(d):
    c = d["Close"]
    ma = c.rolling(200).mean()
    return bool(c.iloc[-1] > ma.iloc[-1]), float((c.iloc[-1] / ma.iloc[-1] - 1) * 100)


def metrics(d):
    c, h, l, v = d["Close"], d["High"], d["Low"], d["Volume"]
    ma20, ma50, ma200 = c.rolling(20).mean(), c.rolling(50).mean(), c.rolling(200).mean()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    px = float(c.iloc[-1])
    rs = 0.4 * (px / c.iloc[-64] - 1) + 0.3 * (px / c.iloc[-127] - 1) + 0.3 * (px / c.iloc[-253] - 1)
    return {"close": px, "ma20": float(ma20.iloc[-1]), "ma50": float(ma50.iloc[-1]),
            "ma200": float(ma200.iloc[-1]), "ma200_up": bool(ma200.iloc[-1] > ma200.iloc[-21]),
            "hi52": float(c.tail(252).max()), "atr": float(tr.rolling(14).mean().iloc[-1]),
            "rs": float(rs), "liq": float((c * v).tail(50).mean())}


def scan(data: dict) -> pd.DataFrame:
    rows = []
    for t, d in data.items():
        if t not in SECTOR_OF:
            continue
        try:
            m = metrics(d)
        except Exception:
            continue
        m["ticker"], m["sector"] = t, SECTOR_OF[t]
        rows.append(m)
    df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty:
        return df
    df["rs_rank"] = df["rs"].rank(pct=True) * 100
    df["c_trend"] = (df["close"] > df["ma50"]) & (df["ma50"] > df["ma200"])
    df["c_up200"] = df["ma200_up"]
    df["c_high"] = df["close"] >= df["hi52"] * 0.85
    df["c_rs"] = df["rs_rank"] >= 70
    df["c_liq"] = df["liq"] >= 2e7
    df["ok"] = df[["c_trend", "c_up200", "c_high", "c_rs", "c_liq"]].all(axis=1)
    df["hot"] = df["close"] / df["ma50"] - 1 > 0.25
    df["pull"] = (df["close"] / df["ma20"] - 1).abs() <= 0.04
    return df


def pick(df, n, cap, skip_hot):
    c = df[df["ok"]].copy()
    if skip_hot:
        c = c[~c["hot"]]
    c = c.sort_values(["hot", "rs_rank"], ascending=[True, False])
    cnt, sel = {}, []
    for _, r in c.iterrows():
        if cnt.get(r["sector"], 0) >= cap:
            continue
        sel.append(r)
        cnt[r["sector"]] = cnt.get(r["sector"], 0) + 1
        if len(sel) >= n:
            break
    return pd.DataFrame(sel)


def money(x):
    return f"${x:,.0f}"


# ============================================================ 사이드바
with st.sidebar:
    st.title("🧭 코어+위성")
    total = st.number_input("총 투자금 (달러)", min_value=1000.0, max_value=1e9, value=10000.0, step=1000.0)
    prof = st.radio("투자 성향", list(PROFILES), index=1)
    n_slots = st.slider("위성 종목 수", 5, 15, 10, help="많을수록 한 종목 사고의 영향이 작아집니다.")
    cap = st.slider("섹터당 최대 종목", 1, 4, 2, help="같은 업종에 몰리지 않게 제한합니다.")
    skip_hot = st.checkbox("과열 종목 제외", True, help="50일선보다 25% 넘게 위에 있는 종목은 뺍니다.")
    if st.button("🔄 데이터 새로고침"):
        st.cache_data.clear()
        st.rerun()
    st.caption("참고용입니다. 매수 권유가 아니며 수익을 보장하지 않습니다.")

# ============================================================ 시장 상태
try:
    with st.spinner("시장 데이터 불러오는 중..."):
        mk = fetch(("SPY", "QQQ"))
    spy_ok, spy_gap = above200(mk["SPY"])
    qqq_ok, qqq_gap = above200(mk["QQQ"])
except Exception as e:
    st.error(f"시장 데이터를 불러오지 못했습니다: {e}")
    st.info("1~2분 뒤 '데이터 새로고침'을 누르세요. 계속 안 되면 Manage app → Reboot app 후 다시 시도하세요.")
    st.stop()

if not spy_ok:
    scale, light = 0.0, "🔴"
    st.error(f"🔴 SPY가 200일선 아래({spy_gap:+.1f}%) → 위성 신규 매수 쉬기, 추세 슬리브는 현금")
elif not qqq_ok:
    scale, light = 0.5, "🟡"
    st.warning(f"🟡 SPY는 200일선 위({spy_gap:+.1f}%)지만 QQQ는 아래({qqq_gap:+.1f}%) → 위성 매수는 절반만")
else:
    scale, light = 1.0, "🟢"
    st.success(f"🟢 SPY {spy_gap:+.1f}% · QQQ {qqq_gap:+.1f}% (200일선 대비) → 정상 운용")

# ============================================================ 종목 스캔
uni, uni_err = None, None
try:
    with st.spinner("종목 분석 중... (첫 실행 1~2분)"):
        uni = scan(fetch(TICKERS))
    if uni.empty:
        uni, uni_err = None, "분석 가능한 종목이 없습니다."
except Exception as e:
    uni_err = str(e)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🧭 오늘의 계획", "🎯 위성 분산 추천", "🔍 종목 조회", "✅ 검증 결과(솔직판)", "📖 용어 설명"])

# ============================================================ 탭1: 오늘의 계획
with tab1:
    core_p, slv_p, sat_p, hi_p = PROFILES[prof]
    st.subheader(f"{prof} 배분 계획  {light}")
    plan = [
        ("① 코어", core_p, "VOO(또는 SPY) 60% + QQQ 40%", "계속 보유. 팔지 않고 꾸준히 적립"),
        ("② 추세 슬리브", slv_p, "QQQ (200일선 아래면 현금·단기채)",
         "QQQ 보유" if qqq_ok else "현금·단기채로 대기 (QQQ가 200일선 아래)"),
        ("③ 위성", sat_p, f"개별 분산 {n_slots}종목",
         {1.0: "추천 종목 매수 가능", 0.5: "절반만 매수", 0.0: "신규 매수 쉬기"}[scale]),
        ("④ 고위험", hi_p, "SPCX, TQQQ 등 (잃어도 되는 돈)", "한도 안에서만. 비중이 커지면 일부를 코어로 이동"),
    ]
    rows = []
    for nm, p, what, act in plan:
        rows.append({"구분": nm, "목표 비중": f"{p}%", "금액": money(total * p / 100), "무엇을": what, "오늘 행동": act})
    wait = total * (slv_p * (0 if qqq_ok else 1) + sat_p * (1 - scale)) / 100
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    if wait > 0:
        st.info(f"현금·단기채로 대기하는 금액: {money(wait)} (시장 신호가 돌아오면 투입)")
    st.markdown(
        "**운용 순서**  \n1. 신호등이 🟢면 계획대로, 🟡면 위성 절반, 🔴면 위성은 쉬고 추세 슬리브는 현금.  \n"
        "2. 코어는 신호와 상관없이 매달 같은 날 적립합니다.  \n"
        "3. 위성 종목은 '위성 분산 추천' 탭의 금액·손절가를 그대로 씁니다.  \n"
        "4. 신호등 확인은 주 1회면 충분합니다. 자주 볼수록 흔들립니다.")
    st.caption("고위험 구간은 전부 잃어도 생활에 지장이 없는 금액이어야 합니다. 환전·수수료·세금은 본인 증권사 기준으로 확인하세요.")

# ============================================================ 탭2: 위성 분산 추천
with tab2:
    st.subheader("위성: 추세 강한 종목을 분산해서")
    if uni is None:
        st.error(f"종목 데이터를 불러오지 못했습니다: {uni_err}")
    else:
        sat_amt = total * PROFILES[prof][2] / 100 * scale
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("분석 종목", len(uni))
        c2.metric("조건 통과", int(uni["ok"].sum()), help="추세·200일선 상승·52주 고점 근접·상대강도 상위·거래대금 5개를 모두 만족")
        c3.metric("위성 투입 가능 금액", money(sat_amt))
        c4.metric("신호등", light)
        if scale == 0:
            st.warning("시장이 약해 신규 매수를 쉬는 구간입니다. 아래 목록은 관심종목(대기용)으로만 보세요.")
        pk = pick(uni, n_slots, cap, skip_hot)
        if pk.empty:
            st.warning("조건을 통과한 종목이 없습니다. 관망도 전략입니다.")
        else:
            slot = (total * PROFILES[prof][2] / 100) / n_slots   # 신호등과 무관한 종목당 기준 금액
            buy = slot * scale
            out = []
            for _, r in pk.iterrows():
                dist = float(np.clip(2.5 * r["atr"] / r["close"], 0.07, 0.15))
                out.append({
                    "종목": r["ticker"], "섹터": r["sector"], "현재가": round(r["close"], 2),
                    "강도순위": int(round(r["rs_rank"])),
                    "상태": ("🟠 과열" if r["hot"] else ("🎯 눌림 근처" if r["pull"] else "🟢 추세")),
                    "매수금액": money(buy), "수량": int(buy // r["close"]),
                    "손절가": round(r["close"] * (1 - dist), 2), "손절폭(%)": round(dist * 100, 1),
                    "손절 시 계좌손실(%)": round(buy / total * dist * 100, 2),
                })
            st.dataframe(pd.DataFrame(out), hide_index=True)
            worst = sum(o["손절 시 계좌손실(%)"] for o in out)
            st.caption(f"전 종목이 동시에 손절되는 최악의 경우 계좌 손실은 약 {worst:.1f}%입니다. "
                       "손절은 종가가 손절가 아래로 마감하면 정리하는 기준입니다. 50일선 종가 이탈도 정리 신호로 참고하세요.")
            st.caption("수량이 0으로 나오면 매수 금액보다 주가가 비싼 경우입니다. 소수점 매수를 쓰거나 위성 종목 수를 줄이거나 투자금을 늘리세요.")
        with st.expander("섹터별 통과 종목 수"):
            st.bar_chart(uni[uni["ok"]].groupby("sector")["ticker"].count())
        st.warning("이 목록의 매수 신호는 과거 검증에서 무작위 진입보다 뚜렷하게 낫지 않았습니다. "
                   "'검증 결과' 탭을 먼저 읽고, 위성은 비중을 작게 유지하세요.")

# ============================================================ 탭3: 종목 조회
with tab3:
    q = st.text_input("티커", "NVDA").strip().upper()
    if q:
        try:
            with st.spinner("불러오는 중..."):
                d = fetch((q,))[q]
            m = metrics(d)
            rank = float((uni["rs"] < m["rs"]).mean() * 100) if uni is not None else None
            dist = float(np.clip(2.5 * m["atr"] / m["close"], 0.07, 0.15))
            stop = m["close"] * (1 - dist)
            a, b, c_ = st.columns(3)
            a.metric("현재가", f"{m['close']:,.2f}")
            b.metric("손절가", f"{stop:,.2f}", f"-{dist * 100:.1f}%")
            c_.metric("강도순위", f"{rank:.0f}" if rank is not None else "-", help="내 종목 목록 안에서의 상대강도 백분위")
            checks = [
                ("종가 > 50일선 > 200일선", m["close"] > m["ma50"] > m["ma200"]),
                ("200일선이 상승 중", m["ma200_up"]),
                ("52주 고점의 85% 이상", m["close"] >= m["hi52"] * 0.85),
                ("상대강도 상위 30%", rank is not None and rank >= 70),
                ("거래대금 충분 (50일 평균 2천만 달러 이상)", m["liq"] >= 2e7),
                ("과열 아님 (50일선 +25% 이내)", m["close"] / m["ma50"] - 1 <= 0.25),
            ]
            for nm, ok in checks:
                st.write(("✅ " if ok else "❌ ") + nm)
            dd = d.tail(250)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dd.index, y=dd["Close"], name="종가", line=dict(color="#222", width=1.6)))
            fig.add_trace(go.Scatter(x=dd.index, y=d["Close"].rolling(50).mean().tail(250), name="50일선", line=dict(color="#fb8c00")))
            fig.add_trace(go.Scatter(x=dd.index, y=d["Close"].rolling(200).mean().tail(250), name="200일선", line=dict(color="#1e88e5")))
            fig.add_hline(y=stop, line_dash="dot", line_color="#e53935", annotation_text="손절가")
            fig.update_layout(title=f"{q} 최근 1년", height=480, margin=dict(l=10, r=10, t=50, b=10),
                              legend=dict(orientation="h", y=1.08))
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"{q} 데이터를 불러오지 못했습니다: {e}")

# ============================================================ 탭4: 검증 결과
with tab4:
    st.subheader("우리가 직접 돌려본 결과 (2020~2026, 비용 편도 0.1% 가정)")
    st.markdown("**① 개별 종목 규칙 vs SPY** — 목록: 미국 대형주 약 120종목, 동시 10종목 보유")
    st.dataframe(pd.DataFrame([
        ["B 돌파·눌림 + 손절", "19.1%", "6.7%", "-24.5%"],
        ["F B + SPY 200일선 위에서만", "19.4%", "7.1%", "-20.7%"],
        ["G 무작위 진입 (대조군)", "11.1%", "3.1%", "-41.1%"],
        ["A 곽경일식 구현 (2% 익절, 20일)", "-0.5%", "5.8%", "-59.5%"],
        ["SPY 그냥 보유", "15.0%", "15.0%", "-33.7%"],
    ], columns=["전략", "연수익 CAGR (전체 종목)", "CAGR (사후 대박 종목 제외)", "최대낙폭(전체)"]), hide_index=True)
    st.markdown("""
- 대박 종목을 빼면 모든 개별 종목 전략이 SPY보다 낮았습니다.
- 신호가 무작위 진입보다 거래당 수익이 높지 않았습니다. 시장 필터와 손절 구조의 효과가 더 컸습니다.
- 목록이 지금 살아남은 종목이라 실제 성과는 더 낮을 가능성이 큽니다.
""")
    st.markdown("**② 지수 200일선 타이밍** — 현금 구간 수익 0% 가정(2%를 넣은 재검증은 별도)")
    st.dataframe(pd.DataFrame([
        ["2004~ SPY", "10.8%", "7.4%", "-55.2%", "-22.4%"],
        ["2004~ QQQ", "14.6%", "9.8%", "-53.4%", "-27.8%"],
        ["2020~ SPY", "15.2%", "11.0%", "-33.7%", "-21.6%"],
        ["2020~ QQQ", "19.6%", "17.5%", "-35.1%", "-22.1%"],
    ], columns=["구간", "보유 CAGR", "타이밍 CAGR", "보유 낙폭", "타이밍 낙폭"]), hide_index=True)
    st.success("결론: 개별 종목으로 지수를 이긴다는 증거는 없습니다. 효과가 확인된 것은 '약세장에 쉬어서 낙폭을 줄이는 것'입니다. "
               "그래서 이 앱은 지수 코어를 중심에 두고, 위성은 작게 분산하는 구조입니다.")
    st.caption("과거 결과는 미래를 보장하지 않습니다. 세금(해외주식 양도세 등)은 반영하지 않았습니다.")

# ============================================================ 탭5: 용어 설명
with tab5:
    GLOSS = {
        "코어": "오래 들고 가는 중심 자산(VOO·QQQ). 전체의 절반 이상.",
        "위성": "코어 주변에서 작게 굴리는 개별 종목. 틀려도 전체가 흔들리지 않게 비중을 제한.",
        "추세 슬리브": "QQQ가 200일선 위일 때만 보유하는 부분. 아래면 현금·단기채로 피신.",
        "200일선": "최근 200거래일 평균가. 종가가 위면 장기 상승 추세로 봅니다.",
        "50일선": "최근 50거래일 평균가. 중기 추세 기준이자 정리 신호로 참고.",
        "신호등": "SPY·QQQ의 200일선 위치로 만든 시장 상태. 🟢정상 🟡위성 절반 🔴위성 쉼.",
        "강도순위(RS)": "최근 3·6·12개월 수익률을 섞어 목록 안에서 매긴 백분위. 높을수록 시장보다 강한 종목.",
        "52주 고점 85%": "1년 최고가의 85% 이상. 위쪽에 팔려는 물량이 적은 종목을 고르는 조건.",
        "과열": "50일선보다 25% 넘게 위. 추격매수 위험이 커 제외합니다.",
        "눌림 근처": "20일선 ±4% 안. 급등 뒤 쉬어가는 자리로 진입 부담이 작습니다.",
        "ATR": "하루 평균 변동폭. 손절폭을 종목 성격에 맞추는 데 씁니다(ATR×2.5, 7~15%로 제한).",
        "손절가": "이 가격 아래로 종가가 마감하면 정리하는 기준. 미리 정해두면 감정이 개입하지 않습니다.",
        "분산": "섹터당 최대 종목 수를 제한하고 같은 금액으로 나눠 한 종목의 사고를 줄이는 방법.",
        "TQQQ": "QQQ의 하루 변동을 3배로 따라가는 ETF. 하락장에서 -80%도 가능하므로 고위험 구간에서만.",
        "SPCX": "스페이스X 본주(2026년 상장). 한 종목에 모든 기대가 걸려 있어 변동성이 큽니다.",
        "CAGR": "연평균 수익률.",
        "최대낙폭": "고점에서 저점까지 가장 크게 빠진 비율. 버틸 수 있는지의 기준.",
        "생존편향": "지금 살아남은 종목만으로 과거를 검증해 결과가 좋게 나오는 왜곡.",
    }
    kw = st.text_input("용어 검색", "").strip()
    for k, v in GLOSS.items():
        if not kw or kw.lower() in (k + v).lower():
            with st.expander(k):
                st.write(v)
