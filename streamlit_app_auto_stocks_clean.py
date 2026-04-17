
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.interpolate import CubicSpline

try:
    import yfinance as yf
except Exception:
    yf = None


st.set_page_config(
    page_title="Quant ATR Lab Auto",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
:root {
    --bg: #07111f;
    --panel: rgba(16, 25, 41, 0.78);
    --panel-2: rgba(12, 20, 34, 0.92);
    --line: rgba(148, 163, 184, 0.14);
    --text: #eef4ff;
    --muted: #94a3b8;
}
html, body, [data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 10% 10%, rgba(86, 204, 242, 0.12), transparent 28%),
        radial-gradient(circle at 90% 5%, rgba(124, 58, 237, 0.10), transparent 25%),
        linear-gradient(180deg, #06101d 0%, #081221 100%);
    color: var(--text);
}
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
#MainMenu, footer {
    visibility: hidden;
    height: 0;
}
[data-testid="stAppViewContainer"] > .main .block-container {
    max-width: 1400px;
    padding-top: 1.4rem;
    padding-bottom: 2rem;
}
.hero-wrap {
    border: 1px solid var(--line);
    background:
        linear-gradient(135deg, rgba(86, 204, 242, 0.14), rgba(124, 58, 237, 0.10)),
        var(--panel);
    backdrop-filter: blur(16px);
    border-radius: 28px;
    padding: 1.2rem 1.35rem 1.25rem 1.35rem;
    box-shadow: 0 20px 60px rgba(0,0,0,0.24);
    margin-bottom: 1rem;
}
.kicker {
    display: inline-block;
    padding: 0.35rem 0.7rem;
    border-radius: 999px;
    background: rgba(86, 204, 242, 0.14);
    border: 1px solid rgba(86, 204, 242, 0.22);
    color: #9ee7ff;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.hero-title {
    margin: 0.8rem 0 0.25rem 0;
    color: var(--text);
    font-size: 2.15rem;
    font-weight: 800;
    line-height: 1.02;
}
.hero-sub {
    margin: 0.25rem 0 0 0;
    color: #b7c5dd;
    font-size: 1rem;
    max-width: 800px;
}
.hero-chip-row {
    display: flex;
    gap: 0.55rem;
    flex-wrap: wrap;
    margin-top: 0.9rem;
}
.hero-chip {
    padding: 0.48rem 0.8rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.05);
    border: 1px solid var(--line);
    color: #d6e5ff;
    font-size: 0.84rem;
}
.control-slab {
    border: 1px solid var(--line);
    background: var(--panel-2);
    border-radius: 24px;
    padding: 0.95rem 1rem 0.4rem 1rem;
    margin-bottom: 1rem;
    box-shadow: 0 18px 44px rgba(0,0,0,0.22);
}
.panel {
    border: 1px solid var(--line);
    background: var(--panel);
    border-radius: 24px;
    padding: 1rem 1rem 0.9rem 1rem;
    box-shadow: 0 18px 42px rgba(0,0,0,0.22);
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
}
.metric-grid-title {
    color: #d8e6ff;
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 0.65rem;
}
.metric-card {
    border: 1px solid var(--line);
    background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.015));
    border-radius: 22px;
    padding: 0.95rem 1rem;
    min-height: 122px;
}
.metric-label {
    color: var(--muted);
    font-size: 0.84rem;
    margin-bottom: 0.35rem;
}
.metric-value {
    color: var(--text);
    font-size: 1.9rem;
    font-weight: 800;
    line-height: 1.05;
}
.metric-foot {
    color: #b9cae5;
    font-size: 0.82rem;
    margin-top: 0.45rem;
}
.soft-divider {
    height: 1px;
    background: var(--line);
    margin: 0.9rem 0 0.8rem 0;
}
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
[data-testid="stNumberInput"] > div > div,
[data-testid="stTextInput"] > div > div {
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--line);
    border-radius: 14px;
}
[data-testid="stFileUploader"] section {
    background: rgba(255,255,255,0.03);
    border: 1px dashed rgba(148,163,184,0.3);
    border-radius: 18px;
}
.stButton > button, .stDownloadButton > button {
    border-radius: 14px;
    border: 1px solid rgba(86, 204, 242, 0.25);
    background: linear-gradient(135deg, rgba(86,204,242,0.18), rgba(124,58,237,0.18));
    color: white;
    font-weight: 700;
    padding: 0.58rem 1rem;
}
.small-muted {
    color: var(--muted);
    font-size: 0.88rem;
}
.info-pill {
    display: inline-block;
    padding: 0.36rem 0.7rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.05);
    color: #d8e6ff;
    border: 1px solid var(--line);
    margin-right: 0.45rem;
    margin-bottom: 0.45rem;
    font-size: 0.82rem;
}
.choice-note {
    color: #9fb2cf;
    font-size: 0.83rem;
    margin-top: -0.2rem;
    margin-bottom: 0.3rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PRESET_STOCKS = {
    "Big Tech": {
        "Apple — AAPL": "AAPL",
        "Microsoft — MSFT": "MSFT",
        "NVIDIA — NVDA": "NVDA",
        "Amazon — AMZN": "AMZN",
        "Alphabet — GOOGL": "GOOGL",
        "Meta — META": "META",
        "Tesla — TSLA": "TSLA",
        "Netflix — NFLX": "NFLX",
    },
    "Finance": {
        "JPMorgan — JPM": "JPM",
        "Bank of America — BAC": "BAC",
        "Goldman Sachs — GS": "GS",
        "Morgan Stanley — MS": "MS",
        "Visa — V": "V",
        "Mastercard — MA": "MA",
    },
    "Industry & Energy": {
        "Exxon Mobil — XOM": "XOM",
        "Chevron — CVX": "CVX",
        "Caterpillar — CAT": "CAT",
        "Boeing — BA": "BA",
        "General Electric — GE": "GE",
    },
    "Consumer & Pharma": {
        "Coca-Cola — KO": "KO",
        "PepsiCo — PEP": "PEP",
        "McDonald's — MCD": "MCD",
        "Johnson & Johnson — JNJ": "JNJ",
        "Pfizer — PFE": "PFE",
        "Procter & Gamble — PG": "PG",
    },
    "ETFs / Index proxies": {
        "SPY — S&P 500 ETF": "SPY",
        "QQQ — Nasdaq 100 ETF": "QQQ",
        "DIA — Dow Jones ETF": "DIA",
        "IWM — Russell 2000 ETF": "IWM",
    },
}


def build_demo_series(seed: int = 28) -> pd.DataFrame:
    np.random.seed(seed)
    n = 261
    dates = pd.bdate_range("2025-09-01", periods=n)
    mu = np.concatenate([
        np.full(60, 0.0009),
        np.full(40, -0.0001),
        np.full(70, 0.0016),
        np.full(40, 0.0002),
        np.full(51, 0.0011),
    ])
    sigma = np.concatenate([
        np.full(60, 0.009),
        np.full(40, 0.017),
        np.full(70, 0.011),
        np.full(40, 0.008),
        np.full(51, 0.010),
    ])
    returns = np.random.normal(mu, sigma)
    close = 180 * np.exp(np.cumsum(returns))
    spread = np.abs(np.random.normal(0.012, 0.005, n))
    high = close * (1 + spread / 2)
    low = close * (1 - spread / 2)
    open_ = close * (1 + np.random.normal(0, 0.002, n))
    return pd.DataFrame({"Date": dates, "Open": open_, "High": high, "Low": low, "Close": close})


def normalize_column_name(col) -> str:
    """Приводит имя столбца к строке, даже если это tuple из MultiIndex."""
    if isinstance(col, tuple):
        parts = [str(x) for x in col if x is not None and str(x) != ""]
        return "_".join(parts).strip().lower()
    return str(col).strip().lower()


def clean_price_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    norm_cols = {normalize_column_name(c): c for c in df.columns}

    def pick_column(*aliases: str):
        for alias in aliases:
            alias = alias.lower()
            if alias in norm_cols:
                return norm_cols[alias]
        for key, original in norm_cols.items():
            for alias in aliases:
                alias = alias.lower()
                if key == alias or key.startswith(alias + "_") or ("_" + alias) in key:
                    return original
        return None

    high_col = pick_column("high")
    low_col = pick_column("low")
    close_col = pick_column("close")
    date_col = pick_column("date", "datetime")

    missing = []
    if high_col is None:
        missing.append("High")
    if low_col is None:
        missing.append("Low")
    if close_col is None:
        missing.append("Close")
    if missing:
        raise ValueError(
            f"В данных не найдены обязательные столбцы: {', '.join(missing)}. "
            f"Проверь CSV или формат ответа Yahoo Finance."
        )

    work = pd.DataFrame()
    if date_col is not None:
        work["Date"] = pd.to_datetime(df[date_col], errors="coerce")
    else:
        work["Date"] = pd.RangeIndex(start=0, stop=len(df), step=1)

    work["High"] = pd.to_numeric(df[high_col], errors="coerce")
    work["Low"] = pd.to_numeric(df[low_col], errors="coerce")
    work["Close"] = pd.to_numeric(df[close_col], errors="coerce")

    work = work.dropna().reset_index(drop=True)
    if len(work) < 40:
        raise ValueError("Слишком мало наблюдений после очистки.")
    return work


@st.cache_data(show_spinner=False)
def load_from_yahoo(ticker: str, period: str) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("Библиотека yfinance недоступна. Установи её: pip install yfinance")
    raw = yf.download(ticker, period=period, auto_adjust=True, progress=False, group_by="column")
    if raw is None or raw.empty:
        raise ValueError(f"Не удалось загрузить данные по тикеру {ticker}.")
    raw = raw.reset_index()

    # На некоторых версиях yfinance колонки могут приходить как MultiIndex.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [
            "_".join([str(x) for x in col if x is not None and str(x) != ""]).strip("_")
            for col in raw.columns.to_flat_index()
        ]

    return clean_price_dataframe(raw)


def compute_atr(df: pd.DataFrame, window: int = 14) -> Tuple[pd.Series, pd.Series]:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = pd.Series(tr, index=df.index).rolling(window=window).mean().dropna()
    aligned_close = df["Close"].iloc[-len(atr):]
    natr = 100.0 * atr / aligned_close
    if len(natr) < 25:
        raise ValueError("После расчёта ATR осталось слишком мало точек.")
    return atr.reset_index(drop=True), natr.reset_index(drop=True)


def make_odd(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) % 2 == 0:
        values = values[:-1]
    return values


def periodize_series(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.arange(len(values), dtype=float)
    line = values[0] + (values[-1] - values[0]) * x / (len(values) - 1)
    return values - line, line


def trig_coefficients(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    n_points = len(values)
    if n_points % 2 == 0:
        raise ValueError("Для классической формулы нужно нечётное число узлов.")
    degree = (n_points - 1) // 2
    nodes = 2 * np.pi * np.arange(n_points) / n_points
    a = np.zeros(degree + 1)
    b = np.zeros(degree + 1)
    a[0] = 2.0 / n_points * np.sum(values)
    for k in range(1, degree + 1):
        a[k] = 2.0 / n_points * np.sum(values * np.cos(k * nodes))
        b[k] = 2.0 / n_points * np.sum(values * np.sin(k * nodes))
    return a, b


def evaluate_trig_poly(a: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    result = np.full_like(x, a[0] / 2.0, dtype=float)
    for k in range(1, len(a)):
        result += a[k] * np.cos(k * x) + b[k] * np.sin(k * x)
    return result


def trig_interpolant(values: np.ndarray, refine: int = 8):
    values = make_odd(values)
    residual, line = periodize_series(values)
    a, b = trig_coefficients(residual)
    n_points = len(values)
    x_nodes = 2 * np.pi * np.arange(n_points) / n_points
    t_dense = np.linspace(0, n_points - 1, n_points * refine)
    x_dense = 2 * np.pi * t_dense / n_points
    line_dense = values[0] + (values[-1] - values[0]) * t_dense / (n_points - 1)
    nodes_interp = evaluate_trig_poly(a, b, x_nodes) + line
    dense_interp = evaluate_trig_poly(a, b, x_dense) + line_dense
    return values, nodes_interp, dense_interp, t_dense, a, b


def spline_dense(values: np.ndarray, refine: int = 8):
    values = make_odd(values)
    x = np.arange(len(values), dtype=float)
    spline = CubicSpline(x, values, bc_type="natural")
    x_dense = np.linspace(0, len(values) - 1, len(values) * refine)
    return values, spline(x), spline(x_dense), x_dense


def find_signals(dense_values: np.ndarray, threshold: float, refine: int, close_prices: np.ndarray, horizon: int = 5):
    idx = []
    dense_idx = []
    for i in range(1, len(dense_values) - 1):
        is_min = dense_values[i] <= dense_values[i - 1] and dense_values[i] < dense_values[i + 1]
        if is_min and dense_values[i] < threshold:
            j = i // refine
            if j < len(close_prices) - horizon and (not idx or j != idx[-1]):
                idx.append(j)
                dense_idx.append(i)
    idx = np.array(idx, dtype=int)
    dense_idx = np.array(dense_idx, dtype=int)
    if len(idx) == 0:
        returns = np.array([])
        success = np.array([], dtype=bool)
    else:
        returns = close_prices[idx + horizon] / close_prices[idx] - 1
        success = returns > 0
    return idx, dense_idx, returns, success


@dataclass
class Stats:
    signals: int
    success_count: int
    hit_rate: float
    mean_return: float
    median_return: float
    std_return: float


def summarize(returns: np.ndarray, success: np.ndarray) -> Stats:
    if len(returns) == 0:
        return Stats(0, 0, 0.0, 0.0, 0.0, 0.0)
    return Stats(
        signals=len(returns),
        success_count=int(success.sum()),
        hit_rate=float(success.mean()),
        mean_return=float(np.mean(returns)),
        median_return=float(np.median(returns)),
        std_return=float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0,
    )


def make_signals_table(dates, close_prices, signal_idx, returns, success, horizon):
    if len(signal_idx) == 0:
        return pd.DataFrame(columns=["Дата сигнала", "Цена в сигнале", f"Доходность через {horizon} дн., %", "Исход"])
    return pd.DataFrame({
        "Дата сигнала": pd.to_datetime(dates.iloc[signal_idx]).dt.strftime("%Y-%m-%d"),
        "Цена в сигнале": np.round(close_prices[signal_idx], 4),
        f"Доходность через {horizon} дн., %": np.round(100 * returns, 4),
        "Исход": np.where(success, "успешный", "неуспешный"),
    })


def kpi_card(label: str, value: str, foot: str = ""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-foot">{foot}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def darkify_axes(ax):
    ax.set_facecolor("#0b1524")
    ax.figure.set_facecolor((0, 0, 0, 0))
    for spine in ax.spines.values():
        spine.set_color("#2d3b54")
    ax.tick_params(colors="#d9e6fb")
    ax.xaxis.label.set_color("#d9e6fb")
    ax.yaxis.label.set_color("#d9e6fb")
    ax.title.set_color("#eef4ff")
    ax.grid(True, color="#22324c", alpha=0.55)


st.markdown(
    """
    <div class="hero-wrap">
        <span class="kicker">Quant ATR Lab</span>
        <div class="hero-title">Автоматический выбор акций и готовые данные<br>для практической части диплома</div>
        <div class="hero-sub">
            Выбирай акцию из готового списка — данные подтягиваются автоматически.
            Интерфейс оставлен только с актуальными сценариями: готовые акции и демо-данные.
        </div>
        <div class="hero-chip-row">
            <span class="hero-chip">Авто-загрузка котировок</span>
            <span class="hero-chip">Готовый список акций</span>
            <span class="hero-chip">ATR / NATR</span>
            <span class="hero-chip">Интерполяционный тригонометрический полином</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="control-slab">', unsafe_allow_html=True)
r1c1, r1c2, r1c3, r1c4 = st.columns([1.0, 1.1, 1.1, 0.9])

with r1c1:
    source = st.selectbox("Источник данных", ["Готовые акции", "Демо-данные"])

with r1c2:
    category = st.selectbox("Категория", list(PRESET_STOCKS.keys()), disabled=(source != "Готовые акции"))

with r1c3:
    preset_name = st.selectbox(
        "Акция",
        list(PRESET_STOCKS[category].keys()),
        disabled=(source != "Готовые акции"),
    )
    st.markdown('<div class="choice-note">При выборе акция загружается автоматически.</div>', unsafe_allow_html=True)

with r1c4:
    period = st.selectbox("Период", ["3mo", "6mo", "1y", "2y", "5y"], index=2)

r2c1, r2c2, r2c3, r2c4 = st.columns([1.0, 1.0, 1.0, 1.0])

with r2c1:
    atr_window = st.slider("Окно ATR", min_value=5, max_value=40, value=14)

with r2c2:
    alpha = st.slider("Порог α", min_value=0.70, max_value=1.10, value=0.96, step=0.01)

with r2c3:
    horizon = st.slider("Горизонт, дней", min_value=1, max_value=20, value=5)

with r2c4:
    refine = st.slider("Плотность сетки", min_value=4, max_value=20, value=8)

run = True

st.markdown('</div>', unsafe_allow_html=True)

view = st.radio("Раздел", ["Командный центр", "Аналитика", "Сигналы", "Методика"], horizontal=True, label_visibility="collapsed")

need_recompute = run or "results_auto" not in st.session_state

if need_recompute:
    try:
        if source == "Готовые акции":
            ticker = PRESET_STOCKS[category][preset_name]
            df = load_from_yahoo(ticker=ticker, period=period)
            source_label = f"{preset_name} / Yahoo Finance"
        else:
            df = clean_price_dataframe(build_demo_series())
            ticker = "DEMO"
            source_label = "Демо-данные"

        atr, natr = compute_atr(df, window=atr_window)
        dates = pd.Series(df["Date"].iloc[-len(natr):].reset_index(drop=True))
        close_prices = df["Close"].iloc[-len(natr):].reset_index(drop=True).to_numpy()

        values, trig_nodes, trig_dense, t_dense, a, b = trig_interpolant(natr.to_numpy(), refine=refine)
        _, spline_nodes, spline_dense_values, spline_dense_x = spline_dense(natr.to_numpy(), refine=refine)

        dates = dates.iloc[:len(values)].reset_index(drop=True)
        close_prices = close_prices[:len(values)]

        threshold = alpha * float(np.mean(values))

        trig_idx, trig_dense_idx, trig_returns, trig_success = find_signals(
            trig_dense, threshold, refine, close_prices, horizon=horizon
        )
        spline_idx, spline_dense_idx, spline_returns, spline_success = find_signals(
            spline_dense_values, threshold, refine, close_prices, horizon=horizon
        )

        st.session_state["results_auto"] = {
            "dates": dates,
            "close_prices": close_prices,
            "values": values,
            "trig_dense": trig_dense,
            "spline_dense_values": spline_dense_values,
            "threshold": threshold,
            "trig_idx": trig_idx,
            "trig_returns": trig_returns,
            "trig_success": trig_success,
            "trig_stats": summarize(trig_returns, trig_success),
            "spline_stats": summarize(spline_returns, spline_success),
            "table": make_signals_table(dates, close_prices, trig_idx, trig_returns, trig_success, horizon),
            "ticker": ticker,
            "source": source_label,
            "period": period,
            "atr_window": atr_window,
            "alpha": alpha,
            "horizon": horizon,
            "refine": refine,
            "coef_count": len(a) - 1,
        }
        st.session_state.pop("error_auto", None)
    except Exception as e:
        st.session_state["error_auto"] = str(e)

if "error_auto" in st.session_state and "results_auto" not in st.session_state:
    st.error(st.session_state["error_auto"])
    st.stop()

if "results_auto" not in st.session_state:
    st.info("Выбери параметры сверху.")
    st.stop()

r = st.session_state["results_auto"]

if view == "Командный центр":
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="metric-grid-title">Ключевые показатели эксперимента</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Актив", r["ticker"], r["source"])
    with c2:
        kpi_card("Сигналов", f"{r['trig_stats'].signals}", "Тригонометрический полином")
    with c3:
        kpi_card("Доля успеха", f"{100*r['trig_stats'].hit_rate:.2f}%", f"Горизонт {r['horizon']} дн.")
    with c4:
        kpi_card("Средняя доходность", f"{100*r['trig_stats'].mean_return:.2f}%", f"Период {r['period']}")
    st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)
    info_l, info_r = st.columns([1.2, 1.0])
    with info_l:
        st.markdown(
            f"""
            <div class="panel" style="margin-bottom:0;">
                <div class="metric-grid-title">Паспорт расчёта</div>
                <span class="info-pill">ATR window: {r['atr_window']}</span>
                <span class="info-pill">α: {r['alpha']:.2f}</span>
                <span class="info-pill">Horizon: {r['horizon']}</span>
                <span class="info-pill">Refine: {r['refine']}</span>
                <span class="info-pill">Степень полинома: {r['coef_count']}</span>
                <p class="small-muted" style="margin-top:0.8rem;">
                    Данные загружаются автоматически при выборе акции из списка.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with info_r:
        st.markdown(
            f"""
            <div class="panel" style="margin-bottom:0;">
                <div class="metric-grid-title">Сравнение методов</div>
                <p style="margin:0.25rem 0;">Тригонометрический полином: <b>{r['trig_stats'].signals}</b> сигналов, <b>{100*r['trig_stats'].hit_rate:.2f}%</b> успеха.</p>
                <p style="margin:0.25rem 0;">Кубический сплайн: <b>{r['spline_stats'].signals}</b> сигналов, <b>{100*r['spline_stats'].hit_rate:.2f}%</b> успеха.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(12, 4.6))
    darkify_axes(ax)
    ax.plot(r["dates"], r["close_prices"], label="Цена закрытия", linewidth=2.2, color="#9ad8ff")
    if len(r["trig_idx"]) > 0:
        good = r["trig_idx"][r["trig_success"]]
        bad = r["trig_idx"][~r["trig_success"]]
        if len(good) > 0:
            ax.scatter(r["dates"].iloc[good], r["close_prices"][good], s=38, label="Успешные", marker="o", color="#34d399")
        if len(bad) > 0:
            ax.scatter(r["dates"].iloc[bad], r["close_prices"][bad], s=42, label="Неуспешные", marker="x", color="#fb7185")
    ax.set_title(f"Сигналы по тригонометрическому интерполянту — {r['ticker']}")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Цена")
    ax.legend(facecolor="#0b1524", edgecolor="#2d3b54", labelcolor="#eef4ff")
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

elif view == "Аналитика":
    left, right = st.columns([1.25, 0.95])
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(12, 4.8))
        darkify_axes(ax)
        ax.plot(r["dates"], r["values"], label="NATR", linewidth=1.5, color="#dce7ff")
        dense_dates = pd.date_range(r["dates"].iloc[0], r["dates"].iloc[len(r["values"]) - 1], periods=len(r["trig_dense"]))
        spline_dates = pd.date_range(r["dates"].iloc[0], r["dates"].iloc[len(r["values"]) - 1], periods=len(r["spline_dense_values"]))
        ax.plot(dense_dates, r["trig_dense"], linestyle="--", linewidth=2.1, color="#56ccf2", label="Тригонометрический полином")
        ax.plot(spline_dates, r["spline_dense_values"], linewidth=1.9, color="#a78bfa", label="Кубический сплайн")
        ax.axhline(r["threshold"], linestyle=":", linewidth=1.4, color="#fbbf24", label="Порог сигнала")
        ax.set_title(f"Непрерывные модели ряда NATR — {r['ticker']}")
        ax.set_xlabel("Дата")
        ax.set_ylabel("NATR, %")
        ax.legend(facecolor="#0b1524", edgecolor="#2d3b54", labelcolor="#eef4ff")
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        if len(r["trig_returns"]) > 0:
            fig2, ax2 = plt.subplots(figsize=(8, 4.4))
            darkify_axes(ax2)
            ax2.hist(100 * r["trig_returns"], bins=12, color="#56ccf2", edgecolor="#081221")
            ax2.set_title("Распределение доходности после сигнала")
            ax2.set_xlabel("Доходность, %")
            ax2.set_ylabel("Частота")
            st.pyplot(fig2)
        else:
            st.warning("Сигналы не найдены — распределение построить нельзя.")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="metric-grid-title">Статистика метода</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <p><b>Медианная доходность:</b> {100*r['trig_stats'].median_return:.2f}%</p>
            <p><b>Стандартное отклонение:</b> {100*r['trig_stats'].std_return:.2f}%</p>
            <p><b>Порог:</b> {r['threshold']:.4f}</p>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

elif view == "Сигналы":
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="metric-grid-title">Таблица найденных сигналов</div>', unsafe_allow_html=True)
    st.dataframe(r["table"], use_container_width=True, height=460)
    csv_bytes = r["table"].to_csv(index=False).encode("utf-8-sig")
    st.download_button("Скачать CSV", data=csv_bytes, file_name=f"signals_{r['ticker']}.csv", mime="text/csv")
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="metric-grid-title">Как это работает</div>', unsafe_allow_html=True)
    st.latex(r"TR_t = \max\{H_t-L_t,\ |H_t-C_{t-1}|,\ |L_t-C_{t-1}|\}")
    st.latex(r"ATR_t = \frac{1}{m}\sum_{j=0}^{m-1} TR_{t-j}")
    st.latex(r"NATR_t = 100\frac{ATR_t}{C_t}")
    st.latex(r"T_n(x)=\frac{a_0}{2}+\sum_{k=1}^{n}(a_k\cos kx+b_k\sin kx)")
    st.markdown(
        """
        Главный сценарий использования:
        выбери акцию из готового списка, дождись автоматической загрузки данных и сразу анализируй результаты.
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)
