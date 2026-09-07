import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import re
import time
from io import StringIO
from datetime import datetime

st.set_page_config(page_title="Stock-Screener", page_icon="🔎", layout="wide")

st.title("🔎 Stock-Screener")
st.caption("V3 – fundamenty → charakter firmy → investiční příběh")

NASDAQ_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
NYSE_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
XETRA_URL = "https://www.cashmarket.deutsche-boerse.com/resource/blob/1528/8e34798266f78fe8811bd24387445b2b/data/t7-xetr-allTradableInstruments.csv"

PARAMS = [
    "Market Cap", "P/E", "Forward P/E", "P/S", "ROE",
    "Revenue Growth", "Earnings Growth", "Free Cash Flow", "Debt/Equity"
]

def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

def safe_float(x):
    try:
        if x is None:
            return np.nan
        if isinstance(x, (list, tuple, dict)):
            return np.nan
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan

def first_valid(*values):
    for v in values:
        n = safe_float(v)
        if not pd.isna(n):
            return n
    return np.nan

def normalize_us_ticker(ticker):
    t = clean_text(ticker).upper()
    # Yahoo Finance uses '-' for share classes such as BRK-B.
    if "." in t:
        t = t.replace(".", "-")
    return t

def yahoo_xetra_ticker(mnemonic):
    m = clean_text(mnemonic).upper()
    return f"{m}.DE" if m else ""

def looks_like_us_equity(name):
    s = clean_text(name).lower()

    # Explicitly reject instrument types we do not want in a stock universe.
    negative = [
        r"\bpreferred\b", r"\bwarrant\b", r"\brights?\b", r"\bunit\b",
        r"\bnotes?\b", r"\bdebenture\b", r"\bbond\b", r"\bsenior notes?\b",
        r"\bsubordinated notes?\b", r"\btrust\b", r"\bfund\b", r"\betf\b",
        r"\bspac\b", r"\bacquisition\b", r"\bsubscription\b",
        r"\bdepositary (?:shares?|receipts?)\b.*\bpreferred\b"
    ]
    if any(re.search(p, s) for p in negative):
        return False

    positive = [
        r"\bcommon stock\b", r"\bcommon shares?\b", r"\bordinary shares?\b",
        r"\bcapital stock\b", r"\bregistered shares?\b",
        r"\bamerican depositary shares?\b", r"\bamerican depositary receipts?\b",
        r"\bads\b", r"\badr\b", r"\bclass [a-z0-9]+ common\b",
        r"\bclass [a-z0-9]+ ordinary\b", r"\bvoting shares?\b",
        r"\bsubordinate voting shares?\b", r"\bnew common stock\b"
    ]
    return any(re.search(p, s) for p in positive)

def find_col(df, candidates):
    lower = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    for c in df.columns:
        lc = str(c).strip().lower()
        for wanted in candidates:
            if wanted.lower() in lc:
                return c
    return None

@st.cache_data(ttl=3600, show_spinner=False)
def load_nasdaq():
    df = pd.read_csv(NASDAQ_URL, sep="|", dtype=str, skipfooter=1, engine="python")
    df.columns = [clean_text(c) for c in df.columns]
    sym = find_col(df, ["Symbol"])
    name = find_col(df, ["Security Name"])
    if sym is None or name is None:
        raise ValueError("NASDAQ: neočekávaná struktura souboru.")
    if "Test Issue" in df.columns:
        df = df[df["Test Issue"].fillna("N") != "Y"]
    if "ETF" in df.columns:
        df = df[df["ETF"].fillna("N") != "Y"]
    if "NextShares" in df.columns:
        df = df[df["NextShares"].fillna("N") != "Y"]
    out = pd.DataFrame({
        "Ticker": df[sym].map(normalize_us_ticker),
        "Name": df[name].map(clean_text),
        "Exchange": "NASDAQ",
        "Source": "Nasdaq Trader"
    })
    out = out[out["Name"].map(looks_like_us_equity)].copy()
    return out.drop_duplicates("Ticker")

@st.cache_data(ttl=3600, show_spinner=False)
def load_nyse():
    df = pd.read_csv(NYSE_URL, sep="|", dtype=str, skipfooter=1, engine="python")
    df.columns = [clean_text(c) for c in df.columns]
    sym = find_col(df, ["ACT Symbol", "Symbol"])
    name = find_col(df, ["Security Name"])
    exch = find_col(df, ["Exchange"])
    if sym is None or name is None or exch is None:
        raise ValueError("NYSE: neočekávaná struktura souboru.")
    if "Test Issue" in df.columns:
        df = df[df["Test Issue"].fillna("N") != "Y"]
    if "ETF" in df.columns:
        df = df[df["ETF"].fillna("N") != "Y"]
    if "NextShares" in df.columns:
        df = df[df["NextShares"].fillna("N") != "Y"]
    df = df[df[exch].fillna("") == "N"]
    out = pd.DataFrame({
        "Ticker": df[sym].map(normalize_us_ticker),
        "Name": df[name].map(clean_text),
        "Exchange": "NYSE",
        "Source": "Nasdaq Trader / NYSE"
    })
    out = out[out["Name"].map(looks_like_us_equity)].copy()
    return out.drop_duplicates("Ticker")

@st.cache_data(ttl=3600, show_spinner=False)
def load_xetra():
    raw = requests.get(
        XETRA_URL,
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    raw.raise_for_status()
    text = raw.content.decode("utf-8-sig", errors="replace")
    lines = text.splitlines()

    header_idx = None
    for i, line in enumerate(lines[:20]):
        if "Instrument Type" in line and ("Mnemonic" in line or "ISIN" in line):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("XETRA: hlavička CSV nebyla nalezena.")

    df = pd.read_csv(StringIO("\n".join(lines[header_idx:])), sep=";", dtype=str)
    df.columns = [clean_text(c) for c in df.columns]

    typ = find_col(df, ["Instrument Type"])
    mnemonic = find_col(df, ["Mnemonic"])
    isin = find_col(df, ["ISIN"])
    instrument = find_col(df, ["Instrument"])
    status = find_col(df, ["Instrument Status"])
    market_status = find_col(df, ["Market Segment Status"])

    if typ is None or mnemonic is None:
        raise ValueError("XETRA: chybí Instrument Type nebo Mnemonic.")

    # Deutsche Börse defines CS as Common Stock / Equity.
    df = df[df[typ].fillna("").str.upper().eq("CS")].copy()

    if status is not None:
        active = df[status].fillna("").str.lower()
        active_mask = active.eq("") | active.str.contains("active")
        df = df[active_mask]
    if market_status is not None:
        ms = df[market_status].fillna("").str.lower()
        active_mask = ms.eq("") | ms.str.contains("active")
        df = df[active_mask]

    out = pd.DataFrame({
        "Ticker": df[mnemonic].map(yahoo_xetra_ticker),
        "Name": df[instrument].map(clean_text) if instrument else "",
        "Exchange": "XETRA",
        "ISIN": df[isin].map(clean_text) if isin else "",
        "Source": "Deutsche Börse Xetra"
    })
    out = out[out["Ticker"].str.len() > 3].copy()
    out = out.drop_duplicates(["Ticker", "ISIN"])
    return out

@st.cache_data(ttl=3600, show_spinner=False)
def load_universe(exchanges):
    parts = []
    if "NASDAQ" in exchanges:
        parts.append(load_nasdaq())
    if "NYSE" in exchanges:
        parts.append(load_nyse())
    if "XETRA" in exchanges:
        parts.append(load_xetra())
    if not parts:
        return pd.DataFrame(columns=["Ticker","Name","Exchange","ISIN","Source"])
    df = pd.concat(parts, ignore_index=True, sort=False)
    for c in ["ISIN"]:
        if c not in df.columns:
            df[c] = ""
    df["Ticker"] = df["Ticker"].map(clean_text)
    return df.drop_duplicates(["Exchange", "Ticker"]).reset_index(drop=True)

@st.cache_data(ttl=86400, show_spinner=False)
def resolve_xetra_symbol(ticker, name, isin):
    """
    First try mnemonic.DE. If Yahoo returns no usable equity data,
    search Yahoo by ISIN/name and select a .DE equity result.
    """
    candidates = [ticker]
    base = ticker[:-3] if ticker.endswith(".DE") else ticker
    candidates += [f"{base}.DE"]

    for c in candidates:
        try:
            t = yf.Ticker(c)
            fi = getattr(t, "fast_info", {}) or {}
            price = safe_float(fi.get("last_price")) if hasattr(fi, "get") else np.nan
            if not pd.isna(price) and price > 0:
                return c, "mnemonic.DE"
            inf = t.info or {}
            if inf.get("symbol") and inf.get("quoteType", "").upper() == "EQUITY":
                return c, "mnemonic.DE"
        except Exception:
            pass

    queries = [q for q in [isin, name] if clean_text(q)]
    for q in queries:
        try:
            url = "https://query1.finance.yahoo.com/v1/finance/search"
            r = requests.get(
                url,
                params={"q": q, "quotesCount": 10, "newsCount": 0},
                timeout=10,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            r.raise_for_status()
            data = r.json()
            for item in data.get("quotes", []):
                sym = clean_text(item.get("symbol"))
                qt = clean_text(item.get("quoteType")).upper()
                if sym.endswith(".DE") and qt in ("EQUITY", "STOCK"):
                    return sym, f"Yahoo Search ({'ISIN' if q == isin and isin else 'name'})"
        except Exception:
            continue

    return ticker, "unresolved"

@st.cache_data(ttl=1800, show_spinner=False)
def yahoo_annual_growth_data(yahoo_ticker):
    """Fallback for annual revenue/net income when yfinance statement rows are incomplete."""
    empty = (np.nan, np.nan, np.nan, np.nan)
    try:
        end = int(time.time())
        start = end - 5 * 365 * 24 * 3600
        url = f"https://query1.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/timeseries/{yahoo_ticker}"
        params = {
            "symbol": yahoo_ticker,
            "type": "annualTotalRevenue,annualNetIncome",
            "period1": start,
            "period2": end,
        }
        r = requests.get(url, params=params, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        result = r.json().get("timeseries", {}).get("result", [])
        revenues, incomes = [], []
        for item in result:
            for key, target in (("annualTotalRevenue", revenues), ("annualNetIncome", incomes)):
                for v in item.get(key, []) or []:
                    raw = v.get("reportedValue", {}) if isinstance(v, dict) else {}
                    val = raw.get("raw") if isinstance(raw, dict) else None
                    date = v.get("asOfDate", "") if isinstance(v, dict) else ""
                    if date and val is not None:
                        target.append((date, safe_float(val)))
        def latest_two(values):
            clean = [(d, v) for d, v in values if d and not pd.isna(v)]
            clean.sort(key=lambda x: x[0], reverse=True)
            out, seen = [], set()
            for d, v in clean:
                if d in seen:
                    continue
                seen.add(d); out.append(v)
                if len(out) == 2:
                    break
            return (out[0], out[1]) if len(out) == 2 else (np.nan, np.nan)
        rc, rp = latest_two(revenues)
        ic, ip = latest_two(incomes)
        return rc, rp, ic, ip
    except Exception:
        return empty

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_fundamentals(ticker, exchange, name="", isin=""):
    time.sleep(0.20)
    requested_ticker = ticker
    yahoo_ticker = ticker
    resolution = "direct"

    try:
        if exchange in ("NASDAQ", "NYSE"):
            yahoo_ticker = normalize_us_ticker(ticker)
        elif exchange == "XETRA":
            yahoo_ticker, resolution = resolve_xetra_symbol(ticker, name, isin)

        t = yf.Ticker(yahoo_ticker)
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}

        fast = {}
        try:
            fast = dict(t.fast_info)
        except Exception:
            fast = {}

        income = pd.DataFrame()
        ttm_income = pd.DataFrame()
        balance = pd.DataFrame()
        cashflow = pd.DataFrame()

        try:
            candidate = t.income_stmt
            if isinstance(candidate, pd.DataFrame) and not candidate.empty:
                income = candidate
        except Exception:
            pass

        try:
            candidate = t.ttm_income_stmt
            if isinstance(candidate, pd.DataFrame) and not candidate.empty:
                ttm_income = candidate
        except Exception:
            pass

        try:
            balance = t.balance_sheet
        except Exception:
            balance = pd.DataFrame()

        try:
            cashflow = t.cashflow
        except Exception:
            cashflow = pd.DataFrame()

        def row_series(df, labels):
            if df is None or df.empty:
                return pd.Series(dtype=float)
            idx = {str(x).strip().lower(): x for x in df.index}
            for label in labels:
                key = str(label).strip().lower()
                if key in idx:
                    s = pd.to_numeric(df.loc[idx[key]], errors="coerce").dropna()
                    if not s.empty:
                        return s
            # Accept yfinance's compact row names as well as spaced variants.
            compact = {re.sub(r"[^a-z0-9]", "", str(x).lower()): x for x in df.index}
            for label in labels:
                key = re.sub(r"[^a-z0-9]", "", str(label).lower())
                if key in compact:
                    s = pd.to_numeric(df.loc[compact[key]], errors="coerce").dropna()
                    if not s.empty:
                        return s
            return pd.Series(dtype=float)

        revenue_s = row_series(income, ["Total Revenue", "TotalRevenue", "Operating Revenue", "OperatingRevenue"])
        net_income_s = row_series(income, ["Net Income", "NetIncome", "Net Income Common Stockholders", "NetIncomeCommonStockholders"])
        equity_s = row_series(balance, ["Stockholders Equity", "StockholdersEquity", "Common Stock Equity", "CommonStockEquity", "Total Equity Gross Minority Interest"])
        debt_s = row_series(balance, ["Total Debt", "TotalDebt"])
        ocf_s = row_series(cashflow, ["Operating Cash Flow", "OperatingCashFlow", "Total Cash From Operating Activities"])
        capex_s = row_series(cashflow, ["Capital Expenditure", "CapitalExpenditure", "Capital Expenditure Reported"])

        price = first_valid(info.get("currentPrice"), fast.get("last_price"), info.get("regularMarketPrice"))
        shares = first_valid(info.get("sharesOutstanding"), info.get("impliedSharesOutstanding"))

        market_cap = first_valid(
            info.get("marketCap"),
            fast.get("market_cap"),
            price * shares if not pd.isna(price) and not pd.isna(shares) else np.nan
        )

        # Use Yahoo's explicit valuation ratios when available.
        pe = first_valid(info.get("trailingPE"))
        fpe = first_valid(info.get("forwardPE"))
        ps = first_valid(info.get("priceToSalesTrailing12Months"))

        # Robust derivations from the financial statements.
        revenue = safe_float(revenue_s.iloc[0]) if not revenue_s.empty else np.nan
        previous_revenue = safe_float(revenue_s.iloc[1]) if len(revenue_s) > 1 else np.nan
        net_income = safe_float(net_income_s.iloc[0]) if not net_income_s.empty else np.nan
        previous_net_income = safe_float(net_income_s.iloc[1]) if len(net_income_s) > 1 else np.nan

        api_rev_cur, api_rev_prev, api_ni_cur, api_ni_prev = yahoo_annual_growth_data(yahoo_ticker)
        if not (revenue > 0 and previous_revenue > 0):
            revenue = api_rev_cur if not pd.isna(api_rev_cur) else revenue
            previous_revenue = api_rev_prev if not pd.isna(api_rev_prev) else previous_revenue
        if not (net_income > 0 and previous_net_income > 0):
            net_income = api_ni_cur if not pd.isna(api_ni_cur) else net_income
            previous_net_income = api_ni_prev if not pd.isna(api_ni_prev) else previous_net_income
        equity = safe_float(equity_s.iloc[0]) if not equity_s.empty else np.nan
        debt = safe_float(debt_s.iloc[0]) if not debt_s.empty else np.nan
        ocf = safe_float(ocf_s.iloc[0]) if not ocf_s.empty else np.nan
        capex = safe_float(capex_s.iloc[0]) if not capex_s.empty else np.nan

        if pd.isna(ps) and not pd.isna(market_cap) and revenue > 0:
            ps = market_cap / revenue

        if pd.isna(pe) and not pd.isna(market_cap) and net_income > 0:
            pe = market_cap / net_income

        forward_eps = first_valid(info.get("forwardEps"))
        if pd.isna(fpe) and not pd.isna(price) and price > 0 and not pd.isna(forward_eps) and forward_eps > 0:
            fpe = price / forward_eps

        roe = first_valid(info.get("returnOnEquity"))
        if not pd.isna(roe):
            roe = roe * 100 if abs(roe) <= 3 else roe
        if pd.isna(roe) and net_income > 0 and equity > 0:
            roe = net_income / equity * 100

        # Revenue growth is meaningful when both revenue observations are positive.
        revenue_growth = np.nan
        if revenue > 0 and previous_revenue > 0:
            revenue_growth = (revenue / previous_revenue - 1) * 100

        # Earnings growth: deliberately NOT using Yahoo's potentially misleading
        # percentage when the base earnings are zero/negative.
        earnings_growth = np.nan
        if net_income > 0 and previous_net_income > 0:
            earnings_growth = (net_income / previous_net_income - 1) * 100

        fcf = first_valid(info.get("freeCashflow"))
        if pd.isna(fcf) and not pd.isna(ocf) and not pd.isna(capex):
            # Yahoo cashflow CapEx is normally negative.
            fcf = ocf + capex if capex < 0 else ocf - capex

        de = first_valid(info.get("debtToEquity"))
        if not pd.isna(de) and equity <= 0:
            de = np.nan
        if pd.isna(de) and debt >= 0 and equity > 0:
            de = debt / equity * 100

        values = {
            "Market Cap": market_cap,
            "P/E": pe,
            "Forward P/E": fpe,
            "P/S": ps,
            "ROE": roe,
            "Revenue Growth": revenue_growth,
            "Earnings Growth": earnings_growth,
            "Free Cash Flow": fcf,
            "Debt/Equity": de,
        }

        available = sum(not pd.isna(v) for v in values.values())
        if available == len(PARAMS):
            status = "OK"
        elif available > 0:
            status = "PARTIAL"
        else:
            status = "NO DATA"

        source_parts = ["Yahoo info"]
        if not income.empty:
            source_parts.append("annual income")
        if not ttm_income.empty:
            source_parts.append("TTM income")
        if not balance.empty:
            source_parts.append("balance")
        if not cashflow.empty:
            source_parts.append("cashflow")
        if resolution != "direct":
            source_parts.append(f"mapping:{resolution}")

        return {
            "Ticker": requested_ticker,
            "Yahoo Ticker": yahoo_ticker,
            "Name": name,
            "Exchange": exchange,
            **values,
            "Status": status,
            "Data Source": " + ".join(source_parts),
            "Mapping": resolution,
            "Error": "",
        }

    except Exception as e:
        return {
            "Ticker": requested_ticker,
            "Yahoo Ticker": yahoo_ticker,
            "Name": name,
            "Exchange": exchange,
            **{p: np.nan for p in PARAMS},
            "Status": "ERROR",
            "Data Source": "Yahoo",
            "Mapping": resolution,
            "Error": str(e)[:300],
        }

def build_candidate_sample(universe, max_candidates):
    if universe.empty:
        return universe
    groups = []
    exchanges = list(universe["Exchange"].dropna().unique())
    n = len(exchanges)
    base = max_candidates // n
    remainder = max_candidates % n

    for i, ex in enumerate(exchanges):
        part = universe[universe["Exchange"] == ex].copy()
        quota = base + (1 if i < remainder else 0)
        quota = min(quota, len(part))
        if quota:
            groups.append(part.sample(n=quota, random_state=42))
    result = pd.concat(groups, ignore_index=True) if groups else universe.head(0)
    return result.sample(frac=1, random_state=42).reset_index(drop=True)

# -----------------------------------------------------------------------------
# V3 – příběhy
# -----------------------------------------------------------------------------

def band(score):
    if pd.isna(score):
        return "⚪ N/A"
    if score >= 70:
        return "🟢 Silné"
    if score >= 50:
        return "🟡 Střední"
    return "🔴 Slabé"

def tier_score(v, cuts, reverse=False):
    """Map a metric to 0–100 using transparent threshold bands."""
    if pd.isna(v):
        return np.nan
    for threshold, score in cuts:
        if (v <= threshold) if reverse else (v >= threshold):
            return float(score)
    return 0.0

def calc_scores(r):
    # Value: lower valuation is better.
    value_parts = [
        tier_score(r["P/E"], [(10,100),(15,85),(20,70),(30,50),(45,25)], reverse=True),
        tier_score(r["Forward P/E"], [(10,100),(15,85),(20,70),(30,50),(45,25)], reverse=True),
        tier_score(r["P/S"], [(1,100),(2,85),(3,70),(5,50),(8,25)], reverse=True),
    ]
    # Quality: profitability, positive cash generation, and reasonable leverage.
    quality_parts = [
        tier_score(r["ROE"], [(25,100),(20,90),(15,75),(10,55),(5,30)], reverse=False),
        100.0 if (not pd.isna(r["Free Cash Flow"]) and r["Free Cash Flow"] > 0) else (0.0 if not pd.isna(r["Free Cash Flow"]) else np.nan),
        tier_score(r["Debt/Equity"], [(25,100),(50,85),(100,65),(150,45),(250,20)], reverse=True),
    ]
    # Growth: positive revenue and earnings growth are the core signals.
    growth_parts = [
        tier_score(r["Revenue Growth"], [(20,100),(10,85),(5,70),(0,50),(-5,25)], reverse=False),
        tier_score(r["Earnings Growth"], [(25,100),(15,85),(10,75),(5,60),(0,45)], reverse=False),
    ]
    def avg_available(parts):
        vals = [x for x in parts if not pd.isna(x)]
        return round(sum(vals)/len(vals), 1) if vals else np.nan
    return avg_available(value_parts), avg_available(quality_parts), avg_available(growth_parts)

def classify_story(r):
    v, q, g = r["Value Score"], r["Quality Score"], r["Growth Score"]
    # These are deliberately simple first-generation rules. We will refine them later.
    if pd.isna(v) or pd.isna(q) or pd.isna(g):
        return "⚪ Nedostatek dat"
    if v >= 70 and q < 50 and g < 50:
        return "🪤 Value Trap – varování"
    if q >= 70 and g >= 65 and v >= 45:
        return "🏆 Quality Compounder"
    if q >= 70 and v >= 60 and g >= 45:
        return "💎 Kvalita za rozumnou cenu"
    if g >= 70 and v >= 50 and q >= 50:
        return "🚀 Růst za rozumnou cenu"
    if v >= 65 and q >= 50 and g < 50:
        return "💰 Value / levná firma"
    # Turnaround signal: revenue can still be weak, but earnings are improving strongly.
    if q < 60 and r["Earnings Growth"] >= 25 and (pd.isna(r["Revenue Growth"]) or r["Revenue Growth"] < 10):
        return "🔄 Kandidát na turnaround"
    if g >= 60 and q < 50 and v < 50:
        return "🔥 High Growth / dražší příběh"
    if v < 40 and q < 50 and g < 50:
        return "⚠️ Slabý fundamentální obraz"
    return "🔎 Smíšený příběh"

def story_priority(r):
    story = r["Story"]
    v, q, g = r["Value Score"], r["Quality Score"], r["Growth Score"]
    targets = {
        "🏆 Quality Compounder": (q, g, v),
        "💎 Kvalita za rozumnou cenu": (q, v, g),
        "🚀 Růst za rozumnou cenu": (g, q, v),
        "💰 Value / levná firma": (v, q, g),
        "🔄 Kandidát na turnaround": (g, v, q),
        "🔥 High Growth / dražší příběh": (g, q, v),
        "🪤 Value Trap – varování": (v, q, g),
        "⚠️ Slabý fundamentální obraz": (100-max(v or 0,0), 100-max(q or 0,0), 100-max(g or 0,0)),
    }
    vals = targets.get(story, (v,q,g))
    vals = [x for x in vals if not pd.isna(x)]
    return round(sum(vals)/len(vals), 1) if vals else np.nan

# Sidebar
st.sidebar.header("⚙️ Nastavení")
selected_exchanges = st.sidebar.multiselect("Burzy", ["NASDAQ", "NYSE", "XETRA"], default=["NASDAQ", "NYSE", "XETRA"])
min_cap_b = st.sidebar.number_input("Min. Market Cap (mld.)", min_value=0.0, value=1.0, step=0.5)
max_candidates = st.sidebar.slider("Max. titulů pro načtení", 25, 1000, 150, 25)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Jaký příběh hledám?")
story_options = [
    "🏆 Quality Compounder",
    "💎 Kvalita za rozumnou cenu",
    "🚀 Růst za rozumnou cenu",
    "💰 Value / levná firma",
    "🔄 Kandidát na turnaround",
    "🔥 High Growth / dražší příběh",
    "🪤 Value Trap – varování",
]
selected_stories = st.sidebar.multiselect("Příběhy", story_options, default=story_options[:6], help="Neatraktivní příběhy nemusíš hledat; Value Trap zde slouží jako výjimka – upozornění na levnou firmu se slabými základy.")
min_data = st.sidebar.slider("Min. počet dostupných parametrů", 3, len(PARAMS), 7, 1)

with st.sidebar.expander("⚙️ Klasické filtry (volitelné)"):
    use_classic = st.checkbox("Použít klasické filtry", value=False)
    max_pe = st.number_input("Max. P/E", min_value=0.0, value=25.0, step=1.0)
    max_fpe = st.number_input("Max. Forward P/E", min_value=0.0, value=20.0, step=1.0)
    min_roe = st.number_input("Min. ROE (%)", value=10.0, step=1.0)
    min_rev_growth = st.number_input("Min. Revenue Growth (%)", value=0.0, step=1.0)
    min_earn_growth = st.number_input("Min. Earnings Growth (%)", value=0.0, step=1.0)
    min_fcf_m = st.number_input("Min. FCF (mil.)", value=0.0, step=50.0)
    max_de = st.number_input("Max. Debt/Equity (%)", value=150.0, step=25.0)

run = st.sidebar.button("🚀 Spustit screening", type="primary")
clear = st.sidebar.button("🧹 Vyčistit výsledky")
refresh = st.sidebar.button("🔄 Obnovit zdroje")

if refresh:
    st.cache_data.clear(); st.rerun()
if clear:
    st.session_state.pop("screening_results", None); st.rerun()
if not selected_exchanges:
    st.warning("Vyber alespoň jednu burzu."); st.stop()

with st.spinner("Načítám aktuální seznam titulů z oficiálních zdrojů…"):
    try:
        universe = load_universe(selected_exchanges)
    except Exception as e:
        st.error(f"Chyba při načtení univerza: {e}"); st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Celkem v univerzu", f"{len(universe):,}".replace(",", " "))
for i, ex in enumerate(selected_exchanges[:3], start=2):
    [c2, c3, c4][i-2].metric(ex, f"{int((universe['Exchange'] == ex).sum()):,}".replace(",", " "))

st.markdown("### 🌍 Univerzum")
st.caption("NASDAQ/NYSE jsou získávány z Nasdaq Trader; XETRA z oficiálního seznamu Deutsche Börse. XETRA je omezeno na Instrument Type = CS (Common Stock / Equity).")

if not run and "screening_results" not in st.session_state:
    st.info("Nastav příběhy a stiskni **🚀 Spustit screening**."); st.stop()

if run:
    candidates = build_candidate_sample(universe, max_candidates)
    rows = []; progress = st.progress(0); status_text = st.empty()
    for i, row in candidates.iterrows():
        status_text.write(f"Načítám {i+1}/{len(candidates)}: **{row['Ticker']}**")
        rows.append(fetch_fundamentals(row["Ticker"], row["Exchange"], row.get("Name", ""), row.get("ISIN", "")))
        progress.progress((i+1)/len(candidates))
    progress.empty(); status_text.empty()
    st.session_state["screening_results"] = pd.DataFrame(rows)

results_df = st.session_state.get("screening_results", pd.DataFrame())
if results_df.empty:
    st.warning("Pro vybrané nastavení nebyla načtena žádná data."); st.stop()

# Scores and stories
scores = results_df.apply(calc_scores, axis=1, result_type="expand")
scores.columns = ["Value Score", "Quality Score", "Growth Score"]
results_df = pd.concat([results_df.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)
results_df["Story"] = results_df.apply(classify_story, axis=1)
results_df["Story Priority"] = results_df.apply(story_priority, axis=1)
results_df["Available Params"] = results_df[PARAMS].notna().sum(axis=1)

# Optional classic filters. Missing data never passes a requested filter.
def passes_classic(r):
    checks = [
        not pd.isna(r["Market Cap"]) and r["Market Cap"] >= min_cap_b * 1e9,
        not pd.isna(r["P/E"]) and r["P/E"] > 0 and r["P/E"] <= max_pe,
        not pd.isna(r["Forward P/E"]) and r["Forward P/E"] > 0 and r["Forward P/E"] <= max_fpe,
        not pd.isna(r["ROE"]) and r["ROE"] >= min_roe,
        not pd.isna(r["Revenue Growth"]) and r["Revenue Growth"] >= min_rev_growth,
        not pd.isna(r["Earnings Growth"]) and r["Earnings Growth"] >= min_earn_growth,
        not pd.isna(r["Free Cash Flow"]) and r["Free Cash Flow"] >= min_fcf_m * 1e6,
        not pd.isna(r["Debt/Equity"]) and r["Debt/Equity"] <= max_de,
    ]
    return all(checks)

results_df["Pass"] = results_df.apply(passes_classic, axis=1) if use_classic else True
results_df["Story Selected"] = results_df["Story"].isin(selected_stories) if selected_stories else True
results_df["Eligible"] = (results_df["Available Params"] >= min_data) & results_df["Pass"] & results_df["Story Selected"]
results_df = results_df.sort_values(["Eligible", "Story Priority"], ascending=[False, False], na_position="last").reset_index(drop=True)
st.session_state["screening_results"] = results_df

# Summary
st.markdown("## 📊 Výsledek screeningu")
a,b,c,d,e = st.columns(5)
a.metric("Načteno", len(results_df))
b.metric("Kompletní data", int((results_df["Status"] == "OK").sum()))
c.metric("≥ min. dat", int((results_df["Available Params"] >= min_data).sum()))
d.metric("Vybraný příběh", int(results_df["Story Selected"].sum()))
e.metric("Kandidáti", int(results_df["Eligible"].sum()))

st.markdown("### 🧭 Mapa investičních příběhů")
st.caption("Skóre není predikce výnosu. Je to první, transparentní způsob, jak převést devět fundamentů do charakteru firmy. Pravidla budeme společně zpřesňovat.")

story_counts = results_df[results_df["Available Params"] >= min_data]["Story"].value_counts().rename_axis("Příběh").reset_index(name="Počet")
st.dataframe(story_counts, use_container_width=True, hide_index=True)

st.markdown("### 🎯 Kandidáti")
passed = results_df[results_df["Eligible"]].copy()
display_cols = [
    "Ticker","Name","Exchange","Story","Value Score","Quality Score","Growth Score",
    "Market Cap","P/E","Forward P/E","P/S","ROE","Revenue Growth","Earnings Growth",
    "Free Cash Flow","Debt/Equity","Status"
]
if passed.empty:
    st.info("Pro zvolený příběh a nastavení dat nebyl nalezen žádný kandidát.")
else:
    st.dataframe(passed[display_cols], use_container_width=True, hide_index=True, height=650,
        column_config={
            "Value Score": st.column_config.NumberColumn("Hodnota", format="%.0f"),
            "Quality Score": st.column_config.NumberColumn("Kvalita", format="%.0f"),
            "Growth Score": st.column_config.NumberColumn("Růst", format="%.0f"),
            "Market Cap": st.column_config.NumberColumn("Market Cap", format="%.0f"),
            "P/E": st.column_config.NumberColumn("P/E", format="%.1f"),
            "Forward P/E": st.column_config.NumberColumn("Forward P/E", format="%.1f"),
            "P/S": st.column_config.NumberColumn("P/S", format="%.1f"),
            "ROE": st.column_config.NumberColumn("ROE %", format="%.1f"),
            "Revenue Growth": st.column_config.NumberColumn("Revenue Growth %", format="%.1f"),
            "Earnings Growth": st.column_config.NumberColumn("Earnings Growth %", format="%.1f"),
            "Free Cash Flow": st.column_config.NumberColumn("FCF", format="%.0f"),
            "Debt/Equity": st.column_config.NumberColumn("D/E %", format="%.1f"),
        })

# Availability
st.markdown("### 📊 Dostupnost fundamentálních parametrů")
availability = []
for p in PARAMS:
    n = int(results_df[p].notna().sum())
    availability.append({
        "Parametr": p,
        "Dostupných hodnot": n,
        "Celkem": len(results_df),
        "Dostupnost %": round(n / len(results_df) * 100, 1)
    })
availability_df = pd.DataFrame(availability)
st.dataframe(availability_df, use_container_width=True, hide_index=True)

# Exchange quality
st.markdown("### 🏛️ Kvalita dat podle burzy")
exchange_rows = []
for ex, g in results_df.groupby("Exchange"):
    complete = int((g["Status"] == "OK").sum())
    at_least_half = int((g[PARAMS].notna().sum(axis=1) >= len(PARAMS)/2).sum())
    at_least_one = int((g[PARAMS].notna().sum(axis=1) >= 1).sum())
    avg_avail = round(g[PARAMS].notna().mean().mean() * 100, 1)
    exchange_rows.append({
        "Burza": ex,
        "Testováno": len(g),
        "Kompletní data": complete,
        "≥ 50 % parametrů": at_least_half,
        "≥ 1 parametr": at_least_one,
        "Průměrná dostupnost %": avg_avail
    })
st.dataframe(pd.DataFrame(exchange_rows), use_container_width=True, hide_index=True)

# Diagnostics
with st.expander("🔍 Detail všech načtených titulů", expanded=False):
    detail_cols = [
        "Ticker", "Yahoo Ticker", "Name", "Exchange",
        *PARAMS, "Value Score", "Quality Score", "Growth Score", "Story", "Available Params", "Status", "Mapping", "Data Source", "Error"
    ]
    st.dataframe(
        results_df[detail_cols],
        use_container_width=True,
        hide_index=True,
        height=800
    )

with st.expander("⚠️ Tituly s neúplnými nebo chybějícími daty", expanded=False):
    bad = results_df[results_df["Status"] != "OK"].copy()
    if bad.empty:
        st.success("Všechny načtené tituly mají kompletní sadu parametrů.")
    else:
        st.dataframe(
            bad[["Ticker","Yahoo Ticker","Name","Exchange","Status","Mapping","Data Source","Error",*PARAMS]],
            use_container_width=True,
            hide_index=True,
            height=700
        )

with st.expander("🧭 Kontrola XETRA mappingu", expanded=False):
    x = results_df[results_df["Exchange"] == "XETRA"].copy()
    if x.empty:
        st.info("V tomto běhu nebyl testován XETRA.")
    else:
        st.dataframe(
            x[["Ticker","Yahoo Ticker","Name","Mapping","Status","Market Cap","P/E"]],
            use_container_width=True,
            hide_index=True
        )

st.info(
    "V3 záměrně odděluje surové fundamenty od interpretace. Hodnota, Kvalita a Růst jsou "
    "první transparentní skóre; z jejich kombinace vzniká investiční příběh. Chybějící hodnoty "
    "se nepřevádějí na nulu. Earnings Growth se počítá pouze při kladném zisku v obou letech."
)

st.caption("Zdroje: Nasdaq Trader, Deutsche Börse Xetra a Yahoo Finance/yfinance. Data jsou získávána při screeningu a mohou být zpožděná či nedostupná.")
