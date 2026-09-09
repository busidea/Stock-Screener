import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import re
import time
from html import unescape
from io import StringIO
from datetime import datetime

st.set_page_config(page_title="Stock-Screener", page_icon="🔎", layout="wide")

st.title("🔎 Stock-Screener")
st.caption("V5.1 – fundament → charakter → trend → investiční příběh → textové signály")

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
    """Load current fundamentals plus a small annual history used by the story engine."""
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
        try: info = t.info or {}
        except Exception: pass
        fast = {}
        try: fast = dict(t.fast_info)
        except Exception: pass

        income = pd.DataFrame(); ttm_income = pd.DataFrame(); balance = pd.DataFrame(); cashflow = pd.DataFrame()
        try:
            x = t.income_stmt
            if isinstance(x, pd.DataFrame) and not x.empty: income = x
        except Exception: pass
        try:
            x = t.ttm_income_stmt
            if isinstance(x, pd.DataFrame) and not x.empty: ttm_income = x
        except Exception: pass
        try: balance = t.balance_sheet
        except Exception: pass
        try: cashflow = t.cashflow
        except Exception: pass

        def row_series(df, labels):
            if df is None or df.empty: return pd.Series(dtype=float)
            idx = {str(x).strip().lower(): x for x in df.index}
            for label in labels:
                key = str(label).strip().lower()
                if key in idx:
                    s = pd.to_numeric(df.loc[idx[key]], errors="coerce").dropna()
                    if not s.empty: return s
            compact = {re.sub(r"[^a-z0-9]", "", str(x).lower()): x for x in df.index}
            for label in labels:
                key = re.sub(r"[^a-z0-9]", "", str(label).lower())
                if key in compact:
                    s = pd.to_numeric(df.loc[compact[key]], errors="coerce").dropna()
                    if not s.empty: return s
            return pd.Series(dtype=float)

        revenue_s = row_series(income, ["Total Revenue", "TotalRevenue", "Operating Revenue", "OperatingRevenue"])
        net_income_s = row_series(income, ["Net Income", "NetIncome", "Net Income Common Stockholders", "NetIncomeCommonStockholders"])
        equity_s = row_series(balance, ["Stockholders Equity", "StockholdersEquity", "Common Stock Equity", "CommonStockEquity", "Total Equity Gross Minority Interest"])
        debt_s = row_series(balance, ["Total Debt", "TotalDebt"])
        ocf_s = row_series(cashflow, ["Operating Cash Flow", "OperatingCashFlow", "Total Cash From Operating Activities"])
        capex_s = row_series(cashflow, ["Capital Expenditure", "CapitalExpenditure", "Capital Expenditure Reported"])

        def latest_values(s, n=5):
            vals = [safe_float(x) for x in s.iloc[:n].tolist()] if not s.empty else []
            return [x for x in vals if not pd.isna(x)]

        rev_hist = latest_values(revenue_s)
        ni_hist = latest_values(net_income_s)
        # yfinance statement columns are normally newest -> oldest.
        rev_current = rev_hist[0] if rev_hist else np.nan
        rev_old = rev_hist[3] if len(rev_hist) >= 4 else (rev_hist[-1] if len(rev_hist) >= 2 else np.nan)
        ni_current = ni_hist[0] if ni_hist else np.nan
        ni_old = ni_hist[3] if len(ni_hist) >= 4 else (ni_hist[-1] if len(ni_hist) >= 2 else np.nan)

        api_rev_cur, api_rev_prev, api_ni_cur, api_ni_prev = yahoo_annual_growth_data(yahoo_ticker)
        if pd.isna(rev_current) or pd.isna(rev_old):
            rev_current = api_rev_cur if not pd.isna(api_rev_cur) else rev_current
            rev_old = api_rev_prev if not pd.isna(api_rev_prev) else rev_old
        if pd.isna(ni_current) or pd.isna(ni_old):
            ni_current = api_ni_cur if not pd.isna(api_ni_cur) else ni_current
            ni_old = api_ni_prev if not pd.isna(api_ni_prev) else ni_old

        price = first_valid(info.get("currentPrice"), fast.get("last_price"), info.get("regularMarketPrice"))
        shares = first_valid(info.get("sharesOutstanding"), info.get("impliedSharesOutstanding"))
        market_cap = first_valid(info.get("marketCap"), fast.get("market_cap"), price * shares if not pd.isna(price) and not pd.isna(shares) else np.nan)
        pe = first_valid(info.get("trailingPE")); fpe = first_valid(info.get("forwardPE")); ps = first_valid(info.get("priceToSalesTrailing12Months"))
        revenue = rev_current; previous_revenue = rev_hist[1] if len(rev_hist) >= 2 else np.nan
        net_income = ni_current; previous_net_income = ni_hist[1] if len(ni_hist) >= 2 else np.nan
        equity = safe_float(equity_s.iloc[0]) if not equity_s.empty else np.nan
        debt = safe_float(debt_s.iloc[0]) if not debt_s.empty else np.nan
        ocf = safe_float(ocf_s.iloc[0]) if not ocf_s.empty else np.nan
        capex = safe_float(capex_s.iloc[0]) if not capex_s.empty else np.nan

        if pd.isna(ps) and not pd.isna(market_cap) and revenue > 0: ps = market_cap / revenue
        if pd.isna(pe) and not pd.isna(market_cap) and net_income > 0: pe = market_cap / net_income
        forward_eps = first_valid(info.get("forwardEps"))
        if pd.isna(fpe) and price > 0 and not pd.isna(forward_eps) and forward_eps > 0: fpe = price / forward_eps
        roe = first_valid(info.get("returnOnEquity"))
        if not pd.isna(roe): roe = roe * 100 if abs(roe) <= 3 else roe
        if pd.isna(roe) and net_income > 0 and equity > 0: roe = net_income / equity * 100
        revenue_growth = (revenue / previous_revenue - 1) * 100 if revenue > 0 and previous_revenue > 0 else np.nan
        earnings_growth = (net_income / previous_net_income - 1) * 100 if net_income > 0 and previous_net_income > 0 else np.nan
        fcf = first_valid(info.get("freeCashflow"))
        if pd.isna(fcf) and not pd.isna(ocf) and not pd.isna(capex): fcf = ocf + capex if capex < 0 else ocf - capex
        de = first_valid(info.get("debtToEquity"))
        if not pd.isna(de) and equity <= 0: de = np.nan
        if pd.isna(de) and debt >= 0 and equity > 0: de = debt / equity * 100

        # Multi-year trend metrics: deliberately simple and auditable.
        revenue_cagr_3y = np.nan
        if len(rev_hist) >= 4 and rev_hist[3] > 0 and rev_hist[0] > 0:
            revenue_cagr_3y = ((rev_hist[0] / rev_hist[3]) ** (1/3) - 1) * 100
        net_income_cagr_3y = np.nan
        if len(ni_hist) >= 4 and ni_hist[3] > 0 and ni_hist[0] > 0:
            net_income_cagr_3y = ((ni_hist[0] / ni_hist[3]) ** (1/3) - 1) * 100
        current_margin = np.nan
        if revenue > 0 and not pd.isna(net_income): current_margin = net_income / revenue * 100
        old_margin = np.nan
        if len(rev_hist) >= 4 and rev_hist[3] > 0 and len(ni_hist) >= 4 and not pd.isna(ni_hist[3]): old_margin = ni_hist[3] / rev_hist[3] * 100
        margin_change = current_margin - old_margin if not pd.isna(current_margin) and not pd.isna(old_margin) else np.nan
        # Previous-year change is crucial for distinguishing a recovery from ordinary growth.
        revenue_prior_yoy = np.nan
        if len(rev_hist) >= 3 and rev_hist[2] != 0:
            revenue_prior_yoy = (rev_hist[1] / rev_hist[2] - 1) * 100
        net_income_prior_yoy = np.nan
        if len(ni_hist) >= 3 and ni_hist[2] != 0:
            net_income_prior_yoy = (ni_hist[1] / ni_hist[2] - 1) * 100
        net_income_sign_recovery = bool(len(ni_hist) >= 2 and ni_hist[0] > 0 and ni_hist[1] <= 0)

        values = {
            "Market Cap": market_cap, "P/E": pe, "Forward P/E": fpe, "P/S": ps, "ROE": roe,
            "Revenue Growth": revenue_growth, "Earnings Growth": earnings_growth, "Free Cash Flow": fcf, "Debt/Equity": de,
            "Revenue CAGR 3Y": revenue_cagr_3y, "Net Income CAGR 3Y": net_income_cagr_3y,
            "Net Margin": current_margin, "Margin Change 3Y": margin_change,
            "Revenue Prior YoY": revenue_prior_yoy, "Net Income Prior YoY": net_income_prior_yoy,
            "Net Income Sign Recovery": net_income_sign_recovery,
            "Revenue Trend": ";".join([f"{x/1e9:.2f}" for x in rev_hist[:4]]) if rev_hist else "",
            "Net Income Trend": ";".join([f"{x/1e9:.2f}" for x in ni_hist[:4]]) if ni_hist else "",
        }
        available = sum(not pd.isna(values[p]) for p in PARAMS)
        status = "OK" if available == len(PARAMS) else ("PARTIAL" if available > 0 else "NO DATA")
        source_parts = ["Yahoo info"]
        if not income.empty: source_parts.append("annual income")
        if not ttm_income.empty: source_parts.append("TTM income")
        if not balance.empty: source_parts.append("balance")
        if not cashflow.empty: source_parts.append("cashflow")
        if resolution != "direct": source_parts.append(f"mapping:{resolution}")

        sector = clean_text(info.get("sector")); industry = clean_text(info.get("industry")); quote_type = clean_text(info.get("quoteType"))
        return {"Ticker": requested_ticker, "Yahoo Ticker": yahoo_ticker, "Name": name, "Exchange": exchange,
                **values, "Sector": sector, "Industry": industry, "Quote Type": quote_type,
                "Status": status, "Data Source": " + ".join(source_parts), "Mapping": resolution, "Error": ""}
    except Exception as e:
        return {"Ticker": requested_ticker, "Yahoo Ticker": yahoo_ticker, "Name": name, "Exchange": exchange,
                **{p: np.nan for p in PARAMS}, "Revenue CAGR 3Y": np.nan, "Net Income CAGR 3Y": np.nan,
                "Net Margin": np.nan, "Margin Change 3Y": np.nan, "Revenue Prior YoY": np.nan,
                "Net Income Prior YoY": np.nan, "Net Income Sign Recovery": False, "Revenue Trend": "", "Net Income Trend": "",
                "Sector": "", "Industry": "", "Quote Type": "", "Status": "ERROR", "Data Source": "Yahoo", "Mapping": resolution, "Error": str(e)[:300]}

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
# V4 – charakter firmy → trend → investiční příběh
# -----------------------------------------------------------------------------

def band(score):
    if pd.isna(score): return "⚪ N/A"
    if score >= 70: return "🟢 Silné"
    if score >= 50: return "🟡 Střední"
    return "🔴 Slabé"

def tier_score(v, cuts, reverse=False):
    if pd.isna(v): return np.nan
    for threshold, score in cuts:
        if (v <= threshold) if reverse else (v >= threshold): return float(score)
    return 0.0

def avg_available(parts):
    vals = [x for x in parts if not pd.isna(x)]
    return round(sum(vals)/len(vals), 1) if vals else np.nan

def calc_scores(r):
    value_parts = [
        tier_score(r["P/E"], [(10,100),(15,85),(20,70),(30,50),(45,25)], True),
        tier_score(r["Forward P/E"], [(10,100),(15,85),(20,70),(30,50),(45,25)], True),
        tier_score(r["P/S"], [(1,100),(2,85),(3,70),(5,50),(8,25)], True)]
    quality_parts = [
        tier_score(r["ROE"], [(25,100),(20,90),(15,75),(10,55),(5,30)]),
        100.0 if (not pd.isna(r["Free Cash Flow"]) and r["Free Cash Flow"] > 0) else (0.0 if not pd.isna(r["Free Cash Flow"]) else np.nan),
        tier_score(r["Debt/Equity"], [(25,100),(50,85),(100,65),(150,45),(250,20)], True)]
    growth_parts = [
        tier_score(r["Revenue Growth"], [(20,100),(10,85),(5,70),(0,50),(-5,25)]),
        tier_score(r["Earnings Growth"], [(25,100),(15,85),(10,75),(5,60),(0,45)])]
    return avg_available(value_parts), avg_available(quality_parts), avg_available(growth_parts)

def company_type(r):
    s = (clean_text(r.get("Sector")) + " " + clean_text(r.get("Industry")) + " " + clean_text(r.get("Name"))).lower()
    if "reit" in s or "real estate investment trust" in s:
        return "REIT / real estate"
    if any(x in s for x in ["asset management", "capital markets", "investment management", "investment holding", "financial services"]):
        return "Financial / investment company"
    if any(x in s for x in ["bank", "insurance", "credit", "mortgage"]):
        return "Financial institution"
    if any(x in s for x in ["oil", "gas", "energy", "mining", "steel", "metals", "chemicals", "coal"]):
        return "Cyclical / commodity"
    return "Operating company"

def turnaround_score(r):
    """Score a *change of direction*, not simply strong current growth."""
    score = 0.0; evidence = []
    rg = r["Revenue Growth"]; eg = r["Earnings Growth"]
    prior_rg = r["Revenue Prior YoY"]; prior_eg = r["Net Income Prior YoY"]
    rc = r["Revenue CAGR 3Y"]; mc = r["Margin Change 3Y"]
    sign_recovery = bool(r.get("Net Income Sign Recovery", False))

    # Gate: there must be evidence that the business was weak/deteriorating before improving.
    deterioration = False
    if (not pd.isna(prior_eg) and prior_eg < 0):
        deterioration = True; score += 25; evidence.append("předchozí pokles zisku")
    if (not pd.isna(prior_rg) and prior_rg < 0):
        deterioration = True; score += 15; evidence.append("předchozí pokles tržeb")
    if sign_recovery:
        deterioration = True; score += 30; evidence.append("návrat ztráty do zisku")
    if not pd.isna(mc) and mc >= 3 and (not pd.isna(prior_eg) and prior_eg < 10):
        deterioration = True; score += 20; evidence.append("výrazné zlepšení marže po slabším období")

    if not deterioration:
        return 0.0, "růst bez prokázané změny směru"

    # Then require current improvement as the second half of the turnaround story.
    if not pd.isna(eg) and eg >= 15: score += 15; evidence.append("aktuální růst zisku")
    if not pd.isna(rg) and rg >= 3: score += 10; evidence.append("aktuální stabilizace/ růst tržeb")
    if not pd.isna(mc) and mc >= 2: score += 10; evidence.append("zlepšení marže")
    if not pd.isna(r["Free Cash Flow"]) and r["Free Cash Flow"] > 0: score += 10; evidence.append("kladný FCF")
    if not pd.isna(rc) and rc < 3: score += 5; evidence.append("tržby dlouhodobě slabé")
    return min(score,100), "; ".join(evidence)

def classify_story(r):
    v,q,g = r["Value Score"],r["Quality Score"],r["Growth Score"]
    ctype = r["Company Type"]; ts = r["Turnaround Score"]
    # Special asset/financial companies should not be forced into operating-company stories.
    if ctype in ("Financial / investment company", "Financial institution", "REIT / real estate"):
        if ts >= 65 and r["Earnings Growth"] >= 15:
            return "🏗️ Asset / financial recovery"
        if ctype == "REIT / real estate" and q >= 60 and v >= 55:
            return "🏢 Real-estate value"
    if ts >= 65 and q < 75:
        return "🔄 Operating turnaround"
    if not pd.isna(v) and not pd.isna(q) and not pd.isna(g):
        if v >= 70 and q < 50 and g < 50: return "🪤 Value Trap – varování"
        if q >= 70 and g >= 65 and v >= 45: return "🏆 Quality Compounder"
        if q >= 70 and v >= 60 and g >= 45: return "💎 Kvalita za rozumnou cenu"
        if g >= 70 and v >= 50 and q >= 50: return "🚀 Růst za rozumnou cenu"
        if v >= 65 and q >= 50 and g < 50: return "💰 Value / levná firma"
        if g >= 60 and q < 50 and v < 50: return "🔥 High Growth / dražší příběh"
        if v < 40 and q < 50 and g < 50: return "⚠️ Slabý fundamentální obraz"
    return "🔎 Smíšený příběh"

def story_priority(r):
    s=r["Story"]; v,q,g=r["Value Score"],r["Quality Score"],r["Growth Score"]; ts=r["Turnaround Score"]
    targets={"🏆 Quality Compounder":(q,g,v),"💎 Kvalita za rozumnou cenu":(q,v,g),"🚀 Růst za rozumnou cenu":(g,q,v),
             "💰 Value / levná firma":(v,q,g),"🔄 Operating turnaround":(ts,q,g),"🏗️ Asset / financial recovery":(ts,q,v),
             "🏢 Real-estate value":(v,q,g),"🔥 High Growth / dražší příběh":(g,q,v),"🪤 Value Trap – varování":(v,100-(q or 0),100-(g or 0))}
    vals=[x for x in targets.get(s,(v,q,g)) if not pd.isna(x)]
    return round(sum(vals)/len(vals),1) if vals else np.nan

# -----------------------------------------------------------------------------
# V5 – Text Evidence Engine
# Druhá fáze: pouze pro úzký výběr kandidátů. Nepoužívá se pro celé univerzum.
# -----------------------------------------------------------------------------

TEXT_RULES = {
    "🔄 Operating turnaround": {
        "positive": {
            "turnaround": 3, "recovery": 2, "restructuring": 2,
            "cost reduction": 2, "cost savings": 2, "margin recovery": 3,
            "return to profitability": 3, "operational improvement": 2,
            "operating improvement": 2, "deleveraging": 2, "new management": 1,
            "strategic review": 1, "transformation": 1, "profitability improved": 2,
            "cash flow improved": 2
        },
        "negative": {
            "continued decline": 3, "deterioration": 2, "liquidity pressure": 3,
            "covenant breach": 3, "going concern": 3, "cash burn": 2,
            "declining demand": 2, "margin pressure": 2, "failed turnaround": 3,
            "turnaround efforts have not": 3, "restructuring costs": 2
        }
    },
    "🏗️ Asset / financial recovery": {
        "positive": {
            "net asset value": 3, "nav per share": 3, "discount to nav": 3,
            "portfolio value": 2, "fair value": 2, "monetization": 2,
            "realization": 2, "asset value": 2, "recovery": 2,
            "investment gains": 2, "portfolio gains": 2, "deleveraging": 2
        },
        "negative": {
            "impairment": 2, "write-down": 3, "liquidity pressure": 3,
            "discount widened": 3, "portfolio loss": 2, "realization risk": 2
        }
    },
    "🏢 Real-estate value": {
        "positive": {
            "net asset value": 3, "nav": 2, "occupancy": 2, "rent growth": 2,
            "same-store noi": 3, "noi growth": 3, "leasing spread": 2,
            "development pipeline": 1, "asset value": 2, "discount to nav": 3
        },
        "negative": {
            "vacancy": 2, "occupancy decline": 3, "rent decline": 2,
            "impairment": 2, "refinancing risk": 3, "higher interest expense": 2
        }
    },
    "🚀 Růst za rozumnou cenu": {
        "positive": {
            "organic growth": 3, "accelerating growth": 3, "market share": 2,
            "capacity expansion": 2, "backlog": 2, "bookings growth": 2,
            "demand growth": 2, "new markets": 2, "international expansion": 2,
            "pipeline": 1, "pricing power": 2
        },
        "negative": {
            "slowing growth": 3, "declining demand": 2, "market share loss": 3,
            "competitive pressure": 2, "guidance cut": 3, "weak bookings": 2
        }
    },
    "🏆 Quality Compounder": {
        "positive": {
            "recurring revenue": 3, "recurring cash flow": 3, "pricing power": 3,
            "competitive advantage": 3, "market leadership": 2, "high margins": 2,
            "free cash flow": 1, "long-term growth": 2, "capital allocation": 2,
            "customer retention": 2, "strong balance sheet": 2
        },
        "negative": {
            "customer churn": 3, "margin pressure": 2, "competitive pressure": 2,
            "market share loss": 3, "cash burn": 3, "weak balance sheet": 3
        }
    },
    "💎 Kvalita za rozumnou cenu": {
        "positive": {
            "undervalued": 3, "attractive valuation": 3, "discount to peers": 2,
            "free cash flow": 2, "pricing power": 2, "competitive advantage": 2,
            "capital return": 2, "share buyback": 2, "strong balance sheet": 2
        },
        "negative": {
            "overvalued": 3, "valuation premium": 2, "margin pressure": 2,
            "competitive pressure": 2, "declining demand": 2
        }
    },
    "💰 Value / levná firma": {
        "positive": {
            "undervalued": 3, "intrinsic value": 3, "discount to peers": 2,
            "asset value": 2, "sum of the parts": 3, "share buyback": 2,
            "capital return": 2, "non-core assets": 1, "monetization": 2
        },
        "negative": {
            "structural decline": 3, "secular decline": 3, "excess capacity": 2,
            "debt burden": 3, "liquidity pressure": 3, "cash burn": 3,
            "declining market share": 3, "impairment": 2
        }
    },
    "🔥 High Growth / dražší příběh": {
        "positive": {
            "accelerating growth": 3, "organic growth": 2, "market share": 2,
            "expansion": 1, "backlog": 2, "pipeline": 1, "new markets": 2
        },
        "negative": {
            "overvalued": 3, "valuation premium": 2, "cash burn": 3,
            "dilution": 2, "slowing growth": 3
        }
    },
    "🪤 Value Trap – varování": {
        "positive": {
            "structural decline": 3, "secular decline": 3, "declining market share": 3,
            "excess capacity": 2, "debt burden": 3, "cash burn": 3,
            "competitive pressure": 2, "liquidity pressure": 3, "impairment": 2
        },
        "negative": {
            "turnaround": 2, "recovery": 2, "margin recovery": 2,
            "return to profitability": 3, "strong balance sheet": 2
        }
    }
}

NEGATION_WORDS = {"not", "no", "without", "unlikely", "failed", "fails", "fail", "never", "neither"}


def text_clean(x):
    s = unescape(clean_text(x)).lower()
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def sentence_chunks(text):
    text = text_clean(text)
    if not text:
        return []
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+", text) if x.strip()]


def keyword_context(sentence, phrase):
    """Return (negated, short human-readable context)."""
    pos = sentence.find(phrase)
    if pos < 0:
        return False, sentence[:240]
    before = sentence[max(0, pos - 80):pos]
    words = re.findall(r"[a-z]+", before)
    negated = any(w in NEGATION_WORDS for w in words[-7:])
    start = max(0, pos - 65)
    end = min(len(sentence), pos + len(phrase) + 95)
    snippet = sentence[start:end]
    if start > 0:
        snippet = "…" + snippet
    if end < len(sentence):
        snippet += "…"
    return negated, snippet


def score_text_evidence(text, story):
    rules = TEXT_RULES.get(story)
    if not rules:
        return np.nan, "⚪ Bez textové vrstvy", 0, 0, [], []
    chunks = sentence_chunks(text)
    joined = " ".join(chunks)
    positive_score = 0
    negative_score = 0
    support = []
    warnings = []
    seen = set()

    def scan(bucket, is_positive):
        nonlocal positive_score, negative_score
        for phrase, weight in bucket.items():
            # Count at most twice per phrase: repeated boilerplate should not dominate.
            count = min(2, len(re.findall(r"(?<![a-z])" + re.escape(phrase) + r"(?![a-z])", joined)))
            if count == 0:
                continue
            for sent in chunks:
                if phrase not in sent:
                    continue
                negated, snippet = keyword_context(sent, phrase)
                key = (phrase, snippet[:160])
                if key in seen:
                    continue
                seen.add(key)
                effective_positive = is_positive and not negated
                effective_negative = (not is_positive) or negated
                if effective_positive:
                    positive_score += weight
                    support.append(f"+ {phrase}: {snippet}")
                elif effective_negative:
                    negative_score += weight
                    warnings.append(f"− {phrase}: {snippet}")
                if len(support) >= 8 and len(warnings) >= 8:
                    return

    scan(rules["positive"], True)
    scan(rules["negative"], False)

    raw = 50 + positive_score * 7 - negative_score * 9
    score = float(max(0, min(100, raw)))
    total = positive_score + negative_score
    if total == 0:
        label = "⚪ Bez textového důkazu"
    elif score >= 70 and positive_score > negative_score:
        label = "🟢 Text podporuje příběh"
    elif score >= 55 and positive_score >= negative_score:
        label = "🟡 Text spíše podporuje"
    elif score <= 30 and negative_score > positive_score:
        label = "🔴 Text příběh zpochybňuje"
    else:
        label = "🟠 Text je smíšený"
    return score, label, positive_score, negative_score, support[:8], warnings[:8]


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_text_evidence(yahoo_ticker, story):
    """Fetch only lightweight public text for shortlisted names.

    Sources: Yahoo Finance business summary + recent Yahoo Finance news.
    This is evidence, not an LLM verdict. Missing text is treated as neutral.
    """
    try:
        t = yf.Ticker(yahoo_ticker)
        parts = []
        try:
            info = t.info or {}
            for key in ("longBusinessSummary", "sector", "industry"):
                val = info.get(key)
                if val:
                    parts.append(clean_text(val))
        except Exception:
            pass
        try:
            news = t.news or []
            for item in news[:12]:
                content = item.get("content", item) if isinstance(item, dict) else {}
                title = content.get("title") if isinstance(content, dict) else None
                summary = content.get("summary") if isinstance(content, dict) else None
                if title:
                    parts.append(clean_text(title))
                if summary:
                    parts.append(clean_text(summary))
        except Exception:
            pass
        text = " ".join(parts)
        score, label, pos, neg, support, warnings = score_text_evidence(text, story)
        return {
            "Text Score": score,
            "Text Evidence": label,
            "Text Positive": pos,
            "Text Negative": neg,
            "Text Support": "\n".join(support),
            "Text Warnings": "\n".join(warnings),
            "Text Sources": "Yahoo Finance business summary + recent news" if text else ""
        }
    except Exception as e:
        return {
            "Text Score": np.nan, "Text Evidence": "⚪ Text nedostupný",
            "Text Positive": 0, "Text Negative": 0, "Text Support": "",
            "Text Warnings": "", "Text Sources": str(e)[:180]
        }


# -----------------------------------------------------------------------------
# V5.2 – Price Recovery Engine
# -----------------------------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_price_history(yahoo_ticker):
    """Load up to five years of daily prices for shortlisted names."""
    try:
        hist = yf.Ticker(yahoo_ticker).history(period="5y", interval="1d", auto_adjust=True)
        if hist is None or hist.empty or "Close" not in hist.columns:
            return pd.DataFrame()
        out = hist[["Close"]].copy().dropna()
        out.index = pd.to_datetime(out.index)
        return out
    except Exception:
        return pd.DataFrame()


def calc_price_pattern(yahoo_ticker):
    """Describe price shape; price alone never defines a turnaround."""
    empty = {"Price Score": np.nan, "Price View": "⚪ Cena nedostupná",
             "Drawdown 3Y": np.nan, "Drawdown 5Y": np.nan,
             "Recovery from 3Y Low": np.nan, "Recovery from 5Y Low": np.nan,
             "6M Return": np.nan, "12M Return": np.nan,
             "Days Since 3Y Low": np.nan, "MA50 vs MA200": np.nan,
             "Price Trend": "", "Price Evidence": ""}
    hist = fetch_price_history(yahoo_ticker)
    if hist.empty: return empty
    s = hist["Close"].astype(float).dropna()
    if len(s) < 60: return empty
    now, current = s.index[-1], float(s.iloc[-1])
    result = empty.copy()
    def window(days): return s[s.index >= now - pd.Timedelta(days=days)]
    s3, s5 = window(365*3), window(365*5)
    low3, high3 = float(s3.min()), float(s3.max())
    low5, high5 = float(s5.min()), float(s5.max())
    result["Drawdown 3Y"] = (current/high3-1)*100 if high3 > 0 else np.nan
    result["Drawdown 5Y"] = (current/high5-1)*100 if high5 > 0 else np.nan
    result["Recovery from 3Y Low"] = (current/low3-1)*100 if low3 > 0 else np.nan
    result["Recovery from 5Y Low"] = (current/low5-1)*100 if low5 > 0 else np.nan
    for label, days in (("6M Return",183),("12M Return",365)):
        w=window(days)
        if len(w)>=2 and float(w.iloc[0])>0: result[label]=(current/float(w.iloc[0])-1)*100
    low_date=s3.idxmin(); result["Days Since 3Y Low"] = max(0,(now-low_date).days)
    ma50=float(s.tail(50).mean()); ma200=float(s.tail(200).mean()) if len(s)>=200 else np.nan
    result["MA50 vs MA200"]=(ma50/ma200-1)*100 if ma200>0 else np.nan
    w6=window(183); slope_pct=np.nan
    if len(w6)>=40:
        y=w6.values; x=np.arange(len(y)); slope=np.polyfit(x,y,1)[0]
        slope_pct=slope*len(y)/max(float(y.mean()),1e-9)*100
    score=50.0; positive=negative=0; evidence=[]
    dd=result["Drawdown 3Y"]
    if not pd.isna(dd):
        if dd<=-60: score+=12; positive+=1; evidence.append("velký 3Y propad")
        elif dd<=-35: score+=8; positive+=1; evidence.append("výrazný 3Y propad")
        elif dd<=-20: score+=4; evidence.append("mírnější 3Y propad")
        elif dd>-10: score-=4; negative+=1; evidence.append("cena blízko 3Y maxima")
    rec=result["Recovery from 3Y Low"]
    if not pd.isna(rec):
        if 20<=rec<=100: score+=8; positive+=1; evidence.append("zotavení od 3Y minima")
        elif rec>100: score+=5; evidence.append("výrazné zotavení od minima")
        elif rec<5: score-=6; negative+=1; evidence.append("cena u 3Y minima")
    r12=result["12M Return"]
    if not pd.isna(r12):
        if r12>=20: score+=10; positive+=1; evidence.append("silný růst za 12M")
        elif r12>=5: score+=5; evidence.append("kladný vývoj za 12M")
        elif r12<=-25: score-=10; negative+=1; evidence.append("silný pokles za 12M")
        elif r12<0: score-=4; negative+=1; evidence.append("pokles za 12M")
    ma=result["MA50 vs MA200"]
    if not pd.isna(ma):
        if ma>=5: score+=8; positive+=1; evidence.append("50D průměr nad 200D")
        elif ma<-10: score-=7; negative+=1; evidence.append("50D průměr pod 200D")
    if not pd.isna(slope_pct):
        if slope_pct>=8: score+=7; positive+=1
        elif slope_pct<=-8: score-=7; negative+=1
    result["Price Score"]=round(max(0,min(100,score)),1)
    if positive>=3 and negative==0: view,trend="🟢 Obrat / rostoucí trend","obrat"
    elif positive>=2 and negative<=1: view,trend="🟡 Stabilizace / první recovery","stabilizace"
    elif negative>=2 and positive==0: view,trend="🔴 Stále klesá","pokles"
    elif not pd.isna(r12) and r12>=25 and not pd.isna(dd) and dd>-25: view,trend="🔵 Trh už příběh zřejmě anticipuje","anticipace"
    else: view,trend="⚪ Smíšený cenový obraz","smíšený"
    result["Price View"],result["Price Trend"]=view,trend
    result["Price Evidence"]="; ".join(evidence[:7])
    return result


def add_price_analysis(df,max_price_candidates):
    if df.empty or max_price_candidates<=0: return df
    out=df.copy()
    defaults={"Price Score":np.nan,"Price View":"⚪ Nehodnoceno","Drawdown 3Y":np.nan,"Drawdown 5Y":np.nan,
              "Recovery from 3Y Low":np.nan,"Recovery from 5Y Low":np.nan,"6M Return":np.nan,"12M Return":np.nan,
              "Days Since 3Y Low":np.nan,"MA50 vs MA200":np.nan,"Price Trend":"","Price Evidence":""}
    for c,v in defaults.items(): out[c]=v
    todo=out[out["Eligible"]].sort_values("Story Priority",ascending=False).head(max_price_candidates)
    if todo.empty: return out
    progress=st.progress(0); status=st.empty()
    for i,(idx,row) in enumerate(todo.iterrows(),1):
        status.write(f"Cenová fáze {i}/{len(todo)}: **{row['Ticker']}**")
        ev=calc_price_pattern(row["Yahoo Ticker"])
        for k,v in ev.items(): out.at[idx,k]=v
        progress.progress(i/len(todo))
    progress.empty(); status.empty(); return out


def market_fundamental_view(row):
    ts,ps=safe_float(row.get("Turnaround Score")),safe_float(row.get("Price Score"))
    if pd.isna(ts) or pd.isna(ps): return "⚪ Nedostatek dat"
    if ts>=65 and ps<40: return "🟢 Fundamenty se zlepšují, cena zaostává"
    if ts>=65 and ps>=70: return "🔵 Fundamenty i cena potvrzují obrat"
    if ts<40 and ps>=70: return "🟠 Cena předbíhá fundamenty"
    if ts<40 and ps<40: return "🔴 Fundamenty ani cena obrat nepotvrzují"
    return "🟡 Smíšený signál"


def compact_verdict(row):
    ts = safe_float(row.get("Turnaround Score"))
    text = clean_text(row.get("Text Evidence"))
    conf = safe_float(row.get("Final Confidence"))
    if "zpochybňuje" in text or (not pd.isna(ts) and ts >= 80 and text.startswith("🔴")):
        return "🔴 Zpochybněno"
    if "smíšený" in text and not pd.isna(conf) and conf < 65:
        return "🟠 Smíšený / rizikový"
    if not pd.isna(ts) and ts >= 85 and not pd.isna(conf) and conf >= 70:
        return "🟢 Silný adept"
    if not pd.isna(ts) and ts >= 65:
        return "🟡 Turnaround kandidát"
    return "⚪ Spíše zlepšení"

def compact_trend(row):
    ev = clean_text(row.get("Turnaround Evidence"))
    ts = safe_float(row.get("Turnaround Score"))
    if "změny směru" in ev or (not pd.isna(ts) and ts < 40):
        return "➡️ Bez jasného obratu"
    if "recovery" in ev.lower() or "sign" in ev.lower() or "návrat" in ev.lower():
        return "🔄 Obrat / recovery"
    if not pd.isna(ts) and ts >= 65:
        return "🔄 Známky obratu"
    return "🟡 Částečné zlepšení"

def compact_warning(row):
    txt = clean_text(row.get("Text Evidence"))
    warnings = clean_text(row.get("Text Warnings"))
    if "zpochybňuje" in txt:
        return "🔴 Text varuje"
    if "smíšený" in txt or warnings:
        return "🟠 Rizika / smíšené"
    return "🟢 Bez výrazného varování"


def add_text_evidence(df, max_text_candidates):
    """Run the expensive text layer only on the highest-priority candidates."""
    if df.empty or max_text_candidates <= 0:
        return df
    out = df.copy()
    for c, default in {
        "Text Score": np.nan, "Text Evidence": "⚪ Nehodnoceno",
        "Text Positive": 0, "Text Negative": 0, "Text Support": "",
        "Text Warnings": "", "Text Sources": ""
    }.items():
        out[c] = default

    eligible = out[out["Eligible"]].copy() if "Eligible" in out.columns else out.copy()
    if eligible.empty:
        return out
    todo = eligible.sort_values("Story Priority", ascending=False).head(max_text_candidates)
    progress = st.progress(0)
    status = st.empty()
    for i, (idx, row) in enumerate(todo.iterrows(), start=1):
        status.write(f"Textová fáze {i}/{len(todo)}: **{row['Ticker']}**")
        ev = fetch_text_evidence(row["Yahoo Ticker"], row["Story"])
        for k, v in ev.items():
            out.at[idx, k] = v
        progress.progress(i / len(todo))
    progress.empty(); status.empty()
    return out


def final_story_confidence(r):
    """Combine quantitative story priority with text evidence when available."""
    q = safe_float(r.get("Story Priority"))
    t = safe_float(r.get("Text Score"))
    if pd.isna(q):
        return t
    if pd.isna(t):
        return q
    # Quant 65 %, text 35 %. Text cannot completely override fundamentals.
    return round(0.65 * q + 0.35 * t, 1)

# Sidebar
st.sidebar.header("⚙️ Nastavení")
selected_exchanges = st.sidebar.multiselect("Burzy", ["NASDAQ", "NYSE", "XETRA"], default=["NASDAQ", "NYSE", "XETRA"])
min_cap_b = st.sidebar.number_input("Min. Market Cap (mld.)", min_value=0.0, value=1.0, step=0.5)
max_candidates = st.sidebar.slider("Max. titulů pro hlubší analýzu", 25, 1000, 250, 25)
max_text_candidates = st.sidebar.slider("Max. titulů pro textovou fázi", 0, 50, 30, 5)
max_price_candidates = st.sidebar.slider("Max. titulů pro cenovou fázi", 0, 50, 30, 5)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Jaký příběh hledám?")
story_options = [
    "🏆 Quality Compounder",
    "💎 Kvalita za rozumnou cenu",
    "🚀 Růst za rozumnou cenu",
    "💰 Value / levná firma",
    "🔄 Operating turnaround",
    "🏗️ Asset / financial recovery",
    "🏢 Real-estate value",
    "🔥 High Growth / dražší příběh",
    "🪤 Value Trap – varování",
]
selected_stories = st.sidebar.multiselect("Příběhy", story_options, default=story_options[:7], help="Neatraktivní příběhy nemusíš hledat; Value Trap zde slouží jako výjimka – upozornění na levnou firmu se slabými základy.")
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
results_df["Company Type"] = results_df.apply(company_type, axis=1)
turns = results_df.apply(turnaround_score, axis=1, result_type="expand")
turns.columns = ["Turnaround Score", "Turnaround Evidence"]
results_df = pd.concat([results_df, turns], axis=1)
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

# V5.2: text + price evidence are second-stage layers. Existing results survive UI reruns.
if run:
    if max_text_candidates > 0:
        results_df = add_text_evidence(results_df, max_text_candidates)
    else:
        for c, default in {"Text Score":np.nan,"Text Evidence":"⚪ Nehodnoceno","Text Positive":0,"Text Negative":0,"Text Support":"","Text Warnings":"","Text Sources":""}.items(): results_df[c]=default
    if max_price_candidates > 0:
        results_df = add_price_analysis(results_df, max_price_candidates)
    else:
        for c, default in {"Price Score":np.nan,"Price View":"⚪ Nehodnoceno","Drawdown 3Y":np.nan,"Drawdown 5Y":np.nan,"Recovery from 3Y Low":np.nan,"Recovery from 5Y Low":np.nan,"6M Return":np.nan,"12M Return":np.nan,"Days Since 3Y Low":np.nan,"MA50 vs MA200":np.nan,"Price Trend":"","Price Evidence":""}.items(): results_df[c]=default
if "Text Score" not in results_df.columns:
    results_df["Text Score"]=np.nan; results_df["Text Evidence"]="⚪ Nehodnoceno"; results_df["Text Positive"]=0; results_df["Text Negative"]=0; results_df["Text Support"]=""; results_df["Text Warnings"]=""; results_df["Text Sources"]=""
if "Price Score" not in results_df.columns:
    results_df["Price Score"]=np.nan; results_df["Price View"]="⚪ Nehodnoceno"; results_df["Drawdown 3Y"]=np.nan; results_df["Drawdown 5Y"]=np.nan; results_df["Recovery from 3Y Low"]=np.nan; results_df["Recovery from 5Y Low"]=np.nan; results_df["6M Return"]=np.nan; results_df["12M Return"]=np.nan; results_df["Days Since 3Y Low"]=np.nan; results_df["MA50 vs MA200"]=np.nan; results_df["Price Trend"]=""; results_df["Price Evidence"]=""
results_df["Final Confidence"] = results_df.apply(final_story_confidence, axis=1)
results_df["Market / Fundamental View"] = results_df.apply(market_fundamental_view, axis=1)
results_df = results_df.sort_values(["Eligible","Final Confidence","Story Priority"], ascending=[False,False,False], na_position="last").reset_index(drop=True)
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
if passed.empty:
    st.info("Pro zvolený příběh a nastavení dat nebyl nalezen žádný kandidát.")
else:
    # V5.1: hlavní tabulka je záměrně kompaktní. Ekonomické detaily jsou níže.
    compact = passed.copy()
    compact["Verdikt"] = compact.apply(compact_verdict, axis=1)
    compact["Trend"] = compact.apply(compact_trend, axis=1)
    compact["Varování"] = compact.apply(compact_warning, axis=1)
    compact = compact[[
        "Ticker", "Name", "Story", "Final Confidence", "Verdikt", "Trend",
        "Price View", "Market / Fundamental View", "Text Evidence", "Value Score", "Quality Score", "Growth Score", "Varování"
    ]].rename(columns={
        "Name": "Firma", "Story": "Příběh", "Final Confidence": "Síla příběhu",
        "Price View": "Cenový obraz", "Market / Fundamental View": "Fundamenty vs. cena",
        "Text Evidence": "Textové signály", "Value Score": "Value",
        "Quality Score": "Quality", "Growth Score": "Growth"
    })
    st.dataframe(
        compact, use_container_width=True, hide_index=True, height=560,
        column_config={
            "Síla příběhu": st.column_config.NumberColumn("Síla příběhu", format="%.0f"),
            "Value": st.column_config.NumberColumn("Value", format="%.0f"),
            "Quality": st.column_config.NumberColumn("Quality", format="%.0f"),
            "Growth": st.column_config.NumberColumn("Growth", format="%.0f"),
            "Firma": st.column_config.TextColumn("Firma", width="large"),
            "Příběh": st.column_config.TextColumn("Příběh", width="medium"),
            "Verdikt": st.column_config.TextColumn("Verdikt", width="medium"),
            "Trend": st.column_config.TextColumn("Trend", width="medium"),
            "Textové signály": st.column_config.TextColumn("Textové signály", width="medium"),
            "Varování": st.column_config.TextColumn("Varování", width="medium"),
        })

    st.markdown("#### 🔎 Detail vybraného kandidáta")
    options = [f"{r['Ticker']} — {r['Name']}" for _, r in passed.iterrows()]
    selected_label = st.selectbox("Vyber titul", options, label_visibility="collapsed")
    selected_ticker = selected_label.split(" — ", 1)[0]
    r = passed[passed["Ticker"] == selected_ticker].iloc[0]

    verdict = compact_verdict(r)
    trend = compact_trend(r)
    warning = compact_warning(r)
    st.markdown(f"### {r['Ticker']} — {r['Name']}")
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Příběh", r["Story"])
    m2.metric("Síla příběhu", f"{safe_float(r['Final Confidence']):.0f}/100" if not pd.isna(safe_float(r['Final Confidence'])) else "—")
    m3.metric("Trend", trend)
    m4.metric("Verdikt", verdict)

    st.caption(f"Textové signály: {clean_text(r.get('Text Evidence')) or 'nehodnoceno'} · {warning}")
    if clean_text(r.get("Turnaround Evidence")):
        st.info("**Proč se titul dostal mezi kandidáty:** " + clean_text(r.get("Turnaround Evidence")))

    q1,q2,q3 = st.columns(3)
    q1.metric("Value", f"{safe_float(r['Value Score']):.0f}" if not pd.isna(safe_float(r['Value Score'])) else "—")
    q2.metric("Quality", f"{safe_float(r['Quality Score']):.0f}" if not pd.isna(safe_float(r['Quality Score'])) else "—")
    q3.metric("Growth", f"{safe_float(r['Growth Score']):.0f}" if not pd.isna(safe_float(r['Growth Score'])) else "—")

    with st.expander("💰 Valuace", expanded=False):
        st.dataframe(pd.DataFrame([{
            "Market Cap": r["Market Cap"], "P/E": r["P/E"], "Forward P/E": r["Forward P/E"], "P/S": r["P/S"]
        }]), use_container_width=True, hide_index=True, column_config={
            "Market Cap": st.column_config.NumberColumn("Market Cap", format="%.0f"),
            "P/E": st.column_config.NumberColumn("P/E", format="%.1f"),
            "Forward P/E": st.column_config.NumberColumn("Forward P/E", format="%.1f"),
            "P/S": st.column_config.NumberColumn("P/S", format="%.2f")
        })
    with st.expander("🏭 Kvalita a finanční zdraví", expanded=False):
        st.dataframe(pd.DataFrame([{
            "ROE %": r["ROE"], "FCF": r["Free Cash Flow"], "D/E %": r["Debt/Equity"], "Net Margin %": r["Net Margin"]
        }]), use_container_width=True, hide_index=True, column_config={
            "ROE %": st.column_config.NumberColumn("ROE %", format="%.1f"),
            "FCF": st.column_config.NumberColumn("FCF", format="%.0f"),
            "D/E %": st.column_config.NumberColumn("D/E %", format="%.1f"),
            "Net Margin %": st.column_config.NumberColumn("Net Margin %", format="%.1f")
        })
    with st.expander("📈 Růst a obrat trendu", expanded=False):
        st.dataframe(pd.DataFrame([{
            "Revenue Growth %": r["Revenue Growth"], "Earnings Growth %": r["Earnings Growth"],
            "Revenue CAGR 3Y %": r["Revenue CAGR 3Y"], "Net Income CAGR 3Y %": r["Net Income CAGR 3Y"],
            "Margin Change 3Y %": r["Margin Change 3Y"], "Revenue Prior YoY %": r["Revenue Prior YoY"],
            "Net Income Prior YoY %": r["Net Income Prior YoY"]
        }]), use_container_width=True, hide_index=True)
    with st.expander("📉 Chování ceny", expanded=False):
        pc1, pc2, pc3, pc4 = st.columns(4)
        ps = safe_float(r.get("Price Score")); dd = safe_float(r.get("Drawdown 3Y")); r12 = safe_float(r.get("12M Return")); ma = safe_float(r.get("MA50 vs MA200"))
        pc1.metric("Price Score", f"{ps:.0f}" if not pd.isna(ps) else "—")
        pc2.metric("3Y propad", f"{dd:.1f}%" if not pd.isna(dd) else "—")
        pc3.metric("12M výnos", f"{r12:.1f}%" if not pd.isna(r12) else "—")
        pc4.metric("MA50 vs MA200", f"{ma:.1f}%" if not pd.isna(ma) else "—")
        st.write(f"**Cenový obraz:** {r.get('Price View', '—')}")
        st.write(f"**Fundamenty vs. cena:** {r.get('Market / Fundamental View', '—')}")
        st.write(f"**Evidence:** {r.get('Price Evidence', '') or '—'}")
    with st.expander("📰 Textové signály a varování", expanded=False):
        if clean_text(r.get("Text Support")):
            st.markdown("**Podpůrné signály**")
            st.write(r["Text Support"])
        if clean_text(r.get("Text Warnings")):
            st.markdown("**Varovné signály**")
            st.write(r["Text Warnings"])
        if clean_text(r.get("Text Sources")):
            st.caption("Zdroj textu: " + r["Text Sources"])

    with st.expander("🔬 Kompletní technický záznam", expanded=False):
        detail_cols = [
            "Ticker","Yahoo Ticker","Name","Exchange","Company Type","Sector","Industry",
            *PARAMS,"Revenue CAGR 3Y","Net Income CAGR 3Y","Net Margin","Margin Change 3Y",
            "Revenue Prior YoY","Net Income Prior YoY","Turnaround Score","Turnaround Evidence",
            "Story Priority","Text Score","Final Confidence","Status","Mapping","Data Source","Error"
        ]
        detail_cols = [c for c in detail_cols if c in r.index]
        st.dataframe(pd.DataFrame([r[detail_cols].to_dict()]), use_container_width=True, hide_index=True)


# Text evidence detail
with st.expander("📰 Textové důkazy k příběhu", expanded=False):
    text_cols = ["Ticker", "Name", "Story", "Story Priority", "Text Score", "Text Evidence", "Final Confidence", "Text Support", "Text Warnings", "Text Sources"]
    if "Text Score" in results_df.columns:
        text_view = results_df[(results_df["Text Evidence"] != "⚪ Nehodnoceno") & results_df["Text Score"].notna()].copy()
        if text_view.empty:
            st.info("Textová fáze zatím nebyla provedena nebo pro kandidáty nebyl dostupný text.")
        else:
            st.dataframe(text_view[text_cols], use_container_width=True, hide_index=True, height=700)

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
        *PARAMS, "Revenue CAGR 3Y", "Net Income CAGR 3Y", "Net Margin", "Margin Change 3Y", "Revenue Prior YoY", "Net Income Prior YoY", "Value Score", "Quality Score", "Growth Score", "Turnaround Score", "Company Type", "Story", "Turnaround Evidence", "Text Score", "Text Evidence", "Price Score", "Price View", "Drawdown 3Y", "Recovery from 3Y Low", "6M Return", "12M Return", "MA50 vs MA200", "Market / Fundamental View", "Final Confidence", "Available Params", "Status", "Mapping", "Data Source", "Error"
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
    "V5.2 pracuje ve třech vrstvách: (1) kvantitativní screening, (2) textové signály, (3) chování ceny jen u nejlepších kandidátů. "
    "Textová vrstva příběh nepotvrzuje automaticky; hledá podpůrné i varovné signály a může výsledný příběh zpochybnit. "
    "Příběhy finančních/investičních společností a REIT jsou posuzovány odděleně, aby se na ně "
    "mechanicky nepřenášela logika běžné provozní firmy. Chybějící hodnoty se nepřevádějí na nulu."
)

st.caption("Zdroje: Nasdaq Trader, Deutsche Börse Xetra a Yahoo Finance/yfinance. Data jsou získávána při screeningu a mohou být zpožděná či nedostupná.")
