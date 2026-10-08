import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import re
import time
import hashlib
import json
from pathlib import Path
from html import unescape
from io import StringIO
from datetime import datetime

st.set_page_config(page_title="Stock-Screener", page_icon="🔎", layout="wide")

RUNTIME_LOG = Path("/tmp/stock_screener_runtime.json")

@st.cache_resource(show_spinner=False)
def get_runtime_state():
    state = {
        "status": "idle",
        "stage": "",
        "message": "",
        "updated_at": "",
        "run_started_at": "",
        "run_id": "",
        "last_error": "",
    }
    try:
        if RUNTIME_LOG.exists():
            saved = json.loads(RUNTIME_LOG.read_text(encoding="utf-8"))
            if isinstance(saved, dict):
                state.update(saved)
    except Exception:
        pass
    return state


def update_runtime(status=None, stage=None, message=None, error=None, run_id=None):
    state = get_runtime_state()
    if status is not None: state["status"] = status
    if stage is not None: state["stage"] = stage
    if message is not None: state["message"] = str(message)
    if error is not None: state["last_error"] = str(error)
    if run_id is not None: state["run_id"] = str(run_id)
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    try:
        RUNTIME_LOG.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    return state


st.title("📊 Stock-Screener")
st.caption("V6.28.9 – Screener · samostatný modul Analytik je dostupný v menu vlevo")

NASDAQ_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
NYSE_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
XETRA_URL = "https://www.cashmarket.deutsche-boerse.com/resource/blob/1528/8e34798266f78fe8811bd24387445b2b/data/t7-xetr-allTradableInstruments.csv"

PARAMS = [
    "Market Cap", "P/E", "Forward P/E", "P/S", "ROE",
    "Revenue Growth", "Earnings Growth", "Free Cash Flow", "Debt/Equity"
]

def clean_text(x):
    if x is None:
        return ""
    if isinstance(x, (pd.Series, pd.DataFrame, list, tuple, dict)):
        return "" if len(x) == 0 else str(x).strip()
    try:
        if pd.isna(x):
            return ""
    except (TypeError, ValueError):
        pass
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
    """Load the official Xetra universe without materializing the huge CSV in memory.

    Important for Streamlit Cloud: the Deutsche Börse file can be large enough
    that pandas' C tokenizer raises `Error tokenizing data: out of memory`.
    We therefore stream the HTTP response and parse rows with Python's csv
    module, retaining only CS (common stock/equity) records.
    """
    import csv
    from itertools import chain

    raw = requests.get(
        XETRA_URL,
        timeout=60,
        headers={"User-Agent": "Mozilla/5.0"},
        stream=True,
    )
    raw.raise_for_status()

    # Read only a tiny prefix to locate the real CSV header.
    prefix = []
    for line in raw.iter_lines(decode_unicode=True):
        if line is None:
            continue
        line = line.lstrip("\ufeff")
        prefix.append(line)
        if len(prefix) >= 20:
            break

    header_idx = None
    for i, line in enumerate(prefix):
        if "Instrument Type" in line and ("Mnemonic" in line or "ISIN" in line):
            header_idx = i
            break

    if header_idx is None:
        raw.close()
        raise ValueError("XETRA: hlavička CSV nebyla nalezena.")

    header_line = prefix[header_idx]
    reader = csv.reader([header_line], delimiter=";", quotechar='"')
    header = next(reader)
    header_clean = [clean_text(c) for c in header]

    def header_index(names):
        for name in names:
            target = clean_text(name).lower()
            for i, col in enumerate(header_clean):
                if col.lower() == target:
                    return i
        return None

    typ_i = header_index(["Instrument Type"])
    mnemonic_i = header_index(["Mnemonic"])
    isin_i = header_index(["ISIN"])
    instrument_i = header_index(["Instrument"])
    status_i = header_index(["Instrument Status"])
    market_status_i = header_index(["Market Segment Status"])

    if typ_i is None or mnemonic_i is None:
        raw.close()
        raise ValueError("XETRA: chybí Instrument Type nebo Mnemonic.")

    # Reconstruct the stream from the header and all lines after it.
    # No giant StringIO and no pandas C tokenizer are used.
    remaining_prefix = prefix[header_idx + 1:]
    row_lines = chain(remaining_prefix, raw.iter_lines(decode_unicode=True))
    csv_reader = csv.reader(row_lines, delimiter=";", quotechar='"')

    rows = []
    for row in csv_reader:
        if not row or len(row) <= max(typ_i, mnemonic_i):
            continue

        typ = clean_text(row[typ_i]).upper()
        if typ != "CS":
            continue

        if status_i is not None and status_i < len(row):
            status = clean_text(row[status_i]).lower()
            if status and "active" not in status:
                continue

        if market_status_i is not None and market_status_i < len(row):
            market_status = clean_text(row[market_status_i]).lower()
            if market_status and "active" not in market_status:
                continue

        mnemonic = clean_text(row[mnemonic_i])
        if not mnemonic:
            continue

        ticker = yahoo_xetra_ticker(mnemonic)
        if len(ticker) <= 3:
            continue

        name = clean_text(row[instrument_i]) if instrument_i is not None and instrument_i < len(row) else ""
        isin = clean_text(row[isin_i]) if isin_i is not None and isin_i < len(row) else ""

        rows.append((ticker, name, "XETRA", isin, "Deutsche Börse Xetra"))

    raw.close()

    if not rows:
        return pd.DataFrame(columns=["Ticker", "Name", "Exchange", "ISIN", "Source"])

    out = pd.DataFrame(
        rows,
        columns=["Ticker", "Name", "Exchange", "ISIN", "Source"]
    )
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



@st.cache_data(ttl=1800, show_spinner=False)
def prefilter_by_market_data(universe, target, _progress_callback=None):
    """Stage 0/1: use batched one-year prices for the whole universe.
    This is deliberately not a valuation filter: it keeps both beaten-down and
    recovering names so turnaround and value stories are not systematically removed.
    """
    if universe.empty or target >= len(universe):
        return universe.copy()
    tickers = universe["Ticker"].dropna().astype(str).unique().tolist()
    rows=[]
    chunk_size=80
    started_at = time.time()
    for i in range(0,len(tickers),chunk_size):
        chunk=tickers[i:i+chunk_size]
        batch_started = time.time()
        if _progress_callback:
            _progress_callback(i / max(1, len(tickers)), f"Čekám na Yahoo · dávka {i+1}–{min(i+len(chunk), len(tickers))} · celkem {i:,}/{len(tickers):,}")
        try:
            data=yf.download(chunk, period="1y", interval="1d", auto_adjust=True,
                             progress=False, threads=False, group_by="column")
            close=data["Close"] if isinstance(data,pd.DataFrame) and "Close" in data else pd.DataFrame()
            if isinstance(close,pd.Series):
                close=close.to_frame()
                close.columns=[chunk[0]]
            if close.empty: continue
            for t in close.columns:
                s=pd.to_numeric(close[t],errors="coerce").dropna()
                if len(s)<30: continue
                ret=(float(s.iloc[-1])/float(s.iloc[0])-1)*100 if s.iloc[0]>0 else 0
                vol=float(s.pct_change().std()*100) if len(s)>20 else 0
                rows.append((t,ret,vol,len(s)))
        except Exception as exc:
            if _progress_callback:
                _progress_callback(min(1.0, (i + len(chunk)) / max(1, len(tickers)), f"Dávka {i+1}–{min(i+len(chunk), len(tickers))} nedostupná · pokračuji dál · {time.time()-started_at:.0f} s"))
            continue
        if _progress_callback:
            done = min(i + len(chunk), len(tickers))
            _progress_callback(done / max(1, len(tickers)), f"Zpracováno {done:,} / {len(tickers):,} · poslední dávka {time.time()-batch_started:.1f} s · celkem {time.time()-started_at:.0f} s")
    if _progress_callback:
        _progress_callback(1.0, f"Zpracováno {len(tickers):,} / {len(tickers):,} titulů · dokončuji výběr · celkem {time.time()-started_at:.0f} s")
    md=pd.DataFrame(rows,columns=["Ticker","1Y Return","Volatility","Price Days"])
    if md.empty:
        return build_stage1_candidates(universe,target) if 'build_stage1_candidates' in globals() else universe.head(target).copy()
    out=universe.merge(md,on="Ticker",how="inner")
    # Stratified coverage: equal attention to recovery/momentum, beaten-down names,
    # and ordinary/stable names. This is not a momentum ranking.
    out["Bucket"]=pd.cut(out["1Y Return"],[-np.inf,-30,-10,10,30,np.inf],labels=False)
    selected=[]
    per=max(1,target//5)
    for b in range(5):
        part=out[out["Bucket"]==b].sort_values(["Volatility","Ticker"],ascending=[True,True])
        selected.append(part.head(per))
    chosen=pd.concat(selected,ignore_index=True).drop_duplicates("Ticker")
    if len(chosen)<target:
        rest=out[~out["Ticker"].isin(chosen["Ticker"])].sort_values(["Ticker"])
        chosen=pd.concat([chosen,rest.head(target-len(chosen))],ignore_index=True)

    # IMPORTANT: missing Yahoo price data must not shrink the investment universe.
    # In previous versions a market-data failure caused the whole prefilter to
    # return fewer than `target` names (e.g. 399 instead of 800). That meant a
    # large part of the official universe disappeared before fundamentals were
    # even considered. Fill the remaining slots deterministically from names
    # without usable market data. They continue to the fundamental phase; their
    # price fields are simply unavailable until/if price analysis is performed.
    if len(chosen) < target:
        missing = universe[~universe["Ticker"].isin(out["Ticker"])].copy()
        if not missing.empty:
            missing = build_stage1_candidates(missing, target - len(chosen)) if 'build_stage1_candidates' in globals() else missing.head(target-len(chosen))
            chosen = pd.concat([chosen, missing], ignore_index=True)

    return chosen.drop(columns=["Bucket"],errors="ignore").head(target).reset_index(drop=True)

def build_stage1_candidates(universe, max_stage1):
    """Stage 1: deterministic, broad candidate pool from the full official universe.
    We avoid random sampling: larger exchanges contribute proportionally, with a
    hard cap only to keep Yahoo calls technically manageable.
    """
    if universe.empty:
        return universe.copy()
    if max_stage1 >= len(universe):
        return universe.copy().reset_index(drop=True)

    # Prefer liquid-looking ordinary equities by name/instrument hygiene already
    # enforced upstream, then use a stable hash of ticker for deterministic coverage.
    out = universe.copy()
    out["_stable_key"] = out["Ticker"].map(lambda x: int(hashlib.md5(str(x).encode("utf-8")).hexdigest()[:8], 16))
    out = out.sort_values(["Exchange", "_stable_key"]).reset_index(drop=True)

    # Proportional allocation prevents XETRA/NASDAQ/NYSE from being dominated by
    # the largest venue while remaining deterministic.
    groups = []
    total = len(out)
    exchanges = list(out["Exchange"].dropna().unique())
    raw_quota = {ex: max(1, round(max_stage1 * (len(out[out["Exchange"] == ex]) / total))) for ex in exchanges}
    while sum(raw_quota.values()) > max_stage1:
        ex = max(raw_quota, key=lambda k: raw_quota[k])
        if raw_quota[ex] > 1:
            raw_quota[ex] -= 1
        else:
            break
    while sum(raw_quota.values()) < max_stage1:
        ex = max(exchanges, key=lambda k: len(out[out["Exchange"] == k]) - raw_quota[k])
        raw_quota[ex] += 1

    for ex in exchanges:
        part = out[out["Exchange"] == ex]
        groups.append(part.head(min(raw_quota[ex], len(part))))
    result = pd.concat(groups, ignore_index=True) if groups else out.head(max_stage1)
    return result.drop(columns=["_stable_key"], errors="ignore").reset_index(drop=True)


def rank_stage2_candidates(df, stage2_limit):
    """Stage 2: select names where the available fundamentals suggest a change,
    quality/value asymmetry or recovery. This ranking is deliberately broad.
    """
    if df.empty:
        return df.copy()
    out = df.copy()
    def n(x): return safe_float(x)

    def rank_row(r):
        v = q = g = 0.0
        # Cheap/value asymmetry
        pe = n(r.get("P/E")); fpe = n(r.get("Forward P/E")); ps = n(r.get("P/S"))
        if not pd.isna(pe) and pe > 0: v += max(0, min(25, (25-pe)))
        if not pd.isna(fpe) and fpe > 0: v += max(0, min(20, (25-fpe)))
        if not pd.isna(ps) and ps > 0: v += max(0, min(15, (8-ps)*2))
        # Quality
        roe = n(r.get("ROE")); fcf = n(r.get("Free Cash Flow")); de = n(r.get("Debt/Equity"))
        if not pd.isna(roe) and roe > 10: q += min(20, roe/2)
        if not pd.isna(fcf) and fcf > 0: q += 10
        if not pd.isna(de) and de < 150: q += 10
        # Change / recovery signals
        for key, weight in [("Revenue Prior YoY", 0.8), ("Net Income Prior YoY", 1.2)]:
            x = n(r.get(key))
            if not pd.isna(x) and x < 0: g += min(18, abs(x)*weight)
        for key, weight in [("Revenue Growth", 0.8), ("Earnings Growth", 1.0), ("Margin Change 3Y", 2.0)]:
            x = n(r.get(key))
            if not pd.isna(x) and x > 0: g += min(18, x*weight)
        return min(100, v+q+g)

    out["Stage 2 Priority"] = out.apply(rank_row, axis=1)
    return out.sort_values("Stage 2 Priority", ascending=False).head(stage2_limit).reset_index(drop=True)

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

def company_archetype(r):
    """Rough economic archetype used to prevent applying the same story logic to every company."""
    sector = clean_text(r.get("Sector")).lower()
    industry = clean_text(r.get("Industry")).lower()
    name = clean_text(r.get("Name")).lower()
    s = f"{sector} {industry} {name}"

    if "reit" in s or "real estate investment trust" in s:
        return "REIT / real estate"
    if any(x in s for x in ["investment holding", "holding company", "investment company", "closed-end fund"]):
        return "Investment holding"
    if any(x in s for x in ["asset management", "wealth management", "investment management", "capital markets"]):
        return "Asset manager / capital markets"
    if any(x in s for x in ["bank", "insurance", "credit", "mortgage", "financial institution"]):
        return "Financial institution"
    if any(x in s for x in ["utility", "utilities", "electric utilities", "regulated electric", "regulated gas"]):
        return "Utility"
    if any(x in s for x in ["oil", "gas", "coal", "mining", "metals", "steel", "commodity"]):
        return "Commodity / resource"
    if any(x in s for x in ["automotive", "auto parts", "machinery", "construction", "building materials",
                            "aerospace", "defense", "industrial", "transportation", "chemicals", "paper"]):
        return "Cyclical industrial"
    if any(x in s for x in ["biotechnology", "biotech", "pharmaceutical", "drug manufacturers", "drug manufacturing"]):
        return "Pharma / biotech"
    if any(x in s for x in ["software", "semiconductor", "technology", "electronic components", "internet",
                            "information technology", "communication equipment"]):
        return "Technology / high growth"
    if any(x in s for x in ["consumer defensive", "household", "food", "beverage", "tobacco", "personal products"]):
        return "Consumer defensive"
    if any(x in s for x in ["quantum", "space", "hologram", "microcap", "nanotechnology"]):
        return "Speculative / deep tech"
    if any(x in s for x in ["infrastructure", "telecom", "pipeline"]):
        return "Infrastructure"
    return "Other operating company"


# Backward-compatible alias for older diagnostic columns.
def company_type(r):
    return company_archetype(r)


def fundamental_direction(r):
    """Classify what is happening to the business before assigning an investment story."""
    rg = safe_float(r.get("Revenue Growth")); eg = safe_float(r.get("Earnings Growth"))
    prior_rg = safe_float(r.get("Revenue Prior YoY")); prior_eg = safe_float(r.get("Net Income Prior YoY"))
    rc = safe_float(r.get("Revenue CAGR 3Y")); ni_cagr = safe_float(r.get("Net Income CAGR 3Y"))
    mc = safe_float(r.get("Margin Change 3Y")); fcf = safe_float(r.get("Free Cash Flow"))
    sign_recovery = bool(r.get("Net Income Sign Recovery", False))

    prior_problem = 0; current_improvement = 0; evidence = []
    if not pd.isna(prior_eg) and prior_eg < -5:
        prior_problem += 30; evidence.append("předchozí pokles zisku")
    elif not pd.isna(prior_eg) and prior_eg < 0:
        prior_problem += 20; evidence.append("mírný předchozí pokles zisku")
    if not pd.isna(prior_rg) and prior_rg < -5:
        prior_problem += 20; evidence.append("předchozí pokles tržeb")
    elif not pd.isna(prior_rg) and prior_rg < 0:
        prior_problem += 12; evidence.append("mírný předchozí pokles tržeb")
    if sign_recovery:
        prior_problem += 35; evidence.append("návrat ze ztráty do zisku")
    if not pd.isna(rc) and rc < 0:
        prior_problem += 10; evidence.append("tržby mají záporný 3Y trend")
    if not pd.isna(ni_cagr) and ni_cagr < 0:
        prior_problem += 10; evidence.append("zisk má záporný 3Y trend")

    if not pd.isna(eg) and eg >= 10:
        current_improvement += 25; evidence.append("aktuální růst zisku")
    elif not pd.isna(eg) and eg > 0:
        current_improvement += 10; evidence.append("zisk se zlepšuje")
    if not pd.isna(rg) and rg >= 3:
        current_improvement += 20; evidence.append("aktuální růst/stabilizace tržeb")
    elif not pd.isna(rg) and rg >= 0:
        current_improvement += 8; evidence.append("tržby neklesají")
    if not pd.isna(mc) and mc >= 3:
        current_improvement += 20; evidence.append("výrazné zlepšení marže")
    elif not pd.isna(mc) and mc >= 1:
        current_improvement += 10; evidence.append("zlepšení marže")
    if not pd.isna(fcf) and fcf > 0:
        current_improvement += 10; evidence.append("kladný FCF")

    # A turnaround needs both a prior problem and a current improvement.
    if prior_problem >= 35 and current_improvement >= 25:
        score = min(100, prior_problem + current_improvement)
        return "🔄 Recovery / obrat", float(score), "; ".join(evidence)

    # Margin/operating improvement without a sufficiently deep prior problem.
    if current_improvement >= 25 and (not pd.isna(mc) and mc >= 2):
        return "🛠️ Operational improvement", float(min(85, current_improvement)), "; ".join(evidence)

    # Strong growth without prior deterioration is growth, not turnaround.
    if (not pd.isna(rg) and rg >= 10) or (not pd.isna(eg) and eg >= 15):
        return "📈 Growth", float(min(85, current_improvement)), "; ".join(evidence) if evidence else "silný aktuální růst"

    if prior_problem >= 30 and current_improvement < 20:
        return "📉 Deterioration / weak recovery", float(max(0, prior_problem - 15)), "; ".join(evidence)
    if current_improvement >= 15:
        return "➡️ Stabilizace", float(current_improvement), "; ".join(evidence)
    return "⚪ Nejasný směr", 0.0, "; ".join(evidence) or "nedostatek trendových signálů"



def recovery_gates(r):
    """Separate generic recovery from a genuine operating turnaround."""
    direction = clean_text(r.get("Fundamental Direction")); archetype = clean_text(r.get("Company Archetype"))
    prior_eg = safe_float(r.get("Net Income Prior YoY")); prior_rg = safe_float(r.get("Revenue Prior YoY"))
    ni_cagr = safe_float(r.get("Net Income CAGR 3Y")); rev_cagr = safe_float(r.get("Revenue CAGR 3Y"))
    mc = safe_float(r.get("Margin Change 3Y")); eg = safe_float(r.get("Earnings Growth")); rg = safe_float(r.get("Revenue Growth"))
    fcf = safe_float(r.get("Free Cash Flow")); sign = bool(r.get("Net Income Sign Recovery", False))
    cyclical = archetype in ("Commodity / resource", "Cyclical industrial")
    asset_recovery = archetype in ("REIT / real estate", "Investment holding", "Asset manager / capital markets", "Financial institution")
    technology = archetype == "Technology / high growth"
    problem = ((not pd.isna(prior_eg) and prior_eg < -5) or (not pd.isna(prior_rg) and prior_rg < -5) or sign or (not pd.isna(ni_cagr) and ni_cagr < -5))
    bottom = (sign or (not pd.isna(prior_eg) and prior_eg < 0 and not pd.isna(eg) and eg > 0) or (not pd.isna(prior_rg) and prior_rg < 0 and not pd.isna(rg) and rg >= 0) or (not pd.isna(mc) and mc >= 3))
    improvement = ((not pd.isna(eg) and eg > 5) or (not pd.isna(rg) and rg > 3) or (not pd.isna(mc) and mc >= 2) or (not pd.isna(fcf) and fcf > 0 and sign))
    persistence = ((not pd.isna(mc) and mc >= 2) and ((not pd.isna(eg) and eg > 0) or (not pd.isna(rg) and rg >= 0)))
    # Severity is deliberately based on the business, not on the share price.
    # A large price fall is useful later as market context, but it is not proof
    # that the underlying business suffered a serious operational problem.
    severe_profit = ((not pd.isna(ni_cagr) and ni_cagr <= -15) or
                     (not pd.isna(prior_eg) and prior_eg <= -20))
    severe_revenue = ((not pd.isna(rev_cagr) and rev_cagr <= -15) or
                      (not pd.isna(prior_rg) and prior_rg <= -10))
    severe_margin = not pd.isna(mc) and mc <= -8
    severity = bool(severe_profit or severe_revenue or severe_margin)

    # Quantitative mechanism proxy: improvement should have an economic source,
    # not merely a one-period earnings rebound. Text evidence is checked later
    # and remains evidence rather than a hard gate for candidate generation.
    mechanism = bool(
        ((not pd.isna(mc) and mc >= 2) and
         ((not pd.isna(rg) and rg >= 0) or (not pd.isna(eg) and eg > 0) or
          (not pd.isna(fcf) and fcf > 0)))
        or
        ((not pd.isna(rg) and rg >= 3) and (not pd.isna(eg) and eg > 0))
    )

    gates = {"Prior Problem": bool(problem), "Bottom / Stabilization": bool(bottom), "Current Improvement": bool(improvement),
             "Persistence": bool(persistence), "Mechanism / economics": mechanism,
             "Cyclical": bool(cyclical), "Asset / financial": bool(asset_recovery),
             "Technology": bool(technology), "Severity / business": severity}
    score = 25*problem + 15*bottom + 25*improvement + 15*persistence + 20*mechanism
    if not severity: score -= 30
    if cyclical: score -= 15
    if asset_recovery: score -= 20
    if technology: score -= 15
    score = max(0, min(100, score))
    if cyclical and improvement: label = "🔵 Cyklické zotavení"
    elif asset_recovery and improvement: label = "🏢 Aktivové / finanční zotavení"
    elif technology and improvement: label = "🟣 Růstové zotavení, ne klasický turnaround"
    elif problem and bottom and improvement and persistence and mechanism and severity: label = "🟢 Silná struktura turnaroundu"
    elif problem and improvement: label = "🟡 Zotavení – chybí závažnost nebo mechanismus"
    else: label = "⚪ Nedostatek důkazů o turnaroundu"
    return label, float(score), gates

def turnaround_score(r):
    direction, score, evidence = fundamental_direction(r)
    archetype = clean_text(r.get("Company Archetype")); gate = clean_text(r.get("Posouzení zotavení"))
    excluded = ("Commodity / resource", "Cyclical industrial", "REIT / real estate", "Investment holding", "Asset manager / capital markets", "Financial institution", "Technology / high growth")
    if direction != "🔄 Recovery / obrat": return 0.0, evidence or "bez prokázaného obratu"
    if archetype in excluded: return 0.0, evidence or "zotavení jiného typu než klasický provozní turnaround"
    if gate != "🟢 Silná struktura turnaroundu": return 0.0, evidence or "nedostatečná závažnost pro klasický turnaround"
    return min(100.0, score), evidence

def classify_story(r):
    v,q,g = r["Value Score"],r["Quality Score"],r["Growth Score"]
    archetype = r["Company Archetype"]; direction = r["Fundamental Direction"]; ts = r["Turnaround Score"]
    gate_label = clean_text(r.get("Posouzení zotavení"))
    if archetype in ("Investment holding", "Asset manager / capital markets", "Financial institution", "REIT / real estate"):
        if direction == "🔄 Recovery / obrat": return "🏗️ Asset / financial recovery"
        if archetype == "REIT / real estate" and q >= 60 and v >= 55: return "🏢 Real-estate value"
    if archetype in ("Commodity / resource", "Cyclical industrial") and direction == "🔄 Recovery / obrat": return "🌐 Cyclical / commodity recovery"
    if archetype == "Technology / high growth" and direction == "🔄 Recovery / obrat": return "🚀 Growth / recovery" if g >= 60 else "🛠️ Operational improvement"
    if ts >= 75 and gate_label == "🟢 Silná struktura turnaroundu": return "🔄 Operating turnaround"
    if direction == "🔄 Recovery / obrat" and ts >= 55: return "🔄 Recovery candidate"
    if direction == "🛠️ Operational improvement":
        if q >= 65 and g >= 45: return "🏆 Quality Compounder" if g >= 65 else "💎 Kvalita za rozumnou cenu"
        return "🛠️ Operational improvement"
    if not pd.isna(v) and not pd.isna(q) and not pd.isna(g):
        if v >= 70 and q < 50 and g < 50: return "🪤 Value Trap – varování"
        if q >= 70 and g >= 65 and v >= 45: return "🏆 Quality Compounder"
        if q >= 70 and v >= 60 and g >= 45: return "💎 Kvalita za rozumnou cenu"
        if g >= 70 and v >= 50 and q >= 50: return "🚀 Růst za rozumnou cenu"
        if v >= 65 and q >= 50 and g < 50: return "💰 Value / levná firma"
        if g >= 60 and q < 50 and v < 50: return "🔥 High Growth / dražší příběh"
        if v < 40 and q < 50 and g < 50: return "⚠️ Slabý fundamentální obraz"
    return "🔎 Smíšený příběh"

def investment_attractiveness(r):
    """Obecná investiční atraktivita: valuation + quality + growth.
    Není to doporučení ani odhad budoucího výnosu; slouží jen k pořadí kandidátů.
    """
    v = safe_float(r.get("Value Score")); q = safe_float(r.get("Quality Score")); g = safe_float(r.get("Growth Score"))
    vals = [x for x in (v, q, g) if not pd.isna(x)]
    if not vals:
        return np.nan
    # Value gets slightly higher weight because the purpose is to prioritize
    # candidates worth opening at today's price, not simply the fastest growers.
    weights = [(v, 0.40), (q, 0.35), (g, 0.25)]
    num = sum(x*w for x,w in weights if not pd.isna(x)); den = sum(w for x,w in weights if not pd.isna(x))
    return round(num/den, 1) if den else np.nan


def story_fit(r, selected_story):
    """How closely the company matches the story currently being searched.
    Turnaround 1.0 is deliberately stricter than the old categorical Story label.
    """
    if not selected_story:
        return np.nan
    v,q,g = safe_float(r.get("Value Score")), safe_float(r.get("Quality Score")), safe_float(r.get("Growth Score"))
    ts = safe_float(r.get("Turnaround Score")); gate = clean_text(r.get("Posouzení zotavení"))
    direction = clean_text(r.get("Fundamental Direction")); archetype = clean_text(r.get("Company Archetype"))
    price = safe_float(r.get("Skóre ceny"))

    if selected_story == "🔄 Operating turnaround":
        # 35 prior problem, 20 stabilization, 25 current improvement, 20 persistence.
        gates = r.get("Recovery Gates", {})
        if isinstance(gates, dict):
            problem = bool(gates.get("Prior Problem")); bottom = bool(gates.get("Bottom / Stabilization"))
            improvement = bool(gates.get("Current Improvement")); persistence = bool(gates.get("Persistence"))
        else:
            problem = bottom = improvement = persistence = False
            gates = {}
        score = 0
        score += 35 if problem else 0
        score += 20 if bottom else 0
        score += 25 if improvement else 0
        score += 20 if persistence else 0
        score += 10 if bool(gates.get("Mechanism / economics")) else 0
        score -= 15 if not bool(gates.get("Severity / business")) else 0
        if archetype in ("Commodity / resource", "Cyclical industrial", "Technology / high growth",
                         "REIT / real estate", "Investment holding", "Asset manager / capital markets",
                         "Financial institution"):
            score -= 35
        if direction != "🔄 Recovery / obrat":
            score -= 20
        if not pd.isna(ts):
            score = 0.70*score + 0.30*ts
        return round(max(0, min(100, score)), 1)

    if selected_story == "🌐 Cyclical / commodity recovery":
        base = 0
        base += 35 if archetype == "Commodity / resource" else 20 if "Cyclical" in archetype else 0
        base += 30 if direction == "🔄 Recovery / obrat" else 15 if direction == "➡️ Stabilizace" else 0
        base += 20 if (not pd.isna(g) and g >= 50) else 10 if not pd.isna(g) else 0
        base += 15 if (not pd.isna(price) and price >= 50) else 0
        return float(min(100, base))

    mapping = {
        "🏆 Quality Compounder": (q, g, v),
        "💎 Kvalita za rozumnou cenu": (q, v, g),
        "🚀 Růst za rozumnou cenu": (g, q, v),
        "💰 Value / levná firma": (v, q, g),
        "🔄 Recovery candidate": (ts, q, v),
        "🌐 Cyclical / commodity recovery": (g, q, v),
        "🏗️ Asset / financial recovery": (q, v, ts),
        "🏢 Real-estate value": (v, q, g),
        "🛠️ Operational improvement": (q, g, v),
        "🚀 Growth / recovery": (g, q, v),
        "🔥 High Growth / dražší příběh": (g, q, v),
        "🪤 Value Trap – varování": (v, 100-(q or 0), 100-(g or 0)),
    }
    vals = mapping.get(selected_story)
    if not vals:
        # For the remaining recovery stories use the fundamental direction first.
        if selected_story in ("🔄 Recovery candidate", "🚀 Growth / recovery"):
            base = ts if selected_story == "🔄 Recovery candidate" else g
            if pd.isna(base): return np.nan
            return round(float(base), 1)
        return np.nan
    clean = [x for x in vals if not pd.isna(x)]
    return round(sum(clean)/len(clean), 1) if clean else np.nan


def story_priority(r, selected_story=None):
    """Pořadí kandidátů = 60 % shoda s příběhem + 40 % investiční atraktivita."""
    fit = story_fit(r, selected_story) if selected_story else np.nan
    attr = investment_attractiveness(r)
    vals = []
    if not pd.isna(fit): vals.append((fit, 0.60))
    if not pd.isna(attr): vals.append((attr, 0.40))
    if not vals:
        return np.nan
    return round(sum(v*w for v,w in vals) / sum(w for v,w in vals), 1)

# -----------------------------------------------------------------------------
# V5 – Text Evidence Engine
# Druhá fáze: pouze pro úzký výběr kandidátů. Nepoužívá se pro celé univerzum.
# -----------------------------------------------------------------------------

TEXT_RULES = {
    "🔄 Operating turnaround": {
        "positive": {"turnaround":3,"recovery":2,"restructuring":3,"cost reduction":2,"cost savings":2,"margin recovery":3,"return to profitability":3,"operational improvement":2,"deleveraging":2,"new management":2,"strategic review":1,"transformation":1,"profitability improved":2,"cash flow improved":2,"debt reduction":2},
        "negative": {"continued decline":3,"deterioration":3,"liquidity pressure":3,"covenant breach":3,"going concern":3,"cash burn":2,"declining demand":2,"margin pressure":2,"failed turnaround":3,"restructuring costs":2}
    },
    "🔄 Recovery candidate": {
        "positive": {"recovery":2,"turnaround":2,"restructuring":3,"cost reduction":2,"margin recovery":3,"return to profitability":3,"operational improvement":2,"deleveraging":2,"new management":2,"profitability improved":2,"cash flow improved":2,"debt reduction":2},
        "negative": {"continued decline":3,"deterioration":3,"liquidity pressure":3,"going concern":3,"cash burn":2,"declining demand":2,"margin pressure":2}
    },
    "🌐 Cyclical / commodity recovery": {
        "positive": {"commodity prices":3,"pricing":2,"demand recovery":2,"cycle":2,"cyclical recovery":3,"margins":2,"utilization":2,"production growth":2},
        "negative": {"oversupply":3,"weak pricing":3,"lower commodity prices":3,"demand destruction":3}
    },
    "🚀 Growth / recovery": {
        "positive": {"recovery":2,"accelerating growth":3,"organic growth":2,"market share":2,"new products":2,"pricing power":2,"return to growth":3,"profitability improved":2},
        "negative": {"slowing growth":3,"declining demand":2,"guidance cut":3,"cash burn":3,"dilution":2}
    },
    "🏗️ Asset / financial recovery": {
        "positive": {"net asset value":3,"nav per share":3,"discount to nav":3,"portfolio value":2,"fair value":2,"monetization":2,"realization":2,"asset value":2,"recovery":2,"investment gains":2,"portfolio gains":2,"deleveraging":2},
        "negative": {"impairment":2,"write-down":3,"liquidity pressure":3,"discount widened":3,"portfolio loss":2,"realization risk":2}
    },
    "🏢 Real-estate value": {
        "positive": {"net asset value":3,"nav":2,"occupancy":2,"rent growth":2,"same-store noi":3,"noi growth":3,"leasing spread":2,"asset value":2,"discount to nav":3},
        "negative": {"vacancy":2,"occupancy decline":3,"rent decline":2,"impairment":2,"refinancing risk":3,"higher interest expense":2}
    },
    "🚀 Růst za rozumnou cenu": {
        "positive": {"organic growth":3,"accelerating growth":3,"market share":2,"capacity expansion":2,"backlog":2,"bookings growth":2,"demand growth":2,"new markets":2,"international expansion":2,"pricing power":2},
        "negative": {"slowing growth":3,"declining demand":2,"market share loss":3,"competitive pressure":2,"guidance cut":3,"weak bookings":2}
    },
    "🏆 Quality Compounder": {
        "positive": {"recurring revenue":3,"recurring cash flow":3,"pricing power":3,"competitive advantage":3,"market leadership":2,"high margins":2,"free cash flow":1,"long-term growth":2,"capital allocation":2,"customer retention":2,"strong balance sheet":2},
        "negative": {"customer churn":3,"margin pressure":2,"competitive pressure":2,"market share loss":3,"cash burn":3,"weak balance sheet":3}
    },
    "💎 Kvalita za rozumnou cenu": {
        "positive": {"undervalued":3,"attractive valuation":3,"discount to peers":2,"free cash flow":2,"pricing power":2,"competitive advantage":2,"capital return":2,"share buyback":2,"strong balance sheet":2},
        "negative": {"overvalued":3,"valuation premium":2,"margin pressure":2,"competitive pressure":2,"declining demand":2}
    },
    "💰 Value / levná firma": {
        "positive": {"undervalued":3,"intrinsic value":3,"discount to peers":2,"asset value":2,"sum of the parts":3,"share buyback":2,"capital return":2,"non-core assets":1,"monetization":2},
        "negative": {"structural decline":3,"secular decline":3,"excess capacity":2,"debt burden":3,"liquidity pressure":3,"cash burn":3,"declining market share":3,"impairment":2}
    },
    "🔥 High Growth / dražší příběh": {
        "positive": {"accelerating growth":3,"organic growth":2,"market share":2,"expansion":1,"backlog":2,"pipeline":1,"new markets":2},
        "negative": {"overvalued":3,"valuation premium":2,"cash burn":3,"dilution":2,"slowing growth":3}
    },
    "🪤 Value Trap – varování": {
        "positive": {"structural decline":3,"secular decline":3,"declining market share":3,"excess capacity":2,"debt burden":3,"cash burn":3,"competitive pressure":2,"liquidity pressure":3,"impairment":2},
        "negative": {"turnaround":2,"recovery":2,"margin recovery":2,"return to profitability":3,"strong balance sheet":2}
    }
}
NEGATION_WORDS = {"not","no","without","unlikely","failed","fails","fail","never","neither"}

def _known_text_phrases():
    phrases=[]
    for groups in GENERAL_TEXT_RULES.values():
        for vals in groups.values():
            phrases.extend(vals)
    for rules in TEXT_RULES.values():
        for bucket in ("positive", "negative"):
            phrases.extend(rules.get(bucket, {}).keys())
    phrases += [
        "is bce stock worth buying", "execution risk", "bce stock",
        "return to profitability", "free cash flow", "cash flow",
        "strong balance sheet", "higher financing costs", "ai and fiber",
        "asset sales", "rental growth", "funds from operations",
        "heavy ai and fiber spending", "financing costs", "operating results"
    ]
    return sorted(set(p.lower() for p in phrases if p), key=len, reverse=True)


def _spaced_phrase_pattern(phrase):
    """Regex for a known phrase whose characters may be separated by whitespace."""
    chars=[re.escape(ch) for ch in phrase.lower() if ch.isalnum()]
    if not chars:
        return None
    return r"(?<![A-Za-z0-9])" + r"\s*".join(chars) + r"(?![A-Za-z0-9])"


def repair_known_spaced_phrases(s):
    """Repair known phrases BEFORE generic letter-run collapsing."""
    if not s:
        return s
    for phrase in _known_text_phrases():
        pattern=_spaced_phrase_pattern(phrase)
        if pattern:
            s=re.sub(pattern, phrase, s, flags=re.I)
    return s


def collapse_letter_spaced_text(s):
    """Collapse residual single-letter runs after known phrases are repaired."""
    if not s:
        return s
    pattern=r"(?<![A-Za-z0-9])(?:[A-Za-z]\s+){3,}[A-Za-z](?![A-Za-z0-9])"
    def repl(m):
        return re.sub(r"\s+", "", m.group(0))
    return re.sub(pattern, repl, s)


def normalize_text_evidence_output(x):
    """Final safety pass: displayed evidence must not contain feed-level letter spacing."""
    if x is None:
        return ""
    s=unescape(clean_text(x))
    s=re.sub(r"<[^>]+>", " ", s)
    s=repair_known_spaced_phrases(s)
    s=collapse_letter_spaced_text(s)
    s=repair_known_spaced_phrases(s)
    return re.sub(r"\s+", " ", s).strip()


def text_clean(x):
    return normalize_text_evidence_output(x).lower()

def sentence_chunks(text):
    text = text_clean(text)
    if not text: return []
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+", text) if x.strip()]

def keyword_context(sentence, phrase):
    pos = sentence.find(phrase)
    if pos < 0: return False, sentence[:240]
    before = sentence[max(0,pos-100):pos]
    words = re.findall(r"[a-z]+", before)
    negated = any(w in NEGATION_WORDS for w in words[-8:])
    start=max(0,pos-70); end=min(len(sentence),pos+len(phrase)+110)
    snippet=sentence[start:end]
    if start>0: snippet="…"+snippet
    if end<len(sentence): snippet+="…"
    return negated,snippet

def company_identity_terms(name, ticker):
    words = [w for w in re.findall(r"[a-z0-9]+", text_clean(name)) if len(w) >= 3]
    stop={"inc","corp","corporation","company","plc","limited","ltd","ag","ordinary","shares","common","holdings","class","the"}
    words=[w for w in words if w not in stop]
    return words[:4], text_clean(ticker).replace(".de","")

def sentence_is_company_relevant(sentence, name, ticker):
    words, tick = company_identity_terms(name, ticker)
    s=text_clean(sentence)
    if tick and re.search(r"(?<![a-z0-9])"+re.escape(tick)+r"(?![a-z0-9])",s):
        return True
    if not words: return True
    # A single distinctive name word is sufficient for news headlines; for generic
    # names require two words.
    hits=sum(1 for w in words if re.search(r"(?<![a-z0-9])"+re.escape(w)+r"(?![a-z0-9])",s))
    return hits >= (2 if len(words)>=2 else 1)

GENERAL_TEXT_RULES = {
    "problem": {
        "demand weakness": ["declining demand", "weak demand", "soft demand", "demand weakness", "lower demand", "demand slowdown"],
        "margin pressure": ["margin pressure", "margin compression", "compressed margins", "gross margin declined", "operating margin declined"],
        "cost pressure": ["cost inflation", "higher costs", "cost pressure", "input costs", "labor costs", "freight costs"],
        "debt/liquidity": ["high debt", "debt burden", "liquidity pressure", "covenant", "refinancing risk", "cash burn", "liquidity concerns"],
        "restructuring need": ["restructuring", "reorganization", "turnaround plan", "restructuring plan", "cost reduction plan"],
        "competitive pressure": ["market share loss", "competitive pressure", "pricing pressure", "lost customers", "customer churn"],
        "cyclical weakness": ["cyclical downturn", "industry downturn", "downcycle", "sector downturn", "weak cycle", "commodity prices"],
    },
    "change": {
        "demand recovery": ["demand recovery", "demand improved", "demand improving", "orders recovered", "order recovery", "return to growth", "returning demand"],
        "margin recovery": ["margin recovery", "margin expansion", "margins improved", "margin improved", "gross margin improved", "operating margin improved", "profit margin improved"],
        "cost improvement": ["cost savings", "cost reduction", "cost cutting", "lower costs", "efficiency gains", "operating efficiencies"],
        "profit recovery": ["return to profitability", "returned to profitability", "profitability improved", "earnings recovery", "earnings improved", "profit improved", "loss narrowed"],
        "cash-flow improvement": ["free cash flow improved", "cash flow improved", "positive free cash flow", "cash generation improved", "cash flow turned positive"],
        "balance-sheet improvement": ["deleveraging", "debt reduction", "reduced debt", "balance sheet improved", "liquidity improved"],
        "pricing improvement": ["pricing power", "pricing improved", "price increases", "better pricing", "pricing actions"],
    },
    "mechanism": {
        "restructuring": ["restructuring plan", "restructuring program", "reorganization plan", "turnaround plan", "transformation plan"],
        "cost program": ["cost reduction program", "cost savings program", "cost cutting program", "efficiency program", "productivity program"],
        "management change": ["new ceo", "new chief executive", "new cfo", "new management", "management change", "appointed ceo", "appointed cfo"],
        "asset/division action": ["divestiture", "divested", "asset sale", "sold non-core", "sale of non-core", "exit from", "exited the business"],
        "pricing/mix": ["pricing actions", "price increases", "product mix", "favorable mix", "improved mix"],
        "demand/cycle": ["demand recovery", "volume recovery", "market recovery", "industry recovery", "cycle recovery", "cyclical recovery"],
        "refinancing": ["refinancing", "refinanced", "debt maturity", "debt restructuring"],
        "new product/market": ["new product", "product launch", "new market", "market expansion", "capacity expansion"],
    },
    "risk": {
        "structural decline": ["structural decline", "secular decline", "secular pressure", "business model risk", "technology disruption"],
        "demand still weak": ["demand remains weak", "weak demand persists", "demand continued to decline", "declining demand"],
        "margin risk": ["margin pressure", "margin remains under pressure", "gross margin declined", "pricing pressure"],
        "cash/debt risk": ["cash burn", "negative free cash flow", "liquidity pressure", "covenant breach", "refinancing risk", "high debt"],
        "guidance risk": ["guidance cut", "lowered guidance", "reduced outlook", "weak outlook", "outlook deteriorated"],
        "competitive risk": ["market share loss", "competitive pressure", "customer churn", "lost customers"],
        "execution risk": ["execution risk", "turnaround risk", "restructuring risk", "plan is not working", "failed turnaround"],
    }
}


def _scan_general_signals(chunks, name, ticker):
    # Keep the result grouped by label so the downstream formatter can
    # distinguish e.g. "margin pressure" from "demand weakness".
    buckets={k:{label:[] for label in groups} for k, groups in GENERAL_TEXT_RULES.items()}
    for sent in chunks:
        # Business summary is already company-specific; news/RSS sentences must pass identity.
        for bucket, groups in GENERAL_TEXT_RULES.items():
            for label, phrases in groups.items():
                for phrase in phrases:
                    if phrase in sent:
                        negated,snippet=keyword_context(sent, phrase)
                        if bucket in ("problem","risk") and negated:
                            continue
                        if bucket in ("change","mechanism") and negated:
                            continue
                        item=f"{label}: {snippet}"
                        if item not in buckets[bucket][label]:
                            buckets[bucket][label].append(item)
                        break
    return buckets


def score_text_evidence(text, story):
    rules=TEXT_RULES.get(story)
    chunks=sentence_chunks(text)
    general=_scan_general_signals(chunks, "", "")
    positive_score=negative_score=0; support=[]; warnings=[]; seen=set()

    # Story-specific phrases remain useful, but no longer dominate the interpretation.
    if rules:
        for phrase,weight in rules.get("positive",{}).items():
            for sent in chunks:
                if phrase not in sent: continue
                negated,snippet=keyword_context(sent,phrase)
                key=("+",phrase,snippet[:160])
                if not negated and key not in seen:
                    seen.add(key); positive_score += weight; support.append(f"{phrase}: {snippet}")
        for phrase,weight in rules.get("negative",{}).items():
            for sent in chunks:
                if phrase not in sent: continue
                negated,snippet=keyword_context(sent,phrase)
                key=("-",phrase,snippet[:160])
                if not negated and key not in seen:
                    seen.add(key); negative_score += weight; warnings.append(f"{phrase}: {snippet}")

    # General mechanism/change evidence is more valuable than a generic occurrence of 'recovery'.
    mechanism_hits = sum(len(v) for v in general["mechanism"].values())
    change_hits = sum(len(v) for v in general["change"].values())
    problem_hits = sum(len(v) for v in general["problem"].values())
    risk_hits = sum(len(v) for v in general["risk"].values())

    score = 50 + positive_score*5 + change_hits*5 + mechanism_hits*7 - negative_score*6 - risk_hits*8
    score=float(max(0,min(100,score)))

    # Human-readable structured evidence. Deduplicate by label/snippet.
    def flatten(bucket, limit=6):
        out=[]; seen2=set()
        for label,items in bucket.items():
            for item in items:
                if item not in seen2:
                    seen2.add(item); out.append(f"• {item}")
                if len(out)>=limit: return out
        return out

    problem_lines=flatten(general["problem"],5)
    change_lines=flatten(general["change"],5)
    mechanism_lines=flatten(general["mechanism"],5)
    risk_lines=flatten(general["risk"],5)

    structured=[]
    if problem_lines: structured.append("**Předchozí problém**\n"+"\n".join(problem_lines))
    if change_lines: structured.append("**Aktuální změna**\n"+"\n".join(change_lines))
    if mechanism_lines: structured.append("**Mechanismus změny**\n"+"\n".join(mechanism_lines))
    support_text="\n\n".join(structured)
    warning_text="\n".join(risk_lines + [f"• {x}" for x in warnings[:5]])

    if not problem_lines and not change_lines and not mechanism_lines and not warnings and not risk_lines:
        label="⚪ Bez textového důkazu"
    elif mechanism_hits>0 and change_hits>0 and risk_hits==0:
        label="🟢 Text popisuje změnu a její mechanismus"
    elif change_hits>0 or mechanism_hits>0:
        label="🟡 Text naznačuje změnu, mechanismus není úplný"
    elif risk_hits>0:
        label="🟠 Text přináší hlavně rizika"
    else:
        label="🟠 Text je smíšený"

    support_text=normalize_text_evidence_output(support_text)
    warning_text=normalize_text_evidence_output(warning_text)
    label=normalize_text_evidence_output(label)
    return score,label,positive_score+change_hits,negative_score+risk_hits,support_text,warning_text


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_google_news(company_name, ticker):
    """Company-focused RSS search. Titles are filtered again for identity."""
    queries=[
        f'"{company_name}" recovery turnaround restructuring margin',
        f'"{company_name}" cost cutting demand pricing profitability',
        f'"{company_name}" debt refinancing cash flow outlook',
    ]
    parts=[]
    for q in queries:
        try:
            url="https://news.google.com/rss/search"
            r=requests.get(url,params={"q":q,"hl":"en-US","gl":"US","ceid":"US:en"},timeout=12,
                           headers={"User-Agent":"Mozilla/5.0"})
            r.raise_for_status()
            xml=r.text
            items=re.findall(r"<item>(.*?)</item>",xml,re.S|re.I)
            for item in items[:10]:
                title=re.search(r"<title>(.*?)</title>",item,re.S|re.I)
                pub=re.search(r"<pubDate>(.*?)</pubDate>",item,re.S|re.I)
                if title:
                    t=unescape(re.sub(r"<[^>]+>"," ",title.group(1)))
                    if sentence_is_company_relevant(t,company_name,ticker):
                        parts.append(t + (f" ({pub.group(1)})" if pub else ""))
        except Exception:
            continue
    return " ".join(parts[:20])

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_text_evidence(yahoo_ticker, story, company_name="", ticker=""):
    """Evidence layer: company-specific business summary + identity-filtered news.
    Google News is used to find analyst/news labels; Yahoo news is retained only
    when the sentence/title is demonstrably about the company.
    """
    try:
        parts=[]
        try:
            info=yf.Ticker(yahoo_ticker).info or {}
            summary=info.get("longBusinessSummary")
            if summary: parts.append(clean_text(summary))
            sector=info.get("sector"); industry=info.get("industry")
            if sector: parts.append(clean_text(sector))
            if industry: parts.append(clean_text(industry))
        except Exception:
            pass

        try:
            news=yf.Ticker(yahoo_ticker).news or []
            for item in news[:15]:
                content=item.get("content",item) if isinstance(item,dict) else {}
                title=content.get("title") if isinstance(content,dict) else None
                summary=content.get("summary") if isinstance(content,dict) else None
                for val in (title,summary):
                    if val and sentence_is_company_relevant(val,company_name,ticker):
                        parts.append(clean_text(val))
        except Exception:
            pass

        google=fetch_google_news(company_name,ticker) if company_name else ""
        if google: parts.append(google)
        text=" ".join(parts)
        score,label,pos,neg,support,warnings=score_text_evidence(text,story)
        support_text=normalize_text_evidence_output("\n".join(support))
        warning_text=normalize_text_evidence_output("\n".join(warnings))
        source_text="Yahoo Finance business summary + identity-filtered Yahoo news + Google News RSS" if text else ""
        return {"Text Score":score,"Text Evidence":label,"Text Positive":pos,"Text Negative":neg,
                "Text Support":support_text,"Text Warnings":warning_text,
                "Text Sources":normalize_text_evidence_output(source_text)}
    except Exception as e:
        return {"Text Score":np.nan,"Text Evidence":"⚪ Text nedostupný","Text Positive":0,"Text Negative":0,
                "Text Support":"","Text Warnings":"","Text Sources":str(e)[:180]}



# -----------------------------------------------------------------------------
# V5.3 – Price Recovery Engine
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
    """Describe price structure; price is evidence, never the definition of a turnaround."""
    empty = {"Skóre ceny": np.nan, "Price View": "⚪ Cena nedostupná",
             "Drawdown 3Y": np.nan, "Drawdown 5Y": np.nan,
             "Recovery from 3Y Low": np.nan, "Recovery from 5Y Low": np.nan,
             "6M Return": np.nan, "12M Return": np.nan,
             "Days Since 3Y Low": np.nan, "MA50 vs MA200": np.nan,
             "Higher Low": "", "Higher High": "", "Price Trend": "", "Price Evidence": ""}
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

    # Compare the recent 6 months with the preceding 6 months. This is a simple proxy for
    # higher lows / higher highs and is more useful than rewarding a rebound from any low.
    recent = s[s.index >= now - pd.Timedelta(days=183)]
    prior = s[(s.index < now - pd.Timedelta(days=183)) & (s.index >= now - pd.Timedelta(days=366))]
    recent_low = float(recent.min()) if len(recent) else np.nan
    prior_low = float(prior.min()) if len(prior) else np.nan
    recent_high = float(recent.max()) if len(recent) else np.nan
    prior_high = float(prior.max()) if len(prior) else np.nan
    higher_low = (recent_low > prior_low * 1.03) if prior_low > 0 else False
    higher_high = (recent_high > prior_high * 1.03) if prior_high > 0 else False
    lower_low = (recent_low < prior_low * 0.97) if prior_low > 0 else False
    lower_high = (recent_high < prior_high * 0.97) if prior_high > 0 else False
    result["Higher Low"] = "Ano" if higher_low else ("Ne" if prior_low > 0 else "N/A")
    result["Higher High"] = "Ano" if higher_high else ("Ne" if prior_high > 0 else "N/A")

    score=50.0; positive=negative=0; evidence=[]
    dd=result["Drawdown 3Y"]
    if not pd.isna(dd):
        if dd<=-60: score+=8; positive+=1; evidence.append("velký 3Y propad")
        elif dd<=-35: score+=6; positive+=1; evidence.append("výrazný 3Y propad")
        elif dd<=-20: score+=3; evidence.append("mírnější 3Y propad")
        elif dd>-10: score-=5; negative+=1; evidence.append("cena blízko 3Y maxima")
    r12=result["12M Return"]
    if not pd.isna(r12):
        if r12>=20: score+=8; positive+=1; evidence.append("kladný vývoj za 12M")
        elif r12>=5: score+=4; positive+=1; evidence.append("mírný růst za 12M")
        elif r12<=-25: score-=10; negative+=1; evidence.append("silný pokles za 12M")
        elif r12<0: score-=4; negative+=1; evidence.append("pokles za 12M")
    if higher_low: score+=10; positive+=1; evidence.append("vyšší minima")
    if higher_high: score+=10; positive+=1; evidence.append("vyšší maxima")
    if lower_low: score-=8; negative+=1; evidence.append("nižší minima")
    if lower_high: score-=6; negative+=1; evidence.append("nižší maxima")
    ma=result["MA50 vs MA200"]
    if not pd.isna(ma):
        if ma>=5: score+=6; positive+=1; evidence.append("50D průměr nad 200D")
        elif ma<-10: score-=6; negative+=1; evidence.append("50D průměr pod 200D")

    result["Skóre ceny"]=round(max(0,min(100,score)),1)
    if higher_low and higher_high and not pd.isna(r12) and r12>0:
        view,trend="🟢 Trh potvrzuje zlepšení","rostoucí struktura"
    elif higher_low and not lower_low:
        view,trend="🟡 Stabilizace / vyšší minima","stabilizace"
    elif lower_low and lower_high and (pd.isna(r12) or r12<0):
        view,trend="🔴 Trh stále nevěří","klesající struktura"
    elif not pd.isna(r12) and r12>=25 and not pd.isna(dd) and dd>-25:
        view,trend="🔵 Recovery už je z velké části v ceně","anticipace"
    else:
        view,trend="⚪ Smíšený cenový obraz","smíšený"
    result["Price View"],result["Price Trend"]=view,trend
    result["Price Evidence"]="; ".join(evidence[:8])
    return result


def add_price_analysis(df,max_price_candidates):
    if df.empty or max_price_candidates<=0: return df
    out=df.copy()
    defaults={"Skóre ceny":np.nan,"Price View":"⚪ Nehodnoceno","Drawdown 3Y":np.nan,"Drawdown 5Y":np.nan,
              "Recovery from 3Y Low":np.nan,"Recovery from 5Y Low":np.nan,"6M Return":np.nan,"12M Return":np.nan,
              "Days Since 3Y Low":np.nan,"MA50 vs MA200":np.nan,"Higher Low":"","Higher High":"","Price Trend":"","Price Evidence":""}
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
    direction = clean_text(row.get("Fundamental Direction"))
    ps = safe_float(row.get("Skóre ceny"))
    if not direction or pd.isna(ps): return "⚪ Nedostatek dat"
    if direction == "🔄 Recovery / obrat":
        if ps < 40: return "🟢 Fundamenty předbíhají cenu"
        if ps >= 70: return "🔵 Fundamenty i trh potvrzují zlepšení"
        return "🟡 Fundamenty se zlepšují, trh čeká"
    if direction in ("📈 Growth", "🛠️ Operational improvement"):
        if ps >= 70: return "🔵 Zlepšení je trhem potvrzené"
        if ps < 40: return "🟢 Fundamenty jsou lepší než cena"
        return "🟡 Fundamenty se zlepšují, cena je smíšená"
    if direction == "📉 Deterioration / weak recovery":
        if ps >= 70: return "🟠 Cena předbíhá slabé fundamenty"
        return "🔴 Fundamenty obrat nepotvrzují"
    return "🟡 Smíšený signál"


def compact_verdict(row):
    story = clean_text(row.get("Story")); ts = safe_float(row.get("Turnaround Score"))
    text = clean_text(row.get("Text Evidence")); conf = safe_float(row.get("Final Confidence"))
    if "zpochybňuje" in text: return "🔴 Zpochybněno"
    if story in ("🔄 Operating turnaround","🔄 Recovery candidate"):
        if not pd.isna(ts) and ts >= 80 and not pd.isna(conf) and conf >= 70: return "🟢 Silný adept"
        if not pd.isna(ts) and ts >= 65: return "🟡 Turnaround kandidát"
    if story == "🛠️ Operational improvement": return "⚪ Spíše zlepšení"
    if "smíšený" in text and not pd.isna(conf) and conf < 65: return "🟠 Smíšený / rizikový"
    if story in ("🚀 Růst za rozumnou cenu", "🔥 High Growth / dražší příběh"):
        return "📈 Růstový příběh"
    if story == "🏆 Quality Compounder": return "🏆 Kvalitní růst"
    return "🟡 Zajímavý kandidát"

def compact_trend(row):
    direction = clean_text(row.get("Fundamental Direction"))
    if direction == "🔄 Recovery / obrat": return "🔄 Obrat / recovery"
    if direction == "🛠️ Operational improvement": return "🛠️ Provozní zlepšení"
    if direction == "📈 Growth": return "📈 Růst"
    if direction == "📉 Deterioration / weak recovery": return "📉 Zhoršení"
    if direction == "➡️ Stabilizace": return "➡️ Stabilizace"
    return "⚪ Nejasný směr"

def compact_text_signal(row):
    ev=clean_text(row.get("Text Evidence"))
    if "mechanismus" in ev.lower(): return "🟢 Mechanismus nalezen"
    if "změnu" in ev.lower(): return "🟡 Změna nalezena"
    if "rizika" in ev.lower(): return "🟠 Převážně rizika"
    return ev or "⚪ Bez důkazu"

def compact_warning(row):
    warnings = clean_text(row.get("Text Warnings"))
    if warnings:
        return "🟠 Rizika nalezena"
    return "🟢 Bez textového varování"


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
        ev = fetch_text_evidence(row["Yahoo Ticker"], row["Story"], row.get("Name",""), row.get("Ticker",""))
        for k, v in ev.items():
            out.at[idx, k] = v
        progress.progress(i / len(todo))
    progress.empty(); status.empty()
    return out


def final_story_confidence(r):
    """Story strength is quantitative. Text is evidence quality, not a weighted vote."""
    q=safe_float(r.get("Story Priority"))
    return q

def evidence_quality(r):
    """Verbal quality of the text evidence; numeric Text Score remains internal."""
    score=safe_float(r.get("Text Score"))
    pos=safe_float(r.get("Text Positive")) or 0
    neg=safe_float(r.get("Text Negative")) or 0
    if pd.isna(score): return "⚪ Zatím nedoloženo"
    if pos+neg == 0: return "⚪ Zatím nedoloženo"
    if neg > pos and neg >= 3: return "🟠 Slabé / rozporné"
    if pos >= 5 and neg <= 2: return "🟢 Silné"
    if pos >= 2: return "🟡 Střední"
    return "🟠 Slabé"




# ============================================================
# ANALYTIK — independent company research module
# This module deliberately does not consume Screener outputs.
# It shares only the names of the investment-story categories.
# ============================================================

ANALYST_STORIES = [
    "Kvalitní compounder",
    "Kvalita za rozumnou cenu",
    "Růst za rozumnou cenu",
    "Value / levná firma",
    "Provozní turnaround",
    "Cyklické zotavení",
    "Aktivové / finanční zotavení",
    "Realitní hodnota",
    "Provozní zlepšení",
    "Růstové zotavení",
    "Vysoký růst / dražší příběh",
    "Value trap – varování",
    "Nejasný / smíšený příběh",
]



def analyst_yahoo_ticker(ticker, exchange):
    ticker = clean_text(ticker).upper()
    if exchange in ("NASDAQ", "NYSE"):
        return normalize_us_ticker(ticker)
    if ticker.endswith(".DE"):
        return ticker
    return ticker + ".DE"


def analyst_human_number(x, decimals=1):
    x = safe_float(x)
    if pd.isna(x):
        return "—"
    sign = "−" if x < 0 else ""
    v = abs(x)
    if v >= 1_000_000_000:
        return f"{sign}{v/1_000_000_000:.{decimals}f} mld.".replace(".", ",")
    if v >= 1_000_000:
        return f"{sign}{v/1_000_000:.{decimals}f} mil.".replace(".", ",")
    if v >= 10_000:
        return f"{sign}{v:,.0f}".replace(",", " ")
    return f"{sign}{v:.{decimals}f}".replace(".", ",")


def analyst_pct(x, decimals=1):
    x = safe_float(x)
    return "—" if pd.isna(x) else f"{x:+.{decimals}f} %".replace(".", ",")


@st.cache_data(ttl=1800, show_spinner=False)
def _analyst_yahoo_chart_data(yahoo_ticker, range_value="5y", interval="1d"):
    """Direct Yahoo Chart API fallback used by Analytik.

    This endpoint does not depend on yfinance cookies/crumbs and is also the
    source used by a number of current Yahoo Finance wrappers.  It is kept
    ticker-agnostic: US, Xetra and other Yahoo-supported listings use the same
    endpoint.
    """
    symbol = clean_text(yahoo_ticker).strip()
    if not symbol:
        return {}
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/154.0 Safari/537.36"}
    urls = [
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
        f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}",
    ]
    for url in urls:
        try:
            r = requests.get(url, params={"range": range_value, "interval": interval, "events": "history", "includeAdjustedClose": "true"}, headers=headers, timeout=20)
            r.raise_for_status()
            payload = r.json()
            result = ((payload.get("chart") or {}).get("result") or [])
            if result:
                return result[0] or {}
        except Exception:
            continue
    return {}


def _analyst_regex_number(html, field):
    if not html:
        return np.nan
    patterns = [
        rf'"{re.escape(field)}"\s*:\s*\{{\s*"raw"\s*:\s*(-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)',
        rf'"{re.escape(field)}"\s*:\s*(-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)',
    ]
    for pat in patterns:
        m = re.search(pat, html)
        if m:
            return safe_float(m.group(1))
    return np.nan


def _analyst_regex_string(html, field):
    if not html:
        return ""
    patterns = [
        rf'"{re.escape(field)}"\s*:\s*"((?:\\.|[^"\\])*)"',
    ]
    for pat in patterns:
        m = re.search(pat, html)
        if m:
            try:
                return clean_text(json.loads('"' + m.group(1) + '"'))
            except Exception:
                return clean_text(unescape(m.group(1).replace('\\"', '"')))
    return ""


@st.cache_data(ttl=1800, show_spinner=False)
def _analyst_yahoo_quote_page(yahoo_ticker):
    symbol = clean_text(yahoo_ticker).strip()
    if not symbol:
        return {}
    url = f"https://finance.yahoo.com/quote/{symbol}/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/154.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        r = requests.get(url, headers=headers, timeout=25)
        r.raise_for_status()
        html = r.text
    except Exception:
        return {}

    # Yahoo's rendered quote page is normally authoritative for identity,
    # but protect against a stale/misrouted response: only accept identity
    # fields if the page itself contains the requested Yahoo symbol.
    symbol_upper = symbol.upper()
    page_symbol = _analyst_regex_string(html, "symbol").upper()
    identity_ok = page_symbol == symbol_upper or re.search(rf"\b{re.escape(symbol_upper)}\b", html.upper()) is not None

    out = {
        "symbol": page_symbol,
        "identity_ok": identity_ok,
        "name": (_analyst_regex_string(html, "longName") or _analyst_regex_string(html, "shortName")) if identity_ok else "",
        "sector": _analyst_regex_string(html, "sector"),
        "industry": _analyst_regex_string(html, "industry"),
        "country": _analyst_regex_string(html, "country"),
        "website": _analyst_regex_string(html, "website"),
        "ir_website": _analyst_regex_string(html, "irWebsite"),
        "summary": _analyst_regex_string(html, "longBusinessSummary"),
        "currency": _analyst_regex_string(html, "currency"),
        "exchange": _analyst_regex_string(html, "exchange"),
        "market_cap": _analyst_regex_number(html, "marketCap"),
        "pe": _analyst_regex_number(html, "trailingPE"),
        "forward_pe": _analyst_regex_number(html, "forwardPE"),
        "ps": _analyst_regex_number(html, "priceToSalesTrailing12Months"),
        "pb": _analyst_regex_number(html, "priceToBook"),
        "roe": _analyst_regex_number(html, "returnOnEquity"),
        "revenue_growth": _analyst_regex_number(html, "revenueGrowth"),
        "earnings_growth": _analyst_regex_number(html, "earningsGrowth"),
        "free_cash_flow": _analyst_regex_number(html, "freeCashflow"),
        "debt_to_equity": _analyst_regex_number(html, "debtToEquity"),
        "dividend_yield": _analyst_regex_number(html, "dividendYield"),
        "employees": _analyst_regex_number(html, "fullTimeEmployees"),
        "quote_type": _analyst_regex_string(html, "quoteType"),
    }
    return out


@st.cache_data(ttl=1800, show_spinner=False)
def _analyst_yahoo_search_quote(yahoo_ticker):
    """Unauthenticated Yahoo search fallback for company name/profile identity."""
    symbol = clean_text(yahoo_ticker).strip()
    if not symbol:
        return {}
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/154.0 Safari/537.36"}
    try:
        r = requests.get(
            "https://query1.finance.yahoo.com/v1/finance/search",
            params={"q": symbol, "quotesCount": 10, "newsCount": 0},
            headers=headers, timeout=20,
        )
        r.raise_for_status()
        quotes = (r.json() or {}).get("quotes") or []
        # SECURITY/CORRECTNESS: never accept an approximate Yahoo search hit.
        # A search for a short ticker such as SHL can return unrelated US
        # securities (e.g. APLD) when the exact Yahoo symbol is not present.
        # Returning the first result can therefore silently analyze another
        # company under the requested ticker.
        for q in quotes:
            if clean_text(q.get("symbol")).upper() == symbol.upper():
                return q
        return {}
    except Exception:
        return {}


@st.cache_data(ttl=1800, show_spinner=False)
def analyst_get_quote_data(yahoo_ticker):
    """
    V6.28.9 IDENTITY FIREWALL.

    A company may be accepted only when BOTH independent Yahoo layers agree
    on the exact requested Yahoo symbol:
      1) Chart API -> meta.symbol
      2) Yahoo Search -> exact quote.symbol

    The Yahoo quote HTML page is deliberately NOT used here. Its large HTML
    document can contain data for other securities, so global regex extraction
    of fields such as longName/marketCap/P-E is unsafe.
    """
    expected = clean_text(yahoo_ticker).upper()
    if not expected:
        return {"symbol": "", "identity_ok": False, "identity_reason": "Prázdný Yahoo ticker."}

    chart = _analyst_yahoo_chart_data(yahoo_ticker, range_value="5d", interval="1d")
    meta = chart.get("meta") or {}
    search = _analyst_yahoo_search_quote(yahoo_ticker)

    chart_symbol = clean_text(meta.get("symbol")).upper()
    search_symbol = clean_text(search.get("symbol")).upper()

    chart_ok = chart_symbol == expected
    search_ok = search_symbol == expected
    identity_ok = chart_ok and search_ok

    def pick_text(*vals):
        for v in vals:
            x = clean_text(v)
            if x:
                return x
        return ""

    def pick_num(*vals):
        for v in vals:
            x = safe_float(v)
            if not pd.isna(x):
                return x
        return np.nan

    # Search fields are used only from the EXACT symbol-matched quote returned
    # above. This keeps the identity and the descriptive fields bound together.
    name = pick_text(
        search.get("longname"), search.get("longName"),
        search.get("shortname"), search.get("shortName")
    )

    # Yahoo Search uses slightly different field names depending on endpoint
    # version. Accept only fields belonging to the exact matched quote object.
    market_cap = pick_num(search.get("marketCap"), search.get("marketcap"))
    pe = pick_num(search.get("trailingPE"), search.get("trailingPe"))
    forward_pe = pick_num(search.get("forwardPE"), search.get("forwardPe"))
    ps = pick_num(search.get("priceToSalesTrailing12Months"), search.get("priceToSales"))
    pb = pick_num(search.get("priceToBook"), search.get("priceToBookRatio"))
    roe = pick_num(search.get("returnOnEquity"), search.get("roe"))
    revenue_growth = pick_num(search.get("revenueGrowth"))
    earnings_growth = pick_num(search.get("earningsGrowth"))
    free_cash_flow = pick_num(search.get("freeCashflow"), search.get("freeCashFlow"))
    debt_to_equity = pick_num(search.get("debtToEquity"))
    dividend_yield = pick_num(search.get("dividendYield"), search.get("dividendYieldPct"))
    employees = pick_num(search.get("fullTimeEmployees"))

    if not identity_ok:
        reason = (
            f"Yahoo identity mismatch: požadováno {expected}; "
            f"Chart={chart_symbol or '—'}, Search={search_symbol or '—'}."
        )
    elif not name:
        reason = "Yahoo ověřilo ticker, ale neposkytlo bezpečně svázaný název firmy."
        identity_ok = False
    else:
        reason = "OK – Chart API a přesný Yahoo Search quote shodně potvrzují ticker."

    return {
        "symbol": expected if chart_ok else (chart_symbol or search_symbol),
        "identity_ok": identity_ok,
        "identity_reason": reason,
        "identity_chart_symbol": chart_symbol,
        "identity_search_symbol": search_symbol,
        "identity_source": "Yahoo Chart API + exact Yahoo Search",
        "price": pick_num(meta.get("regularMarketPrice"), meta.get("previousClose"), search.get("regularMarketPrice")),
        "market_cap": market_cap,
        "currency": pick_text(meta.get("currency"), search.get("currency")),
        "name": name,
        "sector": pick_text(search.get("sector")),
        "industry": pick_text(search.get("industry")),
        "country": pick_text(search.get("country")),
        "website": "",
        "ir_website": "",
        "summary": "",
        "employees": employees,
        "pe": pe,
        "forward_pe": forward_pe,
        "ps": ps,
        "pb": pb,
        "roe": roe,
        "revenue_growth": revenue_growth,
        "earnings_growth": earnings_growth,
        "free_cash_flow": free_cash_flow,
        "debt_to_equity": debt_to_equity,
        "dividend_yield": dividend_yield,
        "quote_type": pick_text(search.get("quoteType"), meta.get("instrumentType")),
        "exchange": pick_text(meta.get("exchangeName"), search.get("exchange")),
    }


def _analyst_timeseries_to_series(item, requested_key):
    vals = (item.get(requested_key) or []) if isinstance(item, dict) else []
    rows = []
    for v in vals:
        if not isinstance(v, dict):
            continue
        d = pd.to_datetime(v.get("asOfDate"), errors="coerce")
        raw = v.get("reportedValue", {})
        val = raw.get("raw") if isinstance(raw, dict) else None
        val = safe_float(val)
        if not pd.isna(d) and not pd.isna(val):
            rows.append((pd.Timestamp(d).normalize(), val))
    if not rows:
        return pd.Series(dtype=float)
    return pd.Series(dict(rows), dtype=float).sort_index()


@st.cache_data(ttl=1800, show_spinner=False)
def _analyst_yahoo_timeseries_direct(yahoo_ticker, annual=False, trailing=False):
    """Direct Yahoo fundamentals-timeseries data, independent of yfinance."""
    symbol = clean_text(yahoo_ticker).strip()
    if not symbol:
        return {}
    prefix = "trailing" if trailing else ("annual" if annual else "quarterly")
    keys = ["TotalRevenue", "NetIncome", "OperatingIncome", "OperatingCashFlow", "CapitalExpenditure", "TotalDebt", "StockholdersEquity"]
    requested = [prefix + k for k in keys]
    now = int(time.time())
    years = 10 if annual else (7 if not trailing else 3)
    start = now - int(years * 366 * 24 * 3600)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/154.0 Safari/537.36"}
    urls = [
        f"https://query2.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/timeseries/{symbol}",
        f"https://query1.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/timeseries/{symbol}",
    ]
    for url in urls:
        try:
            # First try one combined request. This is fastest and is the normal
            # path for annual/quarterly data.
            r = requests.get(url, params={"symbol": symbol, "type": ",".join(requested), "period1": start, "period2": now}, headers=headers, timeout=25)
            r.raise_for_status()
            result = ((r.json().get("timeseries") or {}).get("result") or [])
            out = {}
            for item in result:
                for key in requested:
                    ss = _analyst_timeseries_to_series(item, key)
                    if not ss.empty:
                        out[key] = ss
            if out:
                # For trailing data Yahoo can occasionally omit one or more
                # fields from a combined request. Keep what arrived, then
                # continue below for the missing fields.
                if not trailing or len(out) == len(requested):
                    return out
        except Exception:
            out = {}

        # Yahoo/yfinance itself sometimes gets better coverage by asking for
        # one metric at a time. This is especially important for TTM because
        # the missing FY-end quarter is reconstructed from TTM minus the next
        # three reported quarters. Never estimate from growth rates.
        for key in requested:
            if key in out and not out[key].empty:
                continue
            # Trailing series are a special Yahoo endpoint case: asking for a
            # very old period can make Yahoo omit the trailing value even though
            # the same key is available without a period restriction. Try the
            # normal bounded request first, then an unrestricted period request.
            trailing_params = [
                {"symbol": symbol, "type": key, "period1": start, "period2": now},
                {"symbol": symbol, "type": key},
            ] if trailing else [
                {"symbol": symbol, "type": key, "period1": start, "period2": now}
            ]
            for params in trailing_params:
                try:
                    r = requests.get(
                        url,
                        params=params,
                        headers=headers,
                        timeout=25,
                    )
                    r.raise_for_status()
                    result = ((r.json().get("timeseries") or {}).get("result") or [])
                    for item in result:
                        ss = _analyst_timeseries_to_series(item, key)
                        if not ss.empty:
                            out[key] = ss
                            break
                    if key in out and not out[key].empty:
                        break
                except Exception:
                    continue
        if out:
            return out
    return {}


@st.cache_data(ttl=1800, show_spinner=False)
def _analyst_yahoo_timeseries_all(yahoo_ticker):
    """Fetch annual, quarterly and trailing Yahoo data once per ticker."""
    return {
        "annual": _analyst_yahoo_timeseries_direct(yahoo_ticker, annual=True),
        "quarterly": _analyst_yahoo_timeseries_direct(yahoo_ticker, annual=False),
        "trailing": _analyst_yahoo_timeseries_direct(yahoo_ticker, trailing=True),
    }


def _analyst_series_from_direct(raw, metric, prefix):
    return raw.get(prefix + metric, pd.Series(dtype=float)).copy()


def _analyst_derive_missing_quarter_from_ttm(series, ttm_series, missing_date):
    """Derive a missing quarter only when TTM and the following 3 quarters exist."""
    if series is None or series.empty or ttm_series is None or ttm_series.empty:
        return series
    s = series.copy()
    s.index = pd.to_datetime(s.index, errors="coerce")
    t = ttm_series.copy()
    t.index = pd.to_datetime(t.index, errors="coerce")
    target = pd.Timestamp(missing_date).normalize()
    if target in s.index and not pd.isna(s.loc[target]):
        return s
    candidates = sorted([d for d in t.index if d > target])
    if not candidates:
        return s
    # Prefer the latest TTM observation; it should contain exactly the target
    # quarter plus the following three reported quarters.
    ttm_end = candidates[-1]
    later = s[(s.index > target) & (s.index <= ttm_end)].dropna()
    if len(later) != 3:
        # If Yahoo exposes an additional stale TTM observation, try each
        # candidate from newest to oldest and accept only an exact 3-quarter
        # decomposition.
        for candidate in reversed(candidates):
            candidate_later = s[(s.index > target) & (s.index <= candidate)].dropna()
            if len(candidate_later) == 3:
                ttm_end = candidate
                later = candidate_later
                break
        if len(later) != 3:
            return s
    ttm_val = safe_float(t.loc[ttm_end])
    if pd.isna(ttm_val):
        return s
    derived = ttm_val - float(later.sum())
    s.loc[target] = derived
    return s.sort_index()


def _analyst_fiscal_end_from_annual_dates(dates):
    valid = [pd.Timestamp(d) for d in dates if not pd.isna(pd.Timestamp(d))]
    if not valid:
        return None
    counts = {}
    for d in valid:
        counts[(d.month, d.day)] = counts.get((d.month, d.day), 0) + 1
    month_day = max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
    return month_day


def _analyst_fy_for_date(d, fy_end):
    d = pd.Timestamp(d)
    if not fy_end:
        return d.year
    return d.year + 1 if (d.month, d.day) > fy_end else d.year


def _analyst_build_history_direct(yahoo_ticker, quarterly=False):
    """Build Analytik history entirely from direct Yahoo data.

    Important: yfinance is intentionally not involved here.  Missing FY-end
    flow data can be derived only when the Yahoo TTM value plus the following
    three reported quarters mathematically determines the missing quarter.
    Such rows are marked as derived in the period label.
    """
    all_data = _analyst_yahoo_timeseries_all(yahoo_ticker)
    annual_raw, q_raw, trailing_raw = all_data.get("annual", {}), all_data.get("quarterly", {}), all_data.get("trailing", {})
    prefix = "quarterly" if quarterly else "annual"
    metric_map = {
        "Revenue": "TotalRevenue", "Net Income": "NetIncome", "Operating Income": "OperatingIncome",
        "Operating Cash Flow": "OperatingCashFlow", "Capital Expenditure": "CapitalExpenditure",
        "Debt": "TotalDebt", "Equity": "StockholdersEquity",
    }
    raw = q_raw if quarterly else annual_raw
    filled = {name: _analyst_series_from_direct(raw, metric, prefix) for name, metric in metric_map.items()}

    # Fiscal-year end must always come from annual statement dates, even when
    # we are currently building the quarterly table. Using quarterly dates here
    # would silently turn a Sep fiscal year into a calendar-quarter assumption.
    annual_rev_for_fy = _analyst_series_from_direct(annual_raw, "TotalRevenue", "annual")
    annual_dates = annual_rev_for_fy.index if not annual_rev_for_fy.empty else _analyst_series_from_direct(annual_raw, "TotalDebt", "annual").index
    fy_end = _analyst_fiscal_end_from_annual_dates(annual_dates)
    if fy_end is None:
        debt_idx = filled["Debt"].index if not filled["Debt"].empty else []
        fy_end = _analyst_fiscal_end_from_annual_dates(debt_idx)

    derived_dates = set()
    if quarterly:
        # Yahoo currently omits some fiscal-year-end flow rows while providing
        # the TTM value. If mathematically determinable, reconstruct that one
        # quarter. This is deliberately conservative: no estimate is made from
        # growth rates, averages, or analyst forecasts.
        trailing_map = {name: _analyst_series_from_direct(trailing_raw, metric, "trailing") for name, metric in metric_map.items()}
        # Look for quarter-end dates immediately before a known next quarter;
        # in practice this catches a missing Sep fiscal quarter for a Sep FY end.
        known_dates = sorted(set().union(*[set(s.index) for s in filled.values() if not s.empty]))
        if known_dates and fy_end:
            years = sorted(set(d.year for d in known_dates))
            candidates = []
            for y in years:
                try:
                    target = pd.Timestamp(year=y, month=fy_end[0], day=fy_end[1])
                except ValueError:
                    continue
                if target not in known_dates:
                    candidates.append(target)
            # Prefer the most recent missing FY-end date that is surrounded by data.
            for target in sorted(candidates, reverse=True):
                changed = False
                for name in metric_map:
                    before = filled[name]
                    after = _analyst_derive_missing_quarter_from_ttm(before, trailing_map[name], target)
                    if target in after.index and (target not in before.index):
                        filled[name] = after
                        changed = True
                if changed:
                    derived_dates.add(target)
                    break

    # Annual FY2025 (or equivalent) may be absent even though all four fiscal
    # quarters are available. Sum exactly those quarters belonging to the FY.
    if not quarterly and fy_end:
        q_data = _analyst_build_history_direct(yahoo_ticker, quarterly=True)
        if q_data is not None and not q_data.empty:
            q_dates = []
            for label in q_data.get("Období", []):
                m = re.search(r"ended (\d{4}-\d{2}-\d{2})", clean_text(label))
                if m:
                    q_dates.append((pd.Timestamp(m.group(1)), clean_text(label)))
            if q_dates:
                q_raw_df = q_data.copy()
                q_raw_df["_date"] = [d for d, _ in q_dates]
                q_raw_df["_fy"] = q_raw_df["_date"].map(lambda d: _analyst_fy_for_date(d, fy_end))
                full_fys = [fy for fy, grp in q_raw_df.groupby("_fy") if len(grp) == 4]
                latest_fy = max(full_fys) if full_fys else None
                qfy = q_raw_df[q_raw_df["_fy"] == latest_fy] if latest_fy is not None else pd.DataFrame()
                if len(qfy) == 4:
                    end_date = max(qfy["_date"])
                    for name in ["Revenue", "Net Income", "Operating Income", "Operating Cash Flow", "Capital Expenditure"]:
                        if name in qfy.columns:
                            vals = pd.to_numeric(qfy[name], errors="coerce")
                            if vals.notna().sum() == 4:
                                filled[name].loc[end_date] = float(vals.sum())
                    # Balance-sheet values are point-in-time, so use the direct
                    # annual Yahoo value when it exists; otherwise quarter-end.
                    for name in ["Debt", "Equity"]:
                        if name in qfy.columns and end_date in qfy["_date"].values:
                            val = qfy.loc[qfy["_date"] == end_date, name].iloc[-1] if qfy[name].notna().any() else np.nan
                            if not pd.isna(val) and end_date not in filled[name].index:
                                filled[name].loc[end_date] = float(val)
                    derived_dates.add(end_date)

    cols = {name: s for name, s in filled.items() if s is not None and not s.empty}
    if not cols:
        return pd.DataFrame()
    dates = sorted(set().union(*[set(pd.to_datetime(s.index, errors="coerce")) for s in cols.values()]))
    dates = [d for d in dates if not pd.isna(d)]
    dates = dates[-8:] if quarterly else dates[-5:]
    out = pd.DataFrame(index=dates)
    for name, s in cols.items():
        ss = s.copy()
        ss.index = pd.to_datetime(ss.index, errors="coerce")
        out[name] = ss.reindex(dates).values

    labels = []
    for d in dates:
        fy = _analyst_fy_for_date(d, fy_end)
        if quarterly:
            month_distance = (d.month - fy_end[0]) % 12 if fy_end else (d.month - 12) % 12
            qnum = 4 if month_distance == 0 else (month_distance + 2) // 3
            label = f"Q{qnum} FY{fy} (ended {d:%Y-%m-%d})"
        else:
            label = f"FY{fy} (ended {d:%Y-%m-%d})"
        if d in derived_dates:
            label += " [odvozeno z dostupných Yahoo dat]"
        labels.append(label)
    out.index = labels

    if "Revenue" in out and "Net Income" in out:
        revs = pd.to_numeric(out["Revenue"], errors="coerce")
        nis = pd.to_numeric(out["Net Income"], errors="coerce")
        out["Net Margin %"] = np.where(revs > 0, nis / revs * 100, np.nan)
    if "Operating Cash Flow" in out and "Capital Expenditure" in out:
        ocf_s = pd.to_numeric(out["Operating Cash Flow"], errors="coerce")
        cap_s = pd.to_numeric(out["Capital Expenditure"], errors="coerce")
        out["FCF"] = np.where(cap_s <= 0, ocf_s + cap_s, ocf_s - cap_s)
    return out.reset_index(names="Období")


@st.cache_data(ttl=3600, show_spinner=False)
def analyst_get_financial_history(yahoo_ticker):
    return _analyst_build_history_direct(yahoo_ticker, quarterly=False)


@st.cache_data(ttl=3600, show_spinner=False)
def analyst_get_quarterly_history(yahoo_ticker):
    return _analyst_build_history_direct(yahoo_ticker, quarterly=True)



def _analyst_statement_series(df, labels):
    if df is None or df.empty:
        return pd.Series(dtype=float)
    idx = {str(x).strip().lower(): x for x in df.index}
    for label in labels:
        if label.lower() in idx:
            return pd.to_numeric(df.loc[idx[label.lower()]], errors="coerce")
    compact = {re.sub(r"[^a-z0-9]", "", str(x).lower()): x for x in df.index}
    for label in labels:
        key = re.sub(r"[^a-z0-9]", "", label.lower())
        if key in compact:
            return pd.to_numeric(df.loc[compact[key]], errors="coerce")
    return pd.Series(dtype=float)


def _analyst_get_statement(t, attr, getter_name, freq):
    frames = []
    try:
        x = getattr(t, attr, None)
        if isinstance(x, pd.DataFrame) and not x.empty:
            frames.append(x)
    except Exception:
        pass
    try:
        getter = getattr(t, getter_name, None)
        if callable(getter):
            x = getter(freq=freq)
            if isinstance(x, pd.DataFrame) and not x.empty:
                frames.append(x)
    except Exception:
        pass
    for x in frames:
        if isinstance(x, pd.DataFrame) and not x.empty:
            return x
    return pd.DataFrame()


def _analyst_period_labels(dates, annual_dates=None, quarterly=False):
    """Create fiscal-period labels from actual statement dates.

    For quarterly data the fiscal year-end month/day is inferred from the
    annual statements. It is deliberately NOT necessary to have the following
    annual year-end already present: this is what lets SHL's Dec-2025, Mar-2026
    and Jun-2026 quarters be labelled FY2026 even when Yahoo exposes only four
    annual columns.
    """
    dates = [pd.Timestamp(d) for d in dates if not pd.isna(pd.Timestamp(d))]
    annual_ends = sorted(pd.Timestamp(d) for d in (annual_dates or []) if not pd.isna(pd.Timestamp(d)))
    if not quarterly:
        return [f"FY{d.year} (ended {d:%Y-%m-%d})" for d in dates]

    if not annual_ends:
        return [f"Q{((d.month - 1) // 3) + 1} FY{d.year} (ended {d:%Y-%m-%d}; FY end unknown)" for d in dates]

    # Infer the fiscal-year end from the annual statement dates. In practice
    # the same month/day repeats every year; mode protects against a malformed
    # single date.
    md_counts = {}
    for a in annual_ends:
        key = (a.month, a.day)
        md_counts[key] = md_counts.get(key, 0) + 1
    fy_month, fy_day = max(md_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]

    labels = []
    for d in dates:
        after_fy_end = (d.month, d.day) > (fy_month, fy_day)
        fy_year = d.year + 1 if after_fy_end else d.year
        # Quarter 4 is the fiscal year-end quarter itself.
        month_distance = (d.month - fy_month) % 12
        qnum = 4 if month_distance == 0 else (month_distance + 2) // 3
        labels.append(f"Q{qnum} FY{fy_year} (ended {d:%Y-%m-%d})")
    return labels


def analyst_ttm_from_quarters(quarterly):
    if quarterly is None or quarterly.empty or len(quarterly) < 4:
        return pd.DataFrame()
    q = quarterly.tail(4)
    end_label = clean_text(q.iloc[-1].get("Období")) if not q.empty else ""
    row = {"Období": f"TTM (4Q ended {end_label})" if end_label else "TTM"}
    for c in ["Revenue", "Net Income", "Operating Income", "Operating Cash Flow", "FCF"]:
        if c in q.columns:
            vals = pd.to_numeric(q[c], errors="coerce")
            if vals.notna().sum() >= 3:
                row[c] = vals.sum(min_count=3)
    rev = safe_float(row.get("Revenue"))
    ni = safe_float(row.get("Net Income"))
    if not pd.isna(rev) and rev > 0 and not pd.isna(ni):
        row["Net Margin %"] = ni / rev * 100
    return pd.DataFrame([row])


@st.cache_data(ttl=3600, show_spinner=False)
def analyst_price_history(yahoo_ticker):
    """Direct Yahoo Chart API price history; no yfinance dependency."""
    data = _analyst_yahoo_chart_data(yahoo_ticker, range_value="5y", interval="1d")
    timestamps = data.get("timestamp") or []
    indicators = data.get("indicators") or {}
    quote = (indicators.get("quote") or [{}])[0] or {}
    closes = quote.get("close") or []
    volumes = quote.get("volume") or []
    if not timestamps or not closes:
        return pd.DataFrame()
    rows = []
    for i, ts in enumerate(timestamps):
        try:
            d = pd.to_datetime(int(ts), unit="s", utc=True).tz_localize(None)
        except Exception:
            continue
        row = {"Date": d, "Close": safe_float(closes[i] if i < len(closes) else np.nan)}
        if volumes:
            row["Volume"] = safe_float(volumes[i] if i < len(volumes) else np.nan)
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def analyst_sec_ticker_map():
    try:
        url = "https://www.sec.gov/files/company_tickers.json"
        r = requests.get(url, timeout=30, headers={"User-Agent": "Stock-Screener research contact research@example.com"})
        r.raise_for_status()
        data = r.json()
        return pd.DataFrame([{
            "ticker": clean_text(item.get("ticker")).upper(),
            "name": clean_text(item.get("title")),
            "cik": str(item.get("cik_str", "")).zfill(10)
        } for item in data.values()])
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner=False)
def analyst_sec_filings(ticker):
    try:
        mp = analyst_sec_ticker_map()
        hit = mp[mp["ticker"].eq(clean_text(ticker).upper())]
        if hit.empty: return pd.DataFrame()
        cik = hit.iloc[0]["cik"]
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        r = requests.get(url, timeout=30, headers={"User-Agent": "Stock-Screener research contact research@example.com"})
        r.raise_for_status()
        recent = r.json().get("filings", {}).get("recent", {})
        rows=[]
        n=min(len(recent.get("form",[])), 35)
        for i in range(n):
            rows.append({"Datum":recent.get("filingDate",[""])[i],"Formulář":recent.get("form",[""])[i],"Datum výkazu":recent.get("reportDate",[""])[i],"Dokument":recent.get("primaryDocument",[""])[i],"Accession":recent.get("accessionNumber",[""])[i]})
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


def _analyst_company_tokens(company):
    stop={"inc","inc.","corp","corp.","corporation","company","co","co.","ltd","limited","plc","sa","ag","se","nv","the","group","holdings","holding"}
    toks=re.findall(r"[a-zA-ZÀ-ž0-9]+", clean_text(company).lower())
    return [x for x in toks if len(x)>=3 and x not in stop]


def _analyst_news_relevant(title, description, company, ticker):
    text=(clean_text(title)+" "+clean_text(description)).lower()
    if not text: return False, 0
    bad={"hockey","nhl","leafs","prospect","training camp","football","soccer","basketball","baseball","cricket","fantasy","match report","recipe","weather"}
    if any(x in text for x in bad): return False, 0
    name=clean_text(company).lower()
    tokens=_analyst_company_tokens(company)
    exact_name=name and name in text
    ticker_clean=clean_text(ticker).lower().replace(".de","")
    token_hits=sum(1 for x in tokens[:4] if x in text)
    score=(8 if exact_name else 0)+(2*token_hits)
    if ticker_clean and re.search(rf"\b{re.escape(ticker_clean)}\b",text): score+=3
    event_terms=["earnings","results","guidance","outlook","revenue","profit","margin","orders","demand","china","tariff","regulation","acquisition","acquire","divest","sale","spin-off","spinoff","restructur","separation","ceo","cfo","management","debt","buyback","dividend","launch","approval","recall","settlement","lawsuit","factory","capacity"]
    score += sum(1 for w in event_terms if w in text)
    # Company name is the strongest identity test. If neither exact name nor multiple name tokens appear, reject.
    if not exact_name and token_hits < 2 and score < 6:
        return False, 0
    return True, score


@st.cache_data(ttl=900, show_spinner=False)
def analyst_google_news(company_name, ticker, ir_website=""):
    import xml.etree.ElementTree as ET
    from urllib.parse import urlparse
    company=clean_text(company_name)
    if not company: return pd.DataFrame()
    domain=""
    try: domain=urlparse(clean_text(ir_website)).netloc.replace("www.","")
    except Exception: pass
    queries=[
        f'"{company}" earnings results guidance outlook',
        f'"{company}" strategy restructuring acquisition divestment separation',
        f'"{company}" demand orders margin China regulation tariff',
        f'"{company}" CEO CFO management investor',
        f'"{company}" product launch approval capacity factory',
    ]
    if domain: queries.append(f'"{company}" site:{domain}')
    rows=[]
    for query in queries:
        try:
            url="https://news.google.com/rss/search?q="+requests.utils.quote(query)+"&hl=en-US&gl=US&ceid=US:en"
            r=requests.get(url,timeout=20,headers={"User-Agent":"Mozilla/5.0"})
            r.raise_for_status(); root=ET.fromstring(r.text)
            for item in root.findall(".//item")[:15]:
                title=clean_text(item.findtext("title")); desc=clean_text(item.findtext("description"));
                ok,score=_analyst_news_relevant(title,desc,company,ticker)
                if not ok: continue
                source_el=item.find("source")
                source=clean_text(source_el.text if source_el is not None else "")
                rows.append({"Název":title,"Popis":re.sub(r"<[^>]+>"," ",desc),"Zdroj":source,"Datum":clean_text(item.findtext("pubDate")),"Odkaz":clean_text(item.findtext("link")),"Relevance":score})
        except Exception:
            continue
    if not rows: return pd.DataFrame()
    out=pd.DataFrame(rows).drop_duplicates(subset=["Název"])
    return out.sort_values(["Relevance","Datum"],ascending=[False,False]).head(35).reset_index(drop=True)


def _analyst_fmt_pct(v):
    return "—" if pd.isna(safe_float(v)) else analyst_pct(v,1)


def analyst_financial_summary(annual, quarterly):
    """Trend commentary that respects the company's actual fiscal periods."""
    if annual is None or annual.empty:
        return "Finanční trend se nepodařilo z veřejných dat spolehlivě sestavit."
    parts=[]
    def series_change(df,col):
        if col not in df.columns: return np.nan
        s=pd.to_numeric(df[col],errors="coerce").dropna()
        if len(s)<2 or s.iloc[0]==0: return np.nan
        return (s.iloc[-1]/s.iloc[0]-1)*100
    for col,label in [("Revenue","tržby"),("Net Income","čistý zisk"),("FCF","volný cash flow")]:
        x=series_change(annual,col)
        if not pd.isna(x):
            parts.append(f"Za dostupné fiskální období {label} {'rostou' if x>5 else 'klesají' if x<-5 else 'jsou zhruba stabilní'} ({analyst_pct(x,0)}).")
    if "Net Margin %" in annual.columns:
        s=pd.to_numeric(annual["Net Margin %"],errors="coerce").dropna()
        if len(s)>=2:
            d=s.iloc[-1]-s.iloc[0]
            parts.append(f"Čistá marže se mezi nejstarším a nejnovějším dostupným FY změnila o {analyst_pct(d,1)} p. b.")
    if "Debt" in annual.columns:
        x=series_change(annual,"Debt")
        if not pd.isna(x): parts.append(f"Dluh se ve stejném fiskálním období změnil o {analyst_pct(x,0)}.")

    if quarterly is not None and not quarterly.empty:
        ttm=analyst_ttm_from_quarters(quarterly)
        latest_q_label = clean_text(quarterly.iloc[-1].get("Období"))
        latest_fy_label = clean_text(annual.iloc[-1].get("Období"))
        # Direct TTM/FY comparison is meaningful only when both end on the same date.
        q_date = re.search(r"ended (\d{4}-\d{2}-\d{2})", latest_q_label)
        fy_date = re.search(r"ended (\d{4}-\d{2}-\d{2})", latest_fy_label)
        if not ttm.empty and q_date and fy_date and q_date.group(1) == fy_date.group(1):
            for col,label in [("Revenue","tržby"),("Net Income","čistý zisk"),("FCF","FCF")]:
                if col in ttm.columns and col in annual.columns:
                    a=safe_float(annual.iloc[-1].get(col)); v=safe_float(ttm.iloc[0].get(col))
                    if not pd.isna(a) and a != 0 and not pd.isna(v):
                        parts.append(f"TTM {label} jsou oproti FY končícímu stejným datem {analyst_pct((v/a-1)*100,0)}.")
        elif not ttm.empty:
            parts.append("TTM je uvedeno samostatně; přímé srovnání s posledním FY nebylo použito, protože konce sledovaných období nejsou shodné nebo nejsou jednoznačně určitelné.")

    if len(annual)<5: parts.append(f"Yahoo Finance poskytlo pouze {len(annual)} celých fiskálních období, nikoli plných pět let.")
    if quarterly is not None and len(quarterly)<8: parts.append(f"Čtvrtletní řada obsahuje pouze {len(quarterly)} fiskálních období; TTM je {'dostupné' if len(quarterly)>=4 else 'nedostupné'}.")
    return " ".join(parts) if parts else "Trend nelze z dostupných údajů spolehlivě určit."


def analyst_price_commentary(price):
    if price is None or price.empty or "Close" not in price.columns: return "Cenovou historii se nepodařilo spolehlivě získat."
    p=pd.to_numeric(price["Close"],errors="coerce").dropna()
    if len(p)<60: return "Cenová historie je příliš krátká pro smysluplný kontext."
    last=p.iloc[-1]; ret12=(last/p.iloc[max(0,len(p)-252)]-1)*100 if len(p)>252 else np.nan; ret3=(last/p.iloc[max(0,len(p)-756)]-1)*100 if len(p)>756 else np.nan
    hi=p.tail(min(756,len(p))).max(); dist=(last/hi-1)*100 if hi else np.nan
    ma50=p.tail(50).mean(); ma200=p.tail(200).mean() if len(p)>=200 else np.nan
    facts=[]
    if not pd.isna(ret12): facts.append(f"12M {analyst_pct(ret12)}")
    if not pd.isna(ret3): facts.append(f"3Y {analyst_pct(ret3)}")
    if not pd.isna(dist): facts.append(f"od 3Y maxima {analyst_pct(dist)}")
    if not pd.isna(ma200): facts.append("50D nad 200D" if ma50>ma200 else "50D pod 200D")
    if not pd.isna(ret12) and ret12>15 and dist>-10: interp="Cena už část zlepšení zřejmě promítá do ocenění; další potvrzení bude muset přijít z výsledků."
    elif not pd.isna(ret12) and ret12<-15 and dist<-20: interp="Cena zůstává výrazně pod nedávnými maximy; trh tedy zatím vývoj plně nepotvrzuje."
    else: interp="Cenový vývoj zatím neposkytuje jednoznačné potvrzení ani vyvrácení fundamentálního příběhu."
    return "; ".join(facts)+". "+interp


_ANALYST_EVENT_GROUPS={
    "Strategie / portfolio":["spin-off","spinoff","separation","divest","divestment","sale","carve-out","acquisition","acquire","merger","portfolio"],
    "Výsledky / provoz":["earnings","results","revenue","profit","margin","cash flow","guidance","outlook","orders","demand","cost","capacity"],
    "Trh / regulace":["china","tariff","regulation","regulatory","sanction","government","approval","reimbursement","pricing"],
    "Management / kapitál":["ceo","cfo","management","appoint","resign","buyback","dividend","debt","refinanc","shareholder"],
    "Produkt / technologie":["launch","product","technology","innovation","approval","clinical","trial","patent","factory"]
}


def _analyst_cluster_news(news):
    clusters={k:[] for k in _ANALYST_EVENT_GROUPS}
    if news is None or news.empty: return clusters
    for _,r in news.iterrows():
        text=(clean_text(r.get("Název"))+" "+clean_text(r.get("Popis"))).lower()
        hits=[]
        for cat,words in _ANALYST_EVENT_GROUPS.items():
            score=sum(1 for w in words if w in text)
            if score: hits.append((score,cat))
        if hits:
            hits.sort(reverse=True)
            clusters[hits[0][1]].append(r)
    for cat in clusters:
        clusters[cat]=clusters[cat][:5]
    return clusters


def _analyst_event_direction(text):
    t=text.lower()
    pos=["raises","raised","growth","strong","record","improve","improved","increase","increased","beat","higher","recovery","expands","orders"]
    neg=["cuts","cut","weak","decline","declining","lower","miss","pressure","slow","slower","loss","warning","challenge","tariff","layoff","restructur"]
    p=sum(1 for x in pos if x in t); n=sum(1 for x in neg if x in t)
    return "pozitivní" if p>n else "negativní" if n>p else "smíšený / nejasný"


def _analyst_source_item(source_type, source, date, claim, period="", url="", extra=""):
    return {
        "type": clean_text(source_type),
        "source": clean_text(source),
        "date": clean_text(date),
        "claim": clean_text(claim),
        "period": clean_text(period),
        "url": clean_text(url),
        "extra": clean_text(extra),
    }


def _analyst_add_profile_evidence(pack, q):
    for label, key in [
        ("Sektor", "sector"), ("Odvětví", "industry"),
        ("Země", "country"), ("Měna", "currency"),
        ("Typ instrumentu", "quote_type")
    ]:
        value = clean_text(q.get(key))
        if value:
            pack.append(_analyst_source_item(
                "profile", "Yahoo Finance", "aktuální data",
                f"{label}: {value}"
            ))

    summary = clean_text(q.get("summary"))
    if summary:
        pack.append(_analyst_source_item(
            "profile", "Yahoo Finance", "aktuální data",
            f"Stručný popis obchodního modelu: {summary[:1800]}",
            extra="Firemní profil z Yahoo Finance; používá se jako kontext, nikoli jako důkaz konkrétní aktuální změny"
        ))

    vals = []
    for key, label in [
        ("pe", "P/E"), ("forward_pe", "Forward P/E"),
        ("ps", "P/S"), ("pb", "P/B"), ("roe", "ROE"),
        ("revenue_growth", "Revenue Growth"),
        ("earnings_growth", "Earnings Growth"),
        ("free_cash_flow", "Free Cash Flow"),
        ("debt_to_equity", "Debt/Equity")
    ]:
        v = safe_float(q.get(key))
        if pd.isna(v):
            continue
        if key in ("roe", "revenue_growth", "earnings_growth"):
            v = v * 100 if abs(v) <= 3 else v
            txt = f"{v:.2f} %"
        else:
            txt = analyst_human_number(v, 2)
        vals.append(f"{label}: {txt}")
    if vals:
        pack.append(_analyst_source_item(
            "market_data", "Yahoo Finance", "aktuální data", "; ".join(vals)
        ))


def _analyst_add_financial_evidence(pack, annual, quarterly):
    """Add compact financial evidence, preserving actual fiscal-period labels."""
    def add_selected(df, extra, limit=6):
        if df is None or df.empty:
            return
        # Only send the most recent periods plus the oldest available point.
        work = df.copy()
        rows = []
        if len(work) > limit:
            rows.append(work.iloc[0])
            rows.extend([work.iloc[i] for i in range(max(1, len(work)-limit+1), len(work))])
        else:
            rows = [work.iloc[i] for i in range(len(work))]
        seen = set()
        for row in rows:
            period = clean_text(row.get("Období"))
            if period in seen:
                continue
            seen.add(period)
            vals = []
            for col, label in [
                ("Revenue", "Tržby"), ("Net Income", "Čistý zisk"),
                ("Operating Income", "Provozní zisk"), ("Operating Cash Flow", "Provozní cash flow"),
                ("Capital Expenditure", "Kapitálové výdaje"), ("FCF", "FCF"),
                ("Debt", "Dluh"), ("Equity", "Vlastní kapitál"),
                ("Net Margin %", "Čistá marže")
            ]:
                if col not in df.columns:
                    continue
                v = safe_float(row.get(col))
                if pd.isna(v):
                    continue
                txt = f"{v:.2f} %" if col.endswith("%") else analyst_human_number(v)
                vals.append(f"{label}: {txt}")
            if vals:
                pack.append(_analyst_source_item(
                    "financial", "Yahoo Finance (direct)", period,
                    "; ".join(vals), period=period, extra=extra
                ))

    add_selected(annual, "Celý účetní rok; období je označeno podle skutečného data konce FY.", limit=5)
    add_selected(quarterly, "Čtvrtletní účetní období; Q je určeno vůči skutečnému konci fiskálního roku, nikoli automaticky podle kalendáře.", limit=6)

    ttm = analyst_ttm_from_quarters(quarterly)
    if not ttm.empty:
        add_selected(ttm, "TTM = poslední čtyři dostupná čtvrtletí; není to automaticky kalendářní rok.", limit=1)


def _analyst_add_news_evidence(pack, news, max_items=10):
    if news is None or news.empty:
        return
    d = news.copy()
    if "Datum" in d.columns:
        d["_dt"] = pd.to_datetime(d["Datum"], errors="coerce", utc=True)
    else:
        d["_dt"] = pd.NaT
    if "Relevance" not in d.columns:
        d["Relevance"] = 0
    d = d.sort_values(["Relevance", "_dt"], ascending=[False, False], na_position="last")

    seen = set()
    count = 0
    for _, row in d.iterrows():
        title = clean_text(row.get("Název"))
        if not title:
            continue
        key = re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        desc = clean_text(row.get("Popis"))
        claim = title + (f" | Popis zdroje: {desc[:500]}" if desc else "")
        pack.append(_analyst_source_item(
            "news", clean_text(row.get("Zdroj")) or "Google News",
            clean_text(row.get("Datum")), claim,
            url=clean_text(row.get("Odkaz")),
            extra="Externí zpráva; titulek/popisek je claim zdroje, nikoli automaticky ověřený fakt"
        ))
        count += 1
        if count >= max_items:
            break


def _analyst_add_sec_evidence(pack, sec):
    if sec is None or sec.empty:
        return
    for _, row in sec.head(6).iterrows():
        form = clean_text(row.get("Formulář"))
        date = clean_text(row.get("Datum"))
        report = clean_text(row.get("Datum výkazu"))
        doc = clean_text(row.get("Dokument"))
        acc = clean_text(row.get("Accession"))
        url = clean_text(row.get("Odkaz"))
        claim = f"SEC podání: {form or 'neuvedeno'}"
        if report:
            claim += f"; období výkazu: {report}"
        if doc:
            claim += f"; dokument: {doc}"
        pack.append(_analyst_source_item(
            "regulatory", "SEC / EDGAR", date, claim,
            period=report, url=url, extra=f"Accession: {acc}" if acc else ""
        ))


def analyst_build_evidence_pack(company, ticker, exchange, q, annual, quarterly, news, sec):
    """Evidence layer. Screener data, score ani důvod výběru se sem nikdy nepředávají."""
    pack = []
    _analyst_add_profile_evidence(pack, q)
    _analyst_add_financial_evidence(pack, annual, quarterly)
    _analyst_add_news_evidence(pack, news)
    _analyst_add_sec_evidence(pack, sec)
    return pack


def _analyst_evidence_text(pack, max_chars=12000):
    rows = []
    for i, item in enumerate(pack, 1):
        line = (
            f"[E{i}] {item['type']} | {item['source']} | {item['date']} | "
            f"{item['period']} | {item['claim']}"
        )
        if item.get("extra"):
            line += f" | {item['extra']}"
        rows.append(line)
    text = "\n".join(rows)
    return text[:max_chars]


def _analyst_financial_change_summary(annual, quarterly):
    parts = []
    if annual is not None and not annual.empty:
        parts.append(f"Dostupná roční řada: {len(annual)} fiskálních období; období jsou označena skutečným datem konce FY.")
        for col, label in [("Revenue", "tržby"), ("Net Income", "čistý zisk"), ("FCF", "FCF"), ("Debt", "dluh")]:
            if col not in annual.columns:
                continue
            s = pd.to_numeric(annual[col], errors="coerce").dropna()
            if len(s) >= 2 and s.iloc[0] != 0:
                change = (s.iloc[-1] / s.iloc[0] - 1) * 100
                parts.append(f"{label}: {change:+.1f} % mezi nejstarším a nejnovějším dostupným FY.")
    if quarterly is not None and not quarterly.empty:
        parts.append(f"Dostupná kvartální řada: {len(quarterly)} fiskálních kvartálů; označení Q vychází z fiskálního roku.")
        for col, label in [("Revenue", "tržby"), ("Net Income", "čistý zisk"), ("FCF", "FCF")]:
            if col not in quarterly.columns:
                continue
            s = pd.to_numeric(quarterly[col], errors="coerce").dropna()
            if len(s) >= 2 and s.iloc[0] != 0:
                change = (s.iloc[-1] / s.iloc[0] - 1) * 100
                parts.append(f"{label}: {change:+.1f} % mezi nejstarším a nejnovějším dostupným kvartálem.")
    ttm = analyst_ttm_from_quarters(quarterly)
    if not ttm.empty:
        parts.append("TTM je součet posledních čtyř dostupných kvartálů; není vydáváno za samostatný fiskální rok.")
    return "\n".join(parts)


def _analyst_price_evidence(price):
    if price is None or price.empty or "Close" not in price.columns:
        return ""
    p = pd.to_numeric(price["Close"], errors="coerce").dropna()
    if len(p) < 60:
        return ""
    last = p.iloc[-1]
    parts = [f"Poslední dostupná cena: {last:.2f}"]
    if len(p) > 252:
        parts.append(f"12M změna ceny: {(last / p.iloc[-253] - 1) * 100:+.1f} %")
    if len(p) > 756:
        parts.append(f"3Y změna ceny: {(last / p.iloc[-757] - 1) * 100:+.1f} %")
    return "; ".join(parts)


@st.cache_data(ttl=60, show_spinner=False)
def groq_connection_diagnostics(api_key, model="openai/gpt-oss-120b"):
    results = []
    base = "https://api.groq.com"
    headers = {"Authorization": f"Bearer {api_key}", "User-Agent": "Stock-Screener/6.17"}
    try:
        r = requests.get(base + "/", headers={"User-Agent": "Stock-Screener/6.17"}, timeout=12)
        results.append({"test": "api.groq.com – základní dostupnost", "status": r.status_code, "detail": r.text[:300]})
    except Exception as e:
        results.append({"test": "api.groq.com – základní dostupnost", "status": "ERROR", "detail": f"{type(e).__name__}: {e}"})
    try:
        r = requests.get(base + "/openai/v1/models", headers=headers, timeout=15)
        results.append({"test": "Groq /openai/v1/models", "status": r.status_code, "detail": r.text[:700]})
    except Exception as e:
        results.append({"test": "Groq /openai/v1/models", "status": "ERROR", "detail": f"{type(e).__name__}: {e}"})
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply exactly with three short lines: OK / MODEL / READY."}],
        "temperature": 0,
        "max_completion_tokens": 600,
        "reasoning_effort": "low",
        "include_reasoning": False
    }
    try:
        r = requests.post(base + "/openai/v1/chat/completions", headers={**headers, "Content-Type": "application/json"}, json=payload, timeout=30)
        detail = r.text[:1600]
        if r.status_code == 200:
            try:
                data = r.json()
                choice = (data.get("choices") or [{}])[0]
                msg = choice.get("message") or {}
                usage = data.get("usage") or {}
                detail = (
                    f"finish_reason={choice.get('finish_reason')}\n"
                    f"completion_tokens={usage.get('completion_tokens')}\n"
                    f"reasoning_tokens={usage.get('reasoning_tokens')}\n\n"
                    f"{clean_text(msg.get('content')) or '(prázdný content)'}"
                )
            except Exception:
                pass
        results.append({"test": f"Groq Chat Completions – {model}", "status": r.status_code, "detail": detail})
    except Exception as e:
        results.append({"test": f"Groq Chat Completions – {model}", "status": "ERROR", "detail": f"{type(e).__name__}: {e}"})
    return results


@st.cache_data(ttl=900, show_spinner=False)
def analyst_ai_synthesis(company, ticker, exchange, q, annual, quarterly, news, sec, price=None):
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        return {"ok": False, "error": "Chybí GROQ_API_KEY ve Streamlit Secrets.", "text": "", "model": "openai/gpt-oss-120b"}

    pack = analyst_build_evidence_pack(company, ticker, exchange, q, annual, quarterly, news, sec)
    evidence = _analyst_evidence_text(pack, max_chars=7000)
    fin = _analyst_financial_change_summary(annual, quarterly)
    price_ctx = _analyst_price_evidence(price)

    allowed_stories = [
        "Kvalitní compounder", "Kvalita za rozumnou cenu", "Růst za rozumnou cenu",
        "Value / levná firma", "Provozní turnaround", "Cyklické zotavení",
        "Aktivové / finanční zotavení", "Realitní hodnota", "Provozní zlepšení",
        "Růstové zotavení", "Vysoký růst / dražší příběh", "Value trap – varování",
        "Nejasný / smíšený příběh"
    ]

    prompt = f"""Jsi seniorní analytik veřejně obchodované společnosti {company} ({ticker}, {exchange}).

HLAVNÍ ÚKOL
Z dostupné evidence vytvoř použitelný pracovní investiční obraz firmy. Nejde jen o výčet zpráv. Hledej kauzální řetězec:
zdroj → událost → změna ekonomiky firmy → dopad na výsledky/cash flow/riziko/ocenění → co to znamená pro pracovní příběh.

DŮLEŽITÉ: ANALYTICKÁ ODVAHA
- Nemusíš být neutrální jen proto, že existují protichůdné signály. Pokud evidence jasně podporuje určitou interpretaci, řekni ji přímo.
- „Nejasný / smíšený příběh“ použij pouze tehdy, když skutečně nelze rozumně určit dominantní ekonomický obraz firmy.
- Přítomnost jednoho nebo dvou negativních faktorů sama o sobě NEZNAMENÁ Nejasný příběh.
- Pokud jsou současně silné pozitivní i negativní signály, urč dominantní základní příběh a popiš, co jej právě mění nebo ohrožuje.
- Odděluj tři věci: (1) základní charakter podniku, (2) aktuální změnu, (3) ocenění. Jedna negativní zpráva nemá automaticky přepsat základní charakter firmy.
- „Inference:“ používej pro vlastní analytický závěr. Nemá být omluvou ani opakováním faktů.
- Opatrnost používej hlavně tam, kde tvrdíš konkrétní kauzalitu, kterou evidence přímo nedokládá. Např. můžeš říct „pravděpodobně zvyšuje tlak na očekávání růstu“, ale ne tvrdit „způsobilo pokles akcie“, pokud to evidence nedokládá.
- Když je ekonomický význam dostatečně zřejmý z čísel a firemních informací, formuluj závěr jasně.
- Posuzuj také tři podpůrné dimenze pracovního příběhu: **konkurenční výhoda/moat, management a odvětví**.
- U moat uváděj jen skutečně doložené mechanismy (např. switching costs, síťový efekt, nákladová výhoda, rozsah, regulace, značka, IP, data, distribuce, zákaznické vazby). Pokud evidence nestačí, napiš **Neznáme:**.
- U managementu hledej pouze doložené signály: plnění/slib vs. skutečnost, změny guidance, konzistence komunikace, přiznání chyb, pobídky, vlastnictví akcií, SBC, insider aktivita nebo personální změny. Nepřisuzuj managementu motivy bez evidence.
- U odvětví odděluj firemně specifické změny od změn celého sektoru. Pokud zdroj dokládá pouze sektorový trend, neprezentuj jej jako specifickou výhodu či problém firmy.
- Sílu pracovního příběhu určuj nejen podle krátkodobého FCF a marží, ale také podle toho, zda dlouhodobou ekonomiku firmy podporují konkurenční výhoda, kvalita kapitálu a udržitelnost ziskovosti. Pokud tyto informace chybí, nesnižuj příběh automaticky na „Nejasný“, ale jasně označ, co zatím není prokázáno.

PŘÍSNÁ EVIDENČNÍ PRAVIDLA
- Používej pouze níže uvedenou evidenci. Nevymýšlej čísla, události, výroky, zdroje ani odkazy.
- Faktická tvrzení označ [E#].
- Logickou interpretaci označ **Inference:**.
- Chybějící zásadní informaci označ **Neznáme:**.
- Pokud zdroje odporují, ukaž konflikt; nevyrob smíření bez opory.
- Nikdy nepřisuzuj růst/pokles konkrétnímu segmentu bez přímé evidence.
- Nepředpokládej, že fiskální rok končí 31.12. Respektuj označení období a skutečná data konce FY.
- TTM jsou poslední 4 dostupná čtvrtletí, nikoli automaticky kalendářní rok.
- Nezaměňuj procentní změnu za absolutní hodnotu.
- Jednotlivá čtvrtletí FCF jsou sezónní a nesmějí být prezentována jako „růst FCF“
  pouze na základě mezikvartálního procenta. Pro trend FCF používej především celé
  FY nebo TTM; pokud porovnáváš dvě čtvrtletí, výslovně uveď, že jde o změnu
  mezi obdobími, nikoli o tempo růstu podniku.
- Pokud je FY2025 nebo jiný fiskální rok matematicky odvozen z TTM minus tří
  následujících reportovaných kvartálů, považuj jej za přesnou rekonstrukci
  dostupných dat, ale ne za nový reportovaný údaj.
- Opakované články o stejné události slouč do jednoho tématu.
- Žádné Buy/Hold/Sell, skóre, pořadí nebo doporučení.

POVOLENÉ NÁZVY PRACOVNÍHO PŘÍBĚHU
{'; '.join(allowed_stories)}

FIRMA: {company}
SEKTOR: {clean_text(q.get('sector'))} | ODVĚTVÍ: {clean_text(q.get('industry'))}

FINANČNÍ KONTEXT:
{fin or 'Není k dispozici.'}

CENOVÝ KONTEXT:
{price_ctx or 'Není k dispozici.'}

EVIDENCE:
{evidence or 'Není k dispozici.'}

VÝSTUP

DŮLEŽITÉ: nejdříve vždy napiš krátké ROZHODOVACÍ JÁDRO. Je povinné a nesmí být vynecháno ani tehdy, když je odpověď dlouhá. Teprve potom pokračuj podrobnou analýzou.

## ROZHODOVACÍ JÁDRO
STORY: [jeden povolený název]
PROFILE: [jedna až dvě stručné české věty o ekonomickém modelu firmy]
THESIS: [jedna stručná věta: proč je tento STORY nyní dominantní]
THREAT: [jedna stručná věta: co jej právě nejvíce ohrožuje]
COUNTER: [jedna stručná věta: nejsilnější protiargument]
CHANGE: [jedna stručná věta: co by muselo nastat, aby se STORY změnil]

## Pracovní investiční příběh
**Základní charakter firmy:** [stručná ekonomická charakteristika, ne povolený štítek]
**Aktuální stav:** [co se právě mění; např. růstové zpomalení, provozní zlepšení, restrukturalizace]
**Valuační kontext:** [co říká dostupné ocenění o očekáváních; bez falešné přesnosti]
**Hlavní pracovní příběh:** **[jeden povolený název]**
**Proč:** 2–4 konkrétní důvody s [E#] + jasná **Inference:**
**Co tento příběh právě ohrožuje:** ...
**Protiargument:** ...
**Alternativní interpretace:** ...
**Co by změnilo můj pracovní příběh:** ...

Pozor: „Protiargument“ ani „Alternativní interpretace“ nesmí automaticky vést k Nejasnému příběhu. Je normální, že pracovní příběh má protiváhu.

## Co se ve firmě právě mění
Vyber nejvýše 3 skutečně odlišné změny. Neopakuj jeden příběh ve více variantách. Piš úsporně; každé téma maximálně 5–7 krátkých řádků.
Pro každé téma:
### 1. [konkrétní změna]
**Co víme:** ... [E#]
**Co se mění:** ...
**Ekonomický dopad:** ...
**Inference:** ...
**Neznáme:** ...
**Charakter změny:** strukturální / cyklická / dočasná / jednorázová / nejasná
**Co by ji potvrdilo nebo vyvrátilo:** ...

## Vztah k finančním výsledkům
**Co vidíme v číslech:** ...
**Co to podle mě znamená:** ...
**Co z čísel nelze zjistit:** ...

## Co si navzájem potvrzují nebo odporují zdroje
Uveď pouze skutečné vazby nebo konflikty. Pokud evidence převážně souhlasí, řekni to stručně.

## Co bych teď sledoval
3–5 konkrétních ověřitelných věcí, pouze v krátkých odrážkách.
"""

    payload = {
        "model": "openai/gpt-oss-120b",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.15,
        "max_completion_tokens": 2800,
        "reasoning_effort": "low",
        "include_reasoning": False
    }
    def _groq_call(current_payload):
        return requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "User-Agent": "Stock-Screener/6.26"},
            json=current_payload, timeout=90
        )

    try:
        r = _groq_call(payload)
        # Groq free-tier context is limited. Retry once with a smaller evidence pack
        # if the combined prompt + completion request is rejected as too large.
        if r.status_code == 413:
            compact_evidence = _analyst_evidence_text(pack, max_chars=5200)
            retry_prompt = prompt.replace(evidence, compact_evidence)
            retry_payload = dict(payload)
            retry_payload["messages"] = [{"role": "user", "content": retry_prompt}]
            retry_payload["max_completion_tokens"] = 2200
            r = _groq_call(retry_payload)
        if r.status_code != 200:
            return {"ok": False, "error": f"Groq HTTP {r.status_code}: {r.text[:1200]}", "text": "", "model": "openai/gpt-oss-120b", "evidence_count": len(pack)}
        data = r.json()
        choice = (data.get("choices") or [{}])[0]
        text = clean_text((choice.get("message") or {}).get("content"))
        finish = choice.get("finish_reason")
        if len(text) < 250:
            return {"ok": False, "error": f"Groq vrátil příliš krátkou odpověď (finish_reason={finish}).", "text": text, "model": "openai/gpt-oss-120b", "evidence_count": len(pack)}
        result = {"ok": True, "error": "", "text": text, "model": "openai/gpt-oss-120b", "evidence_count": len(pack), "finish_reason": finish, "usage": data.get("usage", {})}
        # The long answer can be truncated after the story heading. In that case the old
        # parser incorrectly fell back to 'Nejasný'. Recover only the compact decision fields.
        has_story_title = bool(
            re.search(r"(?im)^\s*STORY\s*:\s*(?:\*{0,2})[^\n]+", text)
            or re.search(r"(?im)^\s*\*{0,2}Hlavní pracovní příběh\s*:\s*\*{0,2}[^\n]+", text)
        )
        has_profile = bool(
            re.search(r"(?im)^\s*PROFILE\s*:\s*[^\n]+", text)
            or re.search(r"(?im)^\s*\*{0,2}Základní charakter firmy\s*:", text)
        )
        has_changes = bool(re.search(r"(?im)^\s*##\s*Co se ve firmě právě mění\s*$", text))
        # Recovery is also needed when the answer contains story/profile but
        # was truncated before the current-developments section.
        if not has_story_title or not has_profile or not has_changes:
            result["recovery"] = analyst_ai_recovery(company, ticker, exchange, q, annual, quarterly, news, sec, price)
        return result
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "text": "", "model": "openai/gpt-oss-120b", "evidence_count": len(pack)}


def _analyst_extract_section(text, heading):
    """Return one markdown H2 section without accidentally swallowing adjacent sections."""
    if not text:
        return ""
    marker = f"## {heading}"
    pos = text.find(marker)
    if pos < 0:
        return ""
    start = pos + len(marker)
    rest = text[start:]
    m = re.search(r"(?m)^##\s+", rest)
    if m:
        rest = rest[:m.start()]
    return rest.strip()


@st.cache_data(ttl=900, show_spinner=False)
def analyst_ai_recovery(company, ticker, exchange, q, annual, quarterly, news, sec, price=None):
    """Compact fallback used when the main AI answer is truncated.
    It recovers the story/profile, the most important current changes, and follow-up checks.
    """
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        return {"ok": False, "text": "", "error": "Chybí GROQ_API_KEY ve Streamlit Secrets."}
    pack = analyst_build_evidence_pack(company, ticker, exchange, q, annual, quarterly, news, sec)
    evidence = _analyst_evidence_text(pack, max_chars=4200)
    prompt = f"""Jsi analytik společnosti {company} ({ticker}, {exchange}).
Použij pouze níže uvedenou evidenci. Nevymýšlej fakta.

Vrať PŘESNĚ tyto čtyři bloky a nic jiného:
STORY: [jeden název z povoleného seznamu]
PROFILE: [jedna až dvě stručné české věty o ekonomickém modelu firmy]
CHANGES:
### 1. [konkrétní změna]
**Co víme:** [fakt s E#]
**Co se mění:** [stručně]
**Ekonomický dopad:** [stručně]
**Inference:** [stručně]
**Neznáme:** [stručně]
**Charakter změny:** [strukturální / cyklická / dočasná / jednorázová / nejasná]
**Co by ji potvrdilo nebo vyvrátilo:** [stručně]
FOLLOW: [3 krátké body oddělené znakem |, co je nejdůležitější dál ověřit]

Pokud je důležitých změn více, přidej nejvýše ještě blok ### 2. Nepiš více než 2 změny. Používej pouze dostupnou evidenci.

Povolené STORY: Kvalitní compounder; Kvalita za rozumnou cenu; Růst za rozumnou cenu; Value / levná firma; Provozní turnaround; Cyklické zotavení; Aktivové / finanční zotavení; Realitní hodnota; Provozní zlepšení; Růstové zotavení; Vysoký růst / dražší příběh; Value trap – varování; Nejasný / smíšený příběh

EVIDENCE:
{evidence}
"""
    payload = {
        "model":"openai/gpt-oss-120b",
        "messages":[{"role":"user","content":prompt}],
        "temperature":0.1,
        "max_completion_tokens":900,
        "reasoning_effort":"low",
        "include_reasoning":False
    }
    try:
        r=requests.post("https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization":f"Bearer {api_key}","Content-Type":"application/json","User-Agent":"Stock-Screener/6.26"},
            json=payload, timeout=60)
        if r.status_code!=200:
            return {"ok":False,"text":"","error":f"Groq recovery HTTP {r.status_code}: {r.text[:800]}"}
        data=r.json(); choice=(data.get("choices") or [{}])[0]
        text=clean_text((choice.get("message") or {}).get("content"))
        return {"ok":bool(text),"text":text,"error":""}
    except Exception as e:
        return {"ok":False,"text":"","error":f"{type(e).__name__}: {e}"}


def _analyst_recovery_changes_text(recovery_text):
    if not recovery_text:
        return ""
    m = re.search(r"(?is)\bCHANGES\s*:\s*(.*?)(?=\n\s*FOLLOW\s*:|\Z)", recovery_text)
    return m.group(1).strip() if m else ""


def analyst_current_developments(company, q, annual, quarterly, news, sec, ticker="", exchange="", price=None, ai_result=None):
    # Reuse the already generated AI result. Calling Groq a second time here
    # wastes quota and can produce a different answer from the one used by the
    # story section.
    result = ai_result if isinstance(ai_result, dict) else analyst_ai_synthesis(company, ticker, exchange, q, annual, quarterly, news, sec, price)
    if result.get("ok"):
        text = result.get("text", "")
        section = _analyst_extract_section(text, "Co se ve firmě právě mění")
        if section:
            return section
        # V6.24: if the long answer was truncated before this section, use the
        # already-available compact recovery result instead of returning an empty section.
        recovery = result.get("recovery") or {}
        recovery_changes = _analyst_recovery_changes_text(recovery.get("text", "") if recovery.get("ok") else "")
        if recovery_changes:
            return recovery_changes
        return "### ⚠️ AI neposkytla sekci „Co se ve firmě právě mění“.\n\nPracovní investiční příběh je dostupný v samostatné části níže."
    return "### ⚠️ AI syntéza není dostupná\n\n" + result.get("error", "Neznámá chyba.")


def analyst_story_hypothesis(q, annual, quarterly, news, sec, ai_result=None):
    """Extract the structured working story without inventing a deterministic score."""
    if ai_result and ai_result.get("ok"):
        text = ai_result.get("text", "")
        recovery = ai_result.get("recovery") or {}
        recovery_text = recovery.get("text", "") if recovery.get("ok") else ""
        # V6.28: the compact decision core is placed first in the main answer,
        # so story extraction remains reliable even when the long answer is truncated.
        mmain = re.search(r"(?m)^\s*STORY\s*:\s*(.+)$", text, flags=re.I)
        if mmain:
            title = mmain.group(1).strip().strip("*# ")
            allowed = {
                "Kvalitní compounder", "Kvalita za rozumnou cenu", "Růst za rozumnou cenu",
                "Value / levná firma", "Provozní turnaround", "Cyklické zotavení",
                "Aktivové / finanční zotavení", "Realitní hodnota", "Provozní zlepšení",
                "Růstové zotavení", "Vysoký růst / dražší příběh", "Value trap – varování",
                "Nejasný / smíšený příběh"
            }
            for candidate in allowed:
                if candidate.lower() == title.lower():
                    return candidate, text
        mrec = re.search(r"(?m)^\s*STORY\s*:\s*(.+)$", recovery_text, flags=re.I)
        if mrec:
            title = mrec.group(1).strip().strip("*# ")
            allowed = {
                "Kvalitní compounder", "Kvalita za rozumnou cenu", "Růst za rozumnou cenu",
                "Value / levná firma", "Provozní turnaround", "Cyklické zotavení",
                "Aktivové / finanční zotavení", "Realitní hodnota", "Provozní zlepšení",
                "Růstové zotavení", "Vysoký růst / dražší příběh", "Value trap – varování",
                "Nejasný / smíšený příběh"
            }
            for candidate in allowed:
                if candidate.lower() == title.lower():
                    return candidate, recovery_text
        
        if "## Pracovní investiční příběh" in text:
            part = _analyst_extract_section(text, "Pracovní investiční příběh")
            lines = [x.strip() for x in part.splitlines() if x.strip()]
            title = "Nejasný / smíšený příběh"
            for line in lines:
                m = re.search(r"Hlavní pracovní příběh\s*:\s*\*\*([^*]+)\*\*", line, flags=re.I)
                if m:
                    title = m.group(1).strip()
                    break
                m = re.search(r"Hlavní pracovní příběh\s*:\s*([^\n]+)", line, flags=re.I)
                if m:
                    title = m.group(1).strip("*# ")
                    break
            allowed = {
                "Kvalitní compounder", "Kvalita za rozumnou cenu", "Růst za rozumnou cenu",
                "Value / levná firma", "Provozní turnaround", "Cyklické zotavení",
                "Aktivové / finanční zotavení", "Realitní hodnota", "Provozní zlepšení",
                "Růstové zotavení", "Vysoký růst / dražší příběh", "Value trap – varování",
                "Nejasný / smíšený příběh"
            }
            if title not in allowed:
                # Backward-compatible fallback for older cached AI output.
                for candidate in allowed:
                    if candidate.lower() in title.lower():
                        title = candidate
                        break
                else:
                    title = "Nejasný / smíšený příběh"
            return title, part.strip()
    return "Nejasný / smíšený příběh", "Pracovní příběh nebyl mechanicky dopočítán, protože AI syntéza nebyla dostupná."



def _analyst_czech_sector(value):
    mapping = {
        "Healthcare": "Zdravotnictví",
        "Medical Devices": "Zdravotnické technologie",
        "Diagnostics & Research": "Diagnostika a výzkum",
        "Technology": "Technologie",
        "Financial Services": "Finanční služby",
        "Industrials": "Průmysl",
        "Consumer Cyclical": "Spotřební cyklické zboží",
        "Consumer Defensive": "Spotřební defenzivní zboží",
        "Energy": "Energetika",
        "Basic Materials": "Základní materiály",
        "Communication Services": "Komunikační služby",
        "Real Estate": "Nemovitosti",
        "Utilities": "Síťová odvětví / utility",
    }
    return mapping.get(clean_text(value), clean_text(value) or "—")


def _analyst_czech_country(value):
    mapping = {
        "Germany": "Německo", "United States": "USA", "United Kingdom": "Velká Británie",
        "France": "Francie", "Netherlands": "Nizozemsko", "Switzerland": "Švýcarsko",
        "Japan": "Japonsko", "China": "Čína", "Canada": "Kanada",
    }
    return mapping.get(clean_text(value), clean_text(value) or "—")


def _analyst_story_profile_text(ai_result):
    if not ai_result or not ai_result.get("ok"):
        return ""
    recovery = ai_result.get("recovery") or {}
    recovery_text = recovery.get("text", "") if recovery.get("ok") else ""
    # V6.28: prefer the main answer's compact PROFILE field.
    m = re.search(r"(?m)^\s*PROFILE\s*:\s*(.+)$", ai_result.get("text", ""), flags=re.I)
    if m:
        return m.group(1).strip()
    m = re.search(r"(?m)^\s*PROFILE\s*:\s*(.+)$", recovery_text, flags=re.I)
    if m:
        return m.group(1).strip()
    part = _analyst_extract_section(ai_result.get("text", ""), "Pracovní investiční příběh")
    for line in part.splitlines():
        if line.strip().lower().startswith("**základní charakter firmy:**"):
            return re.sub(r"^\*\*Základní charakter firmy:\*\*\s*", "", line.strip(), flags=re.I).strip()
    return ""


def _analyst_display_financial_table(df):
    if df is None or df.empty:
        return df
    out = df.copy()
    rename = {
        "Období":"Období", "Revenue":"Tržby", "Net Income":"Čistý zisk",
        "Operating Income":"Provozní zisk", "Operating Cash Flow":"Provozní cash flow",
        "Capital Expenditure":"Kapitálové výdaje", "Debt":"Dluh",
        "Equity":"Vlastní kapitál", "Net Margin %":"Čistá marže", "FCF":"FCF"
    }
    out = out.rename(columns=rename)
    for col in out.columns:
        if col == "Období":
            continue
        if "marže" in col.lower():
            out[col] = out[col].apply(lambda x: "—" if pd.isna(safe_float(x)) else f"{safe_float(x):.1f} %".replace(".", ","))
        else:
            out[col] = out[col].apply(lambda x: analyst_human_number(x, 1))
    def period_label(x):
        x = clean_text(x)
        m = re.search(r"FY(\d{4}).*?ended (\d{4})-(\d{2})-(\d{2})", x, re.I)
        if m:
            return f"FY {m.group(1)} (k {int(m.group(4))}. {int(m.group(3))}. {m.group(2)})"
        m = re.search(r"(Q\d).*?FY(\d{4}).*?ended (\d{4})-(\d{2})-(\d{2})", x, re.I)
        if m:
            return f"{m.group(1)} FY {m.group(2)} (k {int(m.group(5))}. {int(m.group(4))}. {m.group(3)})"
        if x.startswith("TTM"):
            return x.replace("ended ", "k ")
        return x
    if "Období" in out.columns:
        out["Období"] = out["Období"].map(period_label)
    return out


def _analyst_valuation_rows(q, price):
    rows = []
    labels = [("P/E", "pe"), ("Forward P/E", "forward_pe"), ("P/S", "ps"), ("P/B", "pb")]
    for label, key in labels:
        raw = q.get(key)
        v = safe_float(raw)
        if not pd.isna(v):
            rows.append({"Ukazatel": label, "Hodnota": analyst_human_number(v, 1) + "×"})
    return pd.DataFrame(rows, columns=["Ukazatel", "Hodnota"])


def _analyst_valuation_comment(q, price):
    pe, fpe = safe_float(q.get("pe")), safe_float(q.get("forward_pe"))
    ps, pb = safe_float(q.get("ps")), safe_float(q.get("pb"))
    bits = []
    if not pd.isna(pe):
        bits.append(f"P/E {analyst_human_number(pe,1)}× znamená, že trh firmě stále přisuzuje nezanedbatelnou kvalitu a očekávání")
    if not pd.isna(fpe) and not pd.isna(pe):
        if fpe < pe:
            bits.append(f"Forward P/E {analyst_human_number(fpe,1)}× je nižší než současné P/E, takže ocenění předpokládá růst budoucího zisku")
        elif fpe > pe:
            bits.append(f"Forward P/E {analyst_human_number(fpe,1)}× je vyšší než současné P/E, což neukazuje na očekávané zlepšení zisku")
    if not pd.isna(ps):
        bits.append(f"P/S je {analyst_human_number(ps,1)}×")
    if not pd.isna(pb):
        bits.append(f"P/B je {analyst_human_number(pb,1)}×")
    return ". ".join(bits) + "." if bits else "Valuační data nejsou dostatečná pro smysluplný komentář."


def analyst_render(ticker_input):
    st.title("🔎 Analytik")
    st.caption("Nezávislé výzkumné jádro · Analytik nevidí Screener ani důvod, proč byl titul vybrán.")

    c1, c2 = st.columns([3, 1])
    with c1:
        ticker_input = st.text_input("Ticker", value=ticker_input, placeholder="např. SHL, GOOGL, ONC, AT1.DE")
    with c2:
        default_ex = 2 if str(ticker_input).upper().endswith(".DE") else 0
        exchange = st.selectbox("Trh", ["NASDAQ", "NYSE", "XETRA"], index=default_ex)

    analyse = st.button("🔬 Spustit analytické jádro", type="primary")
    with st.expander("🧪 Diagnostika Groq připojení", expanded=False):
        st.write("Tento test nic nemění na API klíči. Ověří pouze, zda Streamlit Cloud dokáže z tohoto serveru oslovit Groq a zda Groq přijme klíč i model.")
        test_groq = st.button("Otestovat Groq připojení", key="groq_diag_button")
        if test_groq:
            try:
                api_key = st.secrets["GROQ_API_KEY"]
                diag = groq_connection_diagnostics(api_key)
                for item in diag:
                    st.markdown(f"**{item['test']}** → `{item['status']}`")
                    st.code(item["detail"] or "(prázdná odpověď)")
                statuses = [x["status"] for x in diag]
                if 403 in statuses and any(x.get("status") == 403 and "Access denied" in x.get("detail", "") for x in diag):
                    st.warning("Groq vrací HTTP 403 s hlášením Access denied. To silně ukazuje na blokaci síťového prostředí/IP, nikoli na chybu analytického promptu.")
                elif 401 in statuses:
                    st.warning("Groq vrací HTTP 401. Pravděpodobný problém je API klíč nebo jeho oprávnění.")
                elif 200 in statuses:
                    st.success("Groq odpovídá HTTP 200 alespoň na jednom rozhodujícím testu. Pokud Analytik přesto selhává, budeme hledat problém v konkrétním požadavku.")
            except Exception as e:
                st.error(f"Diagnostiku se nepodařilo spustit: {type(e).__name__}: {e}")
    if not analyse:
        st.info("Zadej ticker a spusť analytické jádro. Analytik je nezávislý na Screeneru.")
        return

    ticker = clean_text(ticker_input).upper()
    if not ticker:
        st.warning("Zadej ticker.")
        return

    yahoo_ticker = analyst_yahoo_ticker(ticker, exchange)
    status = st.empty()
    status.info("1/5 Ověřuji identitu firmy a načítám veřejná data…")
    # V6.28.7: Analytik identity hardening + direct Yahoo data path.
    # The former V6.28.x diagnostic block is intentionally removed from the
    # normal UI so it cannot interrupt the analytical workflow.

    # V6.28.9: HARD IDENTITY FIREWALL. No AI, financial history or news may run
    # until the requested Yahoo ticker is independently confirmed by Chart +
    # exact Search quote. An ambiguous identity is a hard stop, not a fallback.
    q = analyst_get_quote_data(yahoo_ticker)
    company = q.get("name") or ""
    returned_symbol = clean_text(q.get("symbol")).upper()
    expected_symbol = clean_text(yahoo_ticker).upper()
    if not q.get("identity_ok") or not company or returned_symbol != expected_symbol:
        st.error(
            f"Identitu titulu se nepodařilo bezpečně ověřit: požadováno {ticker} / {yahoo_ticker}. "
            f"{q.get('identity_reason') or 'Neznámý důvod.'} Analýza byla z bezpečnostních důvodů zastavena."
        )
        st.info(
            "Analytik V6.28.9 záměrně nepoužije přibližně nalezenou firmu ani data z jiného titulu. "
            "Pokud se zastavení opakuje, pošli mi diagnostický výpis – nebudeme to obcházet dalším fallbackem."
        )
        return

    status.info("2/5 Sestavuji dlouhodobý finanční trend, poslední kvartály a cenu…")
    annual = analyst_get_financial_history(yahoo_ticker)
    quarterly = analyst_get_quarterly_history(yahoo_ticker)
    price = analyst_price_history(yahoo_ticker)
    sec = analyst_sec_filings(ticker) if exchange in ("NASDAQ", "NYSE") else pd.DataFrame()

    status.info("3/5 Hledám aktuální firemně relevantní události…")
    news = analyst_google_news(company, ticker, q.get("ir_website") or q.get("website"))

    status.info("4/5 Stavím důkazní balíček a provádím AI syntézu…")
    ai_result = analyst_ai_synthesis(company, ticker, exchange, q, annual, quarterly, news, sec, price)
    current = analyst_current_developments(company, q, annual, quarterly, news, sec, ticker, exchange, price, ai_result=ai_result)
    primary, story_reason = analyst_story_hypothesis(q, annual, quarterly, news, sec, ai_result)
    price_comment = analyst_price_commentary(price)
    status.success("5/5 Analytické jádro dokončeno.")

    st.markdown(f"## {company}")
    st.caption(f"Ticker **{ticker}** · Yahoo **{yahoo_ticker}** · {exchange} · načteno {datetime.now().strftime('%d.%m.%Y %H:%M')}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Cena", f"{analyst_human_number(q.get('price'), 2)} {q.get('currency', '')}")
    m2.metric("Tržní kapitalizace", analyst_human_number(q.get("market_cap")))
    m3.metric("P/E", analyst_human_number(q.get("pe"), 1))
    roe = safe_float(q.get("roe"))
    m4.metric("ROE", "—" if pd.isna(roe) else f"{roe * 100:.1f} %".replace(".", ","))

    with st.expander("🏢 1. Základní profil a obchodní model", expanded=True):
        st.write(f"**Sektor:** {_analyst_czech_sector(q.get('sector'))} · **Odvětví:** {q.get('industry') or '—'}")
        if q.get("country"):
            st.write(f"**Země:** {_analyst_czech_country(q.get('country'))}")
        profile_text = _analyst_story_profile_text(ai_result)
        if profile_text:
            st.write(profile_text)
        else:
            sector = _analyst_czech_sector(q.get("sector"))
            industry = clean_text(q.get("industry")) or "daném oboru"
            st.write(f"{company} působí v oblasti {sector.lower()} a zaměřuje se na {industry.lower()}. Detailní český profil z AI nebyl v tomto běhu dostupný.")

    with st.expander("🧠 2. Co se ve firmě právě mění", expanded=True):
        if ai_result.get("ok"):
            st.markdown(current)
        else:
            st.error(ai_result.get("error", "AI syntéza není dostupná."))

    with st.expander("📚 3. Důkazní balíček použitý AI", expanded=False):
        pack = analyst_build_evidence_pack(company, ticker, exchange, q, annual, quarterly, news, sec)
        st.caption(f"AI dostala {len(pack)} evidence položek.")
        for i, item in enumerate(pack, 1):
            st.markdown(f"**[E{i}] {item['source']}** · {item['date']}")
            st.write(item["claim"])
            if item.get("url"):
                st.markdown(f"[Zdroj]({item['url']})")

    with st.expander("📊 4. Finanční vývoj – trend, ne snapshot", expanded=True):
        st.write(analyst_financial_summary(annual, quarterly))
        if not annual.empty:
            st.markdown("**Celé účetní roky dostupné přes Yahoo Finance**")
            st.table(_analyst_display_financial_table(annual))
        if not quarterly.empty:
            st.markdown(f"**Posledních dostupných {len(quarterly)} fiskálních čtvrtletí**")
            st.table(_analyst_display_financial_table(quarterly))
        ttm = analyst_ttm_from_quarters(quarterly)
        if not ttm.empty:
            st.markdown("**TTM – poslední čtyři dostupná čtvrtletí**")
            st.table(_analyst_display_financial_table(ttm))

    with st.expander("💰 5. Valuace – jaká očekávání jsou v ceně", expanded=False):
        valuation_rows = _analyst_valuation_rows(q, price)
        if not valuation_rows.empty:
            st.table(valuation_rows)
            st.write(_analyst_valuation_comment(q, price))
        else:
            st.write("Valuační data nejsou dostupná.")
        st.caption("Valuace je zde kontext očekávání, nikoli výpočet falešně přesné férové ceny.")

    with st.expander("🧩 6. Pracovní investiční příběh", expanded=True):
        st.markdown(f"### {primary}")
        st.markdown(story_reason)
        st.caption("Pracovní hypotéza, nikoli investiční doporučení.")

    with st.expander("🧭 7. Cenový kontext", expanded=True):
        st.write(price_comment)

    with st.expander("🎯 8. Co má smysl dále ověřit", expanded=True):
        if ai_result.get("ok") and "## Co bych teď sledoval" in ai_result.get("text", ""):
            follow = ai_result["text"].split("## Co bych teď sledoval", 1)[1].strip()
            next_h2 = re.search(r"(?m)^##\s+", follow)
            if next_h2:
                follow = follow[:next_h2.start()].strip()
            st.markdown(follow or "AI neuvedla konkrétní body ke sledování.")
        elif ai_result.get("ok") and (ai_result.get("recovery") or {}).get("ok"):
            recovery_text = (ai_result.get("recovery") or {}).get("text", "")
            m = re.search(r"(?m)^\s*FOLLOW\s*:\s*(.+)$", recovery_text, flags=re.I)
            if m:
                st.markdown("\n".join(f"- {x.strip()}" for x in m.group(1).split("|") if x.strip()))
            else:
                st.write("AI neposkytla samostatnou sekci pro další ověření.")
        else:
            st.write("AI neposkytla samostatnou sekci pro další ověření.")

    with st.expander("📚 9. Zdroje a diagnostika", expanded=False):
        st.write(f"Yahoo Finance: {yahoo_ticker}")
        st.write(f"Relevantních zpráv: {len(news) if news is not None else 0}")
        st.write(f"Evidence položek: {ai_result.get('evidence_count', '—')}")
        st.write(f"AI model: {ai_result.get('model', '—')}")
        if not ai_result.get("ok"):
            st.error(ai_result.get("error", ""))
        if exchange in ("NASDAQ", "NYSE"):
            st.write(f"SEC podání načtena: {len(sec) if sec is not None else 0}")




# Page navigation. Analytik is deliberately isolated from the Screener execution path.
st.sidebar.markdown("## 🧭 Modul")
app_page = st.sidebar.radio("", ["📊 Screener", "🔎 Analytik"], index=0, label_visibility="collapsed")
if app_page == "🔎 Analytik":
    analyst_render("")
    st.stop()


# Sidebar
st.sidebar.header("⚙️ Nastavení")
universe_choice = st.sidebar.radio(
    "Univerzum / burza",
    ["Všechny burzy", "NASDAQ", "NYSE", "XETRA"],
    index=0,
    help="Pro rychlejší a cílenější screening zvol jednu burzu. Režim Všechny burzy zachová prohledání celého univerza."
)
selected_exchanges = ["NASDAQ", "NYSE", "XETRA"] if universe_choice == "Všechny burzy" else [universe_choice]
min_cap_b = st.sidebar.number_input("Min. Market Cap (mld.)", min_value=0.0, value=1.0, step=0.5)
max_candidates = st.sidebar.slider("Max. titulů pro fundamentální fázi", 100, 600, 350, 50)
max_text_candidates = st.sidebar.slider("Max. titulů pro textovou fázi", 0, 60, 40, 5)
max_price_candidates = st.sidebar.slider("Max. titulů pro cenovou fázi", 0, 60, 40, 5)
max_stage1 = st.sidebar.slider("Max. titulů z univerza do předvýběru", 200, 1500, 800, 100)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Jaký příběh hledám?")
story_options = [
    "🏆 Quality Compounder",
    "💎 Kvalita za rozumnou cenu",
    "🚀 Růst za rozumnou cenu",
    "💰 Value / levná firma",
    "🔄 Operating turnaround",
    "🔄 Recovery candidate",
    "🌐 Cyclical / commodity recovery",
    "🏗️ Asset / financial recovery",
    "🏢 Real-estate value",
    "🛠️ Operational improvement",
    "🚀 Growth / recovery",
    "🔥 High Growth / dražší příběh",
    "🪤 Value Trap – varování",
]
selected_stories = st.sidebar.multiselect("Příběhy", story_options, default=story_options[:9], help="Neatraktivní příběhy nemusíš hledat; Value Trap zde slouží jako výjimka – upozornění na levnou firmu se slabými základy.")
min_data = st.sidebar.slider("Min. počet dostupných parametrů", 3, len(PARAMS), 7, 1)
min_story_fit = st.sidebar.slider("Min. shoda s příběhem", 0, 100, 60, 5, help="Určuje, jak dobře musí titul odpovídat hledanému příběhu, aby se dostal mezi kandidáty. Nejde o investiční doporučení.")

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
fresh_run = st.sidebar.checkbox("🧪 Čerstvý běh bez cache", value=False, help="Vymaže Streamlit cache před spuštěním. Použij při diagnostice rozdílů mezi běhy.")

if refresh:
    st.cache_data.clear(); st.rerun()
if clear:
    st.session_state.pop("screening_results", None); st.rerun()
if not selected_exchanges:
    st.warning("Vyber alespoň jednu burzu."); st.stop()

st.info("Načítám aktuální seznam titulů z oficiálních zdrojů…")
try:
    universe = load_universe(selected_exchanges)
except Exception as e:
    st.error(f"Chyba při načtení univerza: {e}"); st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Celkem v univerzu", f"{len(universe):,}".replace(",", " "))
for i, ex in enumerate(selected_exchanges[:3], start=2):
    [c2, c3, c4][i-2].metric(ex, f"{int((universe['Exchange'] == ex).sum()):,}".replace(",", " "))
if universe_choice != "Všechny burzy":
    st.caption(f"🎯 Zvoleno cílené univerzum: **{universe_choice}** · {len(universe):,} titulů. Další fáze budou pracovat pouze s tímto trhem.")

st.markdown("### 🌍 Univerzum")
st.caption("NASDAQ/NYSE jsou získávány z Nasdaq Trader; XETRA z oficiálního seznamu Deutsche Börse. XETRA je omezeno na Instrument Type = CS (Common Stock / Equity).")

if not run and "screening_results" not in st.session_state:
    runtime = get_runtime_state()
    if runtime.get("status") in ("running", "error"):
        icon = "🟠" if runtime.get("status") == "running" else "🔴"
        st.warning(
            f"{icon} **Poslední screening nemá v této relaci uložený výsledek.** "
            f"Poslední zaznamenaná fáze: **{runtime.get('stage','—')}** · "
            f"{runtime.get('message','')} · {runtime.get('updated_at','')}"
        )
        if runtime.get("last_error"):
            st.code(runtime.get("last_error"), language="text")
        st.caption("Pokud byl běh přerušen, běžné cache výsledků umožní při novém spuštění přeskočit již načtené fundamenty a předselekci znovu použít.")
    st.info("Nastav příběhy a stiskni **🚀 Spustit screening**."); st.stop()

if run:
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + hashlib.md5(str(time.time_ns()).encode()).hexdigest()[:6]
    update_runtime(status="running", stage="1/6", message=f"Screening spuštěn · univerzum: {universe_choice} · {len(universe):,} titulů", error="", run_id=run_id)
    if fresh_run:
        st.cache_data.clear()
        st.session_state["screening_fresh_run"] = True
    else:
        st.session_state["screening_fresh_run"] = False
    st.info(f"🔄 **Screening běží.** Průběh se aktualizuje po každé dávce. **Stránku během běhu neobnovuj.**")
    stage_status = st.empty()
    stage_progress = st.progress(0, text="🔄 1/6 Předselekce trhu: připravuji…")

    def update_stage1_progress(value, message):
        update_runtime(status="running", stage="1/6", message=message, run_id=run_id)
        stage_progress.progress(max(0.0, min(1.0, float(value))), text=f"🔄 1/6 Předselekce trhu · {message}")

    effective_stage1 = min(int(max_stage1), int(len(universe)))
    stage_status.write(f"🔄 **1/6 Předselekce trhu:** zpracovávám univerzum **{universe_choice}** ({len(universe):,} titulů)…")
    stage1 = prefilter_by_market_data(universe, effective_stage1, _progress_callback=update_stage1_progress)
    update_runtime(status="running", stage="2/6", message=f"Předselekce dokončena · {universe_choice}: {len(stage1):,} titulů", run_id=run_id)
    st.session_state["screening_run_id"] = run_id
    stage_progress.progress(1.0, text=f"✅ 1/6 Předselekce trhu dokončena · {len(stage1):,} titulů")
    stage_status.write(f"✅ **1/6 dokončeno:** {len(stage1):,} titulů pokračuje do fundamentální fáze.")
    candidates = build_stage1_candidates(stage1, max_candidates)
    st.session_state["screening_stage1_tickers"] = stage1["Ticker"].astype(str).tolist() if "Ticker" in stage1.columns else []
    st.session_state["screening_fundamental_tickers"] = candidates["Ticker"].astype(str).tolist() if "Ticker" in candidates.columns else []
    rows = []
    status_text = st.empty()
    for i, row in candidates.iterrows():
        msg = f"{i+1}/{len(candidates)} · aktuálně {row['Ticker']}"
        update_runtime(status="running", stage="2/6", message=msg, run_id=run_id)
        status_text.write(f"🔄 **2/6 Fundamentální data:** {i+1}/{len(candidates)} · aktuálně **{row['Ticker']}**")
        try:
            result_row = fetch_fundamentals(row["Ticker"], row["Exchange"], row.get("Name", ""), row.get("ISIN", ""))
            rows.append(result_row)
        except Exception as exc:
            update_runtime(status="error", stage="2/6", message=msg, error=repr(exc), run_id=run_id)
            status_text.error(f"🔴 Chyba u {row['Ticker']}: {exc}")
            raise
        if (i + 1) % 10 == 0 or i + 1 == len(candidates):
            st.session_state["screening_checkpoint"] = {
                "run_id": run_id, "stage": "2/6", "done": int(i + 1),
                "total": int(len(candidates)), "last_ticker": str(row["Ticker"]),
                "rows": pd.DataFrame(rows).copy(),
            }
        stage_progress.progress((i+1)/max(1,len(candidates)), text=f"🔄 2/6 Fundamentální data · {i+1}/{len(candidates)} · {row['Ticker']}")
    update_runtime(status="running", stage="3/6", message=f"Fundamentální data dokončena: {len(rows)} titulů", run_id=run_id)
    status_text.write(f"✅ **2/6 Fundamentální data dokončena:** {len(rows)} titulů.")
    st.session_state["screening_results"] = pd.DataFrame(rows)
    st.session_state["screening_raw_results"] = pd.DataFrame(rows).copy()
    st.session_state["screening_pipeline_counts"] = {
        "Univerzum": universe_choice,
        "Celé univerzum": int(len(universe)),
        "Předselekce trhu": int(len(stage1)),
        "Fundamentální fáze": int(len(candidates)),
        "Načtené fundamenty": int(len(rows)),
    }

results_df = st.session_state.get("screening_results", pd.DataFrame())
if results_df.empty:
    st.warning("Pro vybrané nastavení nebyla načtena žádná data."); st.stop()

# Scores and stories
stage_progress.progress(0, text="🔄 3/6 Charakter + recovery + investiční příběh: vyhodnocuji…")
stage_status.write("🔄 **3/6 Charakter + recovery + investiční příběh:** vyhodnocuji fundamentální obraz…")
# Keep raw screening data separate from derived/evidence layers. Streamlit reruns
# (for example when changing the selected candidate) must not re-append columns.
raw_results = st.session_state.get("screening_raw_results")
if raw_results is None or raw_results.empty:
    raw_results = results_df.copy()

derived_cols = [
    "Value Score", "Quality Score", "Growth Score", "Company Archetype", "Company Type",
    "Fundamental Direction", "Fundamental Trend Score", "Fundamental Evidence",
    "Turnaround Score", "Turnaround Evidence", "Story", "Shoda s příběhem", "Potenciální shoda s příběhem", "Investiční atraktivita", "Story Priority",
    "Available Params", "Potenciální shoda s příběhem", "Pass", "Story Selected", "Eligible",
    "Text Score", "Text Evidence", "Text Positive", "Text Negative", "Text Support",
    "Text Warnings", "Text Sources", "Skóre ceny", "Price View", "Drawdown 3Y",
    "Drawdown 5Y", "Recovery from 3Y Low", "Recovery from 5Y Low", "6M Return",
    "12M Return", "Days Since 3Y Low", "MA50 vs MA200", "Higher Low", "Higher High",
    "Price Trend", "Price Evidence", "Final Confidence", "Market / Fundamental View"
]
raw_results = raw_results.drop(columns=[c for c in derived_cols if c in raw_results.columns], errors="ignore").copy()
results_df = raw_results.copy()

scores = results_df.apply(calc_scores, axis=1, result_type="expand")
scores.columns = ["Value Score", "Quality Score", "Growth Score"]
results_df = pd.concat([results_df.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)
results_df["Company Archetype"] = results_df.apply(company_archetype, axis=1)
results_df["Company Type"] = results_df["Company Archetype"]
dirs = results_df.apply(fundamental_direction, axis=1, result_type="expand")
dirs.columns = ["Fundamental Direction", "Fundamental Trend Score", "Fundamental Evidence"]
results_df = pd.concat([results_df, dirs], axis=1)
gates = results_df.apply(recovery_gates, axis=1, result_type="expand")
gates.columns = ["Posouzení zotavení", "Skóre zotavení", "Recovery Gates"]
results_df = pd.concat([results_df, gates], axis=1)
turns = results_df.apply(turnaround_score, axis=1, result_type="expand")
turns.columns = ["Turnaround Score", "Turnaround Evidence"]
results_df = pd.concat([results_df, turns], axis=1)
results_df["Story"] = results_df.apply(classify_story, axis=1)
results_df["Investiční atraktivita"] = results_df.apply(investment_attractiveness, axis=1)
results_df["Available Params"] = results_df[PARAMS].notna().sum(axis=1)
def best_selected_story_fit(r):
    if not selected_stories:
        return np.nan
    fits = [story_fit(r, s) for s in selected_stories]
    fits = [x for x in fits if not pd.isna(x)]
    return round(max(fits), 1) if fits else np.nan

# V6.10.1: distinguish a calculated/potential fit from a valid fit.
# A story fit can be mathematically computed from a partial row, but it must
# not be treated as a real candidate signal until the minimum data threshold is met.
results_df["Potenciální shoda s příběhem"] = results_df.apply(best_selected_story_fit, axis=1)
results_df["Shoda s příběhem"] = results_df.apply(
    lambda r: safe_float(r.get("Potenciální shoda s příběhem")) if int(r.get("Available Params", 0)) >= min_data else np.nan,
    axis=1
)
results_df["Story Priority"] = results_df.apply(
    lambda r: story_priority(r, None) if pd.isna(safe_float(r.get("Shoda s příběhem"))) else round(
        0.60 * safe_float(r.get("Shoda s příběhem")) + 0.40 * safe_float(r.get("Investiční atraktivita")), 1
        ) if not pd.isna(safe_float(r.get("Investiční atraktivita"))) else safe_float(r.get("Shoda s příběhem")), axis=1)

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
# V6.6: kandidáta neurčuje název automaticky přiřazeného příběhu.
# Rozhodující je Shoda s příběhem, protože cílem screeneru je vytipovat
# i hraniční firmy, které algoritmus zařadil do jiného, příbuzného příběhu.
if selected_stories:
    results_df["Story Selected"] = results_df["Shoda s příběhem"].notna() & (results_df["Shoda s příběhem"] >= min_story_fit)
else:
    results_df["Story Selected"] = True
results_df["Eligible"] = (results_df["Available Params"] >= min_data) & results_df["Pass"] & results_df["Story Selected"]
update_runtime(status="running", stage="4/6", message=f"Charakter/recovery/příběh dokončen: {int(results_df["Eligible"].sum())}")
stage_progress.progress(1.0, text=f"✅ 3/6 Charakter + recovery + příběh dokončen · {int(results_df["Eligible"].sum())} kandidátů")
stage_status.write(f"✅ **3/6 dokončeno:** {int(results_df["Eligible"].sum())} kandidátů splňuje podmínky.")

# V6.9.1 diagnostics: keep stage counts visible so changes between versions/runs
# can be localized without guessing whether they came from universe, prefilter,
# fundamentals, story-fit filtering, price analysis, or text analysis.
pipeline_counts = st.session_state.get("screening_pipeline_counts", {}).copy()
pipeline_counts["Dostatek dat (min. parametry)"] = int((results_df["Available Params"] >= min_data).sum())
pipeline_counts["Shoda s příběhem ≥ minimum"] = int(results_df["Story Selected"].sum()) if selected_stories else int(len(results_df))
# Explicit CLST trace for the turnaround diagnostic. This is diagnostic only.
clst_stage1 = "CLST" in set(st.session_state.get("screening_stage1_tickers", []))
clst_fund = "CLST" in set(st.session_state.get("screening_fundamental_tickers", []))
clst_row = results_df[results_df["Ticker"].astype(str).eq("CLST")]
clst_has_data = bool(not clst_row.empty and int(clst_row.iloc[0]["Available Params"]) >= min_data)
clst_story = bool(not clst_row.empty and bool(clst_row.iloc[0]["Story Selected"]))
clst_eligible = bool(not clst_row.empty and bool(clst_row.iloc[0]["Eligible"]))
pipeline_counts["CLST – v předvýběru 800"] = int(clst_stage1)
pipeline_counts["CLST – ve fundamentální fázi"] = int(clst_fund)
pipeline_counts["CLST – má dostatek dat"] = int(clst_has_data)
pipeline_counts["CLST – Story Fit ≥ minimum"] = int(clst_story)
pipeline_counts["CLST – Eligible"] = int(clst_eligible)
pipeline_counts["Prošlo klasickým filtrem"] = int(results_df["Pass"].sum())
pipeline_counts["Eligible před cenou/textem"] = int(results_df["Eligible"].sum())
pipeline_counts["Posláno do cenové fáze"] = int(min(max_price_candidates, pipeline_counts["Eligible před cenou/textem"])) if max_price_candidates > 0 else 0
pipeline_counts["Posláno do textové fáze"] = int(min(max_text_candidates, pipeline_counts["Eligible před cenou/textem"])) if max_text_candidates > 0 else 0
st.session_state["screening_pipeline_counts"] = pipeline_counts
results_df = results_df.sort_values(["Eligible", "Story Priority"], ascending=[False, False], na_position="last").reset_index(drop=True)

def empty_evidence_columns(df):
    out = df.copy()
    for c, default in {
        "Text Score": np.nan, "Text Evidence": "⚪ Nehodnoceno", "Text Positive": 0, "Text Negative": 0,
        "Text Support": "", "Text Warnings": "", "Text Sources": "",
        "Skóre ceny": np.nan, "Price View": "⚪ Nehodnoceno", "Drawdown 3Y": np.nan, "Drawdown 5Y": np.nan,
        "Recovery from 3Y Low": np.nan, "Recovery from 5Y Low": np.nan, "6M Return": np.nan, "12M Return": np.nan,
        "Days Since 3Y Low": np.nan, "MA50 vs MA200": np.nan, "Higher Low": "", "Higher High": "",
        "Price Trend": "", "Price Evidence": ""
    }.items():
        if c not in out.columns:
            out[c] = default
    return out

# Evidence is kept separately, keyed by ticker. This survives UI-only reruns.
if run:
    update_runtime(status="running", stage="4/6", message=f"Textová fáze: až {min(max_text_candidates, int(results_df["Eligible"].sum()))} kandidátů")
    stage_progress.progress(0, text=f"🔄 4/6 Textové důkazy · připravuji až {min(max_text_candidates, int(results_df["Eligible"].sum()))} kandidátů")
    stage_status.write(f"🔄 **4/6 Textové důkazy:** prověřuji nejvýše {min(max_text_candidates, int(results_df["Eligible"].sum()))} kandidátů.")
    if max_text_candidates > 0:
        results_df = add_text_evidence(results_df, max_text_candidates)
    else:
        results_df = empty_evidence_columns(results_df)
    update_runtime(status="running", stage="5/6", message="Textová fáze dokončena")
    stage_progress.progress(1.0, text="✅ 4/6 Textové důkazy dokončeny")
    stage_status.write("✅ **4/6 Textové důkazy dokončeny.**")
    stage_progress.progress(0, text=f"🔄 5/6 Cenová fáze · připravuji až {min(max_price_candidates, int(results_df["Eligible"].sum()))} kandidátů")
    update_runtime(status="running", stage="5/6", message=f"Cenová fáze: až {min(max_price_candidates, int(results_df["Eligible"].sum()))} kandidátů")
    stage_status.write(f"🔄 **5/6 Cenová fáze:** prověřuji nejvýše {min(max_price_candidates, int(results_df["Eligible"].sum()))} kandidátů.")
    if max_price_candidates > 0:
        results_df = add_price_analysis(results_df, max_price_candidates)
    else:
        results_df = empty_evidence_columns(results_df)

    evidence_cols = [c for c in [
        "Ticker", "Posouzení zotavení", "Skóre zotavení", "Recovery Gates", "Text Score", "Text Evidence", "Text Positive", "Text Negative", "Text Support", "Text Warnings", "Text Sources",
        "Skóre ceny", "Price View", "Drawdown 3Y", "Drawdown 5Y", "Recovery from 3Y Low", "Recovery from 5Y Low",
        "6M Return", "12M Return", "Days Since 3Y Low", "MA50 vs MA200", "Higher Low", "Higher High", "Price Trend", "Price Evidence"
    ] if c in results_df.columns]
    st.session_state["screening_evidence"] = results_df[evidence_cols].drop_duplicates("Ticker").copy()
    update_runtime(status="finished", stage="6/6", message="Screening dokončen")
    stage_progress.progress(1.0, text="✅ 6/6 Screening dokončen")
    stage_status.write("✅ **6/6 Screening dokončen.** Výsledky jsou připraveny níže.")
else:
    results_df = empty_evidence_columns(results_df)
    cached_evidence = st.session_state.get("screening_evidence")
    if isinstance(cached_evidence, pd.DataFrame) and not cached_evidence.empty and "Ticker" in cached_evidence.columns:
        merge_cols = [c for c in cached_evidence.columns if c != "Ticker"]
        results_df = results_df.drop(columns=[c for c in merge_cols if c in results_df.columns], errors="ignore")
        results_df = results_df.merge(cached_evidence, on="Ticker", how="left", suffixes=("", "_cached"))
        for c in merge_cols:
            cc = f"{c}_cached"
            if cc in results_df.columns:
                results_df[c] = results_df[cc].where(results_df[cc].notna(), results_df[c])
                results_df = results_df.drop(columns=[cc])

for c, default in {
    "Text Score": np.nan, "Text Evidence": "⚪ Nehodnoceno", "Text Positive": 0, "Text Negative": 0, "Text Support": "", "Text Warnings": "", "Text Sources": "",
    "Skóre ceny": np.nan, "Price View": "⚪ Nehodnoceno", "Drawdown 3Y": np.nan, "Drawdown 5Y": np.nan, "Recovery from 3Y Low": np.nan, "Recovery from 5Y Low": np.nan,
    "6M Return": np.nan, "12M Return": np.nan, "Days Since 3Y Low": np.nan, "MA50 vs MA200": np.nan, "Higher Low": "", "Higher High": "", "Price Trend": "", "Price Evidence": ""
}.items():
    if c not in results_df.columns:
        results_df[c] = default
results_df["Final Confidence"] = results_df.apply(final_story_confidence, axis=1)
results_df["Market / Fundamental View"] = results_df.apply(market_fundamental_view, axis=1)
results_df = results_df.sort_values(["Eligible", "Final Confidence", "Story Priority"], ascending=[False, False, False], na_position="last").reset_index(drop=True)

# Raw data and evidence remain separate; this dataframe is only the current display state.
st.session_state["screening_raw_results"] = raw_results.copy()
st.session_state["screening_results"] = results_df.copy()

# Summary
st.markdown("## 📊 Výsledek screeningu")
a,b,c,d,e = st.columns(5)
a.metric("Načteno", len(results_df))
b.metric("Kompletní data", int((results_df["Status"] == "OK").sum()))
c.metric("≥ min. dat", int((results_df["Available Params"] >= min_data).sum()))
# V6.9.1: diagnostic pipeline summary
with st.expander("🔬 Diagnostika průchodu screeningem", expanded=False):
    pc = st.session_state.get("screening_pipeline_counts", {})
    if pc:
        diag_rows = [
            {"Fáze": "Celé univerzum", "Počet titulů": pc.get("Celé univerzum", 0)},
            {"Fáze": "Předselekce trhu", "Počet titulů": pc.get("Předselekce trhu", 0)},
            {"Fáze": "Fundamentální fáze", "Počet titulů": pc.get("Fundamentální fáze", 0)},
            {"Fáze": "Načtené fundamenty", "Počet titulů": pc.get("Načtené fundamenty", 0)},
            {"Fáze": "Dostatek dat", "Počet titulů": pc.get("Dostatek dat (min. parametry)", 0)},
            {"Fáze": "Shoda s příběhem ≥ minimum", "Počet titulů": pc.get("Shoda s příběhem ≥ minimum", 0)},
            {"Fáze": "Prošlo klasickým filtrem", "Počet titulů": pc.get("Prošlo klasickým filtrem", 0)},
            {"Fáze": "Eligible před cenou/textem", "Počet titulů": pc.get("Eligible před cenou/textem", 0)},
            {"Fáze": "Posláno do cenové fáze", "Počet titulů": pc.get("Posláno do cenové fáze", 0)},
            {"Fáze": "Posláno do textové fáze", "Počet titulů": pc.get("Posláno do textové fáze", 0)},
        ]
        st.dataframe(pd.DataFrame(diag_rows), use_container_width=True, hide_index=True)
        st.caption("Tyto počty slouží pouze k diagnostice pipeline. Nemění výběr ani skóre titulů.")
        fresh = st.session_state.get("screening_fresh_run", False)
        st.caption(f"Režim běhu: **{'ČERSTVÝ – cache vymazána' if fresh else 'STANDARDNÍ – cache mohla být použita'}**")
        clst_info = {
            "CLST – předvýběr 800": pc.get("CLST – v předvýběru 800", 0),
            "CLST – fundamentální fáze": pc.get("CLST – ve fundamentální fázi", 0),
            "CLST – dostatek dat": pc.get("CLST – má dostatek dat", 0),
            "CLST – Story Fit ≥ minimum": pc.get("CLST – Story Fit ≥ minimum", 0),
            "CLST – Eligible": pc.get("CLST – Eligible", 0),
        }
        st.markdown("**🔎 Stopa CLST**")
        st.dataframe(pd.DataFrame([{
            "Kontrola": k, "Stav": "ANO" if v else "NE"
        } for k, v in clst_info.items()]), use_container_width=True, hide_index=True)

d.metric("Vybraný příběh", int(results_df["Story Selected"].sum()))
e.metric("Kandidáti", int(results_df["Eligible"].sum()))

st.markdown("### 🧭 Mapa investičních příběhů")
st.caption("Skóre není predikce výnosu ani doporučení. Shoda s příběhem říká, jak moc titul odpovídá hledanému příběhu; Investiční atraktivita pomáhá určit pořadí kandidátů. Kandidátem je titul s minimální nastavenou shodou, nikoli pouze titul, kterému algoritmus přidělil stejný název příběhu. Priorita = 60 % shoda + 40 % atraktivita. Textová vrstva má vysvětlovat hlavně problém, aktuální změnu, mechanismus a rizika; do priority se nezapočítává.")

story_counts = results_df[results_df["Available Params"] >= min_data]["Story"].value_counts().rename_axis("Příběh").reset_index(name="Počet")
st.dataframe(story_counts, use_container_width=True, hide_index=True)

st.markdown("### 🎯 Kandidáti")
passed = results_df[results_df["Eligible"]].copy()
with st.expander("🧩 Diagnostika titulů se Story Fit ≥ minimum", expanded=False):
    sf = results_df[results_df["Story Selected"]].copy()
    if sf.empty:
        st.info("Žádný titul nedosáhl minimální Shody s příběhem.")
    else:
        # V6.9.4: detailní diagnostika všech titulů se Story Fit >= minimum.
        # Nemění výběr ani skóre; pouze ukazuje, které fundamentální parametry
        # konkrétnímu titulu chybí a proč případně neprošel přes min_data.
        def available_params_text(r):
            return ", ".join([p for p in PARAMS if not pd.isna(r.get(p))])

        def missing_params_text(r):
            return ", ".join([p for p in PARAMS if pd.isna(r.get(p))])

        sf["Dostupné parametry"] = sf.apply(available_params_text, axis=1)
        sf["Chybějící parametry"] = sf.apply(missing_params_text, axis=1)
        sf["Počet dat"] = sf["Available Params"].astype(int)
        sf["Min. dat splněno"] = sf["Available Params"] >= min_data
        sf_cols = [
            "Ticker", "Name", "Story", "Company Archetype", "Shoda s příběhem", "Potenciální shoda s příběhem",
            "Investiční atraktivita", "Počet dat", "Min. dat splněno",
            "Dostupné parametry", "Chybějící parametry", "Pass", "Eligible",
            "Posouzení zotavení", "Skóre zotavení", "Turnaround Score"
        ]
        st.caption("Shoda s příběhem je platná pouze při splnění minima dat. Potenciální shoda může být spočtena i z neúplných dat a slouží jen k diagnostice, nikoli k výběru kandidáta.")
        st.dataframe(sf[[c for c in sf_cols if c in sf.columns]], use_container_width=True, hide_index=True)

        # Samostatná krátká kontrola CLST, pokud je mezi tituly se Story Fit >= minimum.
        clst_diag = sf[sf["Ticker"].astype(str).eq("CLST")].copy()
        if not clst_diag.empty:
            r = clst_diag.iloc[0]
            st.markdown("**CLST – proč není kandidátem**")
            st.write(
                f"Dostupná data: **{int(r['Available Params'])}/{len(PARAMS)}** "
                f"(minimum {min_data}). "
                + (f"Chybí: {r['Chybějící parametry']}." if r['Chybějící parametry'] else "Nechybí žádný parametr.")
                + (f" Potenciální shoda s vybraným příběhem je **{r['Potenciální shoda s příběhem']:.0f}/100**, ale kvůli nedostatku dat se jako skutečná shoda nepoužije." if not pd.isna(r.get('Potenciální shoda s příběhem')) and int(r['Available Params']) < min_data else "")
            )
if passed.empty:
    st.info("Pro zvolený příběh a nastavení dat nebyl nalezen žádný kandidát.")
else:
    # V5.1: hlavní tabulka je záměrně kompaktní. Ekonomické detaily jsou níže.
    compact = passed.copy()
    compact["Verdikt"] = compact.apply(compact_verdict, axis=1)
    compact["Trend"] = compact.apply(compact_trend, axis=1)
    compact["Textový signál"] = compact.apply(compact_text_signal, axis=1)
    compact["Varování"] = compact.apply(compact_warning, axis=1)
    compact = compact[[
        "Ticker", "Name", "Story", "Company Archetype", "Shoda s příběhem", "Investiční atraktivita", "Story Priority", "Verdikt", "Trend",
        "Price View", "Market / Fundamental View", "Posouzení zotavení", "Textový signál", "Value Score", "Quality Score", "Growth Score", "Varování"
    ]].rename(columns={
        "Name": "Firma", "Story": "Příběh", "Company Archetype": "Charakter", "Shoda s příběhem": "Shoda s příběhem", "Investiční atraktivita": "Investiční atraktivita", "Story Priority": "Priorita",
        "Price View": "Cenový obraz", "Market / Fundamental View": "Fundamenty vs. cena", "Posouzení zotavení": "Posouzení zotavení",
        "Textový signál": "Textové důkazy", "Value Score": "Value",
        "Quality Score": "Quality", "Growth Score": "Growth"
    })
    st.dataframe(
        compact, use_container_width=True, hide_index=True, height=560,
        column_config={
            "Shoda s příběhem": st.column_config.NumberColumn("Shoda s příběhem", format="%.0f"),
            "Investiční atraktivita": st.column_config.NumberColumn("Investiční atraktivita", format="%.0f"),
            "Priorita": st.column_config.NumberColumn("Priorita", format="%.0f"),
            "Value": st.column_config.NumberColumn("Value", format="%.0f"),
            "Quality": st.column_config.NumberColumn("Quality", format="%.0f"),
            "Growth": st.column_config.NumberColumn("Growth", format="%.0f"),
            "Firma": st.column_config.TextColumn("Firma", width="large"),
            "Charakter": st.column_config.TextColumn("Charakter", width="medium"),
            "Příběh": st.column_config.TextColumn("Příběh", width="medium"),
            "Verdikt": st.column_config.TextColumn("Verdikt", width="medium"),
            "Trend": st.column_config.TextColumn("Trend", width="medium"),
            "Textové důkazy": st.column_config.TextColumn("Textové důkazy", width="medium"),
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
    m2.metric("Shoda s příběhem", f"{safe_float(r['Shoda s příběhem']):.0f}/100" if not pd.isna(safe_float(r['Shoda s příběhem'])) else "—")
    m3.metric("Trend", trend)
    m4.metric("Priorita", f"{safe_float(r['Story Priority']):.0f}/100" if not pd.isna(safe_float(r['Story Priority'])) else "—")

    st.caption(f"Investiční atraktivita: {safe_float(r.get('Investiční atraktivita')):.0f}/100 · Posouzení zotavení: {clean_text(r.get('Posouzení zotavení')) or 'nehodnoceno'} · Textové důkazy: {clean_text(r.get('Text Evidence')) or 'nehodnoceno'} · {warning}")
    st.caption("Posouzení zotavení = kontrola, zda finanční a provozní údaje vykazují znaky zotavení/obratu. Neznamená to, že je firma zdravá ani že jde automaticky o klasický turnaround.")
    if clean_text(r.get("Turnaround Evidence")):
        st.info("**Proč se titul dostal mezi kandidáty:** " + clean_text(r.get("Fundamental Evidence")) + "\n\n**Signály zotavení:** " + clean_text(r.get("Turnaround Evidence")))

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
        st.write(f"**Posouzení zotavení:** {r.get("Posouzení zotavení","—")} · skóre {safe_float(r.get("Skóre zotavení")):.0f}/100" if not pd.isna(safe_float(r.get("Skóre zotavení"))) else "**Posouzení zotavení:** —")
        st.write(f"**Brány:** {r.get("Posouzení zotavení","—")}")
        st.dataframe(pd.DataFrame([{
            "Směr fundamentů": r["Fundamental Direction"], "Fundamentální trend score": r["Fundamental Trend Score"],
            "Revenue Growth %": r["Revenue Growth"], "Earnings Growth %": r["Earnings Growth"],
            "Revenue CAGR 3Y %": r["Revenue CAGR 3Y"], "Net Income CAGR 3Y %": r["Net Income CAGR 3Y"],
            "Margin Change 3Y %": r["Margin Change 3Y"], "Revenue Prior YoY %": r["Revenue Prior YoY"],
            "Net Income Prior YoY %": r["Net Income Prior YoY"]
        }]), use_container_width=True, hide_index=True)
    with st.expander("📉 Chování ceny", expanded=False):
        pc1, pc2, pc3, pc4 = st.columns(4)
        ps = safe_float(r.get("Skóre ceny")); dd = safe_float(r.get("Drawdown 3Y")); r12 = safe_float(r.get("12M Return")); ma = safe_float(r.get("MA50 vs MA200"))
        pc1.metric("Skóre ceny", f"{ps:.0f}" if not pd.isna(ps) else "—")
        pc2.metric("3Y propad", f"{dd:.1f}%" if not pd.isna(dd) else "—")
        pc3.metric("12M výnos", f"{r12:.1f}%" if not pd.isna(r12) else "—")
        pc4.metric("MA50 vs MA200", f"{ma:.1f}%" if not pd.isna(ma) else "—")
        st.write(f"**Struktura ceny:** vyšší minima = {r.get('Higher Low','—')}; vyšší maxima = {r.get('Higher High','—')}" )
        st.write(f"**Cenový obraz:** {r.get('Price View', '—')}")
        st.write(f"**Fundamenty vs. cena:** {r.get('Market / Fundamental View', '—')}")
        st.write(f"**Co naznačuje cena:** {r.get('Price Evidence', '') or '—'}")
    with st.expander("📰 Textové důkazy k příběhu", expanded=False):
        st.write(f"**Doložení příběhu:** {evidence_quality(r)}")
        if clean_text(r.get("Text Evidence")):
            st.write(f"**Textový závěr:** {normalize_text_evidence_output(r.get('Text Evidence'))}")
        if clean_text(r.get("Text Support")):
            st.markdown(r["Text Support"])
        if clean_text(r.get("Text Warnings")):
            st.markdown("**Rizika / co může příběh zpochybnit**")
            st.markdown(r["Text Warnings"])
        elif clean_text(r.get("Text Evidence")) in ("⚪ Bez textového důkazu", "⚪ Zatím nedoloženo"):
            st.info("Textová vrstva zatím nenašla dostatečně konkrétní firemní signál. To není potvrzení ani vyvrácení příběhu.")
        if clean_text(r.get("Text Sources")):
            st.caption("Zdroj textu: " + r["Text Sources"])

    with st.expander("🔬 Kompletní technický záznam", expanded=False):
        detail_cols = [
            "Ticker","Yahoo Ticker","Name","Exchange","Company Archetype","Company Type","Sector","Industry",
            *PARAMS,"Revenue CAGR 3Y","Net Income CAGR 3Y","Net Margin","Margin Change 3Y",
            "Revenue Prior YoY","Net Income Prior YoY","Fundamental Direction","Fundamental Trend Score","Fundamental Evidence","Posouzení zotavení","Skóre zotavení","Recovery Gates","Turnaround Score","Turnaround Evidence",
            "Story Priority","Text Score","Final Confidence","Status","Mapping","Data Source","Error"
        ]
        detail_cols = [c for c in detail_cols if c in r.index]
        st.dataframe(pd.DataFrame([r[detail_cols].to_dict()]), use_container_width=True, hide_index=True)


# Text evidence detail
with st.expander("📰 Textové důkazy k příběhu", expanded=False):
    text_cols = ["Ticker", "Name", "Story", "Story Priority", "Text Score", "Text Evidence", "Doložení příběhu", "Final Confidence", "Text Support", "Text Warnings", "Text Sources"]
    if "Text Score" in results_df.columns:
        text_view = results_df[(results_df["Text Evidence"] != "⚪ Nehodnoceno") & results_df["Text Score"].notna()].copy()
        if text_view.empty:
            st.info("Textová fáze zatím nebyla provedena nebo pro kandidáty nebyl dostupný text.")
        else:
            text_view["Doložení příběhu"] = text_view.apply(evidence_quality, axis=1)
            text_cols = [c for c in text_cols if c in text_view.columns]
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
        *PARAMS, "Revenue CAGR 3Y", "Net Income CAGR 3Y", "Net Margin", "Margin Change 3Y", "Revenue Prior YoY", "Net Income Prior YoY", "Value Score", "Quality Score", "Growth Score", "Turnaround Score", "Company Type", "Story", "Turnaround Evidence", "Text Score", "Text Evidence", "Skóre ceny", "Price View", "Drawdown 3Y", "Recovery from 3Y Low", "6M Return", "12M Return", "MA50 vs MA200", "Higher Low", "Higher High", "Market / Fundamental View", "Final Confidence", "Available Params", "Status", "Mapping", "Data Source", "Error"
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
    "V6 pracuje v navazujících vrstvách: (1) cenový předvýběr celého univerza v dávkách, (2) fundamentální fáze, (3) charakter firmy, (4) směr fundamentů, (5) recovery gates, (6) textová evidence, (7) chování ceny, (8) investiční příběh. True turnaround vyžaduje předchozí problém i současné zlepšení. "
    "Textová vrstva příběh nepotvrzuje automaticky; je to evidence mechanismus. Yahoo/Google text je nejprve filtrován na relevanci ke konkrétní firmě a teprve potom se hledají podpůrné a varovné signály. "
    "Charakter firmy a asset-based společnosti jsou posuzovány odděleně, aby se na ně "
    "mechanicky nepřenášela logika běžné provozní firmy. Chybějící hodnoty se nepřevádějí na nulu."
)

st.caption("Zdroje: Nasdaq Trader, Deutsche Börse Xetra a Yahoo Finance. Data jsou získávána při běhu aplikace a mohou být zpožděná či nedostupná.")
