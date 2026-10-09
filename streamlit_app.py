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
st.caption("V6.28.12 · BUILD 20261009-A – Screener · samostatný modul Analytik je dostupný v menu vlevo")

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
