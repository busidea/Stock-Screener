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
st.caption("V6.10.4 – Screener · samostatný modul Analytik je dostupný v menu vlevo")

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


@st.cache_data(ttl=3600, show_spinner=False)
def analyst_get_quote_data(yahoo_ticker):
    t = yf.Ticker(yahoo_ticker)
    info, fast = {}, {}
    try: info = t.info or {}
    except Exception: pass
    try: fast = dict(t.fast_info)
    except Exception: pass
    return {
        "price": first_valid(info.get("currentPrice"), info.get("regularMarketPrice"), fast.get("last_price")),
        "market_cap": first_valid(info.get("marketCap"), fast.get("market_cap")),
        "currency": clean_text(info.get("currency")),
        "name": clean_text(info.get("longName") or info.get("shortName")),
        "sector": clean_text(info.get("sector")),
        "industry": clean_text(info.get("industry")),
        "country": clean_text(info.get("country")),
        "website": clean_text(info.get("website")),
        "ir_website": clean_text(info.get("irWebsite")),
        "summary": clean_text(info.get("longBusinessSummary")),
        "employees": first_valid(info.get("fullTimeEmployees")),
        "pe": first_valid(info.get("trailingPE")),
        "forward_pe": first_valid(info.get("forwardPE")),
        "ps": first_valid(info.get("priceToSalesTrailing12Months")),
        "pb": first_valid(info.get("priceToBook")),
        "roe": first_valid(info.get("returnOnEquity")),
        "revenue_growth": first_valid(info.get("revenueGrowth")),
        "earnings_growth": first_valid(info.get("earningsGrowth")),
        "free_cash_flow": first_valid(info.get("freeCashflow")),
        "debt_to_equity": first_valid(info.get("debtToEquity")),
        "dividend_yield": first_valid(info.get("dividendYield")),
        "quote_type": clean_text(info.get("quoteType")),
        "exchange": clean_text(info.get("exchange")),
    }


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


def _analyst_build_history(t, quarterly=False):
    freq = "quarterly" if quarterly else "yearly"
    income = _analyst_get_statement(t, "quarterly_income_stmt" if quarterly else "income_stmt", "get_income_stmt", freq)
    balance = _analyst_get_statement(t, "quarterly_balance_sheet" if quarterly else "balance_sheet", "get_balance_sheet", freq)
    cashflow = _analyst_get_statement(t, "quarterly_cashflow" if quarterly else "cashflow", "get_cash_flow", freq)

    rev = _analyst_statement_series(income, ["Total Revenue", "Operating Revenue", "TotalRevenue"])
    ni = _analyst_statement_series(income, ["Net Income", "Net Income Common Stockholders", "NetIncome"])
    op = _analyst_statement_series(income, ["Operating Income", "Operating Income or Loss", "OperatingIncome"])
    ocf = _analyst_statement_series(cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities", "OperatingCashFlow"])
    capex = _analyst_statement_series(cashflow, ["Capital Expenditure", "Capital Expenditure Reported", "CapitalExpenditure"])
    debt = _analyst_statement_series(balance, ["Total Debt", "TotalDebt"])
    equity = _analyst_statement_series(balance, ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest", "StockholdersEquity"])

    cols = {name: s for name, s in [
        ("Revenue", rev), ("Net Income", ni), ("Operating Income", op),
        ("Operating Cash Flow", ocf), ("Capital Expenditure", capex),
        ("Debt", debt), ("Equity", equity)
    ] if not s.empty}
    if not cols:
        return pd.DataFrame()

    dates = []
    for s in cols.values():
        idx = pd.to_datetime(s.index, errors="coerce")
        dates.extend([d for d in idx if not pd.isna(d)])
    dates = sorted(set(dates))
    if not dates:
        return pd.DataFrame()
    # Keep the most recent requested window, but do not fabricate missing years/quarters.
    dates = dates[-8:] if quarterly else dates[-5:]
    out = pd.DataFrame(index=dates)
    for name, s in cols.items():
        ss = s.copy()
        ss.index = pd.to_datetime(ss.index, errors="coerce")
        out[name] = ss.reindex(dates).values
    if quarterly:
        out.index = [f"{d.year} Q{d.quarter}" for d in out.index]
    else:
        out.index = [str(d.year) for d in out.index]
    if "Revenue" in out and "Net Income" in out:
        revs = pd.to_numeric(out["Revenue"], errors="coerce")
        nis = pd.to_numeric(out["Net Income"], errors="coerce")
        out["Net Margin %"] = np.where(revs > 0, nis / revs * 100, np.nan)
    if "Operating Cash Flow" in out and "Capital Expenditure" in out:
        ocf_s = pd.to_numeric(out["Operating Cash Flow"], errors="coerce")
        cap_s = pd.to_numeric(out["Capital Expenditure"], errors="coerce")
        # Yahoo generally reports capex as negative. Handle both conventions.
        out["FCF"] = np.where(cap_s <= 0, ocf_s + cap_s, ocf_s - cap_s)
    return out.reset_index(names="Období")


@st.cache_data(ttl=3600, show_spinner=False)
def analyst_get_financial_history(yahoo_ticker):
    return _analyst_build_history(yf.Ticker(yahoo_ticker), quarterly=False)


@st.cache_data(ttl=3600, show_spinner=False)
def analyst_get_quarterly_history(yahoo_ticker):
    return _analyst_build_history(yf.Ticker(yahoo_ticker), quarterly=True)


def analyst_ttm_from_quarters(quarterly):
    if quarterly is None or quarterly.empty or len(quarterly) < 4:
        return pd.DataFrame()
    q = quarterly.tail(4)
    row = {"Období": "TTM"}
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
    try:
        hist = yf.Ticker(yahoo_ticker).history(period="5y", auto_adjust=False, actions=False)
        if hist is None or hist.empty:
            return pd.DataFrame()
        out = hist[[c for c in ["Close", "Volume"] if c in hist.columns]].copy().reset_index()
        date_col = "Date" if "Date" in out.columns else out.columns[0]
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce", utc=True).dt.tz_localize(None)
        if date_col != "Date": out = out.rename(columns={date_col: "Date"})
        return out.dropna(subset=["Date"])
    except Exception:
        return pd.DataFrame()


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
    """Interpret trends rather than merely displaying ratios."""
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
            parts.append(f"Za dostupné víceleté období {label} {'rostou' if x>5 else 'klesají' if x<-5 else 'jsou zhruba stabilní'} ({analyst_pct(x,0)}).")
    if "Net Margin %" in annual.columns:
        s=pd.to_numeric(annual["Net Margin %"],errors="coerce").dropna()
        if len(s)>=2:
            d=s.iloc[-1]-s.iloc[0]
            parts.append(f"Čistá marže se proti začátku sledovaného období změnila o {analyst_pct(d,1)} p. b.")
    if "Debt" in annual.columns:
        x=series_change(annual,"Debt")
        if not pd.isna(x): parts.append(f"Dluh se ve stejném období změnil o {analyst_pct(x,0)}.")
    if quarterly is not None and not quarterly.empty:
        ttm=analyst_ttm_from_quarters(quarterly)
        if not ttm.empty and not annual.empty:
            for col,label in [("Revenue","tržby"),("Net Income","čistý zisk"),("FCF","FCF")]:
                if col in ttm.columns and col in annual.columns:
                    a=pd.to_numeric(annual[col],errors="coerce").dropna()
                    v=safe_float(ttm.iloc[0].get(col))
                    if len(a) and not pd.isna(v):
                        delta=(v/a.iloc[-1]-1)*100 if a.iloc[-1] not in (0,np.nan) else np.nan
                        if not pd.isna(delta): parts.append(f"TTM {label} jsou oproti poslednímu uzavřenému roku {analyst_pct(delta,0)}.")
    if len(annual)<5: parts.append(f"Pozor: Yahoo Finance poskytlo pouze {len(annual)} celých účetních období, nikoli plných pět let.")
    if quarterly is not None and len(quarterly)<8: parts.append(f"Čtvrtletní řada obsahuje pouze {len(quarterly)} období; TTM je proto {('dostupné' if len(quarterly)>=4 else 'nedostupné')}.")
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


def analyst_current_developments(company, q, annual, quarterly, news, sec):
    """Research core: turn verified company-specific news into themes, not a headline list."""
    clusters=_analyst_cluster_news(news)
    sections=[]
    preferred=["Strategie / portfolio","Výsledky / provoz","Trh / regulace","Management / kapitál","Produkt / technologie"]
    for cat in preferred:
        rows=clusters.get(cat,[])
        if not rows: continue
        lead=rows[0]; titles=[clean_text(r.get("Název")) for r in rows[:3] if clean_text(r.get("Název"))]
        blob=" ".join((clean_text(r.get("Název"))+" "+clean_text(r.get("Popis"))) for r in rows[:3])
        direction=_analyst_event_direction(blob)
        evidence="; ".join(titles[:2])
        impact={
            "Strategie / portfolio":"Může měnit složení firmy, alokaci kapitálu a dlouhodobou ekonomiku jednotlivých segmentů.",
            "Výsledky / provoz":"Může měnit tempo růstu, marže a schopnost převádět růst do cash flow.",
            "Trh / regulace":"Může měnit poptávku, ceny, náklady nebo přístup firmy na konkrétní trhy.",
            "Management / kapitál":"Může měnit kvalitu exekuce, kapitálovou strukturu nebo důvěryhodnost vedení.",
            "Produkt / technologie":"Může měnit konkurenceschopnost, budoucí růst nebo nákladovou pozici."
        }[cat]
        sections.append(f"**{cat} — směr: {direction}.** Z dostupných podkladů vystupuje jako hlavní změna: {evidence}. **Ekonomický význam:** {impact}")

    # Explicitly connect the news themes with the financial trend.
    fin=analyst_financial_summary(annual,quarterly)
    if fin and "nelze" not in fin.lower():
        sections.append(f"**Finanční vazba:** {fin}")

    # SEC is used as a confirmation signal, not as a dump of filing rows.
    if sec is not None and not sec.empty:
        forms=sec["Formulář"].astype(str).value_counts().to_dict() if "Formulář" in sec.columns else {}
        if forms:
            common=", ".join(f"{k} ({v}×)" for k,v in list(forms.items())[:4])
            sections.append(f"**Regulatorní stopa (USA):** poslední veřejná podání zahrnují {common}. Samotný typ podání není důkazem změny; slouží pouze jako kontrola, zda je kolem firmy aktuální reportovací aktivita.")
    if not sections:
        return "Nebylo nalezeno dost relevantních a jednoznačně firemních podkladů, ze kterých by šlo poctivě sestavit aktuální změny."
    return "\n\n".join(sections)


def analyst_story_hypothesis(q, annual, quarterly, news, sec):
    # Only shared story names are used. This is a hypothesis from the Analyst's own data, not Screener output.
    rev=q.get("revenue_growth",np.nan); earn=q.get("earnings_growth",np.nan); pe=q.get("pe",np.nan); fpe=q.get("forward_pe",np.nan); roe=q.get("roe",np.nan)
    primary="Nejasný / smíšený příběh"; why="Dostupné údaje zatím neukazují jednu dominantní ekonomickou změnu."
    if not pd.isna(rev) and not pd.isna(earn) and rev>8 and earn>12:
        primary="Růst za rozumnou cenu" if pd.isna(fpe) or fpe<30 else "Vysoký růst / dražší příběh"
        why="Růst tržeb i zisku je výrazný; hlavní otázkou je, zda tempo růstu dokáže ospravedlnit očekávání v ceně."
    elif not pd.isna(roe) and roe>15 and not pd.isna(rev) and rev>3:
        primary="Kvalita za rozumnou cenu" if pd.isna(pe) or pe<30 else "Kvalitní růst"
        why="Rentabilita a růst naznačují kvalitní ekonomiku; valuace rozhoduje o tom, kolik této kvality je již započteno."
    elif not pd.isna(earn) and earn>10 and not pd.isna(rev) and rev>-2:
        primary="Provozní zlepšení"
        why="Zisk se zlepšuje rychleji než tržby, což může odpovídat zlepšení marží nebo normalizaci nákladů."
    elif not pd.isna(pe) and 0<pe<12:
        primary="Value / levná firma"
        why="Ocenění je nízké; je ale nutné ověřit, zda nejde o strukturálně slabý podnik."
    return primary,why


def analyst_render(ticker_input):
    st.title("🔎 Analytik")
    st.caption("Nezávislé výzkumné jádro · Analytik nevidí Screener ani důvod, proč byl titul vybrán.")

    c1,c2=st.columns([3,1])
    with c1:
        ticker_input=st.text_input("Ticker",value=ticker_input,placeholder="např. SHL, GOOGL, ONC, AT1.DE")
    with c2:
        default_ex=2 if str(ticker_input).upper().endswith(".DE") else 0
        exchange=st.selectbox("Trh",["NASDAQ","NYSE","XETRA"],index=default_ex)
    analyse=st.button("🔬 Spustit analytické jádro",type="primary")
    if not analyse:
        st.info("Zadej ticker. Pro první test doporučuji například SHL na XETRA, protože na něm dobře uvidíme, zda Analytik dokáže oddělit skutečné firemní změny od šumu.")
        return

    ticker=clean_text(ticker_input).upper()
    if not ticker:
        st.warning("Zadej ticker.")
        return
    yahoo_ticker=analyst_yahoo_ticker(ticker,exchange)
    status=st.empty()
    status.info("1/5 Ověřuji identitu firmy a načítám veřejná data…")
    q=analyst_get_quote_data(yahoo_ticker)
    company=q.get("name") or ""
    if not company:
        st.error(f"Ticker {ticker} se nepodařilo jednoznačně načíst přes Yahoo Finance ({yahoo_ticker}).")
        return

    status.info("2/5 Sestavuji dlouhodobý finanční trend, poslední kvartály a cenový kontext…")
    annual=analyst_get_financial_history(yahoo_ticker)
    quarterly=analyst_get_quarterly_history(yahoo_ticker)
    price=analyst_price_history(yahoo_ticker)
    sec=analyst_sec_filings(ticker) if exchange in ("NASDAQ","NYSE") else pd.DataFrame()

    status.info("3/5 Hledám firemně relevantní aktuální události a filtruji kolize tickeru…")
    news=analyst_google_news(company,ticker,q.get("ir_website") or q.get("website"))

    status.info("4/5 Propojuji události s ekonomikou firmy…")
    current=analyst_current_developments(company,q,annual,quarterly,news,sec)
    primary,story_reason=analyst_story_hypothesis(q,annual,quarterly,news,sec)
    price_comment=analyst_price_commentary(price)
    status.success("5/5 Analytické jádro dokončeno.")

    st.markdown(f"## {company}")
    st.caption(f"Ticker **{ticker}** · Yahoo **{yahoo_ticker}** · {exchange} · načteno {datetime.now().strftime('%d.%m.%Y %H:%M')}")

    m1,m2,m3,m4=st.columns(4)
    price_txt=analyst_human_number(q.get("price"),2)
    m1.metric("Cena",f"{price_txt} {q.get('currency','')}")
    m2.metric("Tržní kapitalizace",analyst_human_number(q.get("market_cap")))
    m3.metric("P/E",analyst_human_number(q.get("pe"),1))
    roe=safe_float(q.get("roe"))
    m4.metric("ROE","—" if pd.isna(roe) else f"{roe*100:.1f} %".replace(".",","))

    with st.expander("🏢 1. Základní profil a obchodní model",expanded=True):
        if q.get("sector") or q.get("industry"):
            st.write(f"**Sektor:** {q.get('sector') or '—'} · **Odvětví:** {q.get('industry') or '—'}")
        if q.get("country"):
            st.write(f"**Sídlo / země:** {q.get('country')}")
        if q.get("summary"):
            st.write(q["summary"])

    with st.expander("🧠 Jádro testu – co se ve firmě mění",expanded=True):
        st.markdown(current)
        if news is not None and not news.empty:
            st.markdown("**Podklady, ze kterých byly změny odvozeny:**")
            for _,n in news.head(8).iterrows():
                date=clean_text(n.get("Datum")); source=clean_text(n.get("Zdroj"))
                meta=" · ".join(x for x in [date,source] if x)
                link=clean_text(n.get("Odkaz"))
                if link:
                    st.markdown(f"- [{n.get('Název')}]({link})" + (f" · {meta}" if meta else ""))
                else:
                    st.markdown(f"- {n.get('Název')}" + (f" · {meta}" if meta else ""))
        else:
            st.warning("Nebyly nalezeny dostatečně relevantní firemní zprávy. To je v testu validní výsledek – Analytik nemá doplňovat domněnky.")

    with st.expander("📊 2. Finanční vývoj – trend, ne snapshot",expanded=True):
        st.write(analyst_financial_summary(annual,quarterly))
        if not annual.empty:
            st.markdown("**Celé účetní roky dostupné přes Yahoo Finance**")
            d=annual.copy()
            for c in d.columns[1:]:
                d[c]=d[c].map(lambda x: "—" if pd.isna(safe_float(x)) else (f"{safe_float(x):.1f} %".replace(".",",") if c.endswith("%") else analyst_human_number(x)))
            st.dataframe(d,use_container_width=True,hide_index=True)
        if not quarterly.empty:
            st.markdown("**Posledních dostupných 8 čtvrtletí**")
            d=quarterly.copy()
            for c in d.columns[1:]:
                d[c]=d[c].map(lambda x: "—" if pd.isna(safe_float(x)) else (f"{safe_float(x):.1f} %".replace(".",",") if c.endswith("%") else analyst_human_number(x)))
            st.dataframe(d,use_container_width=True,hide_index=True)
            ttm=analyst_ttm_from_quarters(quarterly)
            if not ttm.empty:
                st.markdown("**TTM – poslední čtyři dostupná čtvrtletí**")
                d=ttm.copy()
                for c in d.columns[1:]:
                    d[c]=d[c].map(lambda x: "—" if pd.isna(safe_float(x)) else (f"{safe_float(x):.1f} %".replace(".",",") if c.endswith("%") else analyst_human_number(x)))
                st.dataframe(d,use_container_width=True,hide_index=True)

    with st.expander("💰 3. Valuace – jaká očekávání jsou v ceně",expanded=False):
        vals=[]
        for k,v in [("P/E",q.get("pe")),("Forward P/E",q.get("forward_pe")),("P/S",q.get("ps")),("P/B",q.get("pb"))]:
            if not pd.isna(safe_float(v)): vals.append(f"**{k}:** {analyst_human_number(v,1)}")
        st.write(" · ".join(vals) if vals else "Valuační data nejsou dostupná.")
        st.caption("Cílem není vyrábět falešně přesnou férovou cenu, ale zjistit, jaké očekávání může být v ocenění obsaženo.")

    with st.expander("🛡️ 4. Konkurenční prostředí a MOAT",expanded=False):
        st.caption("Tato část bude doplněna až po ověření, že výzkumné jádro správně identifikuje firmu a její skutečné aktuální změny.")
    with st.expander("🌍 5. Perspektiva odvětví",expanded=False):
        st.caption("Zatím záměrně bez automatického výkladu. Nechceme přidávat obecné sektorové fráze dříve, než funguje jádro.")
    with st.expander("👔 6. Management a jeho důvěryhodnost",expanded=False):
        st.caption("Další fáze bude sledovat sliby vedení versus realitu, změny guidance, kapitálovou alokaci a změny komunikace v čase.")

    with st.expander("🧩 7. Investiční příběh – pracovní hypotéza",expanded=True):
        st.markdown(f"### {primary}")
        st.write(story_reason)
        st.caption("Je to nezávislá hypotéza Analytika. Nejde o výsledek Screeneru ani o investiční doporučení.")

    with st.expander("🧭 8. Cenový kontext",expanded=True):
        st.write(price_comment)

    with st.expander("🎯 9. Co má smysl dále ověřit",expanded=True):
        st.markdown("- Je hlavní změna **strukturální, cyklická, nebo pouze dočasná**?")
        st.markdown("- Promítá se už do **tržeb, marží a cash flow**, nebo zatím jen do očekávání?")
        st.markdown("- Jak na změnu reaguje **management** a odpovídají jeho kroky komunikaci?")
        st.markdown("- Co by v dalších výsledcích **potvrdilo** hlavní hypotézu a co by ji **vyvrátilo**?")

    with st.expander("📚 Zdroje a diagnostika",expanded=False):
        st.write(f"Yahoo Finance: {yahoo_ticker}")
        st.write(f"Relevantních zpráv po filtrování identity: {len(news) if news is not None else 0}")
        if exchange in ("NASDAQ","NYSE"):
            st.write(f"SEC poslední podání načtena: {len(sec) if sec is not None else 0}")
        st.caption("Analytik používá pouze veřejně dostupné zdroje. Pokud důkaz chybí, výstup jej nemá nahrazovat domněnkou.")


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

st.caption("Zdroje: Nasdaq Trader, Deutsche Börse Xetra a Yahoo Finance/yfinance. Data jsou získávána při screeningu a mohou být zpožděná či nedostupná.")
