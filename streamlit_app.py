import io
import re
import time
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
import yfinance as yf


# ============================================================
# STOCK-SCREENER – 1. FUNKČNÍ VERZE
#
# Fáze 1:
#   1) vytvoření akciového univerza NASDAQ + NYSE + XETRA
#   2) omezený výběr kandidátů
#   3) načtení fundamentálních dat
#   4) numerický screening
#
# Strategická / AI analýza přijde až nad kandidáty,
# kteří projdou numerickým sítem.
# ============================================================

st.set_page_config(
    page_title="Stock-Screener",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 Stock-Screener")
st.caption("První funkční verze – numerický fundamentální screening")


# ============================================================
# ZDROJE UNIVERZA
# ============================================================

NASDAQ_URL = (
    "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
)

NYSE_URL = (
    "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
)

XETRA_URL = (
    "https://www.cashmarket.deutsche-boerse.com/resource/blob/1528/"
    "8e34798266f78fe8811bd24387445b2b/data/"
    "t7-xetr-allTradableInstruments.csv"
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; Stock-Screener/1.0)"
}


FUNDAMENTALS = [
    "Market Cap",
    "P/E",
    "Forward P/E",
    "P/S",
    "ROE",
    "Revenue Growth",
    "Earnings Growth",
    "Free Cash Flow",
    "Debt/Equity",
]


# ============================================================
# POMOCNÉ FUNKCE
# ============================================================

def clean_text(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value).strip()


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None

        if isinstance(value, bool):
            return None

        value = float(value)

        if pd.isna(value):
            return None

        return value

    except Exception:
        return None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [clean_text(c) for c in df.columns]
    return df


def request_text(url: str, timeout: int = 30) -> str:
    response = requests.get(
        url,
        headers=HEADERS,
        timeout=timeout,
    )

    response.raise_for_status()

    response.encoding = response.encoding or "utf-8"

    return response.text


def request_bytes(url: str, timeout: int = 30) -> bytes:
    response = requests.get(
        url,
        headers=HEADERS,
        timeout=timeout,
    )

    response.raise_for_status()

    return response.content


# ============================================================
# ROZPOZNÁNÍ SKUTEČNÉ AKCIE – NASDAQ / NYSE
# ============================================================

def looks_like_equity_name(name: str) -> bool:

    name = clean_text(name).lower()

    if not name:
        return False

    # Tvrdé vyloučení
    negative_patterns = [

        r"preferred",

        r"warrant",

        r"\bright\b",

        r"\bunit\b",

        r"\bnotes?\b",

        r"\bbond\b",

        r"\bdebenture\b",

        r"\bsenior notes?\b",

        r"\bsubordinated notes?\b",

        r"\btrust\b",

        r"\bfund\b",

        r"\betf\b",

        r"\bspac\b",

        r"acquisition company",

        r"subscription",

        r"depositary shares",
    ]

    for pattern in negative_patterns:

        if re.search(pattern, name):
            return False


    # Pozitivní identifikace akcie
    positive_patterns = [

        r"common stock",

        r"common shares?",

        r"ordinary shares?",

        r"ordinary share",

        r"capital stock",

        r"registered shares?",

        r"american depositary shares?",

        r"american depositary receipts?",

        r"\bads\b",

        r"\badr\b",

        r"class [a-z0-9]+ common",

        r"class [a-z0-9]+ ordinary",

        r"new york registry shares",

        r"subordinate voting shares?",

        r"voting shares?",
    ]

    return any(
        re.search(pattern, name)
        for pattern in positive_patterns
    )


# ============================================================
# NASDAQ
# ============================================================

@st.cache_data(
    ttl=3600,
    show_spinner=False
)
def load_nasdaq() -> Tuple[pd.DataFrame, str]:

    try:

        text = request_text(NASDAQ_URL)

        df = pd.read_csv(
            io.StringIO(text),
            sep="|",
            dtype=str,
            skipfooter=1,
            engine="python",
        )

        df = normalize_columns(df)

        if "Test Issue" in df.columns:

            df = df[
                df["Test Issue"].fillna("N") != "Y"
            ]

        if "ETF" in df.columns:

            df = df[
                df["ETF"].fillna("N") != "Y"
            ]

        if "NextShares" in df.columns:

            df = df[
                df["NextShares"].fillna("N") != "Y"
            ]


        df = df.rename(
            columns={
                "Symbol": "Ticker",
                "Security Name": "Name",
            }
        )


        if not {"Ticker", "Name"}.issubset(
            df.columns
        ):

            raise ValueError(
                "NASDAQ soubor nemá očekávané sloupce."
            )


        df["Exchange"] = "NASDAQ"

        df["Yahoo Ticker"] = (
            df["Ticker"]
            .astype(str)
            .str.strip()
        )

        df["IsShare"] = (
            df["Name"]
            .apply(looks_like_equity_name)
        )

        df = df[
            df["IsShare"]
        ].copy()


        result = df[
            [
                "Ticker",
                "Name",
                "Exchange",
                "Yahoo Ticker",
            ]
        ].copy()


        return (
            result,
            f"OK – {len(result):,} akcií"
        )


    except Exception as e:

        return (
            pd.DataFrame(),
            f"CHYBA: {e}"
        )


# ============================================================
# NYSE
# ============================================================

@st.cache_data(
    ttl=3600,
    show_spinner=False
)
def load_nyse() -> Tuple[pd.DataFrame, str]:

    try:

        text = request_text(NYSE_URL)

        df = pd.read_csv(
            io.StringIO(text),
            sep="|",
            dtype=str,
            skipfooter=1,
            engine="python",
        )

        df = normalize_columns(df)


        if "Exchange" in df.columns:

            df = df[
                df["Exchange"] == "N"
            ]


        if "Test Issue" in df.columns:

            df = df[
                df["Test Issue"].fillna("N") != "Y"
            ]


        if "ETF" in df.columns:

            df = df[
                df["ETF"].fillna("N") != "Y"
            ]


        if "NextShares" in df.columns:

            df = df[
                df["NextShares"].fillna("N") != "Y"
            ]


        df = df.rename(
            columns={
                "ACT Symbol": "Ticker",
                "Security Name": "Name",
            }
        )


        if not {"Ticker", "Name"}.issubset(
            df.columns
        ):

            raise ValueError(
                "NYSE soubor nemá očekávané sloupce."
            )


        df["Exchange"] = "NYSE"

        df["Yahoo Ticker"] = (
            df["Ticker"]
            .astype(str)
            .str.strip()
        )

        df["IsShare"] = (
            df["Name"]
            .apply(looks_like_equity_name)
        )

        df = df[
            df["IsShare"]
        ].copy()


        result = df[
            [
                "Ticker",
                "Name",
                "Exchange",
                "Yahoo Ticker",
            ]
        ].copy()


        return (
            result,
            f"OK – {len(result):,} akcií"
        )


    except Exception as e:

        return (
            pd.DataFrame(),
            f"CHYBA: {e}"
        )


# ============================================================
# XETRA
# ============================================================

@st.cache_data(
    ttl=3600,
    show_spinner=False
)
def load_xetra() -> Tuple[pd.DataFrame, str]:

    try:

        raw = request_bytes(XETRA_URL)

        decoded = raw.decode(
            "utf-8-sig",
            errors="replace"
        )

        lines = decoded.splitlines()


        # Automatické nalezení řádku s hlavičkou
        header_index = None

        for i, line in enumerate(lines[:15]):

            low = line.lower()

            if (
                "instrument type" in low
                or "mnemonic" in low
            ):

                header_index = i
                break


        if header_index is None:

            header_index = 2


        csv_text = "\n".join(
            lines[header_index:]
        )


        df = pd.read_csv(
            io.StringIO(csv_text),
            sep=";",
            dtype=str,
        )

        df = normalize_columns(df)


        # Normalizace názvů
        colmap = {}

        for col in df.columns:

            low = col.lower().strip()

            if low == "instrument type":

                colmap[col] = "Instrument Type"

            elif low == "mnemonic":

                colmap[col] = "Mnemonic"

            elif low in {
                "title",
                "instrument name",
                "security name",
            }:

                colmap[col] = "Name"

            elif low == "isin":

                colmap[col] = "ISIN"


        df = df.rename(
            columns=colmap
        )


        required = {
            "Instrument Type",
            "Mnemonic",
        }


        if not required.issubset(
            df.columns
        ):

            raise ValueError(
                "Xetra CSV nemá očekávané sloupce. "
                f"Nalezeno: {list(df.columns)}"
            )


        # CS = Common Stock / Equity
        df["Instrument Type"] = (
            df["Instrument Type"]
            .fillna("")
            .str.upper()
            .str.strip()
        )


        df = df[
            df["Instrument Type"] == "CS"
        ].copy()


        df["Mnemonic"] = (
            df["Mnemonic"]
            .fillna("")
            .str.strip()
        )


        df = df[
            df["Mnemonic"] != ""
        ].copy()


        df["Ticker"] = (
            df["Mnemonic"]
            + ".DE"
        )

        df["Yahoo Ticker"] = (
            df["Ticker"]
        )

        df["Exchange"] = "XETRA"


        if "Name" not in df.columns:

            df["Name"] = df["Ticker"]


        if "ISIN" not in df.columns:

            df["ISIN"] = ""


        result = df[
            [
                "Ticker",
                "Name",
                "Exchange",
                "Yahoo Ticker",
                "ISIN",
            ]
        ].copy()


        result = result.drop_duplicates(
            subset=["Ticker"]
        )


        return (
            result,
            f"OK – {len(result):,} akcií"
        )


    except Exception as e:

        return (
            pd.DataFrame(),
            f"CHYBA: {e}"
        )


# ============================================================
# CELÉ UNIVERZUM
# ============================================================

@st.cache_data(
    ttl=3600,
    show_spinner=False
)
def load_universe() -> Tuple[
    pd.DataFrame,
    Dict[str, str]
]:

    nasdaq, nasdaq_status = (
        load_nasdaq()
    )

    nyse, nyse_status = (
        load_nyse()
    )

    xetra, xetra_status = (
        load_xetra()
    )


    frames = []

    if not nasdaq.empty:
        frames.append(nasdaq)

    if not nyse.empty:
        frames.append(nyse)

    if not xetra.empty:
        frames.append(xetra)


    if not frames:

        return (
            pd.DataFrame(),
            {
                "NASDAQ": nasdaq_status,
                "NYSE": nyse_status,
                "XETRA": xetra_status,
            },
        )


    universe = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )


    universe = universe.drop_duplicates(
        subset=[
            "Ticker",
            "Exchange",
        ]
    )


    universe = universe.sort_values(
        [
            "Exchange",
            "Ticker",
        ]
    ).reset_index(
        drop=True
    )


    return (
        universe,
        {
            "NASDAQ": nasdaq_status,
            "NYSE": nyse_status,
            "XETRA": xetra_status,
        },
    )


# ============================================================
# FINANČNÍ VÝKAZY – POMOCNÉ FUNKCE
# ============================================================

def get_statement_row(
    df: Any,
    candidates: list
) -> Optional[pd.Series]:

    if (
        df is None
        or not isinstance(df, pd.DataFrame)
        or df.empty
    ):

        return None


    normalized = {
        str(idx).strip().lower(): idx
        for idx in df.index
    }


    # Přesná shoda
    for candidate in candidates:

        key = candidate.lower().strip()

        if key in normalized:

            return df.loc[
                normalized[key]
            ]


    # Částečná shoda
    for candidate in candidates:

        key = candidate.lower().strip()

        for idx_key, original_idx in normalized.items():

            if key in idx_key:

                return df.loc[
                    original_idx
                ]


    return None


def latest_two_values(
    df: Any,
    candidates: list
) -> Tuple[
    Optional[float],
    Optional[float]
]:

    row = get_statement_row(
        df,
        candidates
    )


    if row is None:

        return None, None


    values = []

    for value in row.iloc[:5].tolist():

        number = safe_float(value)

        if number is not None:

            values.append(number)

        if len(values) >= 2:

            break


    latest = (
        values[0]
        if len(values) >= 1
        else None
    )

    previous = (
        values[1]
        if len(values) >= 2
        else None
    )


    return latest, previous


def latest_value(
    df: Any,
    candidates: list
) -> Optional[float]:

    latest, _ = latest_two_values(
        df,
        candidates
    )

    return latest


def growth_percent(
    current: Optional[float],
    previous: Optional[float]
) -> Optional[float]:

    if current is None:
        return None

    if previous is None:
        return None

    if previous == 0:
        return None

    # Přechod ze ztráty do zisku
    # není smysluplné vyjádřit jednoduchým %.
    if (
        previous < 0 <= current
        or previous > 0 >= current
    ):

        return None


    return (
        current / previous - 1.0
    ) * 100.0


def get_price(
    info: dict,
    fast_info: Any
) -> Optional[float]:

    for key in [
        "currentPrice",
        "regularMarketPrice",
        "previousClose",
    ]:

        value = safe_float(
            info.get(key)
        )

        if value is not None:

            return value


    try:

        value = safe_float(
            getattr(
                fast_info,
                "last_price",
                None
            )
        )

        if value is not None:

            return value

    except Exception:

        pass


    return None


def get_market_cap(
    info: dict,
    fast_info: Any
) -> Optional[float]:

    value = safe_float(
        info.get("marketCap")
    )

    if value is not None:

        return value


    try:

        value = safe_float(
            getattr(
                fast_info,
                "market_cap",
                None
            )
        )

        if value is not None:

            return value

    except Exception:

        pass


    return None


# ============================================================
# FUNDAMENTÁLNÍ DATA – JEDEN TITUL
# ============================================================

@st.cache_data(
    ttl=1800,
    show_spinner=False
)
def fetch_fundamentals(
    ticker: str
) -> Dict[str, Any]:

    result = {

        "Ticker": ticker,

        "Status": "ERROR",

        "Market Cap": None,

        "P/E": None,

        "Forward P/E": None,

        "P/S": None,

        "ROE": None,

        "Revenue Growth": None,

        "Earnings Growth": None,

        "Free Cash Flow": None,

        "Debt/Equity": None,

        "Data Source": "",

        "Error": "",
    }


    errors = []


    try:

        stock = yf.Ticker(
            ticker
        )


        # ----------------------------------------------------
        # Yahoo info
        # ----------------------------------------------------

        try:

            info = stock.info or {}

        except Exception as e:

            info = {}

            errors.append(
                f"info: {str(e)[:120]}"
            )


        # ----------------------------------------------------
        # fast_info
        # ----------------------------------------------------

        try:

            fast_info = stock.fast_info

        except Exception:

            fast_info = None


        market_cap = get_market_cap(
            info,
            fast_info
        )

        price = get_price(
            info,
            fast_info
        )


        result["Market Cap"] = (
            market_cap
        )


        result["P/E"] = safe_float(
            info.get(
                "trailingPE"
            )
        )


        result["Forward P/E"] = safe_float(
            info.get(
                "forwardPE"
            )
        )


        result["P/S"] = safe_float(
            info.get(
                "priceToSalesTrailing12Months"
            )
        )


        roe = safe_float(
            info.get(
                "returnOnEquity"
            )
        )


        result["ROE"] = (
            roe * 100.0
            if roe is not None
            else None
        )


        revenue_growth = safe_float(
            info.get(
                "revenueGrowth"
            )
        )


        result["Revenue Growth"] = (
            revenue_growth * 100.0
            if revenue_growth is not None
            else None
        )


        earnings_growth = safe_float(
            info.get(
                "earningsGrowth"
            )
        )


        result["Earnings Growth"] = (
            earnings_growth * 100.0
            if earnings_growth is not None
            else None
        )


        result["Free Cash Flow"] = safe_float(
            info.get(
                "freeCashflow"
            )
        )


        result["Debt/Equity"] = safe_float(
            info.get(
                "debtToEquity"
            )
        )


        # ----------------------------------------------------
        # Financial statements
        # ----------------------------------------------------

        income = None
        balance = None
        cashflow = None


        try:

            income = stock.ttm_income_stmt

        except Exception:

            try:

                income = stock.income_stmt

            except Exception as e:

                errors.append(
                    f"income_stmt: {str(e)[:100]}"
                )


        try:

            balance = stock.balance_sheet

        except Exception as e:

            errors.append(
                f"balance_sheet: {str(e)[:100]}"
            )


        try:

            cashflow = stock.cashflow

        except Exception as e:

            errors.append(
                f"cashflow: {str(e)[:100]}"
            )


        # ----------------------------------------------------
        # Revenue
        # ----------------------------------------------------

        revenue, revenue_prev = (
            latest_two_values(
                income,
                [
                    "Total Revenue",
                    "Operating Revenue",
                ],
            )
        )


        # ----------------------------------------------------
        # Net Income
        # ----------------------------------------------------

        net_income, net_income_prev = (
            latest_two_values(
                income,
                [
                    "Net Income",
                    "Net Income Common Stockholders",
                ],
            )
        )


        # ----------------------------------------------------
        # Equity
        # ----------------------------------------------------

        equity = latest_value(
            balance,
            [
                "Stockholders Equity",
                "Stockholders' Equity",
                "Common Stock Equity",
                "Total Equity Gross Minority Interest",
            ],
        )


        # ----------------------------------------------------
        # Debt
        # ----------------------------------------------------

        total_debt = latest_value(
            balance,
            [
                "Total Debt",
                "Total Debt And Capital Lease Obligation",
                "Long Term Debt And Capital Lease Obligation",
            ],
        )


        # ----------------------------------------------------
        # P/S
        # ----------------------------------------------------

        if (
            result["P/S"] is None
            and market_cap is not None
            and revenue is not None
            and revenue > 0
        ):

            result["P/S"] = (
                market_cap / revenue
            )


        # ----------------------------------------------------
        # P/E
        # ----------------------------------------------------

        if (
            result["P/E"] is None
            and market_cap is not None
            and net_income is not None
            and net_income > 0
        ):

            result["P/E"] = (
                market_cap / net_income
            )


        # ----------------------------------------------------
        # ROE
        # ----------------------------------------------------

        if (
            result["ROE"] is None
            and net_income is not None
            and equity is not None
            and equity != 0
        ):

            result["ROE"] = (
                net_income / equity
            ) * 100.0


        # ----------------------------------------------------
        # Revenue Growth
        # ----------------------------------------------------

        if result["Revenue Growth"] is None:

            result["Revenue Growth"] = (
                growth_percent(
                    revenue,
                    revenue_prev
                )
            )


        # ----------------------------------------------------
        # Earnings Growth
        # ----------------------------------------------------

        if result["Earnings Growth"] is None:

            result["Earnings Growth"] = (
                growth_percent(
                    net_income,
                    net_income_prev
                )
            )


        # ----------------------------------------------------
        # Debt / Equity
        # ----------------------------------------------------

        if (
            result["Debt/Equity"] is None
            and total_debt is not None
            and equity is not None
            and equity != 0
        ):

            result["Debt/Equity"] = (
                total_debt / equity
            ) * 100.0


        # ----------------------------------------------------
        # Free Cash Flow
        # ----------------------------------------------------

        if result["Free Cash Flow"] is None:

            direct_fcf = latest_value(
                cashflow,
                [
                    "Free Cash Flow"
                ],
            )


            if direct_fcf is not None:

                result["Free Cash Flow"] = (
                    direct_fcf
                )

            else:

                operating_cf = latest_value(
                    cashflow,
                    [
                        "Operating Cash Flow",
                        "Total Cash From Operating Activities",
                    ],
                )


                capex = latest_value(
                    cashflow,
                    [
                        "Capital Expenditure",
                        "Capital Expenditures",
                    ],
                )


                if (
                    operating_cf is not None
                    and capex is not None
                ):

                    result["Free Cash Flow"] = (
                        operating_cf + capex
                    )


        # ----------------------------------------------------
        # Market Cap z ceny a počtu akcií
        # ----------------------------------------------------

        if (
            result["Market Cap"] is None
            and price is not None
        ):

            shares = safe_float(
                info.get(
                    "sharesOutstanding"
                )
            )


            if (
                shares is not None
                and shares > 0
            ):

                result["Market Cap"] = (
                    price * shares
                )


        # ----------------------------------------------------
        # Stav dat
        # ----------------------------------------------------

        available = sum(
            result[field] is not None
            for field in FUNDAMENTALS
        )


        if available == len(FUNDAMENTALS):

            result["Status"] = "OK"

        elif available > 0:

            result["Status"] = "PARTIAL"

        else:

            result["Status"] = "NO DATA"


        # ----------------------------------------------------
        # Zdroj dat
        # ----------------------------------------------------

        sources = []


        if info:

            sources.append(
                "Yahoo info"
            )


        if (
            income is not None
            and not getattr(
                income,
                "empty",
                True
            )
        ):

            sources.append(
                "income"
            )


        if (
            balance is not None
            and not getattr(
                balance,
                "empty",
                True
            )
        ):

            sources.append(
                "balance"
            )


        if (
            cashflow is not None
            and not getattr(
                cashflow,
                "empty",
                True
            )
        ):

            sources.append(
                "cashflow"
            )


        result["Data Source"] = (
            " + ".join(sources)
        )


        if errors:

            result["Error"] = (
                " | ".join(
                    errors[:3]
                )
            )


        # Ochrana proti throttlingu
        time.sleep(0.25)


        return result


    except Exception as e:

        result["Status"] = "ERROR"

        result["Error"] = str(e)[:300]

        return result


# ============================================================
# NUMERICKÝ SCREENING
# ============================================================

def apply_screening(
    df: pd.DataFrame,
    min_market_cap: float,
    max_pe: float,
    max_forward_pe: float,
    min_roe: float,
    min_revenue_growth: float,
    min_earnings_growth: float,
    min_fcf: float,
    max_debt_equity: float,
) -> pd.DataFrame:

    if df.empty:

        return df.copy()


    mask = pd.Series(
        True,
        index=df.index
    )


    # Market Cap
    if min_market_cap > 0:

        mask &= (
            df["Market Cap"]
            .fillna(-1)
            >= min_market_cap
        )


    # P/E
    if max_pe > 0:

        mask &= (
            df["P/E"].notna()
            & (df["P/E"] > 0)
            & (df["P/E"] <= max_pe)
        )


    # Forward P/E
    if max_forward_pe > 0:

        mask &= (
            df["Forward P/E"].notna()
            & (df["Forward P/E"] > 0)
            & (
                df["Forward P/E"]
                <= max_forward_pe
            )
        )


    # ROE
    if min_roe != -100:

        mask &= (
            df["ROE"].notna()
            & (
                df["ROE"]
                >= min_roe
            )
        )


    # Revenue Growth
    if min_revenue_growth != -100:

        mask &= (
            df["Revenue Growth"].notna()
            & (
                df["Revenue Growth"]
                >= min_revenue_growth
            )
        )


    # Earnings Growth
    if min_earnings_growth != -100:

        mask &= (
            df["Earnings Growth"].notna()
            & (
                df["Earnings Growth"]
                >= min_earnings_growth
            )
        )


    # FCF
    if min_fcf != -1e30:

        mask &= (
            df["Free Cash Flow"].notna()
            & (
                df["Free Cash Flow"]
                >= min_fcf
            )
        )


    # Debt / Equity
    if max_debt_equity > 0:

        mask &= (
            df["Debt/Equity"].notna()
            & (
                df["Debt/Equity"]
                >= 0
            )
            & (
                df["Debt/Equity"]
                <= max_debt_equity
            )
        )


    return df.loc[
        mask
    ].copy()


# ============================================================
# SIDEBAR – NASTAVENÍ
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ Nastavení screeneru"
    )


    selected_exchanges = st.multiselect(
        "Burzy",

        [
            "NASDAQ",
            "NYSE",
            "XETRA",
        ],

        default=[
            "NASDAQ",
            "NYSE",
            "XETRA",
        ],
    )


    st.markdown(
        "### 1. Předvýběr"
    )


    min_market_cap_b = st.number_input(
        "Min. Market Cap (mld.)",

        min_value=0.0,

        max_value=5000.0,

        value=1.0,

        step=0.5,

        help=(
            "Později bude tento filtr použit "
            "na celé předvybrané univerzum."
        ),
    )


    max_candidates = st.slider(
        "Max. titulů pro načtení fundamentů",

        min_value=25,

        max_value=1000,

        value=150,

        step=25,

        help=(
            "Kolik titulů maximálně načteme "
            "z Yahoo Finance v jednom běhu."
        ),
    )


    st.markdown(
        "### 2. Fundamentální filtr"
    )


    max_pe = st.number_input(
        "Max. P/E",

        min_value=0.0,

        max_value=200.0,

        value=25.0,

        step=1.0,
    )


    max_forward_pe = st.number_input(
        "Max. Forward P/E",

        min_value=0.0,

        max_value=200.0,

        value=20.0,

        step=1.0,
    )


    min_roe = st.number_input(
        "Min. ROE (%)",

        min_value=-100.0,

        max_value=200.0,

        value=10.0,

        step=1.0,
    )


    min_revenue_growth = st.number_input(
        "Min. růst tržeb (%)",

        min_value=-100.0,

        max_value=500.0,

        value=0.0,

        step=1.0,
    )


    min_earnings_growth = st.number_input(
        "Min. růst zisku (%)",

        min_value=-100.0,

        max_value=1000.0,

        value=0.0,

        step=1.0,
    )


    min_fcf_m = st.number_input(
        "Min. FCF (mil.)",

        min_value=-1000000.0,

        max_value=1000000.0,

        value=0.0,

        step=50.0,
    )


    max_debt_equity = st.number_input(
        "Max. Debt/Equity (%)",

        min_value=0.0,

        max_value=2000.0,

        value=150.0,

        step=10.0,

        help=(
            "Yahoo uvádí Debt/Equity typicky "
            "v procentech: 150 = 1,50x."
        ),
    )


    st.markdown(
        "### 3. Akce"
    )


    run_screen = st.button(
        "🚀 Spustit screening",

        type="primary",

        use_container_width=True,
    )


    if st.button(
        "🧹 Vyčistit výsledky",
        use_container_width=True
    ):

        st.session_state.pop(
            "screening_results",
            None
        )

        st.session_state.pop(
            "screened_results",
            None
        )

        st.session_state.pop(
            "candidate_universe",
            None
        )

        st.rerun()


    if st.button(
        "🔄 Obnovit zdroje",
        use_container_width=True
    ):

        st.cache_data.clear()

        st.session_state.pop(
            "screening_results",
            None
        )

        st.session_state.pop(
            "screened_results",
            None
        )

        st.session_state.pop(
            "candidate_universe",
            None
        )

        st.rerun()


# ============================================================
# NAČTENÍ UNIVERZA
# ============================================================

with st.spinner(
    "Načítám akciové univerzum…"
):

    universe, source_status = (
        load_universe()
    )


st.markdown("---")

st.subheader(
    "📊 Akciové univerzum"
)


if universe.empty:

    st.error(
        "Nepodařilo se vytvořit "
        "akciové univerzum."
    )

    st.stop()


if selected_exchanges:

    universe_selected = universe[
        universe["Exchange"].isin(
            selected_exchanges
        )
    ].copy()

else:

    universe_selected = (
        universe.iloc[0:0].copy()
    )


c1, c2, c3, c4 = st.columns(4)


with c1:

    st.metric(
        "Celkem",

        f"{len(universe):,}".replace(
            ",",
            " "
        ),
    )


with c2:

    st.metric(
        "NASDAQ",

        f"{(
            universe['Exchange'] == 'NASDAQ'
        ).sum():,}".replace(
            ",",
            " "
        ),
    )


with c3:

    st.metric(
        "NYSE",

        f"{(
            universe['Exchange'] == 'NYSE'
        ).sum():,}".replace(
            ",",
            " "
        ),
    )


with c4:

    st.metric(
        "XETRA",

        f"{(
            universe['Exchange'] == 'XETRA'
        ).sum():,}".replace(
            ",",
            " "
        ),
    )


with st.expander(
    "ℹ️ Stav zdrojů univerza",
    expanded=False
):

    for exchange, status in (
        source_status.items()
    ):

        if status.startswith("OK"):

            st.success(
                f"{exchange}: {status}"
            )

        else:

            st.warning(
                f"{exchange}: {status}"
            )


st.caption(
    "Univerzum obsahuje pouze akcie; "
    "fundamentální data se nestahují pro "
    "všech ~6 000 titulů najednou."
)


# ============================================================
# VÝBĚR VZORKU
# ============================================================

st.markdown("---")

st.subheader(
    "🎯 Předvýběr kandidátů"
)


st.write(
    "Vybrané burzy: "
    f"**{', '.join(selected_exchanges) if selected_exchanges else 'žádná'}**"
)


def build_candidate_sample(
    df: pd.DataFrame,
    limit: int
) -> pd.DataFrame:

    if df.empty:

        return df.copy()


    exchanges = [
        e
        for e in [
            "NASDAQ",
            "NYSE",
            "XETRA",
        ]
        if e in df["Exchange"].unique()
    ]


    if not exchanges:

        return df.head(
            limit
        ).copy()


    base = (
        limit
        // len(exchanges)
    )

    remainder = (
        limit
        % len(exchanges)
    )


    parts = []


    for i, exchange in enumerate(
        exchanges
    ):

        group = df[
            df["Exchange"] == exchange
        ].copy()


        take = (
            base
            + (
                1
                if i < remainder
                else 0
            )
        )


        if (
            take > 0
            and len(group) > take
        ):

            group = group.sample(
                n=take,
                random_state=42
            )


        parts.append(
            group
        )


    if parts:

        result = pd.concat(
            parts,
            ignore_index=True
        )

    else:

        result = df.head(
            limit
        ).copy()


    return (
        result
        .head(limit)
        .reset_index(drop=True)
    )


candidate_universe = (
    build_candidate_sample(
        universe_selected,
        max_candidates
    )
)


if (
    "candidate_universe"
    in st.session_state
    and not st.session_state[
        "candidate_universe"
    ].empty
):

    candidate_universe = (
        st.session_state[
            "candidate_universe"
        ].copy()
    )


st.info(
    f"Pro tento běh bude načteno "
    f"maximálně **{len(candidate_universe)} titulů**. "
    "Po získání fundamentů se aplikuje "
    "skutečný numerický filtr."
)


# ============================================================
# SPUŠTĚNÍ SCREENINGU
# ============================================================

if run_screen:

    if not selected_exchanges:

        st.warning(
            "Vyber alespoň jednu burzu."
        )

        st.stop()


    candidates = (
        candidate_universe.copy()
    )


    results = []


    progress = st.progress(0)

    status_text = st.empty()


    for i, row in (
        candidates
        .reset_index(drop=True)
        .iterrows()
    ):

        ticker = row[
            "Yahoo Ticker"
        ]


        status_text.write(
            f"Načítám {i + 1}/"
            f"{len(candidates)}: "
            f"**{ticker}**"
        )


        result = fetch_fundamentals(
            ticker
        )


        result["Ticker"] = (
            row["Ticker"]
        )

        result["Name"] = (
            row["Name"]
        )

        result["Exchange"] = (
            row["Exchange"]
        )

        result["Yahoo Ticker"] = (
            row["Yahoo Ticker"]
        )


        if "ISIN" in row.index:

            result["ISIN"] = (
                row.get(
                    "ISIN",
                    ""
                )
            )


        results.append(
            result
        )


        progress.progress(
            (i + 1)
            / len(candidates)
        )


    status_text.empty()

    progress.empty()


    results_df = pd.DataFrame(
        results
    )


    # --------------------------------------------------------
    # Aplikace filtru
    # --------------------------------------------------------

    min_market_cap = (
        min_market_cap_b
        * 1e9
    )


    min_fcf = (
        min_fcf_m
        * 1e6
    )


    screened = apply_screening(
        results_df,

        min_market_cap=(
            min_market_cap
        ),

        max_pe=(
            max_pe
        ),

        max_forward_pe=(
            max_forward_pe
        ),

        min_roe=(
            min_roe
        ),

        min_revenue_growth=(
            min_revenue_growth
        ),

        min_earnings_growth=(
            min_earnings_growth
        ),

        min_fcf=(
            min_fcf
        ),

        max_debt_equity=(
            max_debt_equity
        ),
    )


    if (
        not screened.empty
        and "P/E"
        in screened.columns
    ):

        screened = (
            screened
            .sort_values(
                "P/E",
                na_position="last"
            )
        )


    st.session_state[
        "screening_results"
    ] = results_df


    st.session_state[
        "screened_results"
    ] = screened


    st.session_state[
        "candidate_universe"
    ] = candidates


    st.success(
        f"Screening dokončen. "
        f"Načteno {len(results_df)} titulů, "
        f"filtrem prošlo {len(screened)} titulů."
    )


# ============================================================
# VÝSLEDKY
# ============================================================

if (
    "screening_results"
    in st.session_state
):

    results_df = (
        st.session_state[
            "screening_results"
        ].copy()
    )


    screened = (
        st.session_state.get(
            "screened_results",
            pd.DataFrame()
        ).copy()
    )


    st.markdown("---")

    st.subheader(
        "📋 Výsledek screeningu"
    )


    total = len(
        results_df
    )


    ok = int(
        (
            results_df["Status"]
            == "OK"
        ).sum()
    )


    partial = int(
        (
            results_df["Status"]
            == "PARTIAL"
        ).sum()
    )


    no_data = int(
        (
            results_df["Status"]
            == "NO DATA"
        ).sum()
    )


    errors = int(
        (
            results_df["Status"]
            == "ERROR"
        ).sum()
    )


    c1, c2, c3, c4, c5 = (
        st.columns(5)
    )


    with c1:

        st.metric(
            "Načteno titulů",
            total
        )


    with c2:

        st.metric(
            "Kompletní data",
            ok
        )


    with c3:

        st.metric(
            "Částečná data",
            partial
        )


    with c4:

        st.metric(
            "Bez dat",
            no_data
        )


    with c5:

        st.metric(
            "Prošlo filtrem",
            len(screened)
        )


    # ========================================================
    # HLAVNÍ TABULKA
    # ========================================================

    st.markdown(
        "### 🏆 Kandidáti po fundamentálním filtru"
    )


    if screened.empty:

        st.warning(
            "Žádný titul neprošel "
            "současným nastavením. "
            "Zkus nejdříve uvolnit P/E, "
            "růst, ROE nebo Debt/Equity."
        )


    else:

        display_cols = [

            "Ticker",

            "Name",

            "Exchange",

            "Market Cap",

            "P/E",

            "Forward P/E",

            "P/S",

            "ROE",

            "Revenue Growth",

            "Earnings Growth",

            "Free Cash Flow",

            "Debt/Equity",

            "Status",
        ]


        display_cols = [
            c
            for c in display_cols
            if c in screened.columns
        ]


        st.dataframe(

            screened[
                display_cols
            ],

            use_container_width=True,

            hide_index=True,

            height=600,

            column_config={

                "Market Cap":
                    st.column_config.NumberColumn(
                        "Market Cap",
                        format="%.0f"
                    ),

                "P/E":
                    st.column_config.NumberColumn(
                        "P/E",
                        format="%.2f"
                    ),

                "Forward P/E":
                    st.column_config.NumberColumn(
                        "Forward P/E",
                        format="%.2f"
                    ),

                "P/S":
                    st.column_config.NumberColumn(
                        "P/S",
                        format="%.2f"
                    ),

                "ROE":
                    st.column_config.NumberColumn(
                        "ROE (%)",
                        format="%.1f"
                    ),

                "Revenue Growth":
                    st.column_config.NumberColumn(
                        "Revenue Growth (%)",
                        format="%.1f"
                    ),

                "Earnings Growth":
                    st.column_config.NumberColumn(
                        "Earnings Growth (%)",
                        format="%.1f"
                    ),

                "Free Cash Flow":
                    st.column_config.NumberColumn(
                        "Free Cash Flow",
                        format="%.0f"
                    ),

                "Debt/Equity":
                    st.column_config.NumberColumn(
                        "Debt/Equity (%)",
                        format="%.1f"
                    ),
            },
        )


    # ========================================================
    # DOSTUPNOST DAT
    # ========================================================

    st.markdown(
        "### 📊 Dostupnost fundamentálních parametrů"
    )


    availability = []


    for parameter in FUNDAMENTALS:

        available = int(
            results_df[
                parameter
            ].notna().sum()
        )


        availability.append({

            "Parametr":
                parameter,

            "Dostupných hodnot":
                available,

            "Celkem":
                total,

            "Dostupnost %":
                round(
                    available
                    / total
                    * 100,
                    1
                )
                if total
                else 0,
        })


    availability_df = pd.DataFrame(
        availability
    )


    st.dataframe(
        availability_df,

        use_container_width=True,

        hide_index=True,
    )


    # ========================================================
    # KVALITA DAT PODLE BURZY
    # ========================================================

    st.markdown(
        "### 🏛️ Kvalita dat podle burzy"
    )


    exchange_rows = []


    for exchange, group in (
        results_df.groupby(
            "Exchange"
        )
    ):

        completeness = (
            group[
                FUNDAMENTALS
            ].notna().sum(axis=1)
        )


        exchange_rows.append({

            "Burza":
                exchange,

            "Testováno":
                len(group),

            "Kompletní data":
                int(
                    (
                        completeness
                        == len(FUNDAMENTALS)
                    ).sum()
                ),

            "≥ 50 % parametrů":
                int(
                    (
                        completeness
                        >= len(FUNDAMENTALS)
                        / 2
                    ).sum()
                ),

            "≥ 1 parametr":
                int(
                    (
                        completeness
                        >= 1
                    ).sum()
                ),

            "Průměrná dostupnost %":
                round(
                    completeness.mean()
                    / len(FUNDAMENTALS)
                    * 100,
                    1
                )
                if len(group)
                else 0,
        })


    exchange_summary = pd.DataFrame(
        exchange_rows
    )


    st.dataframe(
        exchange_summary,

        use_container_width=True,

        hide_index=True,
    )


    # ========================================================
    # DETAIL
    # ========================================================

    with st.expander(
        "🔍 Detail všech načtených titulů",
        expanded=False
    ):

        detail_cols = [

            "Ticker",

            "Name",

            "Exchange",

            *FUNDAMENTALS,

            "Status",

            "Data Source",

            "Error",
        ]


        detail_cols = [
            c
            for c in detail_cols
            if c in results_df.columns
        ]


        st.dataframe(

            results_df[
                detail_cols
            ],

            use_container_width=True,

            hide_index=True,

            height=650,
        )


    # ========================================================
    # PROBLÉMOVÉ TITULY
    # ========================================================

    problems = results_df[
        results_df["Status"]
        != "OK"
    ].copy()


    if not problems.empty:

        with st.expander(
            "⚠️ Tituly s neúplnými nebo chybějícími daty",
            expanded=False
        ):

            problem_cols = [

                "Ticker",

                "Name",

                "Exchange",

                "Status",

                "Data Source",

                "Error",
            ]


            problem_cols = [
                c
                for c in problem_cols
                if c in problems.columns
            ]


            st.dataframe(

                problems[
                    problem_cols
                ],

                use_container_width=True,

                hide_index=True,
            )


    # ========================================================
    # DALŠÍ FÁZE
    # ========================================================

    st.info(
        "Další fáze může nad kandidáty po numerickém filtru "
        "přidat strategickou klasifikaci: "
        "Growth / Balanced / Value / Turnaround / Speculative "
        "a následně hledat konkrétní příběhy typu "
        "restrukturalizace, nový management, odprodej aktiv "
        "nebo jiný katalyzátor."
    )


else:

    st.markdown("---")

    st.info(
        "Nastav filtry vlevo a klikni na "
        "**🚀 Spustit screening**. "
        "Aplikace potom načte fundamenty pouze "
        "pro omezený počet kandidátů, nikoliv "
        "pro celé univerzum."
    )


# ============================================================
# PATIČKA
# ============================================================

st.caption(
    "Data jsou získávána z veřejných zdrojů a "
    "Yahoo Finance přes yfinance. "
    "Chybějící údaj není nahrazován nulou."
)
