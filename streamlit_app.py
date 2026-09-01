```python
import streamlit as st
import pandas as pd
import yfinance as yf
import re
import time

# ============================================================
# STOCK-SCREENER
# Diagnostická verze fundamentálních dat
#
# Cíl této verze:
# 1. vytvořit akciové univerzum NASDAQ / NYSE / XETRA
# 2. otestovat, zda k vybraným akciím dostaneme fundamentální data
# 3. zjistit, proč předchozí test ukazoval 0 %
#
# Zatím NEDĚLÁME finanční screening ani AI analýzu.
# ============================================================

st.set_page_config(
    page_title="Stock-Screener",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 Stock-Screener")

st.caption(
    "Diagnostická verze – test dostupnosti fundamentálních dat"
)

# ============================================================
# ZDROJE
# ============================================================

NASDAQ_URL = (
    "https://www.nasdaqtrader.com/"
    "dynamic/SymDir/nasdaqlisted.txt"
)

NYSE_URL = (
    "https://www.nasdaqtrader.com/"
    "dynamic/SymDir/otherlisted.txt"
)


# ============================================================
# POMOCNÁ FUNKCE
# ============================================================

def clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


# ============================================================
# ROZPOZNÁNÍ SKUTEČNÉ AKCIE
# ============================================================

def is_real_share_name(name):

    name = clean_text(name).lower()

    # Instrumenty, které nechceme
    negative_patterns = [
        r"preferred",
        r"warrant",
        r"right",
        r"\bunit\b",
        r"notes?",
        r"debenture",
        r"bond",
        r"senior notes?",
        r"subordinated notes?",
        r"trust",
        r"fund",
        r"etf",
        r"spac",
        r"acquisition",
        r"subscription"
    ]

    for pattern in negative_patterns:
        if re.search(pattern, name):
            return False

    # Instrumenty, které chceme
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
        r"class [a-z0-9]+ ordinary"
    ]

    for pattern in positive_patterns:
        if re.search(pattern, name):
            return True

    return False


# ============================================================
# NASDAQ
# ============================================================

@st.cache_data(ttl=3600)
def load_nasdaq():

    try:

        df = pd.read_csv(
            NASDAQ_URL,
            sep="|",
            dtype=str,
            skipfooter=1,
            engine="python"
        )

        df.columns = [clean_text(c) for c in df.columns]

        # Testovací instrumenty
        if "Test Issue" in df.columns:
            df = df[
                df["Test Issue"].fillna("N") != "Y"
            ]

        # ETF
        if "ETF" in df.columns:
            df = df[
                df["ETF"].fillna("N") != "Y"
            ]

        # NextShares
        if "NextShares" in df.columns:
            df = df[
                df["NextShares"].fillna("N") != "Y"
            ]

        df = df.rename(columns={
            "Symbol": "Ticker",
            "Security Name": "Name"
        })

        df["Exchange"] = "NASDAQ"

        df["IsShare"] = df["Name"].apply(
            is_real_share_name
        )

        df = df[df["IsShare"]].copy()

        return df[
            ["Ticker", "Name", "Exchange"]
        ]

    except Exception as e:

        st.error(
            f"Chyba při načítání NASDAQ: {e}"
        )

        return pd.DataFrame()


# ============================================================
# NYSE
# ============================================================

@st.cache_data(ttl=3600)
def load_nyse():

    try:

        df = pd.read_csv(
            NYSE_URL,
            sep="|",
            dtype=str,
            skipfooter=1,
            engine="python"
        )

        df.columns = [clean_text(c) for c in df.columns]

        # Pouze NYSE
        if "Exchange" in df.columns:
            df = df[
                df["Exchange"] == "N"
            ]

        # Testovací instrumenty
        if "Test Issue" in df.columns:
            df = df[
                df["Test Issue"].fillna("N") != "Y"
            ]

        # ETF
        if "ETF" in df.columns:
            df = df[
                df["ETF"].fillna("N") != "Y"
            ]

        # NextShares
        if "NextShares" in df.columns:
            df = df[
                df["NextShares"].fillna("N") != "Y"
            ]

        df = df.rename(columns={
            "ACT Symbol": "Ticker",
            "Security Name": "Name"
        })

        df["Exchange"] = "NYSE"

        df["IsShare"] = df["Name"].apply(
            is_real_share_name
        )

        df = df[df["IsShare"]].copy()

        return df[
            ["Ticker", "Name", "Exchange"]
        ]

    except Exception as e:

        st.error(
            f"Chyba při načítání NYSE: {e}"
        )

        return pd.DataFrame()


# ============================================================
# XETRA
# ============================================================

@st.cache_data(ttl=3600)
def load_xetra():

    # --------------------------------------------------------
    # Záměrně používáme bezpečnější veřejný zdroj přes Yahoo
    # pro diagnostický krok.
    #
    # XETRA univerzum ponecháme z předchozí funkční verze.
    # --------------------------------------------------------

    try:

        # Pro tento krok použijeme známé XETRA tituly,
        # abychom nejdříve ověřili mapování Yahoo tickerů.
        data = [
            ["SAP.DE", "SAP SE", "XETRA"],
            ["SIE.DE", "Siemens AG", "XETRA"],
            ["ALV.DE", "Allianz SE", "XETRA"],
            ["DTE.DE", "Deutsche Telekom AG", "XETRA"],
            ["BMW.DE", "Bayerische Motoren Werke AG", "XETRA"],
        ]

        return pd.DataFrame(
            data,
            columns=[
                "Ticker",
                "Name",
                "Exchange"
            ]
        )

    except Exception as e:

        st.error(
            f"Chyba při přípravě XETRA testu: {e}"
        )

        return pd.DataFrame()


# ============================================================
# VYTVOŘENÍ UNIVERZA
# ============================================================

@st.cache_data(ttl=3600)
def load_universe():

    frames = []

    nasdaq = load_nasdaq()
    nyse = load_nyse()
    xetra = load_xetra()

    if not nasdaq.empty:
        frames.append(nasdaq)

    if not nyse.empty:
        frames.append(nyse)

    if not xetra.empty:
        frames.append(xetra)

    if not frames:
        return pd.DataFrame()

    universe = pd.concat(
        frames,
        ignore_index=True
    )

    universe = universe.drop_duplicates(
        subset=["Ticker", "Exchange"]
    )

    universe = universe.sort_values(
        ["Exchange", "Ticker"]
    ).reset_index(drop=True)

    return universe


# ============================================================
# FUNDAMENTÁLNÍ DATA
# ============================================================

def get_fundamentals(ticker):

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
        "Error": ""
    }

    try:

        stock = yf.Ticker(ticker)

        # ----------------------------------------------------
        # Hlavní test
        # ----------------------------------------------------

        info = stock.info

        if not info:
            result["Error"] = (
                "Ticker.info vrátil prázdná data"
            )
            return result

        # ----------------------------------------------------
        # Načtení hodnot
        # ----------------------------------------------------

        result["Market Cap"] = info.get(
            "marketCap"
        )

        result["P/E"] = info.get(
            "trailingPE"
        )

        result["Forward P/E"] = info.get(
            "forwardPE"
        )

        result["P/S"] = info.get(
            "priceToSalesTrailing12Months"
        )

        result["ROE"] = info.get(
            "returnOnEquity"
        )

        result["Revenue Growth"] = info.get(
            "revenueGrowth"
        )

        result["Earnings Growth"] = info.get(
            "earningsGrowth"
        )

        result["Free Cash Flow"] = info.get(
            "freeCashflow"
        )

        result["Debt/Equity"] = info.get(
            "debtToEquity"
        )

        result["Status"] = "OK"

        return result

    except Exception as e:

        result["Error"] = str(e)

        return result


# ============================================================
# DIAGNOSTICKÝ VZOREK
# ============================================================

def create_test_sample(universe):

    samples = []

    # --------------------------------------------------------
    # NASDAQ – známé velké společnosti
    # --------------------------------------------------------

    nasdaq_test = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "META",
        "NVDA"
    ]

    # --------------------------------------------------------
    # NYSE – známé společnosti
    # --------------------------------------------------------

    nyse_test = [
        "JPM",
        "KO",
        "JNJ",
        "V",
        "WMT"
    ]

    # --------------------------------------------------------
    # XETRA
    # --------------------------------------------------------

    xetra_test = [
        "SAP.DE",
        "SIE.DE",
        "ALV.DE",
        "DTE.DE",
        "BMW.DE"
    ]

    for ticker in nasdaq_test:
        samples.append(
            [ticker, "NASDAQ"]
        )

    for ticker in nyse_test:
        samples.append(
            [ticker, "NYSE"]
        )

    for ticker in xetra_test:
        samples.append(
            [ticker, "XETRA"]
        )

    return pd.DataFrame(
        samples,
        columns=[
            "Ticker",
            "Exchange"
        ]
    )


# ============================================================
# HLAVNÍ PROGRAM
# ============================================================

st.markdown("---")

st.subheader("📊 Akciové univerzum")

universe = load_universe()

if universe.empty:

    st.error(
        "Nepodařilo se vytvořit akciové univerzum."
    )

else:

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Celkem",
            f"{len(universe):,}".replace(",", " ")
        )

    with c2:
        st.metric(
            "NASDAQ",
            len(
                universe[
                    universe["Exchange"] == "NASDAQ"
                ]
            )
        )

    with c3:
        st.metric(
            "NYSE",
            len(
                universe[
                    universe["Exchange"] == "NYSE"
                ]
            )
        )

    with c4:
        st.metric(
            "XETRA",
            len(
                universe[
                    universe["Exchange"] == "XETRA"
                ]
            )
        )

    st.dataframe(
        universe.head(100),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# DIAGNOSTIKA FUNDAMENTÁLNÍCH DAT
# ============================================================

st.markdown("---")

st.subheader(
    "🔬 Diagnostika dostupnosti fundamentálních dat"
)

st.write(
    """
    Nejprve netestujeme tisíce titulů. Testujeme malý vzorek
    známých akcií z každé burzy. Cílem je zjistit, zda správně
    funguje spojení ticker → Yahoo Finance → fundamentální data.
    """
)

sample = create_test_sample(universe)

st.markdown("### Testované tituly")

st.dataframe(
    sample,
    use_container_width=True,
    hide_index=True
)


if st.button(
    "🔬 Spustit diagnostický test",
    type="primary"
):

    results = []

    progress = st.progress(0)

    for i, row in sample.iterrows():

        ticker = row["Ticker"]
        exchange = row["Exchange"]

        result = get_fundamentals(
            ticker
        )

        result["Exchange"] = exchange

        results.append(result)

        progress.progress(
            (i + 1) / len(sample)
        )

        # malá pauza
        time.sleep(0.2)

    results_df = pd.DataFrame(
        results
    )

    st.session_state[
        "diagnostic_results"
    ] = results_df


# ============================================================
# VÝSLEDEK DIAGNOSTIKY
# ============================================================

if "diagnostic_results" in st.session_state:

    results_df = st.session_state[
        "diagnostic_results"
    ]

    st.markdown("---")

    st.subheader(
        "📋 Výsledek diagnostického testu"
    )

    # --------------------------------------------------------
    # Přehled úspěšnosti
    # --------------------------------------------------------

    total = len(results_df)

    successful = (
        results_df["Status"] == "OK"
    ).sum()

    success_percent = (
        successful / total * 100
        if total
        else 0
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Testovaných titulů",
            total
        )

    with c2:
        st.metric(
            "Úspěšně načteno",
            successful
        )

    with c3:
        st.metric(
            "Úspěšnost",
            f"{success_percent:.1f} %"
        )

    # --------------------------------------------------------
    # Podle burzy
    # --------------------------------------------------------

    st.markdown(
        "### Úspěšnost podle burzy"
    )

    exchange_summary = (
        results_df
        .groupby("Exchange")
        .agg(
            Testováno=("Ticker", "count"),
            Úspěšně=("Status", lambda x:
                     (x == "OK").sum())
        )
        .reset_index()
    )

    exchange_summary["Úspěšnost %"] = (
        exchange_summary["Úspěšně"]
        / exchange_summary["Testováno"]
        * 100
    ).round(1)

    st.dataframe(
        exchange_summary,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # Kompletní výsledek
    # --------------------------------------------------------

    st.markdown(
        "### Detailní výsledek"
    )

    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True,
        height=600
    )

    # --------------------------------------------------------
    # Dostupnost jednotlivých parametrů
    # --------------------------------------------------------

    st.markdown(
        "### Dostupnost jednotlivých parametrů"
    )

    parameters = [
        "Market Cap",
        "P/E",
        "Forward P/E",
        "P/S",
        "ROE",
        "Revenue Growth",
        "Earnings Growth",
        "Free Cash Flow",
        "Debt/Equity"
    ]

    availability = []

    for parameter in parameters:

        available = (
            results_df[parameter]
            .notna()
            .sum()
        )

        availability.append({
            "Parametr": parameter,
            "Dostupných hodnot": available,
            "Celkem": total,
            "Dostupnost %": round(
                available / total * 100,
                1
            ) if total else 0
        })

    availability_df = pd.DataFrame(
        availability
    )

    st.dataframe(
        availability_df,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # Chyby
    # --------------------------------------------------------

    errors = results_df[
        results_df["Error"].astype(str).str.len() > 0
    ]

    if not errors.empty:

        st.markdown(
            "### ⚠️ Chyby / problémy"
        )

        st.dataframe(
            errors[
                [
                    "Ticker",
                    "Exchange",
                    "Status",
                    "Error"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

    st.success(
        "Diagnostický test dokončen. "
        "Podle tohoto výsledku můžeme rozhodnout, "
        "jakým způsobem postavit hromadný fundamentální screening."
    )
```
