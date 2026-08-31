import streamlit as st
import pandas as pd
import yfinance as yf
import re
import time

# ============================================================
# STOCK-SCREENER
# První pracovní verze:
# - načte univerzum NASDAQ / NYSE / XETRA
# - ponechá pouze skutečné akciové instrumenty
# - vyřadí ETF, fondy, warranty, trusty, SPAC apod.
# - otestuje dostupnost fundamentálních dat přes yfinance
# ============================================================

st.set_page_config(
    page_title="Stock-Screener",
    page_icon="🔎",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# 1. ZÁKLADNÍ KONSTANTY
# ============================================================

NASDAQ_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"

NYSE_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

XETRA_URL = (
    "https://www.cashmarket.deutsche-boerse.com/"
    "resource/blob/1528/8e34798266f78fe8811bd24387445b2b/"
    "data/t7-xetr-allTradableInstruments.csv"
)


# ============================================================
# 2. POMOCNÉ FUNKCE
# ============================================================

def clean_text(value):
    """Bezpečný převod na text."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def is_real_share_name(name):
    """
    U NASDAQ/NYSE rozhodujeme podle názvu instrumentu.
    Pozitivní seznam = něco, co vypadá jako skutečná akcie.
    Negativní seznam = instrumenty, které nechceme.
    """

    name = clean_text(name).lower()

    # -------------------------------
    # Co chceme
    # -------------------------------
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

    # -------------------------------
    # Co nechceme
    # -------------------------------
    negative_patterns = [
        r"preferred",
        r"warrant",
        r"right",
        r"unit",
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
        r"depositary unit",
        r"subscription"
    ]

    # Nejprve vyřadíme evidentní nežádoucí instrumenty
    for pattern in negative_patterns:
        if re.search(pattern, name):
            return False

    # Potom hledáme známku skutečné akcie
    for pattern in positive_patterns:
        if re.search(pattern, name):
            return True

    return False


# ============================================================
# 3. NASDAQ
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

        # Testovací instrumenty nechceme
        if "Test Issue" in df.columns:
            df = df[df["Test Issue"].fillna("N") != "Y"]

        # ETF nechceme
        if "ETF" in df.columns:
            df = df[df["ETF"].fillna("N") != "Y"]

        # NextShares také ne
        if "NextShares" in df.columns:
            df = df[df["NextShares"].fillna("N") != "Y"]

        df = df.rename(columns={
            "Symbol": "Ticker",
            "Security Name": "Name"
        })

        df["Exchange"] = "NASDAQ"

        # Pouze skutečné akcie
        df["IsShare"] = df["Name"].apply(is_real_share_name)

        df = df[df["IsShare"]].copy()

        return df[["Ticker", "Name", "Exchange"]]

    except Exception as e:
        st.error(f"Chyba při načítání NASDAQ: {e}")
        return pd.DataFrame()


# ============================================================
# 4. NYSE
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

        # Pouze NYSE (N)
        if "Exchange" in df.columns:
            df = df[df["Exchange"] == "N"]

        # Testovací instrumenty
        if "Test Issue" in df.columns:
            df = df[df["Test Issue"].fillna("N") != "Y"]

        # ETF
        if "ETF" in df.columns:
            df = df[df["ETF"].fillna("N") != "Y"]

        # NextShares
        if "NextShares" in df.columns:
            df = df[df["NextShares"].fillna("N") != "Y"]

        df = df.rename(columns={
            "ACT Symbol": "Ticker",
            "Security Name": "Name"
        })

        df["Exchange"] = "NYSE"

        df["IsShare"] = df["Name"].apply(is_real_share_name)

        df = df[df["IsShare"]].copy()

        return df[["Ticker", "Name", "Exchange"]]

    except Exception as e:
        st.error(f"Chyba při načítání NYSE: {e}")
        return pd.DataFrame()


# ============================================================
# 5. XETRA
# ============================================================

@st.cache_data(ttl=3600)
def load_xetra():
    try:
        # Xetra CSV začíná dvěma informačními řádky.
        df = pd.read_csv(
            XETRA_URL,
            sep=";",
            skiprows=2,
            dtype=str
        )

        df.columns = [clean_text(c) for c in df.columns]

        # Najdeme sloupce nezávisle na přesném zápisu
        columns_lower = {
            c.lower(): c for c in df.columns
        }

        def find_column(text):
            for lower_name, original in columns_lower.items():
                if text in lower_name:
                    return original
            return None

        instrument_col = find_column("instrument")
        type_col = find_column("instrument type")
        status_col = find_column("instrument status")
        mnemonic_col = find_column("mnemonic")
        isin_col = find_column("isin")

        if not type_col:
            st.error("Xetra: nebyl nalezen sloupec Instrument Type.")
            return pd.DataFrame()

        # Aktivní instrumenty
        if status_col:
            df = df[df[status_col].astype(str).str.strip() == "Active"]

        # CS = Common Stock / Equity
        df = df[df[type_col].astype(str).str.strip().str.upper() == "CS"]

        if not mnemonic_col:
            st.error("Xetra: nebyl nalezen sloupec Mnemonic.")
            return pd.DataFrame()

        result = pd.DataFrame()

        result["Ticker"] = (
            df[mnemonic_col]
            .astype(str)
            .str.strip()
            .str.upper()
            + ".DE"
        )

        if instrument_col:
            result["Name"] = df[instrument_col].astype(str).str.strip()
        else:
            result["Name"] = result["Ticker"]

        result["Exchange"] = "XETRA"

        if isin_col:
            result["ISIN"] = df[isin_col].astype(str).str.strip()

        return result[["Ticker", "Name", "Exchange"]]

    except Exception as e:
        st.error(f"Chyba při načítání XETRA: {e}")
        return pd.DataFrame()


# ============================================================
# 6. NAČTENÍ UNIVERZA
# ============================================================

def load_universe(exchanges):

    frames = []

    if "NASDAQ" in exchanges:
        frames.append(load_nasdaq())

    if "NYSE" in exchanges:
        frames.append(load_nyse())

    if "XETRA" in exchanges:
        frames.append(load_xetra())

    frames = [df for df in frames if not df.empty]

    if not frames:
        return pd.DataFrame()

    universe = pd.concat(frames, ignore_index=True)

    # Odstranění duplicit
    universe = universe.drop_duplicates(
        subset=["Ticker", "Exchange"]
    )

    universe = universe.sort_values(
        ["Exchange", "Ticker"]
    ).reset_index(drop=True)

    return universe


# ============================================================
# 7. TEST FUNDAMENTÁLNÍCH DAT
# ============================================================

@st.cache_data(ttl=1800)
def test_yfinance_data(tickers):

    results = []

    # Pro první test nechceme načítat celý svět najednou.
    # Počet lze později zvýšit.
    for i, ticker in enumerate(tickers):

        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            results.append({
                "Ticker": ticker,
                "Name": info.get("longName", ""),
                "Market Cap": info.get("marketCap"),
                "P/E": info.get("trailingPE"),
                "Forward P/E": info.get("forwardPE"),
                "P/S": info.get("priceToSalesTrailing12Months"),
                "ROE": info.get("returnOnEquity"),
                "Revenue Growth": info.get("revenueGrowth"),
                "Earnings Growth": info.get("earningsGrowth"),
                "Free Cash Flow": info.get("freeCashflow"),
                "Debt/Equity": info.get("debtToEquity")
            })

        except Exception:
            results.append({
                "Ticker": ticker,
                "Name": "",
                "Market Cap": None,
                "P/E": None,
                "Forward P/E": None,
                "P/S": None,
                "ROE": None,
                "Revenue Growth": None,
                "Earnings Growth": None,
                "Free Cash Flow": None,
                "Debt/Equity": None
            })

        # Malá pauza kvůli stabilitě veřejného zdroje
        time.sleep(0.05)

    return pd.DataFrame(results)


# ============================================================
# 8. HLAVNÍ APLIKACE
# ============================================================

st.title("🔎 Stock-Screener")

st.caption(
    "Pracovní verze – vytvoření čistého univerza skutečných akcií "
    "z NASDAQ, NYSE a XETRA."
)

st.markdown("---")


# ============================================================
# 9. VÝBĚR BURZ
# ============================================================

st.subheader("🏛 Výběr burz")

col1, col2, col3 = st.columns(3)

with col1:
    nasdaq = st.checkbox("NASDAQ", value=True)

with col2:
    nyse = st.checkbox("NYSE", value=True)

with col3:
    xetra = st.checkbox("XETRA", value=True)

selected_exchanges = []

if nasdaq:
    selected_exchanges.append("NASDAQ")

if nyse:
    selected_exchanges.append("NYSE")

if xetra:
    selected_exchanges.append("XETRA")


# ============================================================
# 10. NAČTENÍ
# ============================================================

if st.button("🚀 Načíst akciové univerzum", type="primary"):

    if not selected_exchanges:
        st.warning("Vyber alespoň jednu burzu.")
        st.stop()

    with st.spinner("Načítám seznam obchodovaných akcií..."):
        universe = load_universe(selected_exchanges)

    if universe.empty:
        st.error("Nepodařilo se načíst žádné akcie.")
        st.stop()

    st.session_state["universe"] = universe


# ============================================================
# 11. VÝSLEDKY UNIVERZA
# ============================================================

if "universe" in st.session_state:

    universe = st.session_state["universe"]

    st.markdown("---")

    st.subheader("📊 Akciové univerzum")

    # Statistiky
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Celkem akcií",
            f"{len(universe):,}".replace(",", " ")
        )

    with c2:
        st.metric(
            "NASDAQ",
            len(universe[universe["Exchange"] == "NASDAQ"])
        )

    with c3:
        st.metric(
            "NYSE",
            len(universe[universe["Exchange"] == "NYSE"])
        )

    with c4:
        st.metric(
            "XETRA",
            len(universe[universe["Exchange"] == "XETRA"])
        )

    st.markdown("### Seznam akcií")

    st.dataframe(
        universe,
        use_container_width=True,
        hide_index=True,
        height=600
    )


    # ========================================================
    # 12. TEST FUNDAMENTÁLNÍCH DAT
    # ========================================================

    st.markdown("---")

    st.subheader("🧪 Test dostupnosti fundamentálních dat")

    st.info(
        "Tato část zatím netahá data pro celé univerzum automaticky. "
        "Nejdříve otestujeme menší vzorek, abychom zjistili, "
        "jak dobře veřejná data fungují."
    )

    max_test = st.slider(
        "Počet titulů pro test",
        min_value=10,
        max_value=min(200, len(universe)),
        value=min(30, len(universe)),
        step=10
    )

    sample = universe.head(max_test).copy()

    if st.button("🔬 Spustit test dat"):

        tickers = sample["Ticker"].tolist()

        with st.spinner(
            f"Testuji dostupnost fundamentálních dat pro {len(tickers)} titulů..."
        ):
            test_data = test_yfinance_data(tickers)

        # Přidáme burzu
        test_data = test_data.merge(
            sample[["Ticker", "Exchange"]],
            on="Ticker",
            how="left"
        )

        st.session_state["test_data"] = test_data


    # ========================================================
    # 13. ZOBRAZENÍ TESTU
    # ========================================================

    if "test_data" in st.session_state:

        test_data = st.session_state["test_data"]

        st.markdown("### Výsledky testu")

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

        availability_rows = []

        for parameter in parameters:

            available = test_data[parameter].notna().sum()
            total = len(test_data)

            availability_rows.append({
                "Parametr": parameter,
                "Dostupných hodnot": available,
                "Celkem": total,
                "Dostupnost %": round(
                    available / total * 100, 1
                ) if total else 0
            })

        availability_df = pd.DataFrame(availability_rows)

        st.dataframe(
            availability_df,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("### Ukázka načtených dat")

        st.dataframe(
            test_data,
            use_container_width=True,
            hide_index=True,
            height=500
        )

        st.success(
            "Test dokončen. Pokud bude dostupnost fundamentálních dat "
            "dostatečná, můžeme nad tímto univerzem postavit vlastní "
            "finanční filtry."
        )
