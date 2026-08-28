import streamlit as st
import pandas as pd
import yfinance as yf


# --------------------------------------------------
# ZÁKLADNÍ NASTAVENÍ
# --------------------------------------------------

st.set_page_config(
    page_title="Stock Screener",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 Stock Screener")


# --------------------------------------------------
# FUNKCE PRO NAČTENÍ NASDAQ UNIVERZA
# --------------------------------------------------

@st.cache_data(ttl=3600)
def nacti_nasdaq_universe():

    url = (
        "https://www.nasdaqtrader.com/"
        "dynamic/SymDir/"
        "nasdaqlisted.txt"
    )

    df = pd.read_csv(
        url,
        sep="|"
    )

    # Odstranění technického posledního řádku
    df = df[
        df["Symbol"] != "File Creation Time"
    ]

    puvodni_pocet = len(df)

    # Odstranění testovacích titulů
    if "Test Issue" in df.columns:
        df = df[
            df["Test Issue"] == "N"
        ]

    # Odstranění ETF
    if "ETF" in df.columns:
        df = df[
            df["ETF"] == "N"
        ]

    # Odstranění NextShares
    if "NextShares" in df.columns:
        df = df[
            df["NextShares"] == "N"
        ]

    return df, puvodni_pocet


# --------------------------------------------------
# KLASIFIKACE TITULŮ
# --------------------------------------------------

def klasifikuj_titul(nazev):

    nazev = str(nazev).lower()

    # Priorita 1 – Warrants
    if "warrant" in nazev:
        return "Warrant"

    # Priorita 2 – Rights
    if " right" in nazev:
        return "Right"

    # Priorita 3 – Units
    if " unit" in nazev:
        return "Unit"

    # Priorita 4 – Preferred shares
    if (
        "preferred" in nazev
        or "preference" in nazev
    ):
        return "Preferred Share"

    # Priorita 5 – Funds
    if "fund" in nazev:
        return "Fund"

    # Priorita 6 – Trusts
    if "trust" in nazev:
        return "Trust"

    # Priorita 7 – SPAC / Acquisition
    if (
        "acquisition" in nazev
        or "blank check" in nazev
    ):
        return "SPAC / Acquisition"

    # Priorita 8 – ADR
    if (
        "adr" in nazev
        or "depositary" in nazev
    ):
        return "ADR"

    # Priorita 9 – Ordinary Shares
    if "ordinary shares" in nazev:
        return "Ordinary Shares"

    # Priorita 10 – Common Stock
    if "common stock" in nazev:
        return "Common Stock"

    # Ostatní
    return "Other / Unknown"


# --------------------------------------------------
# ZÁLOŽKY
# --------------------------------------------------

tab1, tab2, tab3 = st.tabs([
    "📊 Test fundamentálních dat",
    "🏛 NASDAQ Universe",
    "🔬 Klasifikace univerza"
])


# ==================================================
# TAB 1 – FUNDAMENTÁLNÍ DATA
# ==================================================

with tab1:

    st.subheader("Test dostupnosti fundamentálních dat")

    TEST_TICKERS = [
        "AAPL", "MSFT", "NVDA", "META", "GOOGL",
        "AMZN", "TSLA", "INTC", "CSCO", "PEP",
        "KO", "JPM", "XOM", "JNJ", "WMT",
        "DIS", "BA", "PYPL", "PLTR", "NFLX"
    ]

    pocet = st.selectbox(
        "Počet testovaných společností:",
        [5, 10, 20],
        index=1
    )

    if st.button(
        "🔄 Spustit test dostupnosti dat",
        key="fundamental_test"
    ):

        tickers = TEST_TICKERS[:pocet]

        vysledky = []

        progress_bar = st.progress(0)
        status = st.empty()

        for i, ticker in enumerate(tickers):

            status.write(f"Načítám {ticker}...")

            try:

                stock = yf.Ticker(ticker)
                info = stock.info

                vysledky.append({
                    "Ticker": ticker,
                    "Firma": info.get("shortName"),
                    "Sektor": info.get("sector"),
                    "Market Cap": info.get("marketCap"),
                    "Trailing P/E": info.get("trailingPE"),
                    "Forward P/E": info.get("forwardPE"),
                    "P/S": info.get(
                        "priceToSalesTrailing12Months"
                    ),
                    "ROE": info.get("returnOnEquity"),
                    "Revenue Growth": info.get(
                        "revenueGrowth"
                    ),
                    "Earnings Growth": info.get(
                        "earningsGrowth"
                    ),
                    "Free Cash Flow": info.get(
                        "freeCashflow"
                    ),
                    "Debt / Equity": info.get(
                        "debtToEquity"
                    )
                })

            except Exception:

                vysledky.append({
                    "Ticker": ticker,
                    "Firma": "CHYBA",
                    "Sektor": None,
                    "Market Cap": None,
                    "Trailing P/E": None,
                    "Forward P/E": None,
                    "P/S": None,
                    "ROE": None,
                    "Revenue Growth": None,
                    "Earnings Growth": None,
                    "Free Cash Flow": None,
                    "Debt / Equity": None
                })

            progress_bar.progress(
                (i + 1) / len(tickers)
            )

        status.success("Hotovo!")

        df = pd.DataFrame(vysledky)

        st.subheader("📊 Načtená data")

        st.dataframe(
            df,
            use_container_width=True
        )

        st.subheader(
            "📈 Dostupnost jednotlivých dat"
        )

        dostupnost = []

        for sloupec in df.columns:

            if sloupec not in ["Ticker", "Firma"]:

                pocet_dostupnych = (
                    df[sloupec]
                    .notna()
                    .sum()
                )

                procento = round(
                    pocet_dostupnych
                    / len(df)
                    * 100,
                    1
                )

                dostupnost.append({
                    "Parametr": sloupec,
                    "Dostupných hodnot":
                        pocet_dostupnych,
                    "Celkem": len(df),
                    "Dostupnost %": procento
                })

        df_dostupnost = pd.DataFrame(
            dostupnost
        )

        st.dataframe(
            df_dostupnost,
            use_container_width=True
        )


# ==================================================
# TAB 2 – NASDAQ UNIVERSE
# ==================================================

with tab2:

    st.subheader(
        "🏛 Test načtení univerza NASDAQ"
    )

    if st.button(
        "📥 Načíst NASDAQ Universe",
        key="nasdaq_universe"
    ):

        try:

            df_nasdaq, puvodni_pocet = (
                nacti_nasdaq_universe()
            )

            st.success(
                "NASDAQ Universe načten!"
            )

            col1, col2 = st.columns(2)

            col1.metric(
                "Titulů ve zdrojovém seznamu",
                puvodni_pocet
            )

            col2.metric(
                "Po základním vyčištění",
                len(df_nasdaq)
            )

            df_display = df_nasdaq.rename(
                columns={
                    "Symbol": "Ticker",
                    "Security Name":
                        "Název společnosti"
                }
            )

            sloupce = [
                "Ticker",
                "Název společnosti"
            ]

            if "Financial Status" in df_display.columns:
                sloupce.append(
                    "Financial Status"
                )

            if "Market Category" in df_display.columns:
                sloupce.append(
                    "Market Category"
                )

            df_display = df_display[
                sloupce
            ]

            st.subheader(
                "🏛 Seznam nalezených titulů"
            )

            st.dataframe(
                df_display,
                use_container_width=True,
                height=600
            )

        except Exception as e:

            st.error(
                "Nepodařilo se načíst NASDAQ Universe."
            )

            st.exception(e)


# ==================================================
# TAB 3 – KLASIFIKACE UNIVERZA
# ==================================================

with tab3:

    st.subheader(
        "🔬 Klasifikace NASDAQ Universe"
    )

    st.write(
        """
        Každému titulu je přiřazena právě jedna hlavní
        kategorie podle prioritních pravidel.
        """
    )

    if st.button(
        "🔬 Spustit klasifikaci",
        key="classify_universe"
    ):

        try:

            df, puvodni_pocet = (
                nacti_nasdaq_universe()
            )

            # Přidání klasifikace
            df["Kategorie"] = df[
                "Security Name"
            ].apply(
                klasifikuj_titul
            )

            # --------------------------------------
            # SOUHRN
            # --------------------------------------

            souhrn = (
                df["Kategorie"]
                .value_counts()
                .reset_index()
            )

            souhrn.columns = [
                "Kategorie",
                "Počet titulů"
            ]

            st.subheader(
                "📊 Klasifikace titulů"
            )

            st.dataframe(
                souhrn,
                use_container_width=True
            )

            # --------------------------------------
            # KANDIDÁTI PRO SCREENING
            # --------------------------------------

            kandidati = df[
                df["Kategorie"].isin([
                    "Common Stock",
                    "Ordinary Shares",
                    "ADR",
                    "Other / Unknown"
                ])
            ]

            st.subheader(
                "🟢 Kandidáti pro Stock Screener"
            )

            col1, col2 = st.columns(2)

            col1.metric(
                "Celkem titulů",
                len(df)
            )

            col2.metric(
                "Potenciální kandidáti",
                len(kandidati)
            )

            # --------------------------------------
            # VÝBĚR KATEGORIE
            # --------------------------------------

            st.subheader(
                "🔍 Prohlédnout kategorii"
            )

            vyber = st.selectbox(
                "Kategorie:",
                sorted(
                    df["Kategorie"]
                    .unique()
                )
            )

            ukazka = df[
                df["Kategorie"] == vyber
            ][[
                "Symbol",
                "Security Name",
                "Kategorie"
            ]]

            st.dataframe(
                ukazka,
                use_container_width=True,
                height=500
            )

            # --------------------------------------
            # UKÁZKA KANDIDÁTŮ
            # --------------------------------------

            st.subheader(
                "🟢 Ukázka kandidátů"
            )

            st.dataframe(
                kandidati[
                    [
                        "Symbol",
                        "Security Name",
                        "Kategorie"
                    ]
                ].head(200),
                use_container_width=True,
                height=500
            )

        except Exception as e:

            st.error(
                "Klasifikaci se nepodařilo provést."
            )

            st.exception(e)
