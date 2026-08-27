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
# ZÁLOŽKY
# --------------------------------------------------

tab1, tab2, tab3 = st.tabs([
    "📊 Test fundamentálních dat",
    "🏛 NASDAQ Universe",
    "🔬 Analýza univerza"
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

            status = st.empty()

            status.success(
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
# TAB 3 – ANALÝZA UNIVERZA
# ==================================================

with tab3:

    st.subheader(
        "🔬 Rentgen NASDAQ Universe"
    )

    st.write(
        """
        Tato analýza se zatím nesnaží
        definitivně určit typ každého instrumentu.
        Pouze hledá typické výrazy v názvech titulů.
        """
    )

    if st.button(
        "🔬 Analyzovat NASDAQ Universe",
        key="analyse_universe"
    ):

        try:

            df, puvodni_pocet = (
                nacti_nasdaq_universe()
            )

            nazvy = df[
                "Security Name"
            ].astype(str)

            # --------------------------------------
            # KATEGORIE
            # --------------------------------------

            categories = {

                "Preferred Shares":
                    r"Preferred|Preference",

                "Warrants":
                    r"Warrant",

                "Rights":
                    r"\bRight\b",

                "Units":
                    r"\bUnit\b",

                "SPAC / Acquisition":
                    r"Acquisition|Blank Check",

                "ADR":
                    r"ADR|Depositary",

                "Trust":
                    r"Trust",

                "Fund":
                    r"Fund",

                "Ordinary Shares":
                    r"Ordinary Shares",

                "Common Stock":
                    r"Common Stock"
            }

            vysledky = []

            for category, pattern in (
                categories.items()
            ):

                pocet = nazvy.str.contains(
                    pattern,
                    case=False,
                    regex=True,
                    na=False
                ).sum()

                vysledky.append({
                    "Kategorie": category,
                    "Počet nalezených titulů":
                        pocet
                })

            df_categories = pd.DataFrame(
                vysledky
            )

            # --------------------------------------
            # VÝSLEDKY
            # --------------------------------------

            st.subheader(
                "📊 Typy instrumentů podle názvu"
            )

            st.dataframe(
                df_categories,
                use_container_width=True
            )

            # --------------------------------------
            # UKÁZKY
            # --------------------------------------

            st.subheader(
                "🔍 Ukázky jednotlivých kategorií"
            )

            vybrana_kategorie = st.selectbox(
                "Vyber kategorii:",
                list(categories.keys())
            )

            pattern = categories[
                vybrana_kategorie
            ]

            ukazka = df[
                nazvy.str.contains(
                    pattern,
                    case=False,
                    regex=True,
                    na=False
                )
            ][[
                "Symbol",
                "Security Name"
            ]]

            st.dataframe(
                ukazka,
                use_container_width=True,
                height=500
            )

            # --------------------------------------
            # NEIDENTIFIKOVANÉ
            # --------------------------------------

            st.subheader(
                "❓ Nezařazené tituly"
            )

            combined_pattern = "|".join(
                categories.values()
            )

            nezarazene = df[
                ~nazvy.str.contains(
                    combined_pattern,
                    case=False,
                    regex=True,
                    na=False
                )
            ][[
                "Symbol",
                "Security Name"
            ]]

            st.write(
                f"Počet nezařazených titulů: "
                f"{len(nezarazene)}"
            )

            st.dataframe(
                nezarazene.head(200),
                use_container_width=True,
                height=500
            )

        except Exception as e:

            st.error(
                "Analýzu se nepodařilo provést."
            )

            st.exception(e)
