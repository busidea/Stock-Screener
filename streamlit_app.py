import streamlit as st
import pandas as pd
import yfinance as yf


# --------------------------------------------------
# ZÁKLADNÍ NASTAVENÍ STRÁNKY
# --------------------------------------------------

st.set_page_config(
    page_title="Stock Screener",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 Stock Screener")

tab1, tab2 = st.tabs([
    "📊 Test fundamentálních dat",
    "🏛 NASDAQ Universe"
])


# ==================================================
# TAB 1 – TEST FUNDAMENTÁLNÍCH DAT
# ==================================================

with tab1:

    st.subheader("Test dostupnosti fundamentálních dat")

    TEST_TICKERS = [
        "AAPL",
        "MSFT",
        "NVDA",
        "META",
        "GOOGL",
        "AMZN",
        "TSLA",
        "INTC",
        "CSCO",
        "PEP",
        "KO",
        "JPM",
        "XOM",
        "JNJ",
        "WMT",
        "DIS",
        "BA",
        "PYPL",
        "PLTR",
        "NFLX"
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


        # ------------------------------------------
        # DOSTUPNOST DAT
        # ------------------------------------------

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

    st.write(
        """
        Tato část zatím pouze zjišťuje,
        zda dokážeme automaticky získat
        seznam titulů obchodovaných na NASDAQu.
        """
    )


    if st.button(
        "📥 Načíst NASDAQ Universe",
        key="nasdaq_universe"
    ):

        status = st.empty()

        status.write(
            "Načítám seznam titulů..."
        )

        try:

            url = (
                "https://www.nasdaqtrader.com/"
                "dynamic/SymDir/"
                "nasdaqlisted.txt"
            )

            df_nasdaq = pd.read_csv(
                url,
                sep="|"
            )


            # --------------------------------------
            # ODSTRANĚNÍ POSLEDNÍHO TECHNICKÉHO ŘÁDKU
            # --------------------------------------

            df_nasdaq = df_nasdaq[
                df_nasdaq["Symbol"]
                != "File Creation Time"
            ]


            # --------------------------------------
            # ZÁKLADNÍ ÚPRAVA
            # --------------------------------------

            puvodni_pocet = len(df_nasdaq)


            # Vyřadíme testovací tituly

            if "Test Issue" in df_nasdaq.columns:

                df_nasdaq = df_nasdaq[
                    df_nasdaq["Test Issue"] == "N"
                ]


            # Vyřadíme ETF

            if "ETF" in df_nasdaq.columns:

                df_nasdaq = df_nasdaq[
                    df_nasdaq["ETF"] == "N"
                ]


            # Vyřadíme NextShares,
            # pokud se ve zdroji objeví

            if "NextShares" in df_nasdaq.columns:

                df_nasdaq = df_nasdaq[
                    df_nasdaq["NextShares"] == "N"
                ]


            # --------------------------------------
            # PŘEJMENOVÁNÍ SLOUPCŮ
            # --------------------------------------

            df_display = df_nasdaq.rename(
                columns={
                    "Symbol": "Ticker",
                    "Security Name": "Název společnosti",
                    "ETF": "ETF",
                    "Test Issue": "Testovací titul"
                }
            )


            # Vybereme hlavní sloupce

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


            status.success(
                "NASDAQ Universe načten!"
            )


            # --------------------------------------
            # ZÁKLADNÍ STATISTIKY
            # --------------------------------------

            st.subheader("📊 Výsledek")

            col1, col2 = st.columns(2)

            col1.metric(
                "Titulů ve zdrojovém seznamu",
                puvodni_pocet
            )

            col2.metric(
                "Po základním vyčištění",
                len(df_display)
            )


            # --------------------------------------
            # TABULKA
            # --------------------------------------

            st.subheader(
                "🏛 Seznam nalezených titulů"
            )

            st.dataframe(
                df_display,
                use_container_width=True,
                height=600
            )


            # --------------------------------------
            # VYHLEDÁNÍ TICKERU
            # --------------------------------------

            st.subheader(
                "🔍 Rychlé vyhledání společnosti"
            )

            hledat = st.text_input(
                "Napiš ticker nebo část názvu:"
            )


            if hledat:

                hledat_upper = hledat.upper()

                vysledek_hledani = df_display[
                    df_display[
                        "Ticker"
                    ].astype(str).str.upper()
                    .str.contains(
                        hledat_upper,
                        na=False
                    )
                    |
                    df_display[
                        "Název společnosti"
                    ].astype(str).str.upper()
                    .str.contains(
                        hledat_upper,
                        na=False
                    )
                ]

                st.dataframe(
                    vysledek_hledani,
                    use_container_width=True
                )


        except Exception as e:

            status.error(
                "Nepodařilo se načíst NASDAQ Universe."
            )

            st.exception(e)
