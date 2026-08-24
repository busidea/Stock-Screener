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
st.subheader("Test dostupnosti fundamentálních dat")


# --------------------------------------------------
# TESTOVACÍ SEZNAM FIREM
# --------------------------------------------------

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


# --------------------------------------------------
# VÝBĚR POČTU TESTOVANÝCH FIREM
# --------------------------------------------------

pocet = st.selectbox(
    "Počet testovaných společností:",
    [5, 10, 20],
    index=1
)


# --------------------------------------------------
# NAČTENÍ DAT
# --------------------------------------------------

if st.button("🔄 Spustit test dostupnosti dat"):

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
                "P/S": info.get("priceToSalesTrailing12Months"),
                "ROE": info.get("returnOnEquity"),
                "Revenue Growth": info.get("revenueGrowth"),
                "Earnings Growth": info.get("earningsGrowth"),
                "Free Cash Flow": info.get("freeCashflow"),
                "Debt / Equity": info.get("debtToEquity")
            })

        except Exception as e:

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

        progress_bar.progress((i + 1) / len(tickers))

    status.success("Hotovo!")

    df = pd.DataFrame(vysledky)

    st.subheader("📊 Načtená data")

    st.dataframe(
        df,
        use_container_width=True
    )


    # --------------------------------------------------
    # DOSTUPNOST DAT
    # --------------------------------------------------

    st.subheader("📈 Dostupnost jednotlivých dat")

    dostupnost = []

    for sloupec in df.columns:

        if sloupec not in ["Ticker", "Firma"]:

            pocet_dostupnych = df[sloupec].notna().sum()

            procento = round(
                pocet_dostupnych / len(df) * 100,
                1
            )

            dostupnost.append({
                "Parametr": sloupec,
                "Dostupných hodnot": pocet_dostupnych,
                "Celkem": len(df),
                "Dostupnost %": procento
            })

    df_dostupnost = pd.DataFrame(dostupnost)

    st.dataframe(
        df_dostupnost,
        use_container_width=True
    )
