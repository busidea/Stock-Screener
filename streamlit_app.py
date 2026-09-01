import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# STOCK SCREENER
# ============================================================

st.set_page_config(
    page_title="Stock Screener",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 Stock Screener")

st.markdown(
    """
    Tento screener postupně vytváří akciové univerzum a testuje,
    jaká fundamentální data lze pro jednotlivé tituly získat.
    """
)

# ============================================================
# KONFIGURACE
# ============================================================

EXCHANGES = ["NASDAQ", "NYSE", "XETRA"]

FUNDAMENTAL_COLUMNS = [
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

TEST_TICKERS = [
    # NASDAQ
    ("AAPL", "NASDAQ"),
    ("MSFT", "NASDAQ"),
    ("GOOGL", "NASDAQ"),
    ("META", "NASDAQ"),
    ("NVDA", "NASDAQ"),

    # NYSE
    ("JPM", "NYSE"),
    ("KO", "NYSE"),
    ("JNJ", "NYSE"),
    ("V", "NYSE"),
    ("WMT", "NYSE"),

    # XETRA
    ("SAP.DE", "XETRA"),
    ("SIE.DE", "XETRA"),
    ("ALV.DE", "XETRA"),
    ("DTE.DE", "XETRA"),
    ("BMW.DE", "XETRA"),
]


# ============================================================
# POMOCNÉ FUNKCE
# ============================================================

def clean_number(value):
    """Převede hodnotu na číslo, jinak None."""
    try:
        if value is None:
            return None

        if pd.isna(value):
            return None

        value = float(value)

        if np.isfinite(value):
            return value

        return None

    except Exception:
        return None


def safe_get(dictionary, key):
    """Bezpečný přístup do dictionary."""
    try:
        return dictionary.get(key)
    except Exception:
        return None


def find_statement_value(statement, possible_names):
    """
    Najde požadovanou položku ve finančním výkazu.
    Yahoo používá pro podobné údaje různé názvy.
    """

    if statement is None or statement.empty:
        return None

    for name in possible_names:

        if name in statement.index:
            try:
                series = statement.loc[name]

                if isinstance(series, pd.Series):
                    for value in series:
                        value = clean_number(value)

                        if value is not None:
                            return value

            except Exception:
                pass

    return None


def get_latest_two(statement, possible_names):
    """
    Vrátí dvě poslední dostupné hodnoty z výkazu.
    """

    if statement is None or statement.empty:
        return None, None

    for name in possible_names:

        if name in statement.index:

            try:
                series = statement.loc[name]

                values = []

                for value in series:
                    value = clean_number(value)

                    if value is not None:
                        values.append(value)

                if len(values) >= 2:
                    return values[0], values[1]

            except Exception:
                pass

    return None, None


# ============================================================
# FUNDAMENTÁLNÍ DATA JEDNÉ AKCIE
# ============================================================

def get_fundamentals(ticker_symbol, exchange):

    result = {
        "Ticker": ticker_symbol,
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
        "Error": "",
        "Exchange": exchange
    }

    try:

        ticker = yf.Ticker(ticker_symbol)

        # ----------------------------------------------------
        # 1. INFO
        # ----------------------------------------------------

        info = {}

        try:
            info = ticker.info
            if not isinstance(info, dict):
                info = {}
        except Exception:
            info = {}

        # ----------------------------------------------------
        # 2. MARKET CAP
        # ----------------------------------------------------

        market_cap = safe_get(info, "marketCap")

        if market_cap is None:

            try:
                fast_info = ticker.fast_info

                market_cap = getattr(
                    fast_info,
                    "market_cap",
                    None
                )

            except Exception:
                market_cap = None

        result["Market Cap"] = clean_number(market_cap)

        # ----------------------------------------------------
        # 3. P/E
        # ----------------------------------------------------

        pe = safe_get(info, "trailingPE")

        if pe is None:
            pe = safe_get(info, "trailingPe")

        result["P/E"] = clean_number(pe)

        # ----------------------------------------------------
        # 4. FORWARD P/E
        # ----------------------------------------------------

        forward_pe = safe_get(info, "forwardPE")

        if forward_pe is None:
            forward_pe = safe_get(info, "forwardPe")

        result["Forward P/E"] = clean_number(forward_pe)

        # ----------------------------------------------------
        # 5. FINANČNÍ VÝKAZY
        # ----------------------------------------------------

        income = None
        balance = None
        cashflow = None

        try:
            income = ticker.income_stmt
        except Exception:
            pass

        try:
            balance = ticker.balance_sheet
        except Exception:
            pass

        try:
            cashflow = ticker.cashflow
        except Exception:
            pass

        # ----------------------------------------------------
        # 6. REVENUE
        # ----------------------------------------------------

        revenue_now, revenue_previous = get_latest_two(
            income,
            [
                "Total Revenue",
                "Operating Revenue",
                "Revenue"
            ]
        )

        revenue_growth = None

        if revenue_now is not None and revenue_previous is not None:
            if revenue_previous != 0:
                revenue_growth = (
                    revenue_now / revenue_previous
                ) - 1

        # pokud máme přímo Yahoo hodnotu, použijeme ji
        yahoo_revenue_growth = safe_get(
            info,
            "revenueGrowth"
        )

        if yahoo_revenue_growth is not None:
            revenue_growth = clean_number(
                yahoo_revenue_growth
            )

        result["Revenue Growth"] = revenue_growth

        # ----------------------------------------------------
        # 7. EARNINGS GROWTH
        # ----------------------------------------------------

        earnings_now, earnings_previous = get_latest_two(
            income,
            [
                "Net Income",
                "Net Income Common Stockholders",
                "Net Income Including Noncontrolling Interests"
            ]
        )

        earnings_growth = None

        if (
            earnings_now is not None
            and earnings_previous is not None
            and earnings_previous != 0
        ):

            earnings_growth = (
                earnings_now / earnings_previous
            ) - 1

        yahoo_earnings_growth = safe_get(
            info,
            "earningsGrowth"
        )

        if yahoo_earnings_growth is not None:
            earnings_growth = clean_number(
                yahoo_earnings_growth
            )

        result["Earnings Growth"] = earnings_growth

        # ----------------------------------------------------
        # 8. ROE
        # ----------------------------------------------------

        net_income = find_statement_value(
            income,
            [
                "Net Income",
                "Net Income Common Stockholders",
                "Net Income Including Noncontrolling Interests"
            ]
        )

        equity = find_statement_value(
            balance,
            [
                "Stockholders Equity",
                "Total Equity Gross Minority Interest",
                "Common Stock Equity"
            ]
        )

        roe = None

        if (
            net_income is not None
            and equity is not None
            and equity != 0
        ):

            roe = net_income / equity

        yahoo_roe = safe_get(info, "returnOnEquity")

        if yahoo_roe is not None:
            roe = clean_number(yahoo_roe)

        result["ROE"] = roe

        # ----------------------------------------------------
        # 9. FREE CASH FLOW
        # ----------------------------------------------------

        operating_cashflow = find_statement_value(
            cashflow,
            [
                "Operating Cash Flow",
                "Total Cash From Operating Activities",
                "Cash Flow From Continuing Operating Activities"
            ]
        )

        capex = find_statement_value(
            cashflow,
            [
                "Capital Expenditure",
                "Capital Expenditures"
            ]
        )

        free_cash_flow = None

        if (
            operating_cashflow is not None
            and capex is not None
        ):

            # Yahoo většinou uvádí CapEx jako zápornou hodnotu.
            # Proto zde používáme OCF + CapEx.
            free_cash_flow = (
                operating_cashflow + capex
            )

        yahoo_fcf = safe_get(
            info,
            "freeCashflow"
        )

        if yahoo_fcf is not None:
            free_cash_flow = clean_number(yahoo_fcf)

        result["Free Cash Flow"] = free_cash_flow

        # ----------------------------------------------------
        # 10. DEBT / EQUITY
        # ----------------------------------------------------

        total_debt = find_statement_value(
            balance,
            [
                "Total Debt",
                "Total Debt And Capital Lease Obligation"
            ]
        )

        debt_to_equity = None

        if (
            total_debt is not None
            and equity is not None
            and equity != 0
        ):

            debt_to_equity = (
                total_debt / equity
            )

        yahoo_de = safe_get(
            info,
            "debtToEquity"
        )

        if yahoo_de is not None:
            yahoo_de = clean_number(yahoo_de)

            if yahoo_de is not None:
                # Yahoo někdy uvádí hodnotu jako 165.3
                # místo 1.653
                if yahoo_de > 10:
                    yahoo_de = yahoo_de / 100

                debt_to_equity = yahoo_de

        result["Debt/Equity"] = debt_to_equity

        # ----------------------------------------------------
        # 11. P/S
        # ----------------------------------------------------

        ps = safe_get(
            info,
            "priceToSalesTrailing12Months"
        )

        if ps is not None:
            ps = clean_number(ps)

        # Pokud Yahoo P/S nemá, dopočítáme ho.
        if (
            ps is None
            and result["Market Cap"] is not None
            and revenue_now is not None
            and revenue_now != 0
        ):

            ps = (
                result["Market Cap"]
                / revenue_now
            )

        result["P/S"] = ps

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        result["Status"] = "OK"

        return result

    except Exception as e:

        result["Status"] = "ERROR"
        result["Error"] = str(e)

        return result


# ============================================================
# TEST FUNDAMENTÁLNÍCH DAT
# ============================================================

st.header("📊 Test dostupnosti fundamentálních dat")

st.markdown(
    """
    Nejprve testujeme malý vzorek 15 titulů.
    Tento test je důležitější než okamžité spuštění nad tisíci akciemi:
    potřebujeme zjistit, která data Yahoo skutečně poskytuje.
    """
)

if st.button("🚀 Spustit test fundamentálních dat", type="primary"):

    progress = st.progress(0)
    status_text = st.empty()

    results = []

    total = len(TEST_TICKERS)

    for i, (ticker, exchange) in enumerate(TEST_TICKERS):

        status_text.write(
            f"Načítám {ticker} ({exchange})..."
        )

        result = get_fundamentals(
            ticker,
            exchange
        )

        results.append(result)

        progress.progress(
            int((i + 1) / total * 100)
        )

        # malá pauza kvůli Yahoo rate limitům
        time.sleep(0.15)

    progress.empty()
    status_text.empty()

    df_test = pd.DataFrame(results)

    st.session_state["df_test"] = df_test

    # --------------------------------------------------------
    # DETAILNÍ VÝSLEDEK
    # --------------------------------------------------------

    st.subheader("🔎 Detailní výsledek")

    display_columns = [
        "Ticker",
        "Status",
        "Market Cap",
        "P/E",
        "Forward P/E",
        "P/S",
        "ROE",
        "Revenue Growth",
        "Earnings Growth",
        "Free Cash Flow",
        "Debt/Equity",
        "Error",
        "Exchange"
    ]

    st.dataframe(
        df_test[display_columns],
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # DOSTUPNOST JEDNOTLIVÝCH DAT
    # --------------------------------------------------------

    st.subheader(
        "📈 Dostupnost jednotlivých parametrů"
    )

    availability_rows = []

    for parameter in FUNDAMENTAL_COLUMNS:

        available = (
            df_test[parameter]
            .notna()
            .sum()
        )

        total = len(df_test)

        percentage = (
            available / total * 100
            if total > 0
            else 0
        )

        availability_rows.append(
            {
                "Parametr": parameter,
                "Dostupných hodnot": available,
                "Celkem": total,
                "Dostupnost %": round(
                    percentage,
                    1
                )
            }
        )

    df_availability = pd.DataFrame(
        availability_rows
    )

    st.dataframe(
        df_availability,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # ÚSPĚŠNOST PODLE BURZY
    # --------------------------------------------------------

    st.subheader(
        "🏛 Úspěšnost podle burzy"
    )

    exchange_rows = []

    for exchange in EXCHANGES:

        subset = df_test[
            df_test["Exchange"] == exchange
        ]

        if len(subset) == 0:
            continue

        ok = (
            subset["Status"] == "OK"
        ).sum()

        percentage = (
            ok / len(subset) * 100
        )

        exchange_rows.append(
            {
                "Burza": exchange,
                "Úspěšné tituly": ok,
                "Testované tituly": len(subset),
                "Úspěšnost %": round(
                    percentage,
                    1
                )
            }
        )

    df_exchange = pd.DataFrame(
        exchange_rows
    )

    st.dataframe(
        df_exchange,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # KOMENTÁŘ
    # --------------------------------------------------------

    st.success(
        "Test dokončen. Nyní můžeme podle skutečné dostupnosti "
        "dat rozhodnout, která pole použijeme ve vlastním screeneru."
    )


# ============================================================
# VÝSLEDKY Z PŘEDCHOZÍHO TESTU
# ============================================================

if "df_test" in st.session_state:

    st.divider()

    st.header("📋 Poslední výsledek testu")

    df = st.session_state["df_test"]

    # --------------------------------------------------------
    # ZÁKLADNÍ STATISTIKA
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Testovaných titulů",
            len(df)
        )

    with col2:
        st.metric(
            "Úspěšně načteno",
            int(
                (df["Status"] == "OK").sum()
            )
        )

    with col3:

        success_rate = (
            (
                df["Status"] == "OK"
            ).sum()
            / len(df)
            * 100
        )

        st.metric(
            "Úspěšnost",
            f"{success_rate:.1f}%"
        )

    # --------------------------------------------------------
    # DOPORUČENÍ PRO DALŠÍ KROK
    # --------------------------------------------------------

    st.info(
        """
        **Další krok:** podle tohoto testu vybereme fundamentální
        parametry, které jsou dostatečně dostupné. Teprve potom
        vytvoříme vlastní filtry Stock Screeneru.

        Záměrně zatím neprovádíme screening tisíců akcií.
        Nejdříve chceme ověřit kvalitu a konzistenci dat.
        """
    )


# ============================================================
# POZNÁMKA
# ============================================================

st.divider()

st.caption(
    "Stock Screener – pracovní vývojová verze. "
    "Data jsou získávána prostřednictvím yfinance / Yahoo Finance. "
    "Dostupnost jednotlivých údajů se může mezi tituly a burzami lišit."
)
