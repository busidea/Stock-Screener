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
ANALYST_FALLBACK_REVISION = "DF45-A · priorita aktuálních zpráv, restrukturalizace a posledního vývoje"

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
st.caption("V6.28.46 · BUILD 20261014-M – Screener · samostatný modul Analytik je dostupný v menu vlevo")

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
