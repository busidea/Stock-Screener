import time
import requests
import streamlit as st


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="AI Backend Test V2",
    layout="wide"
)

st.title("🧪 AI Backend Test V2")
st.caption(
    "Diagnostika skutečné dostupnosti AI backendů pro Stock-Screener."
)


# ============================================================
# TEST PROMPTS
# ============================================================

MINIMAL_PROMPT = "Reply only with OK."

SHL_PROMPT = """Jsi seniorní akciový analytik.

Analyzuj Siemens Healthineers (SHL.DE).

Nechci seznam článků ani obecný profil firmy. Chci vlastní analytickou
syntézu toho, co se ve společnosti skutečně mění.

Identifikuj 3 nejdůležitější probíhající změny.

U každé vysvětli:

1. co se změnilo
2. proč se to děje
3. jaký je ekonomický dopad
4. zda jde spíše o strukturální, cyklickou nebo dočasnou změnu
5. jak se to projevuje ve finančních datech
6. hlavní protiargument

Řekni také, co by tuto hypotézu v dalších kvartálech potvrdilo
a co by ji naopak vyvrátilo.

Na závěr formuluj jeden pracovní investiční příběh.

Nedávej doporučení BUY / HOLD / SELL.

Podklady:

Tržby za víceleté období +24 %.
Čistý zisk +12 %.
FCF -6 %.

TTM tržby proti poslednímu roku -25 %.
TTM čistý zisk -16 %.
TTM FCF -18 %.

V roce 2026 existuje tlak na výhled kvůli čínskému trhu.
Současně se objevují informace o silnějších maržích
a pokračujícím růstu v některých částech podnikání.

Firma působí v:
- Imaging
- Diagnostics
- Varian
- Advanced Therapies

Akcie jsou přibližně:
-17 % za 12 měsíců
-22 % za 3 roky.

Důležité:
Neopakuj pouze vstupní údaje.
Pokud z nich nelze některý závěr spolehlivě odvodit,
výslovně to řekni.
"""


# ============================================================
# ENDPOINTS
# ============================================================

ENDPOINTS = [
    ("G4F / Groq", "https://g4f.space/api/groq"),
    ("G4F / Gemini", "https://g4f.space/api/gemini"),
    ("G4F / NVIDIA", "https://g4f.space/api/nvidia"),
    ("G4F / Ollama", "https://g4f.space/api/ollama"),
    ("G4F / Pollinations", "https://g4f.space/api/pollinations"),
]


HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "Stock-Screener-AI-Test/2.0",
}


# ============================================================
# HELPERS
# ============================================================

def classify_error(exc):
    text = str(exc).lower()

    if "402" in text:
        return "⚠️ HTTP 402 – provider vyžaduje credits / quota"

    if "401" in text or "403" in text:
        return "🔐 AUTH ERROR – autentizace / oprávnění"

    if "timeout" in text:
        return "⏱ TIMEOUT"

    if "connection" in text:
        return "🌐 CONNECTION ERROR"

    return "❌ JINÁ CHYBA"


def get_models(base):
    start = time.time()

    response = requests.get(
        base + "/models",
        timeout=15,
        headers={
            "User-Agent": HEADERS["User-Agent"]
        },
    )

    elapsed = time.time() - start

    if not response.ok:
        raise RuntimeError(
            f"HTTP {response.status_code}: "
            f"{response.text[:1000]}"
        )

    data = response.json()

    if isinstance(data, dict):
        raw = data.get(
            "data",
            data.get("models", [])
        )
    else:
        raw = data

    models = []

    for item in raw:
        if isinstance(item, str):
            models.append(item)

        elif isinstance(item, dict):
            model_id = item.get("id")

            if model_id:
                models.append(model_id)

    return models, elapsed


def choose_model(models):
    """
    Pokusí se vybrat rozumný model.
    Pokud není známý model, použije první dostupný.
    """

    if not models:
        return None

    preferred_patterns = [
        "gpt-oss-20b",
        "llama-3.1-8b",
        "llama-3.2-3b",
        "gemini-2.0-flash",
        "gemini-2.5-flash",
        "qwen",
        "mistral",
    ]

    lowered = [
        (m, m.lower())
        for m in models
    ]

    for pattern in preferred_patterns:
        for original, low in lowered:
            if pattern in low:
                return original

    return models[0]


def chat_test(base, model, prompt, timeout=60):
    start = time.time()

    response = requests.post(
        base + "/chat/completions",
        timeout=timeout,
        json={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": 0.1,
        },
        headers=HEADERS,
    )

    elapsed = time.time() - start

    if not response.ok:
        raise RuntimeError(
            f"HTTP {response.status_code}: "
            f"{response.text[:2000]}"
        )

    data = response.json()

    try:
        content = (
            data["choices"][0]
            ["message"]["content"]
        )
    except Exception:
        raise RuntimeError(
            "HTTP 200, ale odpověď nemá očekávanou strukturu."
        )

    return content, elapsed, data


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
### Co tento test ověřuje

Test postupuje ve třech krocích:

**1. Endpoint**
→ odpovídá `/models`?

**2. Model**
→ existuje použitelný model?

**3. Skutečná inference**
→ model skutečně odpoví na chat request?

Pouhé `/models OK` tedy není považováno za funkční AI backend.
"""
)


# ============================================================
# RUN DIAGNOSTIC
# ============================================================

if st.button(
    "🔎 Spustit kompletní diagnostiku",
    type="primary"
):

    results = []

    progress = st.progress(0)

    for index, (name, base) in enumerate(ENDPOINTS):

        st.divider()

        st.subheader(name)

        result = {
            "name": name,
            "base": base,
            "models_ok": False,
            "chat_ok": False,
            "model": None,
            "models": [],
            "status": "",
            "elapsed_models": None,
            "elapsed_chat": None,
            "response": None,
            "raw": None,
        }

        # ----------------------------------------------------
        # STEP 1 – MODELS
        # ----------------------------------------------------

        try:

            models, elapsed = get_models(base)

            result["models_ok"] = True
            result["models"] = models
            result["elapsed_models"] = elapsed

            st.success(
                f"✅ /models OK — "
                f"{len(models)} modelů "
                f"({elapsed:.1f} s)"
            )

            if models:

                st.code(
                    "\n".join(models[:20])
                )

            else:

                st.warning(
                    "⚠️ Endpoint odpověděl, "
                    "ale seznam modelů je prázdný."
                )

        except Exception as exc:

            result["status"] = classify_error(exc)

            st.error(
                f"{result['status']}\n\n"
                f"{exc}"
            )

            results.append(result)

            progress.progress(
                (index + 1) / len(ENDPOINTS)
            )

            continue

        # ----------------------------------------------------
        # STEP 2 – CHOOSE MODEL
        # ----------------------------------------------------

        model = choose_model(models)

        result["model"] = model

        if not model:

            result["status"] = (
                "❌ Žádný použitelný model"
            )

            st.error(result["status"])

            results.append(result)

            progress.progress(
                (index + 1) / len(ENDPOINTS)
            )

            continue

        st.info(
            f"Vybraný testovací model: `{model}`"
        )

        # ----------------------------------------------------
        # STEP 3 – REAL CHAT
        # ----------------------------------------------------

        st.write(
            "Odesílám minimální test: "
            "`Reply only with OK.`"
        )

        try:

            content, elapsed, raw = chat_test(
                base,
                model,
                MINIMAL_PROMPT,
                timeout=60,
            )

            result["chat_ok"] = True
            result["elapsed_chat"] = elapsed
            result["response"] = content
            result["raw"] = raw
            result["status"] = "OK"

            st.success(
                f"🟢 FUNGUJE — HTTP 200 — "
                f"{elapsed:.1f} s"
            )

            st.write(
                f"**Odpověď:** {content}"
            )

        except Exception as exc:

            result["status"] = classify_error(exc)

            st.error(
                f"{result['status']}\n\n"
                f"{exc}"
            )

        results.append(result)

        progress.progress(
            (index + 1) / len(ENDPOINTS)
        )

    st.session_state["diagnostic_results"] = results


# ============================================================
# SUMMARY
# ============================================================

if "diagnostic_results" in st.session_state:

    results = st.session_state["diagnostic_results"]

    st.divider()

    st.header("📊 Výsledek diagnostiky")

    for r in results:

        if r["chat_ok"]:

            st.success(
                f"🟢 {r['name']} — FUNGUJE — "
                f"model: `{r['model']}` — "
                f"{r['elapsed_chat']:.1f} s"
            )

        elif r["models_ok"]:

            st.warning(
                f"🟠 {r['name']} — "
                f"/models funguje, ale chat NEFUNGUJE — "
                f"{r['status']}"
            )

        else:

            st.error(
                f"🔴 {r['name']} — "
                f"{r['status']}"
            )


    # ========================================================
    # TECHNICAL DETAILS
    # ========================================================

    with st.expander("🔧 Technické detaily"):

        for r in results:

            st.markdown(
                f"### {r['name']}"
            )

            st.write(
                f"Endpoint: `{r['base']}`"
            )

            st.write(
                f"Status: **{r['status']}**"
            )

            if r["model"]:
                st.write(
                    f"Model: `{r['model']}`"
                )

            if r["elapsed_models"] is not None:
                st.write(
                    f"/models čas: "
                    f"{r['elapsed_models']:.1f} s"
                )

            if r["elapsed_chat"] is not None:
                st.write(
                    f"Chat čas: "
                    f"{r['elapsed_chat']:.1f} s"
                )

            if r["response"]:
                st.write(
                    f"Odpověď: {r['response']}"
                )

            if r["raw"]:
                st.json(r["raw"])


# ============================================================
# SHL TEST
# ============================================================

if "diagnostic_results" in st.session_state:

    working = [
        r
        for r in st.session_state["diagnostic_results"]
        if r["chat_ok"]
    ]

    if working:

        st.divider()

        st.header(
            "🧠 Druhý krok – skutečný analytický test"
        )

        st.write(
            "Níže jsou pouze backendy, které "
            "úspěšně prošly minimálním chat testem."
        )

        labels = [
            f"{r['name']} · {r['model']}"
            for r in working
        ]

        selected = st.selectbox(
            "Vyber funkční backend",
            labels,
        )

        selected_result = working[
            labels.index(selected)
        ]

        if st.button(
            "▶️ Spustit SHL analytický test"
        ):

            with st.spinner(
                "AI zpracovává analytický úkol..."
            ):

                try:

                    content, elapsed, raw = chat_test(
                        selected_result["base"],
                        selected_result["model"],
                        SHL_PROMPT,
                        timeout=180,
                    )

                    st.success(
                        f"HTTP 200 · "
                        f"{elapsed:.1f} s · "
                        f"{len(content)} znaků"
                    )

                    st.markdown(
                        "### Výstup AI"
                    )

                    st.markdown(content)

                    with st.expander(
                        "🔧 Technická odpověď"
                    ):
                        st.json(raw)

                except Exception as exc:

                    st.error(
                        f"SHL test selhal:\n\n"
                        f"{classify_error(exc)}\n\n"
                        f"{exc}"
                    )

    else:

        st.info(
            "Zatím nebyl nalezen žádný backend, "
            "který by úspěšně prošel skutečným chat testem."
        )
