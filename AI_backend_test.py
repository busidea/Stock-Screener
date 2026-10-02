import time
import requests
import streamlit as st

st.set_page_config(
    page_title="Groq AI Test",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Groq AI – přímý test")
st.caption(
    "Izolovaný test přímého Groq API. "
    "Hlavní Stock-Screener aplikace se tímto testem nemění."
)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-120b"


def get_api_key():
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
        return key.strip() if key else ""
    except Exception:
        return ""


def groq_chat(prompt, system_prompt=None):
    api_key = get_api_key()

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY nebyl nalezen ve Streamlit Secrets."
        )

    messages = []

    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })

    messages.append({
        "role": "user",
        "content": prompt
    })

    started = time.time()

    response = requests.post(
        GROQ_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": MODEL,
            "messages": messages,
            "temperature": 0.2
        },
        timeout=90
    )

    elapsed = time.time() - started

    if not response.ok:
        raise RuntimeError(
            f"HTTP {response.status_code}: "
            f"{response.text[:2000]}"
        )

    data = response.json()

    try:
        content = data["choices"][0]["message"]["content"]
    except Exception:
        raise RuntimeError(
            "Groq odpověděl, ale odpověď nemá očekávanou strukturu."
        )

    return elapsed, content, data


# ------------------------------------------------------------
# 1. Kontrola Secret
# ------------------------------------------------------------

st.subheader("1. Kontrola API klíče")

api_key = get_api_key()

if api_key:
    st.success(
        f"✅ GROQ_API_KEY je nalezen. "
        f"Klíč má {len(api_key)} znaků a jeho hodnota se nezobrazuje."
    )
else:
    st.error(
        "❌ GROQ_API_KEY nebyl nalezen. "
        "Zkontroluj Settings → Secrets této testovací aplikace."
    )


# ------------------------------------------------------------
# 2. Minimální test
# ------------------------------------------------------------

st.subheader("2. Minimální test komunikace")

if st.button("▶ Otestovat Groq", type="primary"):

    if not api_key:
        st.error("Nejdříve musí být dostupný GROQ_API_KEY.")
    else:
        try:
            elapsed, content, raw = groq_chat(
                "Odpověz pouze dvěma slovy: GROQ OK",
                "Jsi jednoduchý diagnostický test AI API. "
                "Dodrž přesně požadovaný formát odpovědi."
            )

            st.success(
                f"✅ Groq odpověděl. "
                f"Čas: {elapsed:.1f} s"
            )

            st.markdown("### Odpověď AI")
            st.info(content)

            with st.expander("Technické informace"):
                st.write(f"Model: `{MODEL}`")
                st.write(f"Čas odpovědi: {elapsed:.2f} s")
                st.write(
                    f"Počet znaků odpovědi: {len(content)}"
                )

        except Exception as e:
            st.error(f"❌ Groq test selhal: {e}")


# ------------------------------------------------------------
# 3. Skutečný analytický test SHL
# ------------------------------------------------------------

st.divider()
st.subheader("3. Skutečný analytický test – Siemens Healthineers")

st.write(
    "Tento test už neověřuje pouze spojení. "
    "Ověří, zda model dokáže vytvořit analytickou syntézu."
)

SHL_PROMPT = """
Jsi seniorní akciový analytik. Analyzuj Siemens Healthineers (SHL.DE).

Nechci seznam článků ani obecný profil firmy.
Chci vlastní analytickou syntézu toho, co se ve společnosti skutečně mění.

Identifikuj 3 nejdůležitější probíhající změny.

U každé změny vysvětli:

1. Co se změnilo.
2. Proč se to mění.
3. Jaký může být ekonomický dopad.
4. Zda jde především o strukturální, cyklickou,
   dočasnou nebo jednorázovou změnu.
5. Co pro tuto interpretaci mluví.
6. Jaký je hlavní protiargument.
7. Co by hypotézu v dalších výsledcích potvrdilo
   nebo vyvrátilo.

Propoj pokud možno změny s finančním vývojem.

Na závěr formuluj jeden pracovní investiční příběh:
co je dnes hlavní změna oproti dřívějšímu příběhu firmy,
co může být trhem špatně pochopeno a co je naopak
rizikem této interpretace.

Nedávej doporučení BUY/SELL a nedávej číselné skóre.

Dostupné podklady:

- víceleté tržby: přibližně +24 %
- víceletý čistý zisk: přibližně +12 %
- víceletý FCF: přibližně -6 %
- TTM tržby proti poslednímu uzavřenému roku: -25 %
- TTM čistý zisk: -16 %
- TTM FCF: -18 %
- v roce 2026 je tlak na výhled zejména kvůli čínskému trhu
- současně jsou patrné známky silnějších marží
- některé části podnikání pokračují v růstu
- hlavní oblasti: Imaging, Diagnostics, Varian,
  Advanced Therapies
- akcie přibližně -17 % za 12 měsíců
- akcie přibližně -22 % za 3 roky

Důležité:
Nesnaž se pouze zopakovat podklady.
Pokud z nich nelze určit některou skutečnost,
výslovně řekni, že podklady ji nepotvrzují.
"""


if st.button("▶ Spustit skutečný SHL analytický test"):

    if not api_key:
        st.error("Nejdříve musí být dostupný GROQ_API_KEY.")
    else:
        try:
            with st.spinner(
                "Groq analyzuje Siemens Healthineers..."
            ):
                elapsed, content, raw = groq_chat(
                    SHL_PROMPT,
                    """
Jsi seniorní equity analytik.
Piš česky.
Buď kritický, konkrétní a věcný.
Nesnaž se uživatele uklidňovat ani mu doporučovat nákup či prodej.
Odděluj fakta, interpretaci a nejistotu.
"""
                )

            st.success(
                f"✅ Skutečná AI analýza dokončena za {elapsed:.1f} s"
            )

            st.markdown("### Analytická odpověď Groq")

            st.markdown(content)

            with st.expander("Technické informace"):
                st.write(f"Model: `{MODEL}`")
                st.write(f"Čas odpovědi: {elapsed:.2f} s")
                st.write(f"Délka odpovědi: {len(content)} znaků")

        except Exception as e:
            st.error(
                f"❌ SHL analytický test selhal: {e}"
            )


st.divider()

st.caption(
    "Tento soubor je pouze diagnostický test. "
    "Nenahrazuje ani nemění hlavní streamlit_app.py."
)
