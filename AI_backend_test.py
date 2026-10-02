import time
import requests
import streamlit as st

st.set_page_config(
    page_title="Groq AI – evidence test",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Groq AI – evidence-based analytický test")

st.caption(
    "Testuje přímé Groq API a schopnost AI vytvářet analytickou syntézu "
    "bez vymýšlení skutečností, které nejsou v podkladech."
)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-120b"


# ============================================================
# API
# ============================================================

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
            "temperature": 0.1
        },
        timeout=120
    )

    elapsed = time.time() - started

    if not response.ok:
        raise RuntimeError(
            f"HTTP {response.status_code}: "
            f"{response.text[:3000]}"
        )

    data = response.json()

    try:
        content = data["choices"][0]["message"]["content"]
    except Exception:
        raise RuntimeError(
            "Groq odpověděl, ale odpověď nemá očekávanou strukturu."
        )

    return elapsed, content, data


# ============================================================
# 1. SECRET
# ============================================================

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


# ============================================================
# 2. MINIMÁLNÍ KOMUNIKACE
# ============================================================

st.subheader("2. Minimální test komunikace")

if st.button("▶ Otestovat Groq", type="primary"):

    if not api_key:
        st.error("Nejdříve musí být dostupný GROQ_API_KEY.")
    else:
        try:
            elapsed, content, raw = groq_chat(
                "Odpověz pouze dvěma slovy: GROQ OK",
                """
Jsi diagnostický AI systém.
Dodrž přesně požadovaný formát odpovědi.
"""
            )

            st.success(
                f"✅ Groq odpověděl za {elapsed:.1f} s"
            )

            st.markdown("### Odpověď AI")
            st.info(content)

        except Exception as e:
            st.error(f"❌ Groq test selhal: {e}")


# ============================================================
# 3. EVIDENCE-BASED SHL TEST
# ============================================================

st.divider()

st.subheader(
    "3. Evidence-based analytický test – Siemens Healthineers"
)

st.write(
    "Toto je hlavní test. AI dostane pouze explicitně uvedené podklady. "
    "Úkolem není zjistit další fakta z internetu, ale ukázat, "
    "zda dokáže z dostupných faktů vytvořit kvalitní analytickou syntézu."
)


SHL_PROMPT = """
ÚKOL

Jsi seniorní akciový analytik.

Analyzuj Siemens Healthineers pouze na základě podkladů uvedených níže.

Toto je TEST ANALYTICKÉ DISCIPLÍNY.

NESMÍŠ používat žádné jiné konkrétní skutečnosti, které nejsou
obsaženy v podkladech.

To znamená:

- nevymýšlej údaje z výročních zpráv,
- nevymýšlej data z výsledků jednotlivých kvartálů,
- nevymýšlej výroky managementu,
- nevymýšlej procenta,
- nevymýšlej názvy restrukturalizačních programů,
- nevymýšlej cíle úspor,
- nevymýšlej změny zaměstnanosti,
- nevymýšlej změny produktového mixu,
- nevymýšlej konkrétní příčiny, pokud nejsou v podkladech,
- nevymýšlej citace,
- nepoužívej znalosti o Siemens Healthineers z vlastní paměti jako fakta.

Pokud je pro nějaký závěr potřeba informace, která v podkladech není,
napiš:

"Z dostupných podkladů to nelze potvrdit."

Můžeš samozřejmě vytvořit ANALYTICKOU INFERENCI.
Takovou inferenci ale jasně označ jako:

"Inference: ..."

Nikdy nepředstavuj inferenci jako ověřený fakt.

DŮLEŽITÉ:

Údaj "-6 %" u FCF znamená změnu FCF v daném sledovaném období.
Údaj "-18 %" u TTM FCF znamená změnu TTM FCF vůči poslednímu
uzavřenému roku.

Tyto údaje NEZNAMENAJÍ, že Siemens Healthineers má záporný FCF
ve výši -6 % nebo -18 %.

Stejnou disciplínu dodrž u všech ostatních čísel.


ANALYTICKÝ ÚKOL

Identifikuj 3 nejdůležitější změny, které lze z dostupných podkladů
rozumně formulovat.

U každé změny použij tuto strukturu:

1. CO VÍME
   Uveď pouze skutečnosti přímo obsažené v podkladech.

2. CO SE PODLE TĚCHTO FAKTŮ MĚNÍ
   Vysvětli vlastní analytickou interpretaci.

3. PROČ BY TO MOHLO BÝT DŮLEŽITÉ
   Popiš možný ekonomický dopad.

4. CHARAKTER ZMĚNY
   Zvaž:
   - strukturální,
   - cyklickou,
   - dočasnou,
   - jednorázovou,
   - nebo kombinaci.
   
   Pokud to nelze z podkladů rozlišit, řekni to.

5. CO HOVOŘÍ PRO TUTO INTERPRETACI

6. CO HOVOŘÍ PROTI NÍ

7. CO BY JI V BUDOUCNU POTVRDILO NEBO VYVRÁTILO

8. CO NEVÍME
   Uveď důležité informace, které by byly potřeba,
   ale v současných podkladech nejsou.


NA ZÁVĚR

Vytvoř:

A) "Pracovní investiční příběh"

Maximálně 2–3 odstavce.

Popiš:
- co se podle dostupných informací ve firmě právě mění,
- proč je to ekonomicky důležité,
- jaká je hlavní pozitivní interpretace,
- jaká je hlavní negativní interpretace,
- kde je největší nejistota.

B) "Co bychom měli zjistit dál"

Uveď 5 konkrétních otázek / informací,
které by podle tebe nejvíce pomohly pracovní příběh potvrdit
nebo vyvrátit.

NEDÁVEJ:
- BUY / SELL,
- cílovou cenu,
- investiční skóre,
- celkové hodnocení akcie.


------------------------------------------------------------
DODANÉ PODKLADY
------------------------------------------------------------

[F1] DLOUHODOBÝ VÝVOJ

Za dostupné víceleté období:

- tržby: přibližně +24 %
- čistý zisk: přibližně +12 %
- FCF: přibližně -6 %

Tyto údaje jsou relativní změny za sledované období.


[F2] AKTUÁLNÍ TTM VÝVOJ

Ve srovnání s posledním uzavřeným rokem:

- TTM tržby: -25 %
- TTM čistý zisk: -16 %
- TTM FCF: -18 %

Tyto údaje znamenají relativní změnu.
Neznamenají, že příslušná absolutní hodnota je záporná.


[F3] TRŽNÍ VÝVOJ

Akcie:

- přibližně -17 % za posledních 12 měsíců
- přibližně -22 % za 3 roky


[F4] AKTUÁLNÍ PROBLÉM

V roce 2026 je uváděn tlak na výhled firmy
v souvislosti s čínským trhem.


[F5] POZITIVNÍ SIGNÁLY

Současně jsou k dispozici informace o:

- silnějších maržích,
- pokračujícím růstu v některých částech podnikání.


[F6] PODNIKATELSKÉ OBLASTI

Siemens Healthineers působí zejména v oblastech:

- Imaging
- Diagnostics
- Varian
- Advanced Therapies


------------------------------------------------------------
PRAVIDLA DŮVĚRYHODNOSTI
------------------------------------------------------------

U každého konkrétního faktického tvrzení si interně polož otázku:

"Je toto tvrzení skutečně obsaženo v podkladech?"

Pokud ANO:
můžeš ho použít jako fakt.

Pokud NE:
nesmíš ho prezentovat jako fakt.

Pokud jde o logický závěr z dostupných faktů:
označ ho jako "Inference".

Pokud nelze rozhodnout:
řekni "Z dostupných podkladů to nelze potvrdit."

Cílem není vytvořit co nejdelší text.
Cílem je vytvořit co nejpřesnější analytickou syntézu
s jasným oddělením FAKTŮ a INFERENCÍ.
"""


# ============================================================
# SPUŠTĚNÍ HLAVNÍHO TESTU
# ============================================================

if st.button(
    "▶ Spustit evidence-based SHL analýzu",
    type="primary"
):

    if not api_key:
        st.error(
            "Nejdříve musí být dostupný GROQ_API_KEY."
        )

    else:

        try:

            with st.spinner(
                "Groq vytváří evidence-based analýzu SHL..."
            ):

                elapsed, content, raw = groq_chat(
                    SHL_PROMPT,
                    """
Jsi seniorní equity analytik.

Piš česky.

Tvým hlavním úkolem je analytické myšlení,
nikoli produkce dlouhého textu.

Striktně rozlišuj:
1. ověřený fakt z dodaných podkladů,
2. analytickou inferenci,
3. informaci, kterou nelze z podkladů určit.

Nikdy nevymýšlej konkrétní fakta,
čísla, citace, události ani názory managementu.

Pokud něco nevíš, je správná odpověď:
"Z dostupných podkladů to nelze potvrdit."

Buď kritický.
Nesnaž se uživatele uklidňovat.
Nedávej doporučení BUY/SELL.
"""
                )

            st.success(
                f"✅ Analýza dokončena za {elapsed:.1f} s"
            )

            st.markdown("## Analytická odpověď Groq")

            st.markdown(content)

            with st.expander("Technické informace"):
                st.write(f"Model: `{MODEL}`")
                st.write(f"Čas odpovědi: {elapsed:.2f} s")
                st.write(
                    f"Délka odpovědi: {len(content)} znaků"
                )

        except Exception as e:

            st.error(
                f"❌ SHL analytický test selhal: {e}"
            )


st.divider()

st.caption(
    "Diagnostický test Groq. "
    "Hlavní streamlit_app.py se tímto testem nemění."
)
