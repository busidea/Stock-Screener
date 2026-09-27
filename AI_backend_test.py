import json, time, requests, streamlit as st

st.set_page_config(page_title="AI Backend Test", layout="wide")
st.title("🧪 AI Backend Test")
st.caption("Samostatná diagnostika AI připojení pro Stock-Screener.")

PROMPT = """Jsi seniorní akciový analytik. Analyzuj Siemens Healthineers (SHL.DE).

Nechci seznam článků ani obecný profil firmy. Chci vlastní analytickou syntézu toho,
co se ve společnosti skutečně mění.

Identifikuj 3 nejdůležitější probíhající změny. U každé vysvětli:
co se změnilo → proč → ekonomický dopad → zda jde o strukturální, cyklickou
nebo dočasnou změnu. Propoj informace s finančním vývojem, uveď protiargument
a řekni, co by hypotézu potvrdilo nebo vyvrátilo. Na závěr formuluj jeden
pracovní investiční příběh bez doporučení koupit/prodat.

Podklady:
Tržby za víceleté období +24 %, čistý zisk +12 %, FCF -6 %.
TTM tržby proti poslednímu roku -25 %, TTM čistý zisk -16 %, TTM FCF -18 %.
2026: tlak na výhled kvůli čínskému trhu; současně zprávy o silnějších maržích
a pokračujícím růstu v některých částech podnikání. Firma působí v Imaging,
Diagnostics, Varian a Advanced Therapies. Akcie cca -17 % za 12 měsíců a -22 %
za 3 roky."""

ENDPOINTS = [
    ("G4F / Groq", "https://g4f.space/api/groq"),
    ("G4F / Gemini", "https://g4f.space/api/gemini"),
    ("G4F / NVIDIA", "https://g4f.space/api/nvidia"),
    ("G4F / Pollinations", "https://g4f.space/api/pollinations"),
]

def models(base):
    r = requests.get(base + "/models", timeout=15,
                     headers={"User-Agent":"Stock-Screener-AI-Test/1.0"})
    r.raise_for_status()
    d = r.json()
    raw = d.get("data", d.get("models", [])) if isinstance(d, dict) else d
    return [x if isinstance(x,str) else x.get("id") for x in raw if (isinstance(x,str) or x.get("id"))]

def chat(base, model):
    t=time.time()
    r=requests.post(base+"/chat/completions", timeout=90, json={
        "model":model,
        "messages":[
            {"role":"system","content":"You are a rigorous senior equity analyst. Answer in Czech."},
            {"role":"user","content":PROMPT}
        ],
        "temperature":0.2
    }, headers={"Content-Type":"application/json","User-Agent":"Stock-Screener-AI-Test/1.0"})
    elapsed=time.time()-t
    if not r.ok: raise RuntimeError(f"HTTP {r.status_code}: {r.text[:2500]}")
    d=r.json()
    return elapsed, d["choices"][0]["message"]["content"], d

if st.button("🔎 Otestovat endpointy", type="primary"):
    found=[]
    for name,base in ENDPOINTS:
        try:
            ms=models(base)
            st.success(f"{name}: /models OK — {len(ms)} modelů")
            st.code("\n".join(ms[:25]) or "(seznam modelů prázdný)")
            found += [(name,base,m) for m in ms[:25]]
        except Exception as e:
            st.error(f"{name}: {e}")
    st.session_state["found"] = found

if "found" in st.session_state and st.session_state["found"]:
    st.divider()
    labels=[f"{n} · {m}" for n,b,m in st.session_state["found"]]
    choice=st.selectbox("Endpoint + model", labels)
    n,b,m=st.session_state["found"][labels.index(choice)]
    if st.button("▶️ Spustit skutečnou AI odpověď"):
        try:
            elapsed,content,raw=chat(b,m)
            st.success(f"HTTP 200 · {elapsed:.1f} s · {len(content)} znaků")
            st.markdown("### Skutečná odpověď AI")
            st.markdown(content)
            with st.expander("Technická odpověď"):
                st.json(raw)
        except Exception as e:
            st.error(f"AI test selhal: {e}")
else:
    st.info("Nejdřív spusť diagnostiku endpointů.")
