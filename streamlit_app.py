        st.write(f"**Posouzení zotavení:** {r.get("Posouzení zotavení","—")} · skóre {safe_float(r.get("Skóre zotavení")):.0f}/100" if not pd.isna(safe_float(r.get("Skóre zotavení"))) else "**Posouzení zotavení:** —")
        st.write(f"**Brány:** {r.get("Posouzení zotavení","—")}")
        st.dataframe(pd.DataFrame([{
            "Směr fundamentů": r["Fundamental Direction"], "Fundamentální trend score": r["Fundamental Trend Score"],
            "Revenue Growth %": r["Revenue Growth"], "Earnings Growth %": r["Earnings Growth"],
            "Revenue CAGR 3Y %": r["Revenue CAGR 3Y"], "Net Income CAGR 3Y %": r["Net Income CAGR 3Y"],
            "Margin Change 3Y %": r["Margin Change 3Y"], "Revenue Prior YoY %": r["Revenue Prior YoY"],
            "Net Income Prior YoY %": r["Net Income Prior YoY"]
        }]), use_container_width=True, hide_index=True)
    with st.expander("📉 Chování ceny", expanded=False):
        pc1, pc2, pc3, pc4 = st.columns(4)
        ps = safe_float(r.get("Skóre ceny")); dd = safe_float(r.get("Drawdown 3Y")); r12 = safe_float(r.get("12M Return")); ma = safe_float(r.get("MA50 vs MA200"))
        pc1.metric("Skóre ceny", f"{ps:.0f}" if not pd.isna(ps) else "—")
        pc2.metric("3Y propad", f"{dd:.1f}%" if not pd.isna(dd) else "—")
        pc3.metric("12M výnos", f"{r12:.1f}%" if not pd.isna(r12) else "—")
        pc4.metric("MA50 vs MA200", f"{ma:.1f}%" if not pd.isna(ma) else "—")
        st.write(f"**Struktura ceny:** vyšší minima = {r.get('Higher Low','—')}; vyšší maxima = {r.get('Higher High','—')}" )
        st.write(f"**Cenový obraz:** {r.get('Price View', '—')}")
        st.write(f"**Fundamenty vs. cena:** {r.get('Market / Fundamental View', '—')}")
        st.write(f"**Co naznačuje cena:** {r.get('Price Evidence', '') or '—'}")
    with st.expander("📰 Textové důkazy k příběhu", expanded=False):
        st.write(f"**Doložení příběhu:** {evidence_quality(r)}")
        if clean_text(r.get("Text Evidence")):
            st.write(f"**Textový závěr:** {normalize_text_evidence_output(r.get('Text Evidence'))}")
        if clean_text(r.get("Text Support")):
            st.markdown(r["Text Support"])
        if clean_text(r.get("Text Warnings")):
            st.markdown("**Rizika / co může příběh zpochybnit**")
            st.markdown(r["Text Warnings"])
        elif clean_text(r.get("Text Evidence")) in ("⚪ Bez textového důkazu", "⚪ Zatím nedoloženo"):
            st.info("Textová vrstva zatím nenašla dostatečně konkrétní firemní signál. To není potvrzení ani vyvrácení příběhu.")
        if clean_text(r.get("Text Sources")):
            st.caption("Zdroj textu: " + r["Text Sources"])

    with st.expander("🔬 Kompletní technický záznam", expanded=False):
        detail_cols = [
            "Ticker","Yahoo Ticker","Name","Exchange","Company Archetype","Company Type","Sector","Industry",
            *PARAMS,"Revenue CAGR 3Y","Net Income CAGR 3Y","Net Margin","Margin Change 3Y",
            "Revenue Prior YoY","Net Income Prior YoY","Fundamental Direction","Fundamental Trend Score","Fundamental Evidence","Posouzení zotavení","Skóre zotavení","Recovery Gates","Turnaround Score","Turnaround Evidence",
            "Story Priority","Text Score","Final Confidence","Status","Mapping","Data Source","Error"
        ]
        detail_cols = [c for c in detail_cols if c in r.index]
        st.dataframe(pd.DataFrame([r[detail_cols].to_dict()]), use_container_width=True, hide_index=True)


# Text evidence detail
with st.expander("📰 Textové důkazy k příběhu", expanded=False):
    text_cols = ["Ticker", "Name", "Story", "Story Priority", "Text Score", "Text Evidence", "Doložení příběhu", "Final Confidence", "Text Support", "Text Warnings", "Text Sources"]
    if "Text Score" in results_df.columns:
        text_view = results_df[(results_df["Text Evidence"] != "⚪ Nehodnoceno") & results_df["Text Score"].notna()].copy()
        if text_view.empty:
            st.info("Textová fáze zatím nebyla provedena nebo pro kandidáty nebyl dostupný text.")
        else:
            text_view["Doložení příběhu"] = text_view.apply(evidence_quality, axis=1)
            text_cols = [c for c in text_cols if c in text_view.columns]
            st.dataframe(text_view[text_cols], use_container_width=True, hide_index=True, height=700)

# Availability
st.markdown("### 📊 Dostupnost fundamentálních parametrů")
availability = []
for p in PARAMS:
    n = int(results_df[p].notna().sum())
    availability.append({
        "Parametr": p,
        "Dostupných hodnot": n,
        "Celkem": len(results_df),
        "Dostupnost %": round(n / len(results_df) * 100, 1)
    })
availability_df = pd.DataFrame(availability)
st.dataframe(availability_df, use_container_width=True, hide_index=True)

# Exchange quality
st.markdown("### 🏛️ Kvalita dat podle burzy")
exchange_rows = []
for ex, g in results_df.groupby("Exchange"):
    complete = int((g["Status"] == "OK").sum())
    at_least_half = int((g[PARAMS].notna().sum(axis=1) >= len(PARAMS)/2).sum())
    at_least_one = int((g[PARAMS].notna().sum(axis=1) >= 1).sum())
    avg_avail = round(g[PARAMS].notna().mean().mean() * 100, 1)
    exchange_rows.append({
        "Burza": ex,
        "Testováno": len(g),
        "Kompletní data": complete,
        "≥ 50 % parametrů": at_least_half,
        "≥ 1 parametr": at_least_one,
        "Průměrná dostupnost %": avg_avail
    })
st.dataframe(pd.DataFrame(exchange_rows), use_container_width=True, hide_index=True)

# Diagnostics
with st.expander("🔍 Detail všech načtených titulů", expanded=False):
    detail_cols = [
        "Ticker", "Yahoo Ticker", "Name", "Exchange",
        *PARAMS, "Revenue CAGR 3Y", "Net Income CAGR 3Y", "Net Margin", "Margin Change 3Y", "Revenue Prior YoY", "Net Income Prior YoY", "Value Score", "Quality Score", "Growth Score", "Turnaround Score", "Company Type", "Story", "Turnaround Evidence", "Text Score", "Text Evidence", "Skóre ceny", "Price View", "Drawdown 3Y", "Recovery from 3Y Low", "6M Return", "12M Return", "MA50 vs MA200", "Higher Low", "Higher High", "Market / Fundamental View", "Final Confidence", "Available Params", "Status", "Mapping", "Data Source", "Error"
    ]
    st.dataframe(
        results_df[detail_cols],
        use_container_width=True,
        hide_index=True,
        height=800
    )

with st.expander("⚠️ Tituly s neúplnými nebo chybějícími daty", expanded=False):
    bad = results_df[results_df["Status"] != "OK"].copy()
    if bad.empty:
        st.success("Všechny načtené tituly mají kompletní sadu parametrů.")
    else:
        st.dataframe(
            bad[["Ticker","Yahoo Ticker","Name","Exchange","Status","Mapping","Data Source","Error",*PARAMS]],
            use_container_width=True,
            hide_index=True,
            height=700
        )

with st.expander("🧭 Kontrola XETRA mappingu", expanded=False):
    x = results_df[results_df["Exchange"] == "XETRA"].copy()
    if x.empty:
        st.info("V tomto běhu nebyl testován XETRA.")
    else:
        st.dataframe(
            x[["Ticker","Yahoo Ticker","Name","Mapping","Status","Market Cap","P/E"]],
            use_container_width=True,
            hide_index=True
        )

st.info(
    "V6 pracuje v navazujících vrstvách: (1) cenový předvýběr celého univerza v dávkách, (2) fundamentální fáze, (3) charakter firmy, (4) směr fundamentů, (5) recovery gates, (6) textová evidence, (7) chování ceny, (8) investiční příběh. True turnaround vyžaduje předchozí problém i současné zlepšení. "
    "Textová vrstva příběh nepotvrzuje automaticky; je to evidence mechanismus. Yahoo/Google text je nejprve filtrován na relevanci ke konkrétní firmě a teprve potom se hledají podpůrné a varovné signály. "
    "Charakter firmy a asset-based společnosti jsou posuzovány odděleně, aby se na ně "
    "mechanicky nepřenášela logika běžné provozní firmy. Chybějící hodnoty se nepřevádějí na nulu."
)

st.caption("Zdroje: Nasdaq Trader, Deutsche Börse Xetra a Yahoo Finance. Data jsou získávána při běhu aplikace a mohou být zpožděná či nedostupná.")
