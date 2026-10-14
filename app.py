"""TalentLens - RAG resume screener. Run:  streamlit run app.py"""
# Chroma needs a newer sqlite than Streamlit Cloud ships; swap it in when available.
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import os
import tempfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from rag_core import answer_question, load_resumes, safe_unzip
from scoring import Settings, interview_questions, run_pipeline

APP_DIR = Path(__file__).resolve().parent
for _name in (".env", ".env.txt"):  # .env.txt = Notepad's accidental extension
    load_dotenv(APP_DIR / _name, override=True)
st.set_page_config(page_title="TalentLens", page_icon="🔎", layout="wide")
st.markdown("""<style>
.block-container{padding-top:1.5rem}
.hero{background:linear-gradient(90deg,#134e4a,#0f766e);color:#fff;padding:18px 24px;border-radius:12px;margin-bottom:12px}
.hero h1{margin:0;font-size:1.7rem}.hero p{margin:4px 0 0;opacity:.85}
.chip{display:inline-block;padding:2px 10px;margin:2px;border-radius:999px;font-size:.8rem}
.ok{background:#d1fae5;color:#065f46}.miss{background:#fee2e2;color:#991b1b}
</style>""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>🔎 TalentLens</h1><p>Upload resumes, paste a job description, '
            'get a ranked shortlist you can question.</p></div>', unsafe_allow_html=True)

# ---------------- fixed settings (no sidebar) ----------------
try:
    api_key = st.secrets.get("OPENAI_API_KEY")
except Exception:  # no secrets file locally
    api_key = None
api_key = api_key or os.getenv("OPENAI_API_KEY")
if api_key:
    api_key = api_key.strip().strip("\"'")
else:
    found = sorted(p.name for p in APP_DIR.iterdir() if p.name.startswith(".env"))
    st.error(f"OPENAI_API_KEY not found.\n\nThe app looked in: `{APP_DIR}`\n\n"
             f"Files starting with .env there: {found or 'none'}\n\n"
             "You need a file named exactly `.env` containing one line: OPENAI_API_KEY=sk-...")
    api_key = st.text_input("Temporary: paste your key here to continue (not saved)", type="password") or None
chat_model = "gpt-4.1-mini"
embed_model = "text-embedding-3-small"
w = {"Skills": 0.45, "Semantic match": 0.25, "Experience": 0.20, "Education": 0.10}
thr = 60

# ---------------- input ----------------
c1, c2 = st.columns([1, 2])
zip_file = c1.file_uploader("ZIP of resumes (PDF)", type="zip")
jd_text = c2.text_area("Job description", height=170, placeholder="Paste the full job description here…")
go_btn = st.button("🚀 Screen candidates", type="primary", disabled=not (zip_file and jd_text.strip() and api_key))

if go_btn:
    llm = ChatOpenAI(model=chat_model, api_key=api_key, temperature=0)
    emb = OpenAIEmbeddings(model=embed_model, api_key=api_key)
    cfg = Settings(w["Skills"], w["Semantic match"], w["Experience"], w["Education"], float(thr))
    bar, msg = st.progress(0.0), st.empty()
    with tempfile.TemporaryDirectory() as tmp:
        zp = Path(tmp) / "in.zip"
        zp.write_bytes(zip_file.getvalue())
        pdfs = safe_unzip(zp, Path(tmp) / "pdfs")
        texts, unreadable = load_resumes(pdfs)
    if not texts:
        st.error("No readable PDFs found in the ZIP.")
        st.stop()
    try:
        run = run_pipeline(texts, jd_text, llm, emb, cfg, lambda m, p: (msg.info(m), bar.progress(p)))
    except Exception as e:
        st.error(f"Screening failed: {e}")
        st.stop()
    run.update(unreadable=unreadable, llm=llm, jd=jd_text, thr=float(thr), chat=[])
    st.session_state["run"] = run
    msg.success(f"Screened {len(texts)} resumes" + (f" ({len(unreadable)} unreadable/scanned skipped)" if unreadable else ""))

run = st.session_state.get("run")
if not run:
    st.info("Upload a ZIP, paste a job description, then click **Screen candidates**.")
    st.stop()

df, sk, job = run["results"], run["skills"], run["job"]
t_over, t_board, t_deep, t_chat = st.tabs(
    ["📊 Overview", "🏆 Leaderboard", "🧑‍💼 Candidate deep-dive", "💬 Ask the resumes"])

# ---------------- overview ----------------
with t_over:
    k = st.columns(4)
    k[0].metric("Resumes screened", len(df))
    k[1].metric("Shortlisted", int(df.shortlisted.sum()), f"{df.shortlisted.mean():.0%} of pool")
    k[2].metric("Average fit", f"{df.fit_score.mean():.1f}")
    k[3].metric("Top candidate", df.iloc[0]["name"], f"{df.iloc[0]['fit_score']}")
    st.caption(f"Role: **{job.title}** · min experience: {job.min_years_experience:g} yrs")
    a, b = st.columns(2)
    cov = (sk.groupby(["skill", "skill_type"]).matched.mean().mul(100).reset_index()
             .sort_values("matched", ascending=False))
    a.plotly_chart(px.bar(cov, x="matched", y="skill", color="skill_type", orientation="h",
                          title="Skill coverage across the pool (%)",
                          labels={"matched": "% of resumes"}).update_layout(yaxis={"autorange": "reversed"}),
                   use_container_width=True)
    b.plotly_chart(px.histogram(df, x="fit_score", nbins=12, color="tier", title="Fit score distribution")
                   .add_vline(x=run["thr"], line_dash="dash"), use_container_width=True)
    gaps = cov[cov.skill_type == "required"].nsmallest(3, "matched")
    if len(gaps):
        st.warning("Hardest-to-find required skills: " + ", ".join(f"{r.skill} ({r.matched:.0f}%)" for r in gaps.itertuples()))
    if run["unreadable"]:
        with st.expander(f"{len(run['unreadable'])} files could not be read (likely scanned images)"):
            st.write(run["unreadable"])

# ---------------- leaderboard ----------------
with t_board:
    only = st.toggle("Shortlisted only", value=False)
    view = df[df.shortlisted] if only else df
    st.dataframe(view[["rank", "name", "tier", "fit_score", "skills_score", "semantic_score", "experience_score",
                       "years", "education_level", "current_role"]],
                 use_container_width=True, hide_index=True,
                 column_config={"fit_score": st.column_config.ProgressColumn("Fit", min_value=0, max_value=100, format="%.1f")})
    st.download_button("⬇ Download shortlist (CSV)", view.to_csv(index=False), "shortlist.csv", "text/csv")

# ---------------- deep dive ----------------
def chips(items, cls):
    return " ".join(f'<span class="chip {cls}">{i}</span>' for i in items) or "—"

with t_deep:
    picks = st.multiselect("Pick a candidate (or up to 3 to compare)", df.name.tolist(),
                           default=[df.name.iloc[0]], max_selections=3)
    labels = [("Skills", "skills_score"), ("Semantic", "semantic_score"),
              ("Experience", "experience_score"), ("Education", "education_score")]
    for nm in picks:
        r = df[df.name == nm].iloc[0]
        mine = sk[sk.file == r.file]
        st.subheader(f"{r['name']} · {r.fit_score} ({r.tier})")
        st.caption(f"{r.current_role or 'Role not stated'} · {r.years:g} yrs · {r.highest_education} · {r.email or 'no email'}")
        cols = st.columns(4)
        for col, (label, key) in zip(cols, labels):
            col.metric(label, f"{r[key]:.0f}")
            col.progress(min(max(float(r[key]) / 100, 0.0), 1.0))
        st.markdown("**Matched:** " + chips(mine[mine.matched == 1].skill, "ok"), unsafe_allow_html=True)
        missing = mine[mine.matched == 0].skill.tolist()
        st.markdown("**Missing:** " + chips(missing, "miss"), unsafe_allow_html=True)
        st.write(r.summary)
        if r.projects:
            st.markdown("**Projects:** " + r.projects.replace(" | ", " · "))
        if st.button("Generate interview questions", key=f"iq_{r.file}"):
            st.markdown(interview_questions(r["name"], run["profiles"][r.file],
                                            mine[(mine.matched == 0) & (mine.skill_type == "required")].skill.tolist(),
                                            job, run["llm"]))
        st.divider()

# ---------------- chat ----------------
with t_chat:
    st.caption(" ")
    for m in run["chat"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("sources"):
                st.caption("Sources: " + ", ".join(m["sources"]))
    q = st.chat_input("e.g. Who has built a RAG project? What is Vamshi's email?")
    if q:
        run["chat"].append({"role": "user", "content": q})
        reply, docs = answer_question(q, run["retriever"], run["llm"])
        run["chat"].append({"role": "assistant", "content": reply,
                            "sources": sorted({d.metadata["source"] for d in docs})})
        st.rerun()
