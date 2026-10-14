"""Write a Power BI-ready star schema. Each run is APPENDED, so Power BI can compare many job openings."""
from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

import pandas as pd


def _append(path: Path, new: pd.DataFrame, key: list[str]):
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old, new], ignore_index=True).drop_duplicates(subset=key, keep="last")
    new.to_csv(path, index=False)


def export_for_powerbi(run: dict, jd_text: str, threshold: float, out_dir: str = "powerbi_data") -> Path:
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    df, sk, job, run_id = run["results"].copy(), run["skills"].copy(), run["job"], run["run_id"]
    cid = lambda f: hashlib.md5(f.encode()).hexdigest()[:10]
    df["candidate_id"] = df["file"].map(cid)
    sk["candidate_id"] = sk["file"].map(cid)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    dim_c = df[["candidate_id", "name", "email", "file", "current_role", "years",
                "education_level", "highest_education", "summary", "projects"]]
    fact = df[["candidate_id", "rank", "fit_score", "skills_score", "semantic_score",
               "experience_score", "education_score", "tier", "shortlisted"]].copy()
    fact.insert(0, "run_id", run_id)
    fact["shortlisted"] = fact["shortlisted"].astype(bool)
    skills = sk[["candidate_id", "skill", "skill_type", "matched"]].copy()
    skills.insert(0, "run_id", run_id)
    dim_j = pd.DataFrame([{"run_id": run_id, "job_title": job.title, "min_years": job.min_years_experience,
                           "required_skill_count": len(job.required_skills),
                           "threshold": threshold, "screened_at": stamp, "jd_text": jd_text[:2000]}])

    _append(out / "dim_candidate.csv", dim_c, ["candidate_id"])
    _append(out / "dim_job.csv", dim_j, ["run_id"])
    _append(out / "fact_scores.csv", fact, ["run_id", "candidate_id"])
    _append(out / "fact_skill_match.csv", skills, ["run_id", "candidate_id", "skill"])
    return out
