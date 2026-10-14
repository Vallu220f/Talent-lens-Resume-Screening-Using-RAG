"""Structured extraction + multi-signal Fit Score (not a single opaque 'ATS' number)."""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from rag_core import build_index, build_retriever


class JobProfile(BaseModel):
    title: str = Field(description="Job title, or 'Unspecified'")
    required_skills: list[str] = Field(description="Must-have skills/tools, short canonical names")
    preferred_skills: list[str] = Field(description="Nice-to-have skills/tools")
    min_years_experience: float = Field(description="Minimum years of experience, 0 if not stated")


class ResumeProfile(BaseModel):
    name: str = Field(description="Candidate full name, or 'Unknown'")
    email: Optional[str] = None
    current_role: Optional[str] = None
    skills: list[str] = Field(description="Technical skills and tools explicitly listed or used")
    total_years_experience: float = Field(description="Professional experience in years; internships count half; 0 for freshers")
    education_level: Literal["PhD", "Masters", "Bachelors", "Diploma", "Other", "Unknown"]
    highest_education: str = Field(description="Degree + institution, short")
    top_projects: list[str] = Field(description="Up to 3 notable projects, one line each")
    summary: str = Field(description="Two-sentence neutral summary of the candidate")


EDU_POINTS = {"PhD": 100, "Masters": 90, "Bachelors": 75, "Diploma": 50, "Other": 40, "Unknown": 30}


@dataclass
class Settings:
    w_skills: float = 0.45
    w_semantic: float = 0.25
    w_experience: float = 0.20
    w_education: float = 0.10
    shortlist_threshold: float = 60.0
    preferred_weight: float = 0.5  # preferred skills count half as much as required


def _has_skill(skill: str, text_lower: str, skill_set: set[str]) -> bool:
    s = skill.lower().strip()
    if s in skill_set:
        return True
    return re.search(r"(?<![a-z0-9+#])" + re.escape(s) + r"(?![a-z0-9+#])", text_lower) is not None


def tier(score: float) -> str:
    return "Strong fit" if score >= 75 else "Good fit" if score >= 60 else "Borderline" if score >= 45 else "Weak fit"


def parse_job(jd_text: str, llm) -> JobProfile:
    prompt = ("Extract a structured profile from this job description. Use short canonical skill names "
              "(e.g. 'Python', 'Power BI', 'SQL').\n\n" + jd_text)
    return llm.with_structured_output(JobProfile).invoke(prompt)


def parse_resumes(texts: dict[str, str], llm, progress=None) -> dict[str, ResumeProfile]:
    structured = llm.with_structured_output(ResumeProfile)
    names = list(texts)
    prompts = ["Extract a structured profile from this resume. Do not invent anything.\n\n" + texts[n][:9000]
               for n in names]
    out: dict[str, ResumeProfile] = {}
    for i in range(0, len(names), 10):  # small batches so the UI can show progress
        batch = structured.batch(prompts[i:i + 10], config={"max_concurrency": 8})
        for n, prof in zip(names[i:i + 10], batch):
            out[n] = prof
        if progress:
            progress(min(i + 10, len(names)), len(names))
    return out


def semantic_similarity(jd_text: str, texts: dict[str, str], embeddings) -> dict[str, float]:
    names = list(texts)
    jd_vec = np.array(embeddings.embed_query(jd_text))
    vecs = np.array(embeddings.embed_documents([texts[n][:12000] for n in names]))
    sims = vecs @ jd_vec / (np.linalg.norm(vecs, axis=1) * np.linalg.norm(jd_vec) + 1e-9)
    return dict(zip(names, sims.tolist()))


def score_candidates(job: JobProfile, profiles: dict[str, ResumeProfile], texts: dict[str, str],
                     sims: dict[str, float], cfg: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    lo, hi = min(sims.values()), max(sims.values())
    rows, skill_rows = [], []
    total_w = cfg.w_skills + cfg.w_semantic + cfg.w_experience + cfg.w_education
    for fname, p in profiles.items():
        low = texts[fname].lower()
        have = {s.lower() for s in p.skills}
        earned = possible = 0.0
        for kind, skills, w in (("required", job.required_skills, 1.0),
                                ("preferred", job.preferred_skills, cfg.preferred_weight)):
            for s in skills:
                hit = _has_skill(s, low, have)
                possible += w
                earned += w if hit else 0
                skill_rows.append({"file": fname, "skill": s, "skill_type": kind, "matched": int(hit)})
        skills_pts = 100 * earned / possible if possible else 50.0
        sem_pts = 100 * (sims[fname] - lo) / (hi - lo) if hi > lo else 50.0
        exp_pts = 100.0 if job.min_years_experience <= 0 else min(p.total_years_experience / job.min_years_experience, 1) * 100
        edu_pts = EDU_POINTS[p.education_level]
        fit = (cfg.w_skills * skills_pts + cfg.w_semantic * sem_pts +
               cfg.w_experience * exp_pts + cfg.w_education * edu_pts) / total_w
        rows.append({"file": fname, "name": p.name if p.name != "Unknown" else fname.rsplit(".", 1)[0],
                     "email": p.email, "current_role": p.current_role, "years": p.total_years_experience,
                     "education_level": p.education_level, "highest_education": p.highest_education,
                     "summary": p.summary, "projects": " | ".join(p.top_projects),
                     "skills_score": round(skills_pts, 1), "semantic_score": round(sem_pts, 1),
                     "experience_score": round(exp_pts, 1), "education_score": edu_pts,
                     "fit_score": round(fit, 1)})
    df = pd.DataFrame(rows).sort_values("fit_score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    df["tier"] = df["fit_score"].map(tier)
    df["shortlisted"] = df["fit_score"] >= cfg.shortlist_threshold
    skills_df = pd.DataFrame(skill_rows)
    return df, skills_df


def run_pipeline(texts, jd_text, llm, embeddings, cfg: Settings, progress=None):
    """Full run. Returns dict with job, results, skills, retriever, run_id."""
    run_id = uuid.uuid4().hex[:8]
    say = progress or (lambda *a: None)
    say("Reading the job description…", 0.05)
    job = parse_job(jd_text, llm)
    say("Indexing resumes for search (RAG)…", 0.15)
    store, chunks = build_index(texts, embeddings, run_id)
    retriever = build_retriever(store, chunks)
    say("Extracting structured profiles…", 0.35)
    profiles = parse_resumes(texts, llm, lambda d, t: say(f"Profiles {d}/{t}", 0.35 + 0.4 * d / t))
    say("Scoring…", 0.85)
    sims = semantic_similarity(jd_text, texts, embeddings)
    df, skills_df = score_candidates(job, profiles, texts, sims, cfg)
    say("Done", 1.0)
    return {"run_id": run_id, "job": job, "results": df, "skills": skills_df,
            "retriever": retriever, "texts": texts, "profiles": profiles}


def interview_questions(name: str, profile: ResumeProfile, missing: list[str], job: JobProfile, llm) -> str:
    prompt = (f"Write 6 interview questions for {name} applying to '{job.title}'. "
              f"Base them ONLY on this profile and probe the gaps.\n"
              f"Projects: {profile.top_projects}\nSkills: {profile.skills}\nMissing required skills: {missing}\n"
              "Mix: 3 about their real projects, 3 about the missing skills. Number them.")
    return llm.invoke(prompt).content
