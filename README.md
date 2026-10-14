====================================================

Talent-lens-Resume-Screening-Using-RAG

====================================================

Created by: Vallabh Kulkarni

OVERVIEW
----------------------------------------------------

TalentLens is an AI-powered resume screening tool that I built to help automate candidate evaluation. Instead of manually reviewing dozens of resumes, this app lets users upload a ZIP file of PDF resumes and paste a job description. 

It automatically parses the resumes, scores candidates using multiple signals (skills, vector similarity, experience, and education), and creates a ranked shortlist. It also includes a RAG (Retrieval-Augmented Generation) chat feature where you can ask direct questions about the candidate pool.

FEATURES

----------------------------------------------------

- Bulk Resume Processing: Unzips and extracts text from multiple PDF resumes at once.
- Multi-Signal Candidate Scoring: Evaluates candidate fit based on:
  * Skill matching (required vs. preferred skills)
  * Semantic match (OpenAI vector embeddings)
  * Total years of experience vs. job requirement

  * Highest education level
- Candidate Analytics: View candidate leaderboards, missing skills, individual summaries, and auto-generated interview questions.
- Ask the Resumes (RAG Chat): Hybrid search (Vector + BM25 keyword search) allowing natural language questions across all uploaded resumes.
- Power BI Export: Option to export evaluation data as CSV files formatted for Power BI dashboard reporting.

TECH STACK
----------------------------------------------------
- Language: Python 3.11+
- Web Framework: Streamlit
- LLM & Embeddings: OpenAI API (gpt-4o-mini, text-embedding-3-small)
- Orchestration / RAG: LangChain, ChromaDB, Rank-BM25
- Data Analysis & Charts: Pandas, Plotly

PROJECT STRUCTURE
----------------------------------------------------

- app.py            : Streamlit UI and app layout
- rag_core.py       : PDF parsing, Chroma vector database indexing, hybrid retriever, and RAG chat function
- scoring.py        : Structured Pydantic extraction models, skill matching, and scoring algorithms
- export.py         : Formats evaluation runs into star-schema CSVs for Power BI
- requirements.txt  : List of required Python packages
- DEPLOY.md         : Deployment notes
- POWERBI_GUIDE.md  : Instructions for building Power BI visual reports

HOW TO RUN LOCALLY
----------------------------------------------------
1. Clone this repository to your local machine.
2. Open terminal in the project directory and create a virtual environment:
   python -m venv .venv
3. Activate the virtual environment:
   - Windows: .venv\Scripts\activate
   - Mac/Linux: source .venv/bin/activate 
4. Install the required dependencies:
   pip install -r requirements.txt
5. Create a `.env` file in the main folder and add your OpenAI API key:
   OPENAI_API_KEY=sk-your-openai-api-key-here
6. Launch the Streamlit application:
   streamlit run app.py

ACKNOWLEDGEMENTS & PRIVACY
----------------------------------------------------
This project was created for educational/demonstration purposes. Resumes contain Personal Identifiable Information (PII), so sample data should be handled responsibly.
   


ACKNOWLEDGEMENTS & PRIVACY
----------------------------------------------------
This project was created for educational/demonstration purposes. Resumes contain Personal Identifiable Information (PII), so sample data should be handled responsibly.
