"""RAG building blocks: safe unzip -> PDF load -> chunk -> embed -> hybrid retrieve -> grounded answer."""
from __future__ import annotations

import zipfile
from pathlib import Path

from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def safe_unzip(zip_path: Path, dest: Path) -> list[Path]:
    """Extract only PDFs, flattening paths (blocks zip-slip and macOS junk)."""
    dest.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = Path(info.filename)
            if info.is_dir() or name.suffix.lower() != ".pdf" or "__MACOSX" in info.filename:
                continue
            target = dest / name.name
            with zf.open(info) as src, open(target, "wb") as dst:
                dst.write(src.read())
            out.append(target)
    return out


def load_resumes(pdf_paths: list[Path]) -> tuple[dict[str, str], list[str]]:
    """Return ({filename: full_text}, [unreadable filenames]). Scanned PDFs with no text layer are reported."""
    texts: dict[str, str] = {}
    unreadable: list[str] = []
    for p in pdf_paths:
        try:
            pages = PyPDFLoader(str(p)).load()
            text = "\n".join(pg.page_content for pg in pages).strip()
        except Exception:
            text = ""
        if len(text) < 80:
            unreadable.append(p.name)
        else:
            texts[p.name] = text
    return texts, unreadable


def build_index(texts: dict[str, str], embeddings, run_id: str, chunk_size=800, chunk_overlap=120):
    docs = [Document(page_content=t, metadata={"source": f}) for f, t in texts.items()]
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""], chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    store = Chroma(collection_name=f"run_{run_id}", embedding_function=embeddings)
    store.add_documents(chunks)
    return store, chunks


def build_retriever(store: Chroma, chunks: list[Document], k: int = 6):
    """Hybrid = MMR vector search (meaning) + BM25 (exact keywords like 'Power BI', 'FastAPI')."""
    dense = store.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": k * 4, "lambda_mult": 0.7})
    sparse = BM25Retriever.from_documents(chunks)
    sparse.k = k
    return EnsembleRetriever(retrievers=[dense, sparse], weights=[0.6, 0.4])


QA_PROMPT = """You are TalentLens, a recruiting assistant. Resume excerpts below are your ONLY source.

Rules:
- Use only the excerpts. Never guess or invent details.
- Name every candidate you refer to (use the resume filename if the name is unclear).
- If several candidates match, list them as bullets with one line of evidence each.
- If the excerpts don't contain the answer, reply exactly: "I couldn't find that in the uploaded resumes."
- Be concise.

Excerpts:
{context}

Question: {question}

Answer:"""


def answer_question(question: str, retriever, llm) -> tuple[str, list[Document]]:
    docs = retriever.invoke(question)
    context = "\n\n".join(f"[{d.metadata.get('source')}]\n{d.page_content}" for d in docs)
    reply = llm.invoke(QA_PROMPT.format(context=context, question=question))
    return reply.content, docs
