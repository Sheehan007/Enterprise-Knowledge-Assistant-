from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from openai import OpenAI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("enterprise-knowledge-assistant")


def resolve_docs_path(raw_path: Optional[str]) -> Path:
    base_dir = Path(__file__).resolve().parent
    if not raw_path:
        return (base_dir / "data" / "sample_docs.txt").resolve()

    path = Path(raw_path)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


@dataclass(frozen=True)
class Settings:
    embed_model_name: str = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
    docs_file: Path = resolve_docs_path(os.getenv("DOCS_FILE"))
    default_top_k: int = int(os.getenv("TOP_K", "3"))
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")


settings = Settings()


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="User question")
    top_k: Optional[int] = Field(None, ge=1, le=10, description="How many docs to retrieve")


class RetrievedDoc(BaseModel):
    rank: int
    score: float
    document: str


class AskResponse(BaseModel):
    question: str
    retrieved_docs: List[RetrievedDoc]
    answer: str
    model_used: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    documents_loaded: int
    generation_enabled: bool


@dataclass
class RAGService:
    settings: Settings
    embed_model: SentenceTransformer
    documents: List[str]
    index: faiss.Index
    client: Optional[OpenAI]

    @classmethod
    def create(cls, settings: Settings) -> "RAGService":
        logger.info("Loading embedding model: %s", settings.embed_model_name)
        embed_model = SentenceTransformer(settings.embed_model_name)

        logger.info("Loading documents from: %s", settings.docs_file)
        documents = cls.load_documents(settings.docs_file)
        if not documents:
            raise RuntimeError("No documents found. Check DOCS_FILE content.")

        logger.info("Building FAISS index for %d documents", len(documents))
        index = cls.build_vector_store(embed_model, documents)

        client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
        return cls(
            settings=settings,
            embed_model=embed_model,
            documents=documents,
            index=index,
            client=client,
        )

    @staticmethod
    def load_documents(file_path: Path) -> List[str]:
        if not file_path.exists():
            raise FileNotFoundError(f"Document file not found: {file_path}")

        raw_text = file_path.read_text(encoding="utf-8").strip()
        chunks = [chunk.strip() for chunk in raw_text.split("\n\n") if chunk.strip()]
        return chunks

    @staticmethod
    def build_vector_store(embed_model: SentenceTransformer, docs: List[str]) -> faiss.Index:
        embeddings = embed_model.encode(
            docs,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")

        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]

        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        return index

    def retrieve(self, question: str, top_k: int) -> List[RetrievedDoc]:
        if not self.documents:
            raise RuntimeError("Document store is empty.")

        top_k = max(1, min(top_k, len(self.documents)))

        query_embedding = self.embed_model.encode(
            [question],
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)

        results: List[RetrievedDoc] = []
        for rank, doc_idx in enumerate(indices[0], start=1):
            if doc_idx < 0:
                continue
            results.append(
                RetrievedDoc(
                    rank=rank,
                    score=float(scores[0][rank - 1]),
                    document=self.documents[doc_idx],
                )
            )
        return results

    @staticmethod
    def build_prompt(question: str, retrieved_docs: List[RetrievedDoc]) -> str:
        context = "\n\n".join(
            f"[Document {doc.rank} | score={doc.score:.4f}]\n{doc.document}"
            for doc in retrieved_docs
        )

        return f"""
You are a helpful enterprise knowledge assistant.

Rules:
1. Answer only from the retrieved context below.
2. If the answer is missing, reply exactly with:
I could not find that in the provided knowledge base.
3. Be concise and factual.

Retrieved Context:
{context}

User Question:
{question}
""".strip()

    def generate_answer(self, prompt: str, retrieved_docs: List[RetrievedDoc]) -> tuple[str, Optional[str]]:
        if self.client is None:
            fallback = (
                "OPENAI_API_KEY is not configured. Retrieval worked, but generation is disabled.\n\n"
                f"Top retrieved context:\n{retrieved_docs[0].document if retrieved_docs else 'No context found.'}"
            )
            return fallback, None

        try:
            response = self.client.responses.create(
                model=self.settings.openai_model,
                instructions="You are a concise enterprise AI assistant.",
                input=prompt,
                temperature=0.2,
            )

            answer = (response.output_text or "").strip()
            if not answer:
                answer = "I could not find that in the provided knowledge base."

            return answer, self.settings.openai_model

        except Exception as exc:
            logger.exception("OpenAI generation failed")
            raise HTTPException(
                status_code=502,
                detail=f"Text generation failed: {exc}",
            ) from exc


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag_service = RAGService.create(settings)
    logger.info("Application startup complete")
    yield
    logger.info("Application shutdown complete")


app = FastAPI(
    title="Enterprise Knowledge Assistant",
    version="2.0.0",
    lifespan=lifespan,
)


def get_rag_service(request: Request) -> RAGService:
    service = getattr(request.app.state, "rag_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="RAG service not initialized")
    return service


@app.get("/", response_model=dict)
def root():
    return {"message": "Enterprise Knowledge Assistant is running."}


@app.get("/health", response_model=HealthResponse)
def health(service: RAGService = Depends(get_rag_service)):
    return HealthResponse(
        status="ok",
        documents_loaded=len(service.documents),
        generation_enabled=service.client is not None,
    )


@app.post("/ask", response_model=AskResponse)
def ask_question(payload: QueryRequest, service: RAGService = Depends(get_rag_service)):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    top_k = payload.top_k or service.settings.default_top_k
    retrieved_docs = service.retrieve(question, top_k=top_k)
    prompt = service.build_prompt(question, retrieved_docs)
    answer, model_used = service.generate_answer(prompt, retrieved_docs)

    return AskResponse(
        question=question,
        retrieved_docs=retrieved_docs,
        answer=answer,
        model_used=model_used,
    )
