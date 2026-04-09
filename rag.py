from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr


INTENT_KEYWORDS = {
    "loan_support": {"loan", "credit", "bank", "finance", "borrow", "crop"},
    "scholarship_support": {"scholarship", "student", "education", "college", "school", "academic"},
    "housing_support": {"housing", "house", "home", "rural", "awas", "pucca", "shelter"},
    "pension_support": {"pension", "elderly", "senior", "old age", "retirement"},
    "farmer_support": {"farmer", "kisan", "agriculture", "crop"},
}


class RAGAgent:
    def __init__(
        self,
        docs_dir: str | None = None,
        persist_dir: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)
        self.docs_dir = Path(docs_dir or os.getenv("SCHEME_DOCS_DIR", "docs"))
        self.persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "data/chroma")
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME", "jan_sahayak_schemes"
        )
        self.embedding_provider = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "models/embedding-001")
        self.local_embedding_model = os.getenv(
            "LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self._last_index_status: Dict[str, Any] = {}
        self._source_docs: List[Document] | None = None

        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=380,
            chunk_overlap=50,
        )

        self.embeddings = None
        if self.embedding_provider == "gemini" and self.gemini_api_key:
            try:
                self.embeddings = GoogleGenerativeAIEmbeddings(
                    model=self.embedding_model,
                    google_api_key=SecretStr(self.gemini_api_key),
                )
            except Exception:
                self.embeddings = None

        if self.embeddings is None:
            try:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=self.local_embedding_model,
                    model_kwargs={"local_files_only": True},
                )
                self.embedding_provider = "local"
            except Exception:
                self.embeddings = None

        self.available = self.embeddings is not None
        self.vectorstore = self._build_vectorstore()

    def _serialize_tags(self, tags: List[str]) -> str:
        cleaned = [tag.strip() for tag in tags if tag and tag.strip()]
        return ", ".join(cleaned) if cleaned else "general_support"

    def _build_vectorstore(self) -> Chroma:
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir,
        )

    def _reset_vectorstore(self) -> None:
        if self.vectorstore is not None:
            try:
                self.vectorstore.delete_collection()
            except Exception:
                pass
        self.vectorstore = self._build_vectorstore()

    def _load_documents(self) -> List[Document]:
        if self._source_docs is not None:
            return self._source_docs
        if not self.docs_dir.exists():
            self._source_docs = []
            return self._source_docs

        docs: List[Document] = []

        def add_markdown_sections(path: Path, content: str) -> int:
            lines = content.splitlines()
            current_heading = ""
            current_block: List[str] = []
            section_count = 0

            def flush() -> None:
                nonlocal section_count
                block_text = "\n".join(current_block).strip()
                if not block_text:
                    return
                docs.append(
                    Document(
                        page_content=block_text,
                        metadata={
                            "source": str(path),
                            "domain": "local_official_docs",
                            "trusted": True,
                            "scheme_name": current_heading or path.stem,
                            "tags": self._serialize_tags(
                                self._derive_tags(current_heading or path.stem, block_text)
                            ),
                        },
                    )
                )
                section_count += 1

            for raw_line in lines:
                line = raw_line.strip()
                heading = ""
                if line.startswith("### "):
                    heading = line[4:].strip().strip("*")
                elif line.startswith("## "):
                    heading = line[3:].strip().strip("*")

                if heading and "part " not in heading.lower() and "sample welfare" not in heading.lower():
                    flush()
                    current_heading = heading
                    current_block = []
                    continue

                current_block.append(raw_line)

            flush()
            return section_count

        for path in self.docs_dir.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".txt", ".md"}:
                continue
            if path.name.lower().startswith("readme"):
                continue
            content = path.read_text(encoding="utf-8", errors="ignore").strip()
            if not content:
                continue

            if path.suffix.lower() == ".md":
                created = add_markdown_sections(path, content)
                if created > 0:
                    continue

            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": str(path),
                        "domain": "local_official_docs",
                        "trusted": True,
                        "scheme_name": path.stem,
                        "tags": self._serialize_tags(self._derive_tags(path.stem, content)),
                    },
                )
            )
        self._source_docs = docs
        return self._source_docs

    def _derive_tags(self, scheme_name: str, text: str) -> List[str]:
        lowered = text.lower()
        scheme_type_match = re.search(r"scheme type:\s*(.+)", lowered)
        signal_match = re.search(r"good match signals:\s*(.+)", lowered)

        candidates = [scheme_name.lower()]
        if scheme_type_match:
            candidates.append(scheme_type_match.group(1))
        if signal_match:
            candidates.append(signal_match.group(1))

        haystack = " ".join(candidates)
        tags: List[str] = []

        explicit_types = [
            tag.strip()
            for tag in re.split(r"[,/]", scheme_type_match.group(1))
        ] if scheme_type_match else []
        for tag in explicit_types:
            if tag.endswith("_support") and tag not in tags:
                tags.append(tag)

        for tag, keywords in INTENT_KEYWORDS.items():
            if any(re.search(rf"\b{re.escape(keyword)}\b", haystack) for keyword in keywords) and tag not in tags:
                tags.append(tag)
        return tags

    def _query_signals(self, query: str) -> Dict[str, bool]:
        lowered = query.lower()
        return {
            "loan_support": any(term in lowered for term in ["loan", "credit", "bank", "borrow"]),
            "scholarship_support": any(term in lowered for term in ["scholarship", "student", "education", "college", "school"]),
            "housing_support": any(term in lowered for term in ["housing", "house", "home", "rural", "awas", "shelter"]),
            "pension_support": any(term in lowered for term in ["pension", "elderly", "senior", "old age", "retirement"]),
            "farmer_support": any(term in lowered for term in ["farmer", "kisan", "agriculture", "crop"]),
        }

    def _lexical_score(self, query: str, text: str, metadata: Dict[str, Any] | None = None) -> int:
        if not text.strip():
            return 0

        stopwords = {
            "the", "and", "for", "with", "from", "that", "this", "what", "which", "about",
            "need", "looking", "scheme", "schemes", "support", "latest", "today", "recent",
            "current", "apply", "application", "eligible", "eligibility", "help", "have",
        }
        query_normalized = re.sub(r"[^a-z0-9]+", " ", query.lower()).strip()
        query_terms = {
            token for token in re.findall(r"[a-zA-Z]{3,}", query.lower()) if token not in stopwords
        }
        haystack = " ".join(
            [
                text.lower(),
                str((metadata or {}).get("scheme_name", "")).lower(),
                str((metadata or {}).get("source", "")).lower(),
            ]
        )

        score = sum(2 for term in query_terms if term in haystack)
        query_lower = query.lower()
        query_signals = self._query_signals(query)
        scheme_name = str((metadata or {}).get("scheme_name", "")).lower()
        raw_tags = (metadata or {}).get("tags", "")
        if isinstance(raw_tags, str):
            tags = {tag.strip() for tag in raw_tags.split(",") if tag.strip()}
        elif isinstance(raw_tags, list):
            tags = {str(tag).strip() for tag in raw_tags if str(tag).strip()}
        else:
            tags = set()
        if not tags and (text.strip() or scheme_name):
            tags = set(self._derive_tags(scheme_name, text))

        for signal, is_present in query_signals.items():
            if not is_present:
                continue
            if signal in tags:
                score += 5
            elif signal in {"loan_support", "scholarship_support", "housing_support", "pension_support"}:
                score -= 4

        if scheme_name:
            normalized_scheme_name = re.sub(r"[^a-z0-9]+", " ", scheme_name).strip()
            if normalized_scheme_name and normalized_scheme_name in query_normalized:
                score += 10
            else:
                scheme_terms = [
                    term for term in re.findall(r"[a-zA-Z0-9]{3,}", scheme_name)
                    if term not in {"card", "portal", "scheme"}
                ]
                score += sum(2 for term in scheme_terms if term in query_normalized)

        if any(term in query_lower for term in ["farmer", "crop", "kisan"]):
            if any(term in haystack for term in ["farmer", "crop", "kisan", "agriculture"]):
                score += 4
            elif query_signals["farmer_support"]:
                score -= 2
        if "loan" in query_lower or "credit" in query_lower:
            if any(term in haystack for term in ["loan", "credit", "bank"]):
                score += 4
            else:
                score -= 4
        if "scholarship" in query_lower or "student" in query_lower:
            if any(term in haystack for term in ["scholarship", "student", "education"]):
                score += 4
            else:
                score -= 4
        if "pension" in query_lower:
            if "pension" in haystack:
                score += 5
            else:
                score -= 5
        if any(term in query_lower for term in ["house", "housing", "home", "rural"]):
            if any(term in haystack for term in ["housing", "house", "awas", "rural", "pucca"]):
                score += 4
            else:
                score -= 4

        if "sample welfare schemes" in haystack:
            score -= 10

        return score

    def _lexical_retrieve(self, query: str, top_k: int = 4) -> List[Dict[str, str | float]]:
        docs = self._load_documents()
        ranked: List[tuple[int, Document]] = []
        for doc in docs:
            score = self._lexical_score(query, doc.page_content, doc.metadata)
            if score >= 2:
                ranked.append((score, doc))

        ranked.sort(key=lambda item: item[0], reverse=True)
        evidence: List[Dict[str, str | float]] = []
        if ranked:
            best_score = ranked[0][0]
        else:
            best_score = 0

        for score, doc in ranked[:top_k]:
            if best_score >= 8 and score < max(4, best_score - 6):
                continue
            evidence.append(
                {
                    "quote": doc.page_content.strip()[:1400],
                    "source": str(doc.metadata.get("source", "unknown")),
                    "scheme_name": str(doc.metadata.get("scheme_name", "unknown")),
                    "domain": str(doc.metadata.get("domain", "unknown")),
                    "score": float(score),
                    "tags": str(
                        doc.metadata.get(
                            "tags",
                            self._serialize_tags(
                                self._derive_tags(
                                    str(doc.metadata.get("scheme_name", "")),
                                    doc.page_content,
                                )
                            ),
                        )
                    ),
                }
            )
        return evidence

    def ensure_index(self, force_reindex: bool = False) -> Dict[str, Any]:
        if not self.available or self.vectorstore is None or self.splitter is None:
            docs = self._load_documents()
            self._last_index_status = {
                "status": "lexical_only" if docs else "embedding_unavailable",
                "detail": (
                    f"Embeddings unavailable; using lexical retrieval over {len(docs)} local docs."
                    if docs
                    else "No embedding model could be initialized."
                ),
                "sources": len(docs),
            }
            return self._last_index_status

        if force_reindex:
            self._reset_vectorstore()

        try:
            current_count = int(self.vectorstore._collection.count())
        except Exception:
            self._reset_vectorstore()
            current_count = 0

        if current_count > 0 and not force_reindex:
            self._last_index_status = {
                "status": "already_indexed",
                "chunks": current_count,
            }
            return self._last_index_status

        docs = self._load_documents()
        if not docs:
            self._last_index_status = {
                "status": "no_docs",
                "detail": f"No .txt/.md files found in {self.docs_dir}",
            }
            return self._last_index_status

        chunks = self.splitter.split_documents(docs)
        try:
            if chunks:
                self.vectorstore.add_documents(chunks)
        except Exception as exc:
            self._last_index_status = {
                "status": "index_error",
                "detail": f"Embedding/indexing failed: {exc}",
                "embedding_model": self.embedding_model,
            }
            return self._last_index_status

        self._last_index_status = {
            "status": "indexed",
            "chunks": len(chunks),
            "sources": len(docs),
            "embedding_provider": self.embedding_provider,
        }
        return self._last_index_status

    def retrieve(self, query: str, top_k: int = 4) -> List[Dict[str, str | float]]:
        if not self.available or self.vectorstore is None:
            return self._lexical_retrieve(query, top_k=top_k)

        try:
            hits = self.vectorstore.similarity_search_with_relevance_scores(query, k=top_k)
        except Exception as exc:
            self._last_index_status = {
                "status": "retrieve_error",
                "detail": f"Retrieval failed, rebuilding index: {exc}",
            }
            rebuild = self.ensure_index(force_reindex=True)
            if str(rebuild.get("status", "")) not in {"indexed", "already_indexed"}:
                return self._lexical_retrieve(query, top_k=top_k)
            try:
                hits = self.vectorstore.similarity_search_with_relevance_scores(query, k=top_k)
            except Exception:
                return self._lexical_retrieve(query, top_k=top_k)

        evidence: List[Dict[str, str | float]] = []
        for doc, score in hits:
            page_content = getattr(doc, "page_content", "")
            if not isinstance(page_content, str) or not page_content.strip():
                continue
            lexical_score = self._lexical_score(query, page_content, getattr(doc, "metadata", {}))
            if lexical_score < 2:
                continue
            quote = page_content.strip()[:1400]
            evidence.append(
                {
                    "quote": quote,
                    "source": str(doc.metadata.get("source", "unknown")),
                    "scheme_name": str(doc.metadata.get("scheme_name", "unknown")),
                    "domain": str(doc.metadata.get("domain", "unknown")),
                    "score": float(score),
                    "lexical_score": float(lexical_score),
                    "tags": str(
                        doc.metadata.get(
                            "tags",
                            self._serialize_tags(
                                self._derive_tags(
                                    str(doc.metadata.get("scheme_name", "")),
                                    page_content,
                                )
                            ),
                        )
                    ),
                }
            )
        evidence.sort(
            key=lambda item: (
                float(item.get("lexical_score", 0.0)),
                float(item.get("score", 0.0)),
            ),
            reverse=True,
        )
        if evidence:
            best_lexical = float(evidence[0].get("lexical_score", 0.0))
            filtered = [
                item for item in evidence[:top_k]
                if best_lexical < 8 or float(item.get("lexical_score", 0.0)) >= max(4.0, best_lexical - 6.0)
            ]
            return filtered
        return self._lexical_retrieve(query, top_k=top_k)

    @property
    def last_index_status(self) -> Dict[str, Any]:
        return self._last_index_status
