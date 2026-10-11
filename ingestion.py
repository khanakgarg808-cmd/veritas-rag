import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


class HybridRetrieverSystem:
    def __init__(self, pdf_paths: List[str]):
        self.chunks: List[Document] = []
        self.vectorstore = None
        self.bm25_retriever = None
        self.dense_retriever = None
        
        # Free local embeddings running directly on your CPU
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        self._load_and_chunk(pdf_paths)
        self._build_indices()

    def _load_and_chunk(self, pdf_paths: List[str]):
        """Load source PDFs and divide them into overlapping text chunks."""
        all_docs = []
        for path in pdf_paths:
            if os.path.exists(path):
                loader = PyPDFLoader(path)
                all_docs.extend(loader.load())
            else:
                print(f"[!] File not found: {path}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )
        self.chunks = text_splitter.split_documents(all_docs)
        print(f"[+] Total chunks indexed: {len(self.chunks)}")

    def _build_indices(self):
        """Construct BM25 and InMemory Vector search indexes."""
        if not self.chunks:
            raise ValueError("No document chunks available to index.")

        # 1. Sparse Keyword Index (BM25)
        self.bm25_retriever = BM25Retriever.from_documents(self.chunks)
        self.bm25_retriever.k = 4

        # 2. Native In-Memory Vector Store (Fully compatible with Python 3.14 + NumPy 2.x)
        self.vectorstore = InMemoryVectorStore.from_documents(
            documents=self.chunks,
            embedding=self.embeddings
        )
        self.dense_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})

    @staticmethod
    def reciprocal_rank_fusion(
        bm25_docs: List[Document], 
        dense_docs: List[Document], 
        k: int = 60, 
        top_n: int = 4
    ) -> List[Document]:
        """Merges Sparse and Dense retrievals using RRF scoring."""
        doc_scores = {}

        def apply_ranks(docs: List[Document]):
            for rank, doc in enumerate(docs):
                content = doc.page_content
                if content not in doc_scores:
                    doc_scores[content] = {"doc": doc, "score": 0.0}
                doc_scores[content]["score"] += 1.0 / (rank + k)

        apply_ranks(bm25_docs)
        apply_ranks(dense_docs)

        sorted_results = sorted(
            doc_scores.values(), key=lambda x: x["score"], reverse=True
        )
        return [item["doc"] for item in sorted_results[:top_n]]

    def retrieve(self, query: str) -> List[Document]:
        """Performs hybrid retrieval for a given search query."""
        bm25_results = self.bm25_retriever.invoke(query)
        dense_results = self.dense_retriever.invoke(query)
        return self.reciprocal_rank_fusion(bm25_results, dense_results)