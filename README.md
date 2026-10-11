# 🛡️ SentinelRAG: Domain-Specific RAG with LLM-as-a-Judge Factuality Checker

An end-to-end, enterprise-oriented Retrieval-Augmented Generation (RAG) system designed to eliminate ungrounded hallucinations in domain-specific tasks. Built with Streamlit, LangChain, and Groq's high-speed inference engine.

---

## 📌 Overview

Traditional RAG implementations often suffer from hallucinated claims when context is sparse or complex. **SentinelRAG** solves this by adding an automated post-generation audit layer:

- **Dual-Phase Search:** Implements Hybrid Retrieval combining sparse keyword matching (BM25) and dense semantic embeddings via Reciprocal Rank Fusion (RRF).
- **Strict Grounding:** Constrains the generative model to the supplied domain context to prevent knowledge leakage.
- **LLM-as-a-Judge Audit Layer:** Tokenizes the generated answer into individual claims and audits each sentence against the retrieved chunks using NLI (Natural Language Inference) validation.
- **Explainable UI:** Visualizes verification logs in real time, highlighting verified statements in green and ungrounded claims in red with an aggregate Factuality Score.

---

## 🚀 Key Features

- **Hybrid Document Retrieval:** Ensures both exact-keyword recall and semantic context awareness.
- **Sentence-by-Sentence Verification:** Audits specific claims rather than scoring outputs as a black box.
- **Real-Time Factuality Scoring:** Computes precision percentages based on supported vs. ungrounded sentences.
- **Interactive Streamlit Interface:** Allows custom PDF uploads, live question answering, and inspectable source chunk verification.
