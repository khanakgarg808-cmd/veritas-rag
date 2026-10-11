import os
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_groq import ChatGroq

# Reads key from environment variable, falls back to placeholder if unset
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your groq key")


class StrictRAGGenerator:
    def __init__(self, model_name: str = "qwen/qwen3.8-27b"):
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model_name=model_name,
            temperature=0.0
        )

    def generate_response(self, query: str, context_docs: List[Document]) -> Dict[str, Any]:
        """Generates an answer using strictly the supplied context documents."""
        formatted_context = "\n\n".join(
            [f"[Source Chunk {i+1}]:\n{doc.page_content}" for i, doc in enumerate(context_docs)]
        )

        system_prompt = (
            "You are a strict, domain-bound assistant. Answer the user question using ONLY "
            "the provided context chunks below.\n"
            "- Do NOT use prior internal knowledge.\n"
            "- Do NOT extrapolate, guess, or assume missing details.\n"
            "- If the provided context does not contain enough information, reply with: "
            "'I cannot find this information in the provided domain documents.'\n\n"
            f"PROVIDED CONTEXT:\n{formatted_context}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {query}"}
        ]

        response = self.llm.invoke(messages)
        return {
            "answer": response.content.strip(),
            "context_texts": [doc.page_content for doc in context_docs]
        }