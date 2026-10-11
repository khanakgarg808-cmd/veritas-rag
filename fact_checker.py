import os
import json
import nltk
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# Fetches key from environment or falls back safely
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your groq key")


class SentenceVerification(BaseModel):
    sentence: str = Field(description="The sentence evaluated from the generated answer")
    status: str = Field(description="Must be 'VERIFIED' or 'HALLUCINATION'")
    reason: str = Field(description="Brief textual justification for the rating")


class FactualityChecker:
    def __init__(self, model_name: str = "qwen/qwen3.8-27b"):
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model_name=model_name,
            temperature=0.0
        )

    def verify_sentence(self, sentence: str, source_contexts: List[str]) -> SentenceVerification:
        """Compares a single sentence claim against source contexts."""
        joined_sources = "\n---\n".join(source_contexts)
        safe_sentence = sentence.replace('"', "'")

        prompt = f"""You are an audit assistant.
Determine if the STATEMENT below is completely supported by the SOURCE CONTEXT.

SOURCE CONTEXT:
{joined_sources}

STATEMENT TO CHECK:
"{safe_sentence}"

Rules:
1. If every fact in the statement is explicitly supported by the source, mark as "VERIFIED".
2. If any detail in the statement is missing, unsupported, or contradicted by the source, mark as "HALLUCINATION".

Respond ONLY in valid JSON matching this exact structure:
{{
  "sentence": "{safe_sentence}",
  "status": "VERIFIED",
  "reason": "<One sentence justification>"
}}
"""
        response = self.llm.invoke(prompt)
        try:
            text = response.content.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            parsed = json.loads(text)
            return SentenceVerification(**parsed)
        except Exception:
            return SentenceVerification(
                sentence=sentence,
                status="HALLUCINATION" if "HALLUCINATION" in response.content.upper() else "VERIFIED",
                reason="Evaluated from fallback parser."
            )

    def audit_answer(self, answer: str, source_contexts: List[str]) -> Dict[str, Any]:
        """Tokenizes the full answer and runs line-by-line NLI fact verification."""
        sentences = nltk.sent_tokenize(answer)
        audit_records = []
        hallucination_count = 0

        meta_phrases = [
            "cannot find this information",
            "no information provided",
            "does not mention",
            "is not mentioned",
            "not possible to determine",
            "provided context only mentions"
        ]

        for sentence in sentences:
            if len(sentence.split()) < 4:
                continue

            if any(phrase in sentence.lower() for phrase in meta_phrases):
                audit_records.append({
                    "sentence": sentence,
                    "status": "VERIFIED",
                    "reason": "Accurately acknowledges that requested data is absent from the domain text."
                })
                continue

            record = self.verify_sentence(sentence, source_contexts)
            audit_records.append(record.model_dump())
            if record.status == "HALLUCINATION":
                hallucination_count += 1

        total_claims = len(audit_records)
        score = (
            ((total_claims - hallucination_count) / total_claims) * 100
            if total_claims > 0 else 100.0
        )

        return {
            "sentence_audits": audit_records,
            "factuality_score": round(score, 2),
            "contains_hallucinations": hallucination_count > 0
        }