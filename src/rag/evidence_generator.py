"""Evidence-aware Gemini generation using the existing generator configuration."""
import json
import os
from typing import Dict, List

from dotenv import load_dotenv
from google import genai

MODEL_NAME = "gemini-3.5-flash-lite"
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set")

client = genai.Client(api_key=API_KEY)


def build_evidence_prompt(claim: str, evidence: List[Dict]) -> str:
    evidence_parts = []
    for number, item in enumerate(evidence, start=1):
        evidence_parts.append(
            f"[Evidence {number}]\n"
            f"Document ID: {item['corpus_id']}\n"
            f"Original document rank: {item['document_rank']}\n"
            f"Similarity: {item['similarity']:.4f}\n"
            f"Sentence: {item['sentence']}"
        )
    supplied_evidence = "\n\n".join(evidence_parts) or "[No extracted evidence]"
    return f'''You are verifying a scientific claim using ONLY the supplied extracted evidence sentences.

Claim:
{claim}

Extracted evidence sentences (not complete documents):
{supplied_evidence}

Determine whether the evidence supports or contradicts the claim.

Return ONLY valid JSON:
{{
  "verdict": "SUPPORT" | "CONTRADICT" | "INSUFFICIENT_EVIDENCE",
  "explanation": "short explanation based only on the supplied evidence"
}}

Rules:
- Do not use outside knowledge.
- Do not invent facts.
- If the evidence does not clearly support or contradict the claim, return INSUFFICIENT_EVIDENCE.
- The explanation must refer only to the supplied evidence sentences.
- Do not return markdown fences around the JSON.'''


def generate_evidence_verification(claim: str, evidence: List[Dict]) -> Dict[str, str]:
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=build_evidence_prompt(claim, evidence),
        config={"response_mime_type": "application/json"},
    )
    try:
        result = json.loads(response.text)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError("Gemini returned invalid JSON") from error

    verdict = result.get("verdict")
    explanation = result.get("explanation")
    valid_verdicts = {"SUPPORT", "CONTRADICT", "INSUFFICIENT_EVIDENCE"}
    if verdict not in valid_verdicts or not isinstance(explanation, str):
        raise ValueError("Gemini JSON did not match the required schema")
    return {"verdict": verdict, "explanation": explanation}
