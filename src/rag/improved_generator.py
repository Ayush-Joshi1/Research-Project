"""Prompt-only improved generator for evidence-grounded claim verification."""
import json
from typing import Dict, List

from .generator import MODEL_NAME, client


def build_improved_prompt(claim: str, retrieved_documents: List[Dict]) -> str:
    evidence_parts = []
    for number, document in enumerate(retrieved_documents, start=1):
        evidence_parts.append(
            f"[Document {number}]\n"
            f"Title: {document.get('title', '')}\n"
            f"Text: {document.get('text', '')}"
        )
    evidence = "\n\n".join(evidence_parts) or "[No retrieved evidence]"
    return f'''You are verifying a scientific claim using ONLY the supplied retrieved evidence.

CLAIM:
{claim}

RETRIEVED EVIDENCE:
{evidence}

ANALYSIS INSTRUCTIONS:
- Read the claim carefully.
- Examine every retrieved document separately.
- For each document, determine whether it provides evidence relevant to the claim.
- Distinguish evidence that SUPPORTS the claim from evidence that CONTRADICTS the claim.
- Ignore documents that are irrelevant to the claim.
- Compare the claim directly against the relevant evidence before deciding the final verdict.
- Base the final verdict only on the retrieved evidence.
- If relevant evidence is present but does not clearly establish support or contradiction, return INSUFFICIENT_EVIDENCE.
- Do not use outside knowledge.
- Do not invent facts.
- The final explanation must mention the evidence that directly justifies the verdict.
- Do not expose chain-of-thought or hidden reasoning. Provide only the final verdict and concise explanation.

FINAL OUTPUT:
Return ONLY valid JSON:
{{
  "verdict": "SUPPORT" | "CONTRADICT" | "INSUFFICIENT_EVIDENCE",
  "explanation": "short explanation based only on the retrieved evidence"
}}
Do not return markdown fences around the JSON.'''


def generate_verification_improved(claim: str, retrieved_documents: List[Dict]) -> Dict[str, str]:
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=build_improved_prompt(claim, retrieved_documents),
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
