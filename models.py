import os
import base64
import json
import uuid
from typing import Any

from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

load_dotenv()


class MedicalAgentModel:
    """Groq-powered evidence-grounded multimodal medical document assistant."""

    DEFAULT_MODEL = "qwen/qwen3.8-27b"
    # Keep this below the 1,000 output-token limit seen on many free/on-demand tiers.
    MAX_OUTPUT_TOKENS = 900
    MAX_PDF_CHARS = 45000
    MAX_PDF_IMAGE_PAGES = 3

    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not self.api_key or self.api_key == "your_groq_api_key_here":
            raise ValueError(
                "GROQ_API_KEY is missing. Add your new Groq API key to .env"
            )

        self.model_name = os.getenv("GROQ_MODEL", self.DEFAULT_MODEL).strip()
        self.client = Groq(api_key=self.api_key)

    # -----------------------------
    # IMAGE PROCESSING
    # -----------------------------

    def get_image_data_url(self, uploaded_file) -> str:
        image_bytes = uploaded_file.getvalue()
        if not image_bytes:
            raise ValueError("Uploaded image is empty.")

        extension = uploaded_file.name.rsplit(".", 1)[-1].lower()
        mime_types = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "webp": "image/webp",
        }
        if extension not in mime_types:
            raise ValueError(f"Unsupported image format: {extension}")

        encoded = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:{mime_types[extension]};base64,{encoded}"

    # -----------------------------
    # PDF PROCESSING
    # -----------------------------

    def extract_pdf_text(self, uploaded_pdf) -> list[dict[str, Any]]:
        if not uploaded_pdf:
            return []

        raw = uploaded_pdf.getvalue()
        if not raw:
            raise ValueError("Uploaded PDF is empty.")

        reader = PdfReader(uploaded_pdf)
        pages: list[dict[str, Any]] = []
        total_chars = 0

        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue

            remaining = self.MAX_PDF_CHARS - total_chars
            if remaining <= 0:
                break

            text = text[:remaining]
            pages.append({"page": page_number, "text": text})
            total_chars += len(text)

        return pages

    def render_pdf_images(self, uploaded_pdf) -> list[str]:
        """Render a few pages when a PDF is scanned/image-based."""
        if not uploaded_pdf or fitz is None:
            return []

        raw = uploaded_pdf.getvalue()
        if not raw:
            return []

        doc = fitz.open(stream=raw, filetype="pdf")
        data_urls: list[str] = []
        try:
            for page_index in range(min(len(doc), self.MAX_PDF_IMAGE_PAGES)):
                page = doc.load_page(page_index)
                pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
                png_bytes = pix.tobytes("png")
                encoded = base64.b64encode(png_bytes).decode("utf-8")
                data_urls.append(f"data:image/png;base64,{encoded}")
        finally:
            doc.close()
        return data_urls

    def build_pdf_context(self, uploaded_pdf) -> tuple[str, list[str]]:
        pages = self.extract_pdf_text(uploaded_pdf)
        text = "\n\n".join(
            f"[SOURCE: Uploaded PDF | PAGE: {page['page']}]\n{page['text']}"
            for page in pages
        )

        # If extraction is poor, use page images as a multimodal fallback.
        page_images = []
        if uploaded_pdf and len(text.strip()) < 500:
            page_images = self.render_pdf_images(uploaded_pdf)

        return text, page_images

    # -----------------------------
    # STRICT JSON SCHEMA
    # -----------------------------

    @staticmethod
    def response_schema() -> dict:
        # Groq strict structured outputs require all fields to be required and
        # every object to have additionalProperties=false.
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "response_type": {"type": "string", "enum": ["general", "clinical"]},
                "general_answer": {"type": "string"},
                "patient_info": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "patient_name": {"type": "string"},
                        "age": {"type": "string"},
                        "sex": {"type": "string"},
                        "document_type": {"type": "string"},
                        "report_date": {"type": "string"},
                        "admission_date": {"type": "string"},
                        "discharge_date": {"type": "string"},
                    },
                    "required": [
                        "patient_name",
                        "age",
                        "sex",
                        "document_type",
                        "report_date",
                        "admission_date",
                        "discharge_date",
                    ],
                },
                "summary": {"type": "string"},
                "image_findings": {"type": "array", "items": {"type": "string"}},
                "lab_findings": {"type": "array", "items": {"type": "string"}},
                "correlations": {"type": "array", "items": {"type": "string"}},
                "possible_considerations": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "uncertainties": {"type": "array", "items": {"type": "string"}},
                "suggested_next_steps": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "evidence": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "claim": {"type": "string"},
                            "source": {"type": "string"},
                            "page": {"type": "string"},
                        },
                        "required": ["claim", "source", "page"],
                    },
                },
                "safety_note": {"type": "string"},
            },
            "required": [
                "response_type",
                "general_answer",
                "patient_info",
                "summary",
                "image_findings",
                "lab_findings",
                "correlations",
                "possible_considerations",
                "uncertainties",
                "suggested_next_steps",
                "evidence",
                "safety_note",
            ],
        }

    # -----------------------------
    # PROMPTS
    # -----------------------------


    def build_system_prompt(self, view_mode: str, response_type: str) -> str:
        if view_mode == "Doctor (Technical)":
            audience = (
                "Use concise clinical terminology suitable for a clinician. "
                "Separate established medical facts from interpretation."
            )
        else:
            audience = (
                "Use plain, calm language suitable for a patient. Explain important "
                "medical terms briefly and avoid unnecessary alarm."
            )

        if response_type == "general":
            return f"""
You are MedScan Pro, a medical education assistant.
The user is asking a GENERAL MEDICAL KNOWLEDGE question and no patient file was supplied.

AUDIENCE: {view_mode}
{audience}

TASK:
Answer the user's question directly and educationally. Do not pretend that a patient
exists and do not create patient/document information. Give a useful, well-organized
explanation appropriate to the question. Cover the most relevant concepts, common
causes/risk factors, symptoms/signs, diagnosis, and general management when relevant.
For disease questions, explain the major conditions or categories without turning the
answer into a diagnosis of an individual.

SAFETY:
- This is general medical information, not an individual diagnosis or treatment plan.
- Do not recommend medication starts/stops or dosage changes for a specific person.
- Do not invent patient data, test results, imaging findings, or medical history.

OUTPUT:
Return ONLY the JSON object required by the supplied JSON Schema.
Set response_type to "general".
Put the complete educational answer in general_answer.
Set all clinical-analysis fields to empty strings/arrays as required by the schema.
Set safety_note to a brief statement that this is general medical information and not an individual diagnosis or treatment plan.
Do not use Markdown code fences or <think> tags.
"""

        return f"""
You are MedScan Pro, an evidence-grounded clinical decision-support assistant.
You are analyzing uploaded medical evidence supplied in this request.

AUDIENCE: {view_mode}
{audience}

SAFETY AND EVIDENCE RULES:
1. Never invent patient information, dates, diagnoses, symptoms, medications,
   measurements, laboratory values, imaging findings, history, or page numbers.
2. Extract patient information only when explicitly present in the uploaded file.
3. If patient/document metadata is absent, return an empty string for that field.
4. Use the exact PDF page number for evidence when the source is PDF text.
5. Distinguish documented facts from clinical interpretation and uncertainty.
6. Do not make a definitive diagnosis from incomplete evidence.
7. Do not recommend medication starts, stops, or dosage changes.
8. Suggested next steps are discussion points for a qualified clinician.
9. If the evidence is insufficient, state that clearly.
10. Do not expose chain-of-thought or hidden reasoning.

OUTPUT:
Return ONLY the JSON object required by the supplied JSON Schema.
Set response_type to "clinical" and keep general_answer empty.
No Markdown outside string values. No code fences. No <think> tags.
Keep the response concise enough to fit the available output budget.
Prefer a small number of high-value findings over repetition.
"""

    def build_user_content(
        self,
        prompt: str,
        pdf_context: str,
        uploaded_image=None,
        pdf_image_urls: list[str] | None = None,
        response_type: str = "clinical",
    ):
        if response_type == "general":
            instruction = (
                "Answer the user's general medical question as an educational explanation. "
                "Do not invent patient-specific information. "
                "Do not force the answer into a clinical report. "
                "Explain the topic clearly and accurately."
        )
        else:
            instruction = (
                "Analyze the uploaded medical evidence only. "
                "Extract explicit patient and document information. "
                "Provide the most relevant findings and correlations. "
                "Do not fill missing fields by guessing."
        )
        text_content = (
            "USER REQUEST:\n"
            + prompt
            + "\n\n"
            + instruction
        )

        content = [
        {
            "type": "text",
            "text": text_content,
        }
        ]
        if pdf_context:
            content.append(
            {
                "type": "text",
                "text": "UPLOADED PDF TEXT:\n" + pdf_context,
            }
        )

        if uploaded_image:
            content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": self.get_image_data_url(uploaded_image)
                },
            }
        )

        for data_url in pdf_image_urls or []:
            content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": data_url
                },
            }
        )

        return content

    # -----------------------------
    # MAIN ANALYSIS
    # -----------------------------

    def analyze_clinical_data(
        self,
        view_mode: str,
        prompt: str,
        uploaded_image=None,
        uploaded_pdf=None,
    ) -> dict:
        request_id = str(uuid.uuid4())

        if not prompt or not prompt.strip():
            raise ValueError("Please provide a question or upload a medical file.")

        has_uploaded_evidence = bool(uploaded_image or uploaded_pdf)
        response_type = "clinical" if has_uploaded_evidence else "general"
        pdf_context, pdf_images = self.build_pdf_context(uploaded_pdf)

        if not has_uploaded_evidence:
            pdf_context = ""

        messages = [
            {"role": "system", "content": self.build_system_prompt(view_mode, response_type)},
            {
                "role": "user",
                "content": self.build_user_content(
                    prompt=prompt,
                    pdf_context=pdf_context,
                    uploaded_image=uploaded_image,
                    pdf_image_urls=pdf_images,
                    response_type=response_type,
                ),
            },
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,
                max_completion_tokens=self.MAX_OUTPUT_TOKENS,
                reasoning_effort="none",
                reasoning_format="hidden",
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "medical_analysis",
                        "strict": True,
                        "schema": self.response_schema(),
                    },
                },
            )
        except Exception as exc:
            raise RuntimeError(
                f"AI analysis failed. Request ID: {request_id}. "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        message = response.choices[0].message
        raw_content = message.content

        if not raw_content:
            raise RuntimeError(
                f"AI returned an empty response. Request ID: {request_id}"
            )

        try:
            result = json.loads(raw_content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"AI returned invalid structured output. Request ID: {request_id}. "
                f"{exc}"
            ) from exc

        # Lightweight application-level validation after the API's strict schema.
        required = [
            "patient_info",
            "summary",
            "image_findings",
            "lab_findings",
            "correlations",
            "possible_considerations",
            "uncertainties",
            "suggested_next_steps",
            "evidence",
            "safety_note",
        ]
        missing = [key for key in required if key not in result]
        if missing:
            raise RuntimeError(
                f"AI returned incomplete structured output. Request ID: {request_id}. "
                f"Missing: {', '.join(missing)}"
            )

        result["_request_id"] = request_id
        result["_model"] = self.model_name
        return result
