from dataclasses import dataclass


@dataclass
class IntentResult:
    intent: str
    response_type: str


class IntentRouter:
    """
    Lightweight request router for MedScan Pro.

    The router uses the user's request and uploaded evidence to determine
    which workflow should handle the request.
    """

    GENERAL_MEDICAL = "GENERAL_MEDICAL"
    CLINICAL_ANALYSIS = "CLINICAL_ANALYSIS"
    IMAGE_ANALYSIS = "IMAGE_ANALYSIS"
    DOCUMENT_SUMMARY = "DOCUMENT_SUMMARY"
    FOLLOW_UP = "FOLLOW_UP"

    def route(
        self,
        prompt: str,
        uploaded_image=None,
        uploaded_pdf=None,
        has_previous_context: bool = False,
    ) -> IntentResult:

        prompt_lower = (prompt or "").strip().lower()

        # ---------------------------------
        # 1. Follow-up questions
        # ---------------------------------
        if has_previous_context and self._looks_like_follow_up(prompt_lower):
            return IntentResult(
                intent=self.FOLLOW_UP,
                response_type="clinical",
            )

        # ---------------------------------
        # 2. Image analysis
        # ---------------------------------
        if uploaded_image:
            return IntentResult(
                intent=self.IMAGE_ANALYSIS,
                response_type="clinical",
            )

        # ---------------------------------
        # 3. PDF document summary
        # ---------------------------------
        if uploaded_pdf and self._looks_like_summary_request(prompt_lower):
            return IntentResult(
                intent=self.DOCUMENT_SUMMARY,
                response_type="clinical",
            )

        # ---------------------------------
        # 4. Clinical document analysis
        # ---------------------------------
        if uploaded_pdf:
            return IntentResult(
                intent=self.CLINICAL_ANALYSIS,
                response_type="clinical",
            )

        # ---------------------------------
        # 5. General medical question
        # ---------------------------------
        return IntentResult(
            intent=self.GENERAL_MEDICAL,
            response_type="general",
        )

    @staticmethod
    def _looks_like_summary_request(prompt: str) -> bool:
        summary_terms = [
            "summarize",
            "summary",
            "summarise",
            "give me a summary",
            "summarize this report",
            "summarise this report",
            "what does this report say",
            "explain this report",
        ]

        return any(term in prompt for term in summary_terms)

    @staticmethod
    def _looks_like_follow_up(prompt: str) -> bool:
        follow_up_terms = [
            "what about this",
            "what does that mean",
            "can you explain that",
            "why is that",
            "what should i ask",
            "tell me more",
            "explain more",
            "what next",
            "what about the above",
        ]

        return any(term in prompt for term in follow_up_terms)