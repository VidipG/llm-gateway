import logging
import re

from app.config import Settings
from app.gateway.errors import PromptInjectionError
from app.schemas.request import CompletionRequest

logger = logging.getLogger(__name__)

# (pattern, risk_weight) — applied case-insensitively to user messages
_PATTERN_RULES: list[tuple[re.Pattern[str], float]] = [
    # Instruction override
    (re.compile(r"ignore\s+(previous|prior|all)\s+instructions?", re.IGNORECASE), 0.4),
    (re.compile(r"\bdisregard\b", re.IGNORECASE), 0.3),
    (re.compile(r"forget\s+your\s+(instructions|prompt|role)", re.IGNORECASE), 0.4),
    # Role assumption
    (re.compile(r"\byou\s+are\s+now\b", re.IGNORECASE), 0.3),
    (re.compile(r"\bact\s+as\b", re.IGNORECASE), 0.2),
    (re.compile(r"\bpretend\s+(you\s+are|to\s+be)\b", re.IGNORECASE), 0.3),
    (re.compile(r"\byour\s+new\s+(role|persona|instructions)\b", re.IGNORECASE), 0.35),
    # Exfiltration
    (re.compile(r"\b(print|repeat|reveal|show)\s+(your|the)\s+(system\s+|original\s+)?(prompt|instructions|rules)\b", re.IGNORECASE), 0.5),
    (re.compile(r"\bwhat\s+(are\s+you|were\s+you)\s+(told|instructed|trained)\b", re.IGNORECASE), 0.4),
    # Jailbreak scaffolds
    (re.compile(r"\bdeveloper\s+mode\b", re.IGNORECASE), 0.5),
    (re.compile(r"\bdo\s+anything\s+now\b", re.IGNORECASE), 0.5),
    (re.compile(r"\bDAN\b"), 0.4),
    (re.compile(r"\bno\s+restrictions\b", re.IGNORECASE), 0.4),
    (re.compile(r"\bbypass\s+your\b", re.IGNORECASE), 0.45),
    (re.compile(r"\boverride\s+your\b", re.IGNORECASE), 0.45),
]

# Output-side audit patterns
_OUTPUT_AUDIT_RULES: list[re.Pattern[str]] = [
    re.compile(r"\bmy\s+(instructions|prompt|rules)\s+(say|state|tell\s+me)\b", re.IGNORECASE),
    re.compile(r"\bI\s+(was\s+told|am\s+instructed)\s+to\b", re.IGNORECASE),
    re.compile(r"\bas\s+per\s+my\s+(system|instructions)\b", re.IGNORECASE),
]


def _score_message(content: str) -> float:
    """Return cumulative risk score [0, ∞) for a single message."""
    score = 0.0
    for pattern, weight in _PATTERN_RULES:
        if pattern.search(content):
            score += weight
    return score


class PromptGuard:
    def __init__(self, settings: Settings) -> None:
        self._enabled = settings.prompt_guard_enabled
        self._max_messages = settings.prompt_guard_max_messages
        self._max_chars = settings.prompt_guard_max_message_chars
        self._block_threshold = settings.prompt_guard_block_threshold
        self._output_audit = settings.prompt_guard_output_audit

    def check_input(self, request: CompletionRequest) -> None:
        """Validate and scan request messages. Raises PromptInjectionError if blocked."""
        if not self._enabled:
            return

        messages = request.messages

        # --- Structural validation ---
        if len(messages) > self._max_messages:
            raise PromptInjectionError(
                f"Request exceeds maximum message count ({len(messages)} > {self._max_messages})"
            )

        system_seen = False
        for i, msg in enumerate(messages):
            if len(msg.content) > self._max_chars:
                raise PromptInjectionError(
                    f"Message {i} exceeds maximum length ({len(msg.content)} > {self._max_chars})"
                )
            if msg.role == "system":
                if i != 0:
                    raise PromptInjectionError(
                        "System message must be the first message"
                    )
                if system_seen:
                    raise PromptInjectionError(
                        "At most one system message is allowed"
                    )
                system_seen = True

        # --- Pattern filter (user messages only) ---
        total_score = 0.0
        for msg in messages:
            if msg.role == "user":
                total_score += _score_message(msg.content)

        total_score = min(total_score, 1.0)

        if total_score > self._block_threshold:
            raise PromptInjectionError(
                f"Request blocked: injection risk score {total_score:.2f} >= threshold {self._block_threshold}"
            )

        if total_score > 0:
            logger.warning(
                "Prompt guard: suspicious content detected (score=%.2f, threshold=%.2f) — passing through",
                total_score,
                self._block_threshold,
            )

    def audit_output(self, model: str, prompt: str, response: str, request_id: str = "") -> None:
        """Scan assembled response for leakage indicators. Logs warnings; never raises."""
        if not self._enabled or not self._output_audit:
            return

        for pattern in _OUTPUT_AUDIT_RULES:
            if pattern.search(response):
                logger.warning(
                    "Output audit: instruction acknowledgement detected "
                    "request_id=%s model=%s pattern=%r",
                    request_id,
                    model,
                    pattern.pattern,
                )
