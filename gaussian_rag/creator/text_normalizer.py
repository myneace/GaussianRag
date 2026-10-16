"""Text normalization and cleaning engine for Gaussian RAG.

Handles messy, unformatted, or dirty text pasted from arbitrary sources:
  - Web pages (HTML/XML tags, styles, scripts, entities)
  - PDF/OCR scans (soft line wraps, hyphenated word breaks, page counters)
  - Chat logs / transcripts (timestamps, speaker prefixes, quote artifacts)
  - Unicode anomalies (zero-width spaces, ligatures, curly quotes, dashes)
  - Optional AI-assisted normalization via OpenAI-compatible endpoints to resolve
    ambiguous formatting, unknown characters, corrupted words, and OCR mojibake.
"""
from __future__ import annotations

import html
import logging
import re
import unicodedata
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gaussian_rag.creator.llm_assistant import OpenAICorpusAssistant

logger = logging.getLogger(__name__)


class TextNormalizer:
    """Configurable text normalization pipeline for cleaning arbitrary text.

    Parameters
    ----------
    normalize_unicode:
        If True, applies NFKC normalization and cleans invisible/non-standard chars.
    strip_html:
        If True, strips script/style tags, HTML tags, and unescapes entities.
    fix_ocr_hyphenation:
        If True, repairs split hyphenated words across line wraps (e.g. 'infor-\\nmation').
    unwrap_soft_linebreaks:
        If True, unwraps artificial hard line breaks within paragraphs while
        preserving double-newline paragraph boundaries and bullet lists.
    strip_page_artifacts:
        If True, removes running headers, footers, and page numbers (e.g. 'Page 1 of 5').
    strip_timestamps:
        If True, removes timestamp headers commonly found in chat logs and transcripts.
    llm_assistant:
        Optional ``OpenAICorpusAssistant`` (or boolean ``True`` to auto-discover) to
        leverage an OpenAI-compatible LLM for repairing ambiguous unknown characters
        or corrupted OCR text.
    """

    def __init__(
        self,
        *,
        normalize_unicode: bool = True,
        strip_html: bool = True,
        fix_ocr_hyphenation: bool = True,
        unwrap_soft_linebreaks: bool = True,
        strip_page_artifacts: bool = True,
        strip_timestamps: bool = True,
        llm_assistant: OpenAICorpusAssistant | bool = False,
    ) -> None:
        self.normalize_unicode = normalize_unicode
        self.strip_html = strip_html
        self.fix_ocr_hyphenation = fix_ocr_hyphenation
        self.unwrap_soft_linebreaks = unwrap_soft_linebreaks
        self.strip_page_artifacts = strip_page_artifacts
        self.strip_timestamps = strip_timestamps

        if isinstance(llm_assistant, bool):
            if llm_assistant:
                from gaussian_rag.creator.llm_assistant import OpenAICorpusAssistant
                self.llm_assistant: OpenAICorpusAssistant | None = OpenAICorpusAssistant()
            else:
                self.llm_assistant = None
        else:
            self.llm_assistant = llm_assistant

    def clean(self, text: str, *, use_llm: bool | None = None) -> str:
        """Run the text normalization pipeline on *text*.

        Parameters
        ----------
        text:
            Raw text string.
        use_llm:
            If True, runs Tier 2 AI text repair using the configured OpenAI-compatible
            endpoint after rule-based cleaning. If None, uses LLM if configured.
        """
        if not text:
            return ""

        # --- Tier 1: Fast Rule-Based Deterministic Cleaning ---

        # 1. Standardize line endings
        result = text.replace("\r\n", "\n").replace("\r", "\n")

        # 2. HTML / Script / Entity cleaning
        if self.strip_html:
            result = self._clean_html(result)

        # 3. Unicode normalization and character sanitation
        if self.normalize_unicode:
            result = self._clean_unicode(result)

        # 4. Strip timestamps from logs / transcripts
        if self.strip_timestamps:
            result = self._clean_timestamps(result)

        # 5. Page artifacts & running headers/footers
        if self.strip_page_artifacts:
            result = self._clean_page_artifacts(result)

        # 6. OCR hyphenation stitching
        if self.fix_ocr_hyphenation:
            result = self._fix_ocr_hyphens(result)

        # 7. Soft line-break unwrapping
        if self.unwrap_soft_linebreaks:
            result = self._unwrap_linebreaks(result)

        # 8. Final whitespace consolidation
        result = self._consolidate_whitespace(result)
        result = result.strip()

        # --- Tier 2: Optional AI-Assisted Repair ---
        should_use_llm = (use_llm is True) or (use_llm is None and self.llm_assistant and self.llm_assistant.is_configured)
        if should_use_llm and self.llm_assistant:
            try:
                result = self.llm_assistant.repair_ambiguous_text(result)
            except Exception as exc:
                logger.warning("LLM text repair skipped due to error: %s", exc)

        return result.strip()

    # ----------------------------------------------------------------------
    # Pipeline stages
    # ----------------------------------------------------------------------

    @staticmethod
    def _clean_html(text: str) -> str:
        """Strip HTML tags and convert formatting tags to text breaks."""
        text = re.sub(r"(?is)<script.*?>.*?</script>", " ", text)
        text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
        text = re.sub(r"(?i)<br\s*/?>", "\n", text)
        text = re.sub(r"(?i)</(?:p|div|h[1-6]|li|tr)>", "\n", text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = html.unescape(text)
        return text

    @staticmethod
    def _clean_unicode(text: str) -> str:
        """Apply NFKC normalization and normalize typography/invisible symbols."""
        text = unicodedata.normalize("NFKC", text)

        invisible_chars = [
            "\u200b",  # zero-width space
            "\u200c",  # zero-width non-joiner
            "\u200d",  # zero-width joiner
            "\ufeff",  # byte order mark / zero-width no-break space
            "\u2060",  # word joiner
            "\u00ad",  # soft hyphen
        ]
        for ch in invisible_chars:
            text = text.replace(ch, "")

        text = text.replace("\u00a0", " ").replace("\u202f", " ")

        # Normalize quotation marks
        text = re.sub(r"[\u201c\u201d\u201e\u00ab\u00bb]", '"', text)
        text = re.sub(r"[\u2018\u2019\u201a\u2032]", "'", text)

        # Normalize em-dashes and en-dashes
        text = text.replace("\u2014", " -- ").replace("\u2013", "-")

        # Strip remaining non-printable control characters (keep \n and \t)
        text = "".join(ch for ch in text if ch in {"\n", "\t"} or unicodedata.category(ch)[0] != "C")
        return text

    @staticmethod
    def _clean_timestamps(text: str) -> str:
        """Strip transcript/chat timestamps such as [10:15:30] or 14:02."""
        text = re.sub(
            r"(?m)^\[\d{1,2}:\d{2}(?::\d{2})?(?:\s*(?:AM|PM|am|pm))?\]\s*",
            "",
            text,
        )
        text = re.sub(
            r"\[\d{1,2}:\d{2}(?::\d{2})?(?:\s*(?:AM|PM|am|pm))?\]",
            " ",
            text,
        )
        text = re.sub(
            r"\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?Z?\b",
            "",
            text,
        )
        return text

    @staticmethod
    def _clean_page_artifacts(text: str) -> str:
        """Remove page numbers, headers, and footers from documents."""
        text = re.sub(r"(?im)^\s*page\s+\d+(\s+of\s+\d+)?\s*$", "", text)
        text = re.sub(r"(?m)^\s*-\s*\d+\s*-\s*$", "", text)
        text = re.sub(r"(?m)^\s*\d+\s*/\s*\d+\s*$", "", text)
        return text

    @staticmethod
    def _fix_ocr_hyphens(text: str) -> str:
        """Rejoin words split by hyphen at line breaks (e.g. 'com-\\nputer' -> 'computer')."""
        return re.sub(r"([a-zA-Z]{2,})-\n\s*([a-zA-Z]{2,})", r"\1\2", text)

    @staticmethod
    def _unwrap_linebreaks(text: str) -> str:
        """Unwrap single line breaks inside paragraphs while preserving genuine paragraph breaks and headers."""
        paragraphs = re.split(r"\n{2,}", text)
        cleaned_paragraphs: list[str] = []

        header_prefix = re.compile(
            r"^(?:#{1,6}\s+|[A-Z][a-z]+ \d+[:.]\s*|Q:|A:|Question:|Answer:)"
        )
        list_prefix = re.compile(
            r"^(?:[-*•]\s+|\d+[.)]\s+)"
        )

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            lines = para.split("\n")
            if len(lines) <= 1:
                cleaned_paragraphs.append(para)
                continue

            # Check if paragraph has headers or list items
            assembled_lines: list[str] = []
            for line in lines:
                line_str = line.strip()
                if not line_str:
                    continue
                # If current line is a header/list item, or previous line was a header:
                prev_is_header = bool(assembled_lines and header_prefix.match(assembled_lines[-1]))
                if header_prefix.match(line_str) or list_prefix.match(line_str) or prev_is_header or not assembled_lines:
                    assembled_lines.append(line_str)
                else:
                    # Continuation line of standard prose or bullet
                    assembled_lines[-1] += " " + line_str
            cleaned_paragraphs.append("\n".join(assembled_lines))

        return "\n\n".join(cleaned_paragraphs)

    @staticmethod
    def _consolidate_whitespace(text: str) -> str:
        """Collapse redundant horizontal spaces and multiple blank lines."""
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text
