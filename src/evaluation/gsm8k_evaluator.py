"""Robust GSM8K Answer Extraction and Evaluation.

Implements multi-stage numeric extraction, canonical normalization,
and exact numeric comparison against ground-truth answers.
"""

import math
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class EvaluationResult:
    gold_answer: str
    raw_output: str
    extracted_answer: Optional[str]
    answer_parse_success: bool
    answer_correct: bool


class GSM8KEvaluator:
    """Evaluates generated model outputs against GSM8K ground truth."""

    @staticmethod
    def normalize_number_string(val_str: str) -> Optional[float]:
        """Converts a raw string representation of a number to a float."""
        if not val_str:
            return None
        # Clean currency, spaces, commas, percents, trailing periods
        cleaned = re.sub(r"[,\$€£%]", "", val_str).strip()
        cleaned = cleaned.rstrip(".")
        
        # Handle simple fractions like "3/4"
        if "/" in cleaned and len(cleaned.split("/")) == 2:
            parts = cleaned.split("/")
            try:
                num = float(parts[0].strip())
                denom = float(parts[1].strip())
                if denom != 0:
                    return num / denom
            except ValueError:
                pass

        try:
            return float(cleaned)
        except ValueError:
            return None

    @classmethod
    def extract_answer(cls, raw_output: str) -> Optional[str]:
        """Extracts the predicted final numeric answer from model generation.
        
        Order of precedence:
        1. Explicit '#### [number]' marker.
        2. 'The answer is [number]' or 'Answer: [number]' phrases.
        3. Final standalone number in the generated text.
        """
        if not raw_output or not raw_output.strip():
            return None

        # 1. Check for '####' marker
        if "####" in raw_output:
            after_hash = raw_output.split("####")[-1].strip()
            # Match first numeric token following #### (including commas and currencies)
            match = re.search(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+)?", after_hash)
            if match:
                return re.sub(r"[,\$€£%]", "", match.group(0))

        # 2. Check for explicit answer phrases
        phrase_patterns = [
            r"(?:the\s+final\s+answer\s+is|the\s+answer\s+is|answer\s*[:=])\s*([\$€£]?\s*[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?)",
            r"(?:equals?|total\s+is|result\s*[:=])\s*([\$€£]?\s*[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?)"
        ]
        for pattern in phrase_patterns:
            matches = list(re.finditer(pattern, raw_output, re.IGNORECASE))
            if matches:
                # Pick the last occurrence
                candidate = matches[-1].group(1)
                num_match = re.search(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?", re.sub(r"[,\$€£%]", "", candidate))
                if num_match:
                    return num_match.group(0)

        # 3. Fallback: find all numbers in the text and take the last valid number
        # Ignore things like step numbers e.g. "Step 1:" if followed by content
        numbers = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?", raw_output)
        if numbers:
            last_num = numbers[-1]
            clean_last = re.sub(r"[,\$€£%]", "", last_num)
            return clean_last

        return None

    @classmethod
    def evaluate(cls, raw_output: str, gold_answer: str) -> EvaluationResult:
        """Compares model output to ground-truth answer."""
        gold_norm = cls.normalize_number_string(gold_answer)
        extracted = cls.extract_answer(raw_output)

        if extracted is None:
            return EvaluationResult(
                gold_answer=gold_answer,
                raw_output=raw_output,
                extracted_answer=None,
                answer_parse_success=False,
                answer_correct=False
            )

        pred_norm = cls.normalize_number_string(extracted)
        if pred_norm is None:
            return EvaluationResult(
                gold_answer=gold_answer,
                raw_output=raw_output,
                extracted_answer=extracted,
                answer_parse_success=False,
                answer_correct=False
            )

        # Numerical comparison with small epsilon for floating point representation
        if gold_norm is not None:
            is_correct = math.isclose(pred_norm, gold_norm, rel_tol=1e-5, abs_tol=1e-5)
        else:
            is_correct = extracted.strip() == gold_answer.strip()

        return EvaluationResult(
            gold_answer=gold_answer,
            raw_output=raw_output,
            extracted_answer=extracted,
            answer_parse_success=True,
            answer_correct=is_correct
        )
