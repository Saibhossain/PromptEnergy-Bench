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
    generation_truncated: bool = False


class GSM8KEvaluator:
    """Evaluates generated model outputs against GSM8K ground truth."""

    @staticmethod
    def normalize_number_string(val_str: str) -> Optional[float]:
        """Converts a raw string representation of a number to a float."""
        if not val_str:
            return None
        # Clean currency, spaces, commas, percents, trailing periods
        cleaned = re.sub(r"[,\$€£%]", "", str(val_str)).strip()
        cleaned = cleaned.rstrip(".")
        
        # Handle simple fractions like "3/4" or "1/2"
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
    def extract_answer(
        cls,
        raw_output: str,
        raw_response: Optional[str] = None,
        generation_truncated: bool = False
    ) -> Optional[str]:
        """Extracts the predicted final numeric answer from model generation.
        
        Order of precedence:
        1. Explicit '#### [number]' marker (strictly highest priority, ignores earlier numbers).
        2. LaTeX '\\boxed{[number]}' marker.
        3. Explicit answer phrases ('The answer is [number]').
        4. Final standalone number in non-thinking text (only if NOT truncated).
        """
        if not raw_output or not raw_output.strip():
            return None

        # 1. Check for explicit '####' marker anywhere in the output
        if "####" in raw_output:
            after_hash = raw_output.split("####")[-1].strip()
            # Match first numeric token following ####
            match = re.search(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+)?", after_hash)
            if match:
                return re.sub(r"[,\$€£%]", "", match.group(0))

        # Separate thinking trace if present
        non_thinking = raw_response.strip() if (raw_response and raw_response.strip()) else ""
        if not non_thinking:
            non_thinking = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()
            # In case unclosed <think> tag
            if "<think>" in non_thinking:
                non_thinking = non_thinking.split("<think>")[0].strip()

        # 2. Check for LaTeX \boxed{...} pattern
        boxed_match = re.search(r"\\boxed\{([^}]+)\}", raw_output)
        if boxed_match:
            cand = boxed_match.group(1).strip()
            num_match = re.search(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+)?", re.sub(r"[,\$€£%]", "", cand))
            if num_match:
                return num_match.group(0)

        # 3. Check for explicit answer phrases in non-thinking text first, then raw_output
        search_targets = [non_thinking, raw_output] if non_thinking else [raw_output]
        phrase_patterns = [
            r"(?:the\s+final\s+answer\s+is|the\s+answer\s+is|answer\s*[:=])\s*([\$€£]?\s*[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+)?)",
            r"(?:equals?|total\s+is|result\s*[:=])\s*([\$€£]?\s*[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+)?)"
        ]
        for target in search_targets:
            if not target:
                continue
            for pattern in phrase_patterns:
                matches = list(re.finditer(pattern, target, re.IGNORECASE))
                if matches:
                    candidate = matches[-1].group(1)
                    num_match = re.search(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+)?", re.sub(r"[,\$€£%]", "", candidate))
                    if num_match:
                        return num_match.group(0)

        # If generation was truncated before an explicit answer, do NOT pull random numbers from unfinished text
        if generation_truncated:
            return None

        # 4. Fallback: find the last standalone number in non-thinking text, or raw output
        fallback_target = non_thinking if non_thinking else raw_output
        numbers = re.findall(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+)?", fallback_target)
        if numbers:
            last_num = numbers[-1]
            return re.sub(r"[,\$€£%]", "", last_num)

        return None

    @classmethod
    def evaluate(
        cls,
        raw_output: str,
        gold_answer: str,
        raw_response: Optional[str] = None,
        generation_truncated: bool = False
    ) -> EvaluationResult:
        """Compares model output to ground-truth answer with truncation handling."""
        gold_norm = cls.normalize_number_string(gold_answer)
        extracted = cls.extract_answer(
            raw_output=raw_output,
            raw_response=raw_response,
            generation_truncated=generation_truncated
        )

        if extracted is None:
            return EvaluationResult(
                gold_answer=gold_answer,
                raw_output=raw_output,
                extracted_answer=None,
                answer_parse_success=False,
                answer_correct=False,
                generation_truncated=generation_truncated
            )

        pred_norm = cls.normalize_number_string(extracted)
        if pred_norm is None:
            return EvaluationResult(
                gold_answer=gold_answer,
                raw_output=raw_output,
                extracted_answer=extracted,
                answer_parse_success=False,
                answer_correct=False,
                generation_truncated=generation_truncated
            )

        # Numerical comparison with small epsilon for floating point representation
        if gold_norm is not None:
            is_correct = math.isclose(pred_norm, gold_norm, rel_tol=1e-5, abs_tol=1e-5)
        else:
            is_correct = extracted.strip() == str(gold_answer).strip()

        return EvaluationResult(
            gold_answer=gold_answer,
            raw_output=raw_output,
            extracted_answer=extracted,
            answer_parse_success=True,
            answer_correct=is_correct,
            generation_truncated=generation_truncated
        )
