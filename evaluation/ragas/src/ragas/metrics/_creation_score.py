from __future__ import annotations

import asyncio
import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.base import (
    MetricOutputType,
    MetricType,
    MetricWithEmbeddings,
    MetricWithLLM,
    SingleTurnMetric,
)
from ragas.prompt import PydanticPrompt

logger = logging.getLogger(__name__)

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

from dataclasses import dataclass, field
import typing as t
import numpy as np
from ragas.metrics.base import MetricWithLLM, SingleTurnMetric
from langchain_core.prompt_values import StringPromptValue
from ._context_recall import ContextRecallClassificationPrompt, QCA, ContextRecallClassification

import logging
import json
import re

logger = logging.getLogger(__name__)

class QAFacts(BaseModel):
    question: str
    reference_answer: str

class ExtractedFact(BaseModel):
    fact: str

class ExtractedFacts(BaseModel):
    facts: t.List[ExtractedFact]

class ReferenceFactExtractionPrompt(PydanticPrompt[QAFacts, ExtractedFacts]):
    name: str = "reference_fact_extraction"
    instruction: str = (
        "You are given a question and a reference answer. Break down the reference answer into a list of distinct factual statements (facts) that could be independently verified. "
        "Output them as a JSON list of strings under the 'facts' field."
    )
    input_model = QAFacts
    output_model = ExtractedFacts

@dataclass
class CoverageScore(MetricWithLLM, SingleTurnMetric):
    name: str = "coverage_score"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"user_input", "reference", "response"}
        }
    )
    output_type: t.Optional[MetricOutputType] = MetricOutputType.CONTINUOUS

    fact_extraction_prompt: PydanticPrompt = field(
        default_factory=ReferenceFactExtractionPrompt
    )
    fact_coverage_prompt: PydanticPrompt = field(
        default_factory=ContextRecallClassificationPrompt
    )
    max_retries: int = 1

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        row = sample.to_dict()
        return await self._ascore(row, callbacks)

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        assert self.llm is not None, "Set LLM before use."

        # Step 1: Extract facts from reference answer
        fact_response = await self.fact_extraction_prompt.generate(
            data=QAFacts(
                question=row["user_input"],
                reference_answer=row["reference"]
            ),
            llm=self.llm,
            callbacks=callbacks
        )
        facts = [fact.fact for fact in fact_response.facts]

        # Step 2: Check whether each fact is covered in the generated response
        coverage_response = await self.fact_coverage_prompt.generate(
            data=QCA(
                question=row["user_input"],
                context=row["response"],  # now the generation is the "context"
                answer=". ".join(facts),  # facts are checked as if they're statements to be attributed
            ),
            llm=self.llm,
            callbacks=callbacks
        )

        score = self._compute_score(coverage_response.classifications)
        return score

    def _compute_score(self, classifications: t.List[ContextRecallClassification]) -> float:
        response = [1 if item.attributed else 0 for item in classifications]
        denom = len(response)
        numerator = sum(response)
        score = numerator / denom if denom > 0 else np.nan
        return score



@dataclass
class InformativenessScore(MetricWithLLM, SingleTurnMetric):
    """
    Measures how informative the answer is (concrete, useful, and non-trivial).
    Ratings:
        - 4: Highly informative with concrete and relevant details.
        - 2: Partially informative, somewhat vague or generic.
        - 0: Not informative, generic, or just repeating the question.
    """

    name: str = "informativeness_score"

    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"user_input", "response"}
        }
    )

    template = (
        "Instruction: Given a question and a user answer, rate how informative the answer is.\n\n"
        "Informativeness means the answer provides specific, helpful, concrete information that goes beyond repeating the question.\n"
        "- Say 4 if the answer is rich in detail, informative, and goes beyond obvious/general statements.\n"
        "- Say 2 if the answer is somewhat informative but lacks depth or includes vague statements.\n"
        "- Say 0 if the answer is mostly generic, uninformative, or just restates the question.\n\n"
        "Respond ONLY with a number: 0, 2, or 4. No explanations.\n\n"
        "### Question:\n{question}\n\n"
        "### Answer:\n{answer}\n\n"
        "Rating:"
    )

    retry = 3

    def process_score(self, text: str) -> float:
        match = re.search(r"\b([042])\b", text)
        if match:
            return int(match.group(1)) / 4
        return np.nan

    async def _single_turn_ascore(self, sample: SingleTurnSample, callbacks=None) -> float:
        assert self.llm is not None, "LLM is not set"
        assert sample.user_input is not None
        assert sample.response is not None

        prompt = self.template.format(
            question=sample.user_input.strip(),
            answer=sample.response.strip(),
        )

        try:
            for i in range(self.retry):
                formatted_prompt = StringPromptValue(text=prompt)
                result = await self.llm.agenerate_text(
                    formatted_prompt,
                    n=1,
                    temperature=0.2,
                )
                raw = result.generations[0][0].text
                score = self.process_score(raw)
                if not np.isnan(score):
                    return score
                logger.warning(f"Informativeness retry {i}, bad output: {raw}")
        except Exception as e:
            logger.warning(f"Informativeness scoring failed: {e}")

        return np.nan

coverage_score = CoverageScore()
informativeness_score = InformativenessScore()