from typing import List, Callable, Awaitable
import numpy as np
from langchain_core.language_models import BaseLanguageModel
from langchain_core.callbacks import Callbacks
import re

CONTEXT_RELEVANCE_PROMPT = """
### Instructions
You are a world class expert designed to evaluate the relevance score of a Context in order to answer the Question.
Your task is to determine if the Context contains proper information to answer the Question.
Do not rely on your previous knowledge about the Question.
Use only what is written in the Context and in the Question.
Follow the instructions below:
0. If the context does not contains any relevant information to answer the question, say 0.
1. If the context partially contains relevant information to answer the question, say 1.
2. If the context contains any relevant information to answer the question, say 2.
You must provide the relevance score of 0, 1, or 2, nothing else.
Do not explain.
Question: {query}
Context: {context}
Do not try to explain.
Analyzing Context and Question, the Relevance score is
"""

CONTEXT_RELEVANCE_JUDGE_BY_EVIDENCE_PROMPT = """
### Task
You are given a Context, a Question, and a list of Evidence. Your task is to evaluate how relevant the Context is to the Question and Evidence.

Score based on the following criteria:
- **2 (Highly Relevant)**: The Context directly answers the Question or is essential for understanding the Evidence.
- **1 (Partially Relevant)**: The Context provides related background information but does not directly answer the Question, or is only tangentially related.
- **0 (Not Relevant)**: The Context is about a completely different topic.

Respond ONLY with a JSON object containing:
- "reason": A brief explanation (1 sentence) for your score.
- "relevance_score": Your relevance score (0, 1, or 2).

### Example
Input:
Context: "The capital of Australia is Canberra, a planned city located between Sydney and Melbourne."
Evidence: ["Canberra is the capital of Australia"]
Question: "What is the capital of Australia?"

Output:
{{
  "reason": "The context directly confirms that Canberra is the capital of Australia, which is the core of the question and evidence.",
  "score": 2
}}

### Actual Input
Context: "{context}"

Evidence: {evidence}

Question: "{question}"

### Your Response:
"""
 
async def compute_context_relevance(
    question: str,
    contexts: List[str],
    llm: BaseLanguageModel,
    callbacks: Callbacks = None,
    max_retries: int = 3
) -> float:
    """
    Evaluate the relevance of retrieved contexts for answering a question.
    Returns a score between 0.0 (irrelevant) and 1.0 (fully relevant).
    """
    # Handle edge cases
    if not question.strip() or not contexts or not any(c.strip() for c in contexts):
        return 0.0
    
    context_str = "\n".join(contexts)[:20000]  # Truncate long contexts
    
    # Check for exact matches (often indicate low relevance)
    if context_str.strip() == question.strip() or context_str.strip() in question:
        return 0.0
    
    # Get two independent ratings from LLM
    rating1 = await _get_llm_rating(question, context_str, llm, callbacks, max_retries)
    rating2 = await _get_llm_rating(question, context_str, llm, callbacks, max_retries)
    
    # Process ratings (0-2 scale) and convert to 0-1 scale
    scores = [r/2 for r in [rating1, rating2] if r is not None]
    
    # Calculate final score
    if not scores:
        return np.nan
    return sum(scores) / len(scores)  # Average of valid scores

async def _get_llm_rating(
    question: str,
    context: str,
    llm: BaseLanguageModel,
    callbacks: Callbacks,
    max_retries: int
) -> float:
    """Get a single relevance rating from LLM with retries"""
    prompt = CONTEXT_RELEVANCE_PROMPT.format(question=question, context=context)
    
    for _ in range(max_retries):
        try:
            response = await llm.ainvoke(prompt, config={"callbacks": callbacks})
            content = re.sub(r"```json|```", "", response.content).strip()
            return _parse_rating(content)
        except Exception:
            continue
    return None  # Return None after max retries

def _parse_rating(text: str) -> float:
    """Extract rating from LLM response"""
    # Look for first number 0-2 in the response
    for token in text.split()[:8]:  # Check first 8 tokens
        if token.isdigit() and 0 <= int(token) <= 2:
            return float(token)
    return None  # No valid rating found