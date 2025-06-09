import os
os.environ["OPENAI_API_KEY"] = ""
# os.environ["OPENAI_API_BASE"] = "https://fast.ominiai.cn/v1"

from ragas import evaluate
from ragas.llms import LangchainLLMWrapper,BaseRagasLLM
# from src.ragas.llms import PromptValue
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from ragas.run_config import RunConfig, add_async_retry, add_retry
from langchain.embeddings import HuggingFaceBgeEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from openai import AsyncOpenAI, OpenAI
from langchain.callbacks.base import Callbacks
from ragas.metrics import ContextRelevance,SemanticSimilarity,AnswerAccuracy
from ragas.metrics import (
    RougeScore,
    StringPresence,
    BleuScore,
    answer_correctness,
    context_recall,
    informativeness_score,
    faithfulness,
    coverage_score
)
from ragas.metrics._string import NonLLMStringSimilarity
from datasets import Dataset
from langchain.schema import LLMResult, Generation
from openai.types.chat import ChatCompletionMessageParam
import json
import aiohttp
import asyncio
import typing as t
from functools import partial
from dataclasses import dataclass


from dataclasses import dataclass
import typing as t


def main():
    model_type = 'graphrag-global'
    key_name = 'context'
    print(model_type)
    llm = ChatOpenAI(model="", base_url="", api_key="")
    # embeddings = OpenAIEmbeddings()
    bge_embeddings = HuggingFaceBgeEmbeddings(model_name = 'bge-large-en-v1.5')
    embedding = LangchainEmbeddingsWrapper(embeddings=bge_embeddings)
    evaluator_llm = LangchainLLMWrapper(llm)

    with open('', 'r') as f:
        file_data = json.load(f)

    
    for question_type in ['type3','type4']:
        questions = [item['question'] for item in file_data[question_type]]
        ground_truths = [item['gold_answer'] for item in file_data[question_type]]
        answers = [item['generated_answer'] for item in file_data[question_type]]

        contexts = []
        if model_type == 'fast-graphrag':
            for item in file_data[question_type]:
                context = []
                for ctx in item['context']:
                    context.append(ctx[0]['content'])
                contexts.append(context)
        else:
            for item in file_data[question_type]:
                if type(item[key_name]) == list:
                    contexts.append(item[key_name])
                else:
                    contexts.append([item[key_name]])

        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        }
        
        dataset = Dataset.from_dict(data)
        bleu_score = BleuScore()
        rouge_score = RougeScore()
        string_presence_score = StringPresence()
        semantic_similarity_score = SemanticSimilarity(embeddings=embedding)
        context_relevancy_score = ContextRelevance(llm=evaluator_llm)
        answer_acc_score = AnswerAccuracy(llm=evaluator_llm)

        # metric = {}
        # metric['type1'] = [rouge_score,answer_correctness]
        # metric['type2'] = [rouge_score,answer_correctness]
        # metric['type3'] = [answer_correctness,coverage_score]
        # metric['type4'] = [answer_correctness,coverage_score,faithfulness]

        metric = {}
        metric['type1'] = [rouge_score,answer_correctness,context_relevancy_score,context_recall]
        metric['type2'] = [rouge_score,answer_correctness,context_relevancy_score,context_recall]
        metric['type3'] = [answer_correctness,coverage_score,context_relevancy_score,context_recall]
        metric['type4'] = [answer_correctness,coverage_score,faithfulness,context_relevancy_score,context_recall]

        # metric = [context_relevancy_score,context_recall]


        result = evaluate(
            llm=evaluator_llm,
            embeddings=embedding,
            dataset = dataset, 
            metrics=metric[question_type],
            # metrics=metric,
        )
        print(f'question_type: {question_type}, result: {result}')
    print('all saved')

if __name__ == "__main__":
    main()