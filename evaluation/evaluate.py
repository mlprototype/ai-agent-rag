import json
import os
import sys
import asyncio

# Add the project root to sys.path to allow imports from application, domain, etc.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
load_dotenv()
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from application.agents.graph import graph
from infrastructure.retrieval.vector_store import get_vector_store

# Set up LLM as a judge for answer similarity evaluation
evaluator_llm = ChatOpenAI(model="gpt-4o", temperature=0)

EVAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an expert evaluator. Your task is to compare an actual answer to an expected answer. "
               "Rate the similarity/correctness of the actual answer on a scale of 0.0 to 1.0, where 1.0 means perfectly correct/similar. "
               "Output ONLY a single float between 0.0 and 1.0."),
    ("human", "Expected Answer: {expected_answer}\nActual Answer: {actual_answer}\nScore:")
])

evaluator_chain = EVAL_PROMPT | evaluator_llm

def assess_answer_similarity(expected: str, actual: str) -> float:
    try:
        response = evaluator_chain.invoke({"expected_answer": expected, "actual_answer": actual})
        return float(response.content.strip())
    except Exception as e:
        print(f"Error evaluating similarity: {e}")
        return 0.0

def assess_recall_at_k(query: str, expected_snippet: str, k: int = 3) -> bool:
    """Checks if the expected snippet or a significantly matching part of it is in the retrieved docs."""
    vector_store = get_vector_store()
    docs = vector_store.similarity_search(query, k=k)
    
    # We simplify recall check by seeing if a large chunk of expected answer words appears in the document content
    expected_words = set(expected_snippet.lower().split())
    
    # We say recall is positive if at least one doc contains 50% of the significant words
    for doc in docs:
        doc_words = set(doc.page_content.lower().split())
        overlap = len(expected_words.intersection(doc_words))
        if len(expected_words) > 0 and (overlap / len(expected_words)) > 0.5:
            return True
            
    return False

async def run_evaluation():
    dataset_path = os.path.join(os.path.dirname(__file__), "dataset.json")
    with open(dataset_path, "r") as f:
        data = json.load(f)

    total_similarity = 0.0
    total_recall = 0
    num_questions = len(data)

    print(f"Starting evaluation of {num_questions} questions...\n")

    for i, item in enumerate(data):
        question = item["question"]
        expected = item["expected_answer"]

        # 1. Run the agent graph asynchronously because our tools are async now
        inputs = {"messages": [HumanMessage(content=question)]}
        final_message = ""
        async for event in graph.astream(inputs, stream_mode="values"):
            msg = event["messages"][-1]
            
        actual_answer = msg.content
        
        # 2. Evaluate Similarity
        sim_score = assess_answer_similarity(expected, actual_answer)
        total_similarity += sim_score
        
        # 3. Evaluate Recall@k
        recall_hit = assess_recall_at_k(question, expected)
        total_recall += 1 if recall_hit else 0

        print(f"Q{i+1}: {question}")
        print(f"  Similarity: {sim_score:.2f}")
        print(f"  Recall@3: {'Hit' if recall_hit else 'Miss'}\n")

    avg_similarity = total_similarity / num_questions
    recall_rate = total_recall / num_questions

    print("===== EVALUATION RESULTS =====")
    print(f"Average Answer Similarity: {avg_similarity:.2f}")
    print(f"Recall@3 Rate: {recall_rate:.2f}")
    print("==============================")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    asyncio.run(run_evaluation())
