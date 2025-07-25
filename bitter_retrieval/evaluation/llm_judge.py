"""
LLM-based evaluation utilities
"""

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List


def call_llm(prompt: str, system_instruction: str = None) -> str:
    """Call the Gemini LLM with a prompt and return the response."""
    try:
        import google.generativeai as genai

        # Configure the API
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

        # Create the model
        model = genai.GenerativeModel("gemini-2.0-flash")

        # Combine system instruction with prompt if provided
        if system_instruction:
            full_prompt = f"{system_instruction}\n\n{prompt}"
        else:
            full_prompt = prompt

        # Generate response
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        print(f"Error calling Gemini API: {str(e)}")
        return f"Error: {str(e)}"


def llm_judge_answer(question: str, reference_answer: str, generated_answer: str) -> float:
    """Use Gemini to judge if generated answer matches reference answer"""
    prompt = f"""Question: {question}
Reference Answer: {reference_answer}
Generated Answer: {generated_answer}

Does the Generated Answer correctly answer the question in the same way as the Reference Answer? Consider the answers equivalent if they convey the same core information, even if worded differently. The question is provided for additional context.

Answer with only "YES" or "NO":"""

    system_instruction = "You are an expert evaluator. Compare answers for semantic equivalence and respond with only YES or NO."
    
    response = call_llm(prompt, system_instruction)
    response = response.strip().upper()
    
    # Parse response - look for YES/NO
    if "YES" in response:
        return 1.0
    elif "NO" in response:
        return 0.0
    else:
        # Default to 0 if unclear response
        return 0.0


async def llm_judge_answer_async(question: str, reference_answer: str, generated_answer: str) -> float:
    """Async version of LLM judge for batch processing"""
    prompt = f"""Question: {question}
Reference Answer: {reference_answer}
Generated Answer: {generated_answer}

Does the Generated Answer correctly answer the question in the same way as the Reference Answer? Consider the answers equivalent if they convey the same core information, even if worded differently. The question is provided for additional context.

Answer with only "YES" or "NO":"""

    system_instruction = "You are an expert evaluator. Compare answers for semantic equivalence and respond with only YES or NO."
    
    try:
        response = await call_llm_async(prompt, system_instruction)
        response = response.strip().upper()
        
        # Parse response - look for YES/NO
        if "YES" in response:
            return 1.0
        elif "NO" in response:
            return 0.0
        else:
            return 0.0
    except Exception as e:
        print(f"Async LLM judge error: {e}")
        return 0.0


async def call_llm_async(prompt: str, system_instruction: str = None) -> str:
    """Async version of call_llm for parallel API calls"""
    import google.generativeai as genai
    
    # Configure the API
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    
    # Create the model
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    # Combine system instruction with prompt if provided
    if system_instruction:
        full_prompt = f"{system_instruction}\n\n{prompt}"
    else:
        full_prompt = prompt
    
    # Use asyncio to run the sync function in thread pool
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        response = await loop.run_in_executor(executor, model.generate_content, full_prompt)
        return response.text


async def batch_llm_judge(judge_tasks: List) -> List[float]:
    """Process multiple LLM judge tasks in parallel"""
    results = await asyncio.gather(*judge_tasks, return_exceptions=True)
    return [r if not isinstance(r, Exception) else 0.0 for r in results] 