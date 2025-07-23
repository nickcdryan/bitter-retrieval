#!/usr/bin/env python3
"""
Alternative LLM judge using OpenAI API
"""

import os

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def call_openai_llm(prompt, system_instruction=None):
    """Alternative LLM judge using OpenAI"""
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Cheap and fast
            messages=messages,
            max_tokens=10,
            temperature=0
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error calling OpenAI API: {str(e)}")
        return f"Error: {str(e)}"

def test_openai_judge():
    """Test OpenAI as LLM judge"""
    print("🧪 Testing OpenAI LLM Judge...")
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        print(f"✅ OpenAI API key found: {api_key[:8]}...")
    else:
        print("❌ No OpenAI API key found")
        print("Set it with: export OPENAI_API_KEY='your_key_here'")
        return
    
    # Test LLM Judge
    question = "What is the capital of France?"
    reference_answer = "Paris"
    generated_answer = "The capital of France is Paris."
    
    judge_prompt = f"""Question: {question}
Reference Answer: {reference_answer}
Generated Answer: {generated_answer}

Does the Generated Answer correctly answer the question in the same way as the Reference Answer? Consider the answers equivalent if they convey the same core information, even if worded differently.

Answer with only "YES" or "NO":"""

    system_instruction = "You are an expert evaluator. Compare answers for semantic equivalence and respond with only YES or NO."
    
    response = call_openai_llm(judge_prompt, system_instruction)
    print(f"Response: '{response}'")
    
    # Parse the response
    response_clean = response.strip().upper()
    if "YES" in response_clean:
        score = 1.0
        print(f"✅ Parsed score: {score}")
    elif "NO" in response_clean:
        score = 0.0
        print(f"❌ Parsed score: {score}")
    else:
        score = 0.0
        print(f"⚠️  Unclear response, defaulted to: {score}")

if __name__ == "__main__":
    test_openai_judge() 