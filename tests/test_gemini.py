#!/usr/bin/env python3
"""
Test script for Gemini API
"""

import os

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded .env file")
except ImportError:
    print("python-dotenv not installed - using system environment variables only")

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response."""
    try:
        import google.generativeai as genai
        import os

        # Configure the API
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

        # Create the model
        model = genai.GenerativeModel("gemini-2.0-flash-exp")

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


def test_gemini_api():
    """Test the Gemini API with a simple prompt"""
    print("🧪 Testing Gemini API...")
    
    # Check if API key is available
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        print(f"✅ Gemini API key found: {api_key[:8]}...")
    else:
        print("❌ No Gemini API key found")
        print("Set it with: export GEMINI_API_KEY='your_key_here'")
        return
    
    # Test 1: Simple prompt
    print("\n--- Test 1: Simple prompt ---")
    prompt = "What is 2 + 2?"
    response = call_llm(prompt)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    
    # Test 2: With system instruction
    print("\n--- Test 2: With system instruction ---")
    prompt = "What is the capital of France?"
    system_instruction = "You are a helpful assistant. Answer concisely."
    response = call_llm(prompt, system_instruction)
    print(f"System: {system_instruction}")
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    
    # Test 3: LLM Judge test (similar to your use case)
    print("\n--- Test 3: LLM Judge simulation ---")
    question = "What is the capital of France?"
    reference_answer = "Paris"
    generated_answer = "The capital of France is Paris."
    
    judge_prompt = f"""Question: {question}
Reference Answer: {reference_answer}
Generated Answer: {generated_answer}

Does the Generated Answer correctly answer the question in the same way as the Reference Answer? Consider the answers equivalent if they convey the same core information, even if worded differently.

Answer with only "YES" or "NO":"""

    system_instruction = "You are an expert evaluator. Compare answers for semantic equivalence and respond with only YES or NO."
    
    response = call_llm(judge_prompt, system_instruction)
    print(f"Judge prompt: {judge_prompt[:100]}...")
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
    test_gemini_api() 