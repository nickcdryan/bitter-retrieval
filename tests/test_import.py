#!/usr/bin/env python3
"""
Simple import test for Google Generative AI
"""

print("Testing imports...")

try:
    import google
    print("✅ google package found")
    print(f"   Location: {google.__file__}")
    print(f"   Contents: {dir(google)}")
except ImportError as e:
    print(f"❌ google package: {e}")

try:
    import google.generativeai
    print("✅ google.generativeai found")
    print(f"   Version: {google.generativeai.__version__ if hasattr(google.generativeai, '__version__') else 'unknown'}")
except ImportError as e:
    print(f"❌ google.generativeai: {e}")

try:
    import google.generativeai as genai
    print("✅ import google.generativeai as genai works")
    
    # Test basic functionality
    print("Testing genai.configure...")
    print(f"Available methods: {[m for m in dir(genai) if not m.startswith('_')][:10]}...")
    
except ImportError as e:
    print(f"❌ import google.generativeai as genai: {e}")

print("\nInstalled packages with 'google' in name:")
import subprocess
import sys
result = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True)
for line in result.stdout.split('\n'):
    if 'google' in line.lower():
        print(f"  {line}") 