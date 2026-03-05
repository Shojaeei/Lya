#!/bin/bash
# Lya Brain Check Script
# Usage: ./check_brain.sh

echo "Checking Lya's Brain..."

# Check Ollama
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "Ollama not running. Starting..."
    ollama serve &
    sleep 5
fi

# Check models - look for kimi-k2.5:cloud specifically
MODELS=$(ollama list | grep "kimi-k2.5")
if [ -z "$MODELS" ]; then
    echo "kimi-k2.5:cloud not found. Pulling..."
    ollama pull kimi-k2.5:cloud
fi

echo "Brain check complete!"
echo "Models available:"
ollama list | grep -E "(NAME|kimi|gemma|deepseek)"

# Quick test
echo ""
echo "Testing kimi-k2.5:cloud..."
curl -s http://localhost:11434/api/generate -d '{
  "model": "kimi-k2.5:cloud",
  "prompt": "Hi",
  "stream": false
}' | grep -o '"response":"[^"]*"' | head -1
