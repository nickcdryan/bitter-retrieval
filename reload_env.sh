#!/bin/bash
# Quick script to reload shell environment and verify Poetry installation

echo "🔄 Reloading shell environment..."

# Source bashrc to pick up PATH changes
source ~/.bashrc

# Verify Poetry is now available
if command -v poetry >/dev/null 2>&1; then
    echo "✅ Poetry is now available: $(poetry --version)"
    echo "🎯 You can now run: poetry run python setup_env.py"
else
    echo "❌ Poetry still not found. Try one of these:"
    echo "1. Open a new terminal"
    echo "2. Run: export PATH=\"\$HOME/.local/bin:\$PATH\""
    echo "3. Check if Poetry was installed correctly"
fi 