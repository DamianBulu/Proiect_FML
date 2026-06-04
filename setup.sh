#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "=== 🚀 Starting LangChain to LM Studio Bridge Setup ==="

# -----------------------------------------------------------
# SET YOUR PREFERRED PYTHON VERSION HERE
# -----------------------------------------------------------
# Change this to "python3.10", "python3.11", or the full path
# like "/opt/homebrew/bin/python3.11"
PYTHON_EXE="python3.12"

# -----------------------------------------------------------
# 1. Python Backend Dependencies Setup
# -----------------------------------------------------------
echo ""
echo "=== 🐍 Setting up Python Backend Dependencies ==="

# Check if the requested Python version actually exists
if ! command -v $PYTHON_EXE &> /dev/null; then
    echo "❌ Error: $PYTHON_EXE was not found on your system."
    echo "Please install it via Homebrew (brew install python@3.11) or specify a different path."
    exit 1
fi

# Check if running inside a Conda environment, otherwise create a python venv
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "Detected active Conda environment: $CONDA_DEFAULT_ENV"
    echo "Installing Python packages via pip inside Conda..."
else
    echo "Creating virtual environment using: $($PYTHON_EXE --version)"
    # Delete old .venv if it exists to avoid version conflicts
    rm -rf .venv
    $PYTHON_EXE -m venv .venv
    source .venv/bin/activate
fi

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing FastAPI, Uvicorn, and LangChain ecosystem..."
pip install fastapi uvicorn pydantic langchain-core langchain-openai langgraph httpx chromadb requests langchain-community

# -----------------------------------------------------------
# 2. LM Studio Plugin Dependencies Setup
# -----------------------------------------------------------
echo ""
echo "=== 📦 Setting up LM Studio Plugin Dependencies ==="

# Navigate to your plugin folder
if [ -d "my-langchain-bridge" ]; then
    cd my-langchain-bridge
    echo "Entered my-langchain-bridge directory..."
else
    echo "❌ Error: 'my-langchain-bridge' folder not found in current directory."
    echo "Please run this script from the parent folder of your plugin."
    exit 1
fi

# Clean up any bad dependency caching artifacts if they exist
if [ -d "node_modules" ]; then
    echo "Cleaning existing node_modules to avoid version conflicts..."
    rm -rf node_modules package-lock.json
fi

echo "Installing Node.js dependencies and the latest LM Studio SDK..."
npm install

# Force update the SDK locally to guarantee version alignment
npm install @lmstudio/sdk@latest

echo ""
echo "=== 🎉 Setup Complete! ==="
echo "To run your architecture:"
echo "1. Start LM Studio Local Server (Port 1234)"
echo "2. Run Python: python main.py (from your Python directory)"
echo "3. Run Plugin: npx -p @lmstudio/sdk@latest lms dev (inside my-langchain-bridge)"