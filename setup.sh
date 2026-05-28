#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "=== 🚀 Starting LangChain to LM Studio Bridge Setup ==="

# -----------------------------------------------------------
# 1. Python Backend Dependencies Setup
# -----------------------------------------------------------
echo ""
echo "=== 🐍 Setting up Python Backend Dependencies ==="

# Check if running inside a Conda environment, otherwise create a python venv
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "Detected active Conda environment: $CONDA_DEFAULT_ENV"
    echo "Installing Python packages via pip inside Conda..."
else
    echo "No Conda detected. Creating a standard Python virtual environment (.venv)..."
    python3 -m venv .venv
    source .venv/bin/activate
fi

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing FastAPI, Uvicorn, and LangChain ecosystem..."
pip install fastapi uvicorn pydantic langchain-core langchain-openai httpx

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