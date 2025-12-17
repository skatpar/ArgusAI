#!/bin/bash
# ArgusAI Startup Script

echo "Starting ArgusAI Fraud Detection Platform..."
echo ""

# Check if we're in the correct directory
if [ ! -f "app.py" ]; then
    echo "Error: app.py not found. Please run this script from the ArgusAI directory."
    exit 1
fi

# Check if required packages are installed
echo "Checking dependencies..."
python -c "import streamlit" 2>/dev/null || {
    echo "Installing dependencies..."
    pip install -q -r requirements.txt
}

echo "✓ Dependencies installed"
echo ""

# Start Streamlit
echo "Starting Streamlit server..."
echo "Access the app at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

streamlit run app.py
