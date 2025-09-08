#!/bin/bash

# Highlight Detector Setup Script
# This script sets up the development environment

set -e

echo "Setting up Highlight Detector..."

# Check prerequisites
echo "Checking prerequisites..."

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "ERROR: Node.js is not installed. Please install Node.js 18+ and try again."
    exit 1
fi

NODE_VERSION=$(node --version | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    echo "ERROR: Node.js version 18+ is required. Current version: $(node --version)"
    exit 1
fi

echo "SUCCESS: Node.js $(node --version) found"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed. Please install Python 3.9+ and try again."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "SUCCESS: Python $PYTHON_VERSION found"

# Check FFmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "ERROR: FFmpeg is not installed. Please install FFmpeg and try again."
    echo "   macOS: brew install ffmpeg"
    echo "   Ubuntu/Debian: sudo apt install ffmpeg"
    echo "   Windows: Download from https://ffmpeg.org/download.html"
    exit 1
fi

echo "SUCCESS: FFmpeg $(ffmpeg -version | head -n1 | cut -d' ' -f3) found"

# Create necessary directories
echo "Creating directories..."
mkdir -p temp cache output logs models templates data/samples data/fixtures

# Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
npm install

# Install Python dependencies
echo "Installing Python dependencies..."
cd apps/server
pip install -r requirements.txt
cd ../..

# Set up environment file
if [ ! -f .env ]; then
    echo "Creating environment file..."
    cp env.example .env
    echo "SUCCESS: Environment file created. Please review and edit .env as needed."
else
    echo "SUCCESS: Environment file already exists."
fi

# Set up Python path
echo "Setting up Python path..."
export PYTHONPATH="${PYTHONPATH}:$(pwd)/packages"

# Create a simple test script
echo "Creating test script..."
cat > test_setup.py << 'EOF'
#!/usr/bin/env python3
"""Test script to verify setup."""

import sys
import importlib

def test_imports():
    """Test that all required packages can be imported."""
    packages = [
        'fastapi',
        'uvicorn',
        'numpy',
        'opencv-python',
        'librosa',
        'torch',
        'pandas',
        'pyyaml'
    ]
    
    failed = []
    for package in packages:
        try:
            # Handle package name differences
            module_name = package.replace('-', '_')
            if package == 'opencv-python':
                module_name = 'cv2'
            elif package == 'pyyaml':
                module_name = 'yaml'
            
            importlib.import_module(module_name)
            print(f"SUCCESS: {package}")
        except ImportError:
            print(f"ERROR: {package}")
            failed.append(package)
    
    if failed:
        print(f"\nERROR: Failed to import: {', '.join(failed)}")
        print("Please install missing packages with: pip install -r apps/server/requirements.txt")
        return False
    else:
        print("\nSUCCESS: All packages imported successfully!")
        return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
EOF

# Run test
echo "Testing Python setup..."
python3 test_setup.py

if [ $? -eq 0 ]; then
    echo ""
    echo "SUCCESS: Setup completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Review and edit .env file if needed"
    echo "2. Review and edit config.yaml if needed"
    echo "3. Start the development servers:"
    echo "   npm run dev"
    echo ""
    echo "Or start them separately:"
    echo "   Backend:  cd apps/server && python main.py"
    echo "   Frontend: cd apps/web && npm run dev"
    echo ""
    echo "Access the application at:"
    echo "   Frontend: http://localhost:3000"
    echo "   Backend:  http://localhost:8000"
    echo "   API Docs: http://localhost:8000/docs"
else
    echo ""
    echo "ERROR: Setup completed with errors. Please check the output above."
    exit 1
fi

# Clean up test script
rm test_setup.py
