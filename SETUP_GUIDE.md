# Quick Setup Guide

## Step 1: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

## Step 2: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Step 3: Install Package in Development Mode (Optional)

```bash
pip install -e .
```

## Step 4: Verify Installation

```bash
# Test that imports work
python -c "import src; print('Package imported successfully')"

# Run tests (once tests are implemented)
pytest tests/
```

## Step 5: Initialize Git (Already Done)

Git repository has been initialized. You can now commit your changes:

```bash
git add .
git commit -m "Initial project setup - Step 1.1 complete"
```

## Notes

- The logging system is configured in `src/__init__.py` and will create log files in the `logs/` directory
- Cache directory is set up at `data/cache/` for storing yfinance data
- All module placeholders are in place and ready for implementation
