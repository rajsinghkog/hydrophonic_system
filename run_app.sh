#!/bin/bash

# Activate virtual environment and run Streamlit app
cd "$(dirname "$0")"
source venv/bin/activate
streamlit run app.py

