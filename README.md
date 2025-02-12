# Text-minator: AI Plagiarism Detector

## Overview
**Text-minator** is an AI-powered plagiarism detection tool that analyzes text using multiple machine learning models, including **GPT-2, DeBERTa, T5, and BERT**. The application runs on **Streamlit**, providing an interactive web-based interface for analysis.

## Features
- Supports **GPT-2, DeBERTa, T5, and BERT** for text analysis.
- Detects AI-generated content using **perplexity, entropy, semantic coherence, and burstiness**.
- Accepts **text input and file uploads** (PDF and TXT formats).
- Displays detection results with visualizations.
- Maintains a **chat history** of previous analyses.

---

## Installation Guide

### Installing Python (For First-Time Users)
Ensure you have Python installed on your system. If not, follow these steps:

#### Windows
1. Download Python from [python.org](https://www.python.org/downloads/).
2. Run the installer and **check the box** for **"Add Python to PATH"** before installing.
3. Open a command prompt and verify the installation:
   ```sh
   python --version

### macOS/Linux
1. Open a terminal and run:
   ```sh
   sudo apt update && sudo apt install python3 python3-pip -y  # Ubuntu/Debian
   brew install python  # macOS (using Homebrew)
2. Verify installation:
   ```sh
   python3 --version

---

### Running Locally (Using Visual Studio Code)
1. Clone the Repository
   ```sh
   git clone https://github.com/your-repo/Text-minator.git
   cd Text-minator
2. Create a Virtual Environment (Recommended)
   ```sh
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate  # Windows
3. Install Dependencies
   ```sh
   pip install -r requirements.txt
4. Run the Streamlit Application
   ```sh
   streamlit run main.py

---

### Running on Google Colab
1. Open Google Colab
Go to [Google Colab](https://colab.research.google.com/).
2. Install Dependencies
Run the following in a Colab cell:
   ```python
   !pip install -r requirements.txt
3. Upload Files
Upload **main.py**, **plagiarism_detector.py**, **Chat_History.py**, and **requirements.txt** manually using:
   ```python
   from google.colab import files
   uploaded = files.upload()
4. Run the Application
Since Streamlit doesn't run natively on Google Colab, use:
   ```python
   !streamlit run main.py & npx localtunnel --port 8501
Follow the LocalTunnel link to access the Streamlit interface.

---

### File Structure
   ```sh
   Text-minator/
   │── main.py              # Main Streamlit app entry point
   │── plagiarism_detector.py  # AI models for text analysis
   │── Chat_History.py      # Chat history display
   │── requirements.txt     # Python dependencies
   └── README.md            # Documentation
