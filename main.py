# main.py

import streamlit as st
from streamlit_option_menu import option_menu
import Chat_History
import plagiarism_detector
import fitz  # PyMuPDF for PDF processing
import time

st.set_page_config(layout="wide", page_title="Text-minator")

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if 'model' not in st.session_state:
    st.session_state['model'] = 'GPT-J'

if 'uploaded_text' not in st.session_state:
    st.session_state['uploaded_text'] = ""

def read_pdf(file, progress_bar):
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    num_pages = pdf_document.page_count
    for page_num in range(num_pages):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
        progress_bar.progress((page_num + 1) / num_pages)
    return text

def read_txt(file, progress_bar):
    total_size = file.size
    chunk_size = 1024
    text = ""
    bytes_read = 0

    while bytes_read < total_size:
        chunk = file.read(chunk_size)
        text += chunk.decode("utf-8")
        bytes_read += len(chunk)
        progress_bar.progress(min(bytes_read / total_size, 1.0))

    return text

def main():
    st.title("Text-minator: AI Plagiarism Detector")
    st.write("Select model for analysis:")

    model = option_menu(
        menu_title=None,
        options=["GPT-J", "DeBERTa", "T5", "BERT"],
        icons=["robot", "book", "bar-chart-line", "graph-up"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal"
    )
    st.session_state['model'] = model

    text_area = st.text_area("Enter text", st.session_state.get('uploaded_text', ""), label_visibility="collapsed", height=170)
    uploaded_file = st.file_uploader("Upload .PDF or .TXT file", type=["txt", "pdf"])

    if uploaded_file is not None:
        progress_bar = st.progress(0)
        if uploaded_file.type == "text/plain":
            uploaded_text = read_txt(uploaded_file, progress_bar)
        elif uploaded_file.type == "application/pdf":
            uploaded_text = read_pdf(uploaded_file, progress_bar)
        progress_bar.empty()
        st.session_state['uploaded_text'] = uploaded_text
        text_area = uploaded_text  # Update text area with uploaded text

    if (text_area.strip() or 'uploaded_text' in st.session_state) and st.button("Analyze"):
        text_to_analyze = text_area.strip()
        result_entry = {'text': text_to_analyze, 'model': st.session_state['model']}

        if st.session_state['model'] == 'GPT-J':
            result = plagiarism_detector.display_results_gptj(text_to_analyze)
            if result is not None:  # Ensure result is not None before unpacking
                perplexity, burstiness_score = result
                result_entry.update({'perplexity': perplexity, 'burstiness_score': burstiness_score})
                result_entry['result_label'] = 1 if perplexity > 8000 and burstiness_score > 0.2 else 0

        elif st.session_state['model'] == 'DeBERTa':
            result = plagiarism_detector.display_results_deberta(text_to_analyze)
            if result is not None:
                entropy, semantic_coherence = result
                result_entry.update({'entropy': entropy, 'semantic_coherence': semantic_coherence})
                result_entry['result_label'] = 1 if entropy > 4.5 and semantic_coherence < 0.6 else 0

        elif st.session_state['model'] == 'T5':
            result = plagiarism_detector.display_results_t5(text_to_analyze)
            if result is not None:
                perplexity = result
                result_entry.update({'perplexity': perplexity})
                result_entry['result_label'] = 1 if perplexity > 150 else 0

        elif st.session_state['model'] == 'BERT':
            result = plagiarism_detector.display_results_bert(text_to_analyze)
            if result is not None:
                entropy = result
                result_entry.update({'entropy': entropy})
                result_entry['result_label'] = 1 if entropy > 3.8 else 0

        st.session_state['chat_history'].append(result_entry)
        st.success(f"Analysis complete")

class MultiApp:
    def run(self):
        with st.sidebar:
            app = option_menu(
                menu_title="Main Menu",
                options=["Home", "Chat History"],
                icons=["house", "chat-square-text-fill"],
                menu_icon="three-dots",
                default_index=0
            )

        if app == "Home":
            main()
        elif app == "Chat History":
            Chat_History.app()

if __name__ == "__main__":
    app = MultiApp()
    app.run()
