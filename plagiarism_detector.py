# plagiarism_detector.py

import nltk
import streamlit as st
import torch
from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel,
    DebertaV2Tokenizer, DebertaV2ForMaskedLM,
    T5Tokenizer, T5ForConditionalGeneration,
    BertTokenizer, BertModel
)
from nltk.probability import FreqDist
from sklearn.metrics.pairwise import cosine_similarity
import string
import plotly.graph_objects as go
import re
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')

# Load Models
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')

deberta_tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v2-xlarge')
deberta_model = DebertaV2ForMaskedLM.from_pretrained('microsoft/deberta-v2-xlarge', output_hidden_states=True)

t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Utility Function
def clean_text(text):
    text = text.replace('\n', ' ').replace('\r', '')
    text = ' '.join(text.split())
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[^\w\s]', '', text)
    return text

# GPT-2: Perplexity Calculation
def calculate_perplexity_gpt2(text, max_chunk_length=512):
    text = clean_text(text)
    tokens = gpt2_tokenizer.encode(text, add_special_tokens=False)
    n_chunks = len(tokens) // max_chunk_length + (1 if len(tokens) % max_chunk_length != 0 else 0)

    total_loss = 0.0
    total_tokens = 0

    for i in range(n_chunks):
        chunk_tokens = tokens[i * max_chunk_length:(i + 1) * max_chunk_length]
        chunk_tokens = chunk_tokens[:gpt2_model.config.max_position_embeddings]

        input_ids = torch.tensor([chunk_tokens])

        with torch.no_grad():
            outputs = gpt2_model(input_ids)
            logits = outputs.logits

        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), input_ids.view(-1), reduction='sum')
        total_loss += loss.item()
        total_tokens += input_ids.size(1)

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))

    return perplexity.item()

# Burstiness Calculation
def calculate_burstiness(text):
    text = clean_text(text)
    tokens = nltk.word_tokenize(text.lower())
    word_freq = FreqDist(tokens)
    repeated_count = sum(count > 1 for count in word_freq.values())
    burstiness_score = repeated_count / len(word_freq) if len(word_freq) > 0 else 0
    return burstiness_score

# DeBERTa: Entropy Calculation
def calculate_entropy_deberta(text):
    text = clean_text(text)
    tokens = deberta_tokenizer.tokenize(text)
    token_ids = deberta_tokenizer.convert_tokens_to_ids(tokens)

    if len(token_ids) > deberta_model.config.max_position_embeddings:
        token_ids = token_ids[:deberta_model.config.max_position_embeddings]

    input_ids = torch.tensor([token_ids])
    with torch.no_grad():
        outputs = deberta_model(input_ids)
        predictions = outputs.logits

    probs = torch.nn.functional.softmax(predictions, dim=-1)
    log_probs = torch.nn.functional.log_softmax(predictions, dim=-1)

    entropy = -torch.sum(probs * log_probs, dim=-1).mean().item()
    return entropy

# DeBERTa: Semantic Coherence Calculation
def calculate_semantic_coherence_deberta(text):
    text = clean_text(text)
    sentences = nltk.sent_tokenize(text)
    if len(sentences) < 2:
        return 1.0

    encoded_sentences = [deberta_tokenizer(sentence, return_tensors='pt') for sentence in sentences]
    sentence_embeddings = [deberta_model(**encoded_sentence).last_hidden_state.mean(dim=1) for encoded_sentence in encoded_sentences]

    similarities = []
    for i in range(len(sentence_embeddings) - 1):
        for j in range(i + 1, len(sentence_embeddings)):
            cosine_sim = torch.nn.functional.cosine_similarity(sentence_embeddings[i], sentence_embeddings[j])
            similarities.append(cosine_sim.item())

    coherence = sum(similarities) / len(similarities)
    return coherence

# T5: Perplexity Calculation
def calculate_perplexity_t5(text):
    text = clean_text(text)
    input_ids = t5_tokenizer.encode(text, return_tensors='pt')
    with torch.no_grad():
        outputs = t5_model(input_ids, labels=input_ids)
        loss = outputs.loss
    perplexity = torch.exp(loss).item()
    return perplexity

# BERT: Entropy Calculation
def calculate_entropy_bert(text):
    text = clean_text(text)
    tokens = bert_tokenizer.tokenize(text)
    token_ids = bert_tokenizer.convert_tokens_to_ids(tokens)

    if len(token_ids) > bert_model.config.max_position_embeddings:
        token_ids = token_ids[:bert_model.config.max_position_embeddings]

    input_ids = torch.tensor([token_ids])
    with torch.no_grad():
        outputs = bert_model(input_ids)
        predictions = outputs.last_hidden_state

    probs = torch.nn.functional.softmax(predictions, dim=-1)
    log_probs = torch.nn.functional.log_softmax(predictions, dim=-1)

    entropy = -torch.sum(probs * log_probs, dim=-1).mean().item()
    return entropy

# Donut Chart
def plot_probability_donut_chart(score, is_ai_generated):
    # Ensure score remains within valid probability range
    score = max(0.0, min(score, 1.0))

    # Handle edge cases where the score is too small or too large
    if score < 0.01:  # If the score is too small, set it to a minimum value
        score = 0.01
    elif score > 0.99:  # If the score is too large, set it to a maximum value
        score = 0.99

    # Adjust the probability based on the classification result
    if is_ai_generated:
        # If AI-generated, set AI probability to a high value (e.g., 80% - 100%)
        ai_probability = 0.8 + (score * 0.2)  # AI probability ranges from 80% to 100%
        human_probability = 1 - ai_probability
    else:
        # If human-generated, set human probability to a high value (e.g., 80% - 100%)
        human_probability = 0.8 + (score * 0.2)  # Human probability ranges from 80% to 100%
        ai_probability = 1 - human_probability

    # Keep the correct ordering so human-generated content appears correctly
    values = [human_probability, ai_probability]  # Human first, AI second
    colors = ['#008000', '#ff0000']  # Green for Human, Red for AI
    labels = ["Human Generated", "AI Generated"]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        textinfo='percent',  # Show percentage only
        marker=dict(colors=colors),
        sort=False  # Prevent automatic sorting that messes up ordering
    )])

    # Adjust layout to properly display the chart
    fig.update_layout(
        showlegend=True,
        margin=dict(l=30, r=30, t=30, b=30),
    )

    st.plotly_chart(fig, use_container_width=True)

# Display Functions
def display_results_gpt2(text):
    col1, col2, col3 = st.columns([1, 1, 1])
    perplexity = calculate_perplexity_gpt2(text)
    burstiness_score = calculate_burstiness(text)

    # Determine if the content is AI-generated based on thresholds
    is_ai_generated = perplexity > 8000 and burstiness_score > 0.2

    # Normalize score for donut chart (Adjust scaling factor if needed)
    # Use a sigmoid function to map the perplexity to a probability-like score
    score = 1 / (1 + np.exp(-(perplexity - 8000) / 1000))  # Sigmoid function for more realistic probability

    # Ensure the score is within a reasonable range
    score = max(0.01, min(score, 0.99))  # Clamp score between 0.01 and 0.99

    with col1:
        st.info("Basic Details")
        st.metric(label="Word Count", value=len(text.split()))
        plot_probability_donut_chart(score, is_ai_generated)  # Pass probability and classification result

    with col2:
        st.info("Detection Score")
        st.write("Perplexity:", perplexity)
        st.write("Burstiness Score:", burstiness_score)

        if is_ai_generated:
            st.error("Text Analysis Result: AI Generated Content")
        else:
            st.success("Text Analysis Result: Human Generated Content")

    with col3:
        st.info("Original Text")
        st.markdown(text)

    return perplexity, burstiness_score

def display_results_deberta(text):
    col1, col2, col3 = st.columns([1, 1, 1])
    entropy = calculate_entropy_deberta(text)
    coherence = calculate_semantic_coherence_deberta(text)

    # Determine if the content is AI-generated based on thresholds
    is_ai_generated = entropy > 4.5 and coherence < 0.6

    # Normalize score for donut chart
    # Use a sigmoid function to map the entropy to a probability-like score
    score = 1 / (1 + np.exp(-(entropy - 4.5) / 0.5))  # Sigmoid function for more realistic probability

    # Ensure the score is within a reasonable range
    score = max(0.01, min(score, 0.99))  # Clamp score between 0.01 and 0.99

    with col1:
        st.info("Basic Details")
        st.metric(label="Word Count", value=len(text.split()))
        plot_probability_donut_chart(score, is_ai_generated)

    with col2:
        st.info("Detection Score")
        st.write("Entropy:", entropy)
        st.write("Semantic Coherence:", coherence)

        if is_ai_generated:
            st.error("Text Analysis Result: AI Generated Content")
        else:
            st.success("Text Analysis Result: Human Generated Content")

    with col3:
        st.info("Original Text")
        st.markdown(text)

    return entropy, coherence

def display_results_t5(text):
    col1, col2, col3 = st.columns([1, 1, 1])
    perplexity = calculate_perplexity_t5(text)

    # Determine if the content is AI-generated based on thresholds
    is_ai_generated = perplexity > 150

    # Normalize score for donut chart
    # Use a sigmoid function to map the perplexity to a probability-like score
    score = 1 / (1 + np.exp(-(perplexity - 150) / 20))  # Sigmoid function for more realistic probability

    # Ensure the score is within a reasonable range
    score = max(0.01, min(score, 0.99))  # Clamp score between 0.01 and 0.99

    with col1:
        st.info("Basic Details")
        st.metric(label="Word Count", value=len(text.split()))
        plot_probability_donut_chart(score, is_ai_generated)

    with col2:
        st.info("Detection Score")
        st.write("Perplexity:", perplexity)

        if is_ai_generated:
            st.error("Text Analysis Result: AI Generated Content")
        else:
            st.success("Text Analysis Result: Human Generated Content")

    with col3:
        st.info("Original Text")
        st.markdown(text)

    return perplexity

def display_results_bert(text):
    col1, col2, col3 = st.columns([1, 1, 1])
    entropy = calculate_entropy_bert(text)

    # Determine if the content is AI-generated based on thresholds
    is_ai_generated = entropy < 3.8

    # Normalize score for donut chart
    # Use a sigmoid function to map the entropy to a probability-like score
    score = 1 / (1 + np.exp(-(entropy - 3.8) / 0.5))  # Sigmoid function for more realistic probability

    # Ensure the score is within a reasonable range
    score = max(0.01, min(score, 0.99))  # Clamp score between 0.01 and 0.99

    with col1:
        st.info("Basic Details")
        st.metric(label="Word Count", value=len(text.split()))
        plot_probability_donut_chart(score, is_ai_generated)

    with col2:
        st.info("Detection Score")
        st.write("Entropy:", entropy)

        if is_ai_generated:
            st.success("Text Analysis Result: Human Generated Content")
        else:
            st.error("Text Analysis Result: AI Generated Content")

    with col3:
        st.info("Original Text")
        st.markdown(text)

    return entropy
