# Chat_History.py

import streamlit as st

def app():
    """Function to display chat history."""
    st.title("Chat History")
    
    if 'chat_history' in st.session_state and st.session_state['chat_history']:
        for idx, entry in enumerate(st.session_state['chat_history'], start=1):
            st.info(f"**Chat #{idx}:**")
            st.write(entry['text'])
            st.markdown("<hr>", unsafe_allow_html=True)

            st.write(f"**Model Used:** {entry.get('model', 'N/A')}")

            # Display relevant detection metrics based on the model used
            if entry['model'] == "GPT-J":
                if 'perplexity' in entry:
                    st.write(f"**Perplexity:** {entry['perplexity']:.2f}")
                if 'burstiness_score' in entry:
                    st.write(f"**Burstiness Score:** {entry['burstiness_score']:.2f}")

            elif entry['model'] == "DeBERTa":
                if 'entropy' in entry:
                    st.write(f"**Entropy:** {entry['entropy']:.2f}")
                if 'semantic_coherence' in entry:
                    st.write(f"**Semantic Coherence:** {entry['semantic_coherence']:.2f}")

            elif entry['model'] == "T5":
                if 'perplexity' in entry:
                    st.write(f"**Perplexity:** {entry['perplexity']:.2f}")

            elif entry['model'] == "BERT":
                if 'entropy' in entry:
                    st.write(f"**Entropy:** {entry['entropy']:.2f}")

            # Display AI classification result
            if 'result_label' in entry:
                if entry['result_label'] == 1:
                    st.markdown("**Result:** <span style='color: crimson; font-weight: bold;'>AI Generated Content</span>", unsafe_allow_html=True)
                else:
                    st.markdown("**Result:** <span style='color: green; font-weight: bold;'>Human Generated Content</span>", unsafe_allow_html=True)
            
            st.markdown("<hr>", unsafe_allow_html=True)  # Add separator between entries

    else:
        st.write("No history available.")
        st.markdown("<hr>", unsafe_allow_html=True)

if __name__ == "__main__":
    app()
