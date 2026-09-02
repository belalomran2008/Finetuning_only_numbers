import os
import re
import streamlit as st

# Configure Keras backend before importing keras_hub
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras_hub

st.set_page_config(
    page_title="Gemma 3 Numbers-Only AI",
    page_icon="🔢",
    layout="centered"
)

st.title("🔢 Numbers-Only AI")
st.markdown(
    "Ask any question, and the fine-tuned **Gemma 3 (270M)** model will respond with **only the numeric answer**."
)

MODEL_REPO = "hf://BelalOmran/gemma3-numbers-only"
INSTRUCTION = "Answer with only a number. No words, no units, no punctuation.\n\n"

@st.cache_resource(show_spinner="Loading Gemma 3 model from Hugging Face...")
def load_model():
    """Load the KerasHub fine-tuned model preset directly."""
    model = keras_hub.models.Gemma3CausalLM.from_preset(MODEL_REPO)
    return model

# Load model
try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Sidebar
st.sidebar.header("⚙️ Settings")
max_len = st.sidebar.slider("Max Sequence Length", min_value=32, max_value=256, value=128, step=16)

st.sidebar.markdown("---")
st.sidebar.subheader("💡 Example Questions")
examples = [
    "What is the atomic number of uranium?",
    "What is 33 multiplied by 49?",
    "How many letters are in the Arabic alphabet?",
    "What is the square root of 256?",
    "Solve for x: 2x + -43 = -63. What is x?",
    "What is the 10th number in the Fibonacci sequence, starting at 1 with 0?",
    "How many stomachs does a cow have?",
    "What is 20 percent of 95?",
]

if "user_query" not in st.session_state:
    st.session_state["user_query"] = ""

def set_example(q):
    st.session_state["user_query"] = q

for ex in examples:
    if st.sidebar.button(ex, use_container_width=True):
        set_example(ex)

# Input Box
user_input = st.text_area(
    "Enter your question:",
    value=st.session_state["user_query"],
    placeholder="e.g. What is the atomic number of nitrogen?",
    height=100
)

col1, col2 = st.columns([1, 4])
with col1:
    submit = st.button("Generate", type="primary", use_container_width=True)

if submit and user_input.strip():
    with st.spinner("Generating answer..."):
        prompt = f"<start_of_turn>user\n{INSTRUCTION}{user_input.strip()}<end_of_turn>\n<start_of_turn>model\n"
        raw_output = model.generate(prompt, max_length=int(max_len))
        
        if "<start_of_turn>model\n" in raw_output:
            response = raw_output.split("<start_of_turn>model\n")[-1]
        else:
            response = raw_output[len(prompt):]
        response = response.replace("<end_of_turn>", "").strip()
        
        match = re.search(r"-?\d+(?:\.\d+)?", response)
        parsed_number = match.group(0) if match else "No number found"
        
        st.markdown("---")
        st.subheader("🎯 Parsed Result")
        st.metric(label="Answer", value=parsed_number)
        
        with st.expander("🔍 View Raw Output & Prompt"):
            st.write("**Formatted Prompt:**")
            st.code(prompt, language="text")
            st.write("**Raw Model Completion:**")
            st.code(response, language="text")
