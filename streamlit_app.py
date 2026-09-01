import os
from huggingface_hub import login
from peft import PeftModel
import streamlit as st
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
)

# 1. Page Configuration
st.set_page_config(page_title="Numbers-Only AI", page_icon="🔢", layout="centered")

st.title("🔢 Numbers-Only AI")
st.write(
    "Ask any question, and the fine-tuned Gemma 3 model will respond with **only the numeric answer**."
)

BASE_MODEL_NAME = "google/gemma-3-270m-it"
ADAPTER_REPO = "BelalOmran/gemma3-numbers-only-adapter"  # 👈 REPLACE THIS


# 2. Cache Model Loading (Runs only once)
@st.cache_resource
def load_model():
  # Log in with secret token if available
  if "HF_TOKEN" in st.secrets:
    login(token=st.secrets["HF_TOKEN"])
  elif "HF_TOKEN" in os.environ:
    login(token=os.environ["HF_TOKEN"])

  tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
  base_model = AutoModelForCausalLM.from_pretrained(
      BASE_MODEL_NAME,
      torch_dtype=torch.float32,
      low_cpu_mem_usage=True,
  )
  model = PeftModel.from_pretrained(base_model, ADAPTER_REPO)
  model.eval()
  return tokenizer, model


with st.spinner("Loading model into memory... (only takes ~15 seconds)"):
  tokenizer, model = load_model()


# 3. Constrained Decoding: Forces digits only
class DigitsOnlyLogitsProcessor(LogitsProcessor):

  def __init__(self, tokenizer):
    allowed = set()
    for token_id in range(len(tokenizer)):
      piece = tokenizer.decode([token_id])
      stripped = piece.strip()
      if stripped == "" or all(c in "0123456789.-" for c in stripped):
        allowed.add(token_id)
    allowed.add(tokenizer.eos_token_id)
    self.allowed_ids = torch.tensor(sorted(allowed))

  def __call__(self, input_ids, scores):
    mask = torch.full_like(scores, float("-inf"))
    mask[:, self.allowed_ids.to(scores.device)] = scores[
        :, self.allowed_ids.to(scores.device)
    ]
    return mask


digits_processor = DigitsOnlyLogitsProcessor(tokenizer)


# 4. Inference Function
def get_numeric_answer(question):
  messages = [{"role": "user", "content": question}]
  inputs = tokenizer.apply_chat_template(
      messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
  )

  output = model.generate(
      inputs,
      max_new_tokens=8,
      do_sample=False,
      logits_processor=LogitsProcessorList([digits_processor]),
      pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
  )

  generated = output[0][inputs.shape[1] :]
  return tokenizer.decode(generated, skip_special_tokens=True).strip()


# 5. User Interface Form
with st.form("qa_form"):
  question = st.text_input(
      "Question:", placeholder="e.g. How many legs does a spider have?"
  )
  submitted = st.form_submit_button("Get Answer")

if submitted:
  if question.strip():
    with st.spinner("Calculating..."):
      answer = get_numeric_answer(question)
    st.success(f"**Answer:** `{answer}`")
  else:
    st.warning("Please enter a question first.")
