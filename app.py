import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_model():
    return pipeline('text-generation', model='gpt2')

model = load_model()

st.title("Salary Forecasting Text Generator")

prompt = st.text_area("Enter your prompt", "Explain salary forecasting in simple terms.")

if st.button("Generate"):
    with st.spinner("Generating..."):
        results = model(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)
        generated_text = results[0]['generated_text']
        st.success("Done!")
        st.write(generated_text)
