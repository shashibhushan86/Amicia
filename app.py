import streamlit as st

from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    #device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")


from transformers import pipeline

# Create a pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=500,
    do_sample=False
)



# Title
st.title("Amicia")

# Header
st.header("Hello. I am Amicia! ")

instructions = st.text_area("What do you want me to generate for you?")

if len(instructions) < 1000:
    if st.button("Generate"):
        # Generate output
        output = generator(instructions)
        print(output[0]["generated_text"])
        st.info(output)
else:
    st.warning(
        "Input too large. Please write less than 1000 words."
    )





