# agent_reply.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

def generate_reply(email_text):
    # Tokenize the input text
    inputs = tokenizer.encode("reply: " + email_text, return_tensors="pt", max_length=1024, truncation=True)
    
    # Generate the reply
    reply_ids = model.generate(inputs, max_length=200, num_beams=4, early_stopping=True)
    
    # Decode and return the reply
    reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    return reply