# agent_sentiment.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

def analyze_sentiment(email_text):
    # Tokenize the input text
    inputs = tokenizer.encode("sentiment: " + email_text, return_tensors="pt", max_length=1024, truncation=True)
    
    # Generate sentiment analysis result
    sentiment_ids = model.generate(inputs, max_length=100, num_beams=4, early_stopping=True)
    
    # Decode and return sentiment
    sentiment = tokenizer.decode(sentiment_ids[0], skip_special_tokens=True)
    return sentiment