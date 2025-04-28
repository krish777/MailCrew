import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from agents.summarizer_agent import summarize_email
from agents.sentiment_agent import analyze_sentiment
from agents.reply_agent import generate_reply


def main():
    # Title of the app
    st.title("MailCrew - Email Assistant")
    
    # Text area for email input
    email_content = st.text_area("Paste your email content here:", height=200)
    
    if email_content:
        # User selects which action to perform
        task = st.radio("What would you like to do with this email?", 
                        ("Summarize", "Analyze Sentiment", "Generate Reply"))
        
        # Depending on user selection, call the appropriate function
        if task == "Summarize":
            st.subheader("Summary:")
            summary = summarize_email(email_content)
            st.write(summary)
        
        elif task == "Analyze Sentiment":
            st.subheader("Sentiment Analysis:")
            sentiment = analyze_sentiment(email_content)
            st.write(f"Sentiment: {sentiment}")
        
        elif task == "Generate Reply":
            st.subheader("Generated Reply:")
            reply = generate_reply(email_content)
            st.write(reply)
    
if __name__ == "__main__":
    main()