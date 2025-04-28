This app is deployed in streamlit. Please use this URL to access it. https://krishn-agents.streamlit.app/

![image](https://github.com/user-attachments/assets/e309d6a2-086f-4775-91f6-3d6982ade436)

MailCrew, a simple, modular project that demonstrates how easy it is to get started with AI agent concepts.

**Here’s a quick summary of what I built:**

✅ Developed an AI Email Assistant called MailCrew that can:

Summarize an email

Analyze Sentiment of the email

Generate a smart reply to the email

✅ Built using open-source models like FLAN-T5 (for summarization,
sentiment analysis, and reply generation).

✅ Created a simple Streamlit UI where users can paste email content
and get instant outputs.

✅ Each feature is handled by a simple agent function (no orchestration
yet — agents are run independently).

✅ Designed for easy extension — users can swap out the model in the
agent files (agent_summarization.py, agent_sentiment.py,
agent_reply.py) if they want to customize.

✅ Virtual environment setup and requirements.txt are provided to make
it simple to deploy locally or to the cloud (e.g., AWS).

✅ Project is fully open-sourced on GitHub! 🎉
If you'd like to try it out or learn how to work with CrewAI or
open-source HuggingFace models, feel free to fork the project and
experiment.

**Tech Stack:**

Python 3

Streamlit for UI

Simple agent design using CrewAI

HuggingFace Transformers (FLAN-T5 model)

VSCode for development

I'm excited about how lightweight tools like CrewAI and open models 
like FLAN-T5 can empower quick innovations.
