import streamlit as st
from langflow.load import run_flow_from_json
import os
import uuid
from datetime import datetime

# Configure the page
st.set_page_config(page_title="Langflow Chat", layout="wide")

# Check for OpenAI API key
if 'OPENAI_API_KEY' not in st.secrets:
    st.error("Please set the OPENAI_API_KEY in your secrets. Check the app's documentation for instructions.")
    st.stop()

# Set the API key as an environment variable
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

# Initialize session state for chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Create a unique session ID if it doesn't exist
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Define the tweaks configuration with OpenAI API keys
TWEAKS = {
    "File-5WyjM": {},
    "SplitText-M5sZ2": {},
    "Pinecone-Ia2GC": {},
    "OpenAIEmbeddings-pmhCH": {
        "openai_api_key": st.secrets['OPENAI_API_KEY']
    },
    "ChatInput-dtNrJ": {
        "session_id": st.session_state.session_id,
        "sender": "Human",
        "sender_name": "User"
    },
    "Pinecone-Ki9ox": {},
    "OpenAIEmbeddings-aKxV5": {
        "openai_api_key": st.secrets['OPENAI_API_KEY']
    },
    "ParseData-XV7R7": {},
    "Prompt-y8lI9": {},
    "Memory-ZNCLd": {},
    "OpenAIModel-EiWSb": {
        "openai_api_key": st.secrets['OPENAI_API_KEY'],
        "model_name": "gpt-4o-mini"  # You can specify the model here if needed
    },
    "ChatOutput-yudoU": {},
    "File-a7Evd": {},
    "File-7CouN": {},
    "File-UFmKb": {},
    "File-GPZCY": {},
    "File-rBbDn": {}
}

# Rest of your code remains the same...
def extract_message(response):
    """
    Extract the message text from the Langflow response
    """
    try:
        if isinstance(response, list) and len(response) > 0:
            run_output = response[0]
            if hasattr(run_output, 'outputs') and len(run_output.outputs) > 0:
                result = run_output.outputs[0]
                if hasattr(result, 'results'):
                    message_obj = result.results.get('message')
                    if message_obj and hasattr(message_obj, 'text'):
                        return message_obj.text
                    
        str_response = str(response)
        if "'text': '" in str_response:
            start_idx = str_response.find("'text': '") + 8
            end_idx = str_response.find("'", start_idx)
            if start_idx != -1 and end_idx != -1:
                return str_response[start_idx:end_idx]
                
        return "I couldn't process that properly. Could you please try again?"
        
    except Exception as e:
        st.error(f"Error extracting message: {str(e)}")
        return "Sorry, I encountered an error processing that message."

# Create the main UI
st.title("Langflow Chat")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Run the Langflow pipeline with just the prompt text
                response = run_flow_from_json(
                    flow="ohochatflow.json",
                    input_value=prompt,  # Just pass the prompt text directly
                    tweaks=TWEAKS,
                    session_id=st.session_state.session_id
                )
                
                # Extract the message from the response
                message = extract_message(response)
                
                # Display the response
                st.markdown(message)
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": message
                })
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.error("Full error details:")
                st.error(str(e))

# Add a sidebar with information
with st.sidebar:
    st.title("About")
    st.markdown("""
    This is a Langflow-powered chatbot integrated with Streamlit.
    
    The chat history is maintained for your session, and each session has a unique identifier.
    """)
    
    # Display session information for debugging
    st.write("Session ID:", st.session_state.session_id)