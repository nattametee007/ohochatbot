import streamlit as st
from langflow.load import run_flow_from_json
import os
from typing import Dict, Any

# Configure the page
st.set_page_config(page_title="Langflow Chat", layout="wide")

# Initialize session state for chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Define the tweaks configuration
TWEAKS = {
    "File-5WyjM": {},
    "SplitText-M5sZ2": {},
    "Pinecone-Ia2GC": {},
    "OpenAIEmbeddings-pmhCH": {},
    "ChatInput-dtNrJ": {},
    "Pinecone-Ki9ox": {},
    "OpenAIEmbeddings-aKxV5": {},
    "ParseData-XV7R7": {},
    "Prompt-y8lI9": {},
    "Memory-ZNCLd": {},
    "OpenAIModel-EiWSb": {},
    "ChatOutput-yudoU": {},
    "File-a7Evd": {},
    "File-7CouN": {},
    "File-UFmKb": {},
    "File-GPZCY": {},
    "File-rBbDn": {}
}

def extract_message(response: Any) -> str:
    """
    Extract the message text from the Langflow response
    """
    try:
        # Check if response is a list of RunOutputs
        if isinstance(response, list) and len(response) > 0:
            # Get the first RunOutputs object
            run_output = response[0]
            
            # Get the results from the outputs
            if hasattr(run_output, 'outputs') and len(run_output.outputs) > 0:
                # Get the first result
                result = run_output.outputs[0]
                
                # Extract message from ResultData
                if hasattr(result, 'results'):
                    message_obj = result.results.get('message')
                    if message_obj and hasattr(message_obj, 'text'):
                        return message_obj.text
                    
        # If we couldn't extract the message using the above method,
        # try to get it from the string representation
        str_response = str(response)
        if "'text': '" in str_response:
            # Find the text between 'text': ' and the next '
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

# Create a unique session ID for each user
if "session_id" not in st.session_state:
    st.session_state.session_id = os.urandom(16).hex()

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
                # Run the Langflow pipeline
                response = run_flow_from_json(
                    flow="ohochatflow.json",
                    input_value=prompt,
                    session_id=st.session_state.session_id,
                    fallback_to_env_vars=True,
                    tweaks=TWEAKS
                )
                
                # Extract the message from the response
                message = extract_message(response)
                
                # Display the response
                st.markdown(message)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": message})
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Add a sidebar with information
with st.sidebar:
    st.title("About")
    st.markdown("""
    This is a Langflow-powered chatbot integrated with Streamlit.
    
    The chat history is maintained for your session, and each session has a unique identifier.
    """)