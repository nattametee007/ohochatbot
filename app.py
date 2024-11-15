import streamlit as st
from langflow.load import run_flow_from_json
import json
import uuid
import os

# Configure Streamlit page
st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="💬",
    layout="centered"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Create a JSON structure for the input
def create_chat_input(message):
    return json.dumps({
        "text": message,
        "session_id": st.session_state.session_id,
        "sender": "user",
        "sender_name": "User"
    })

# Define the tweaks dictionary
TWEAKS = {
    "ChatInput-dtNrJ": {
        "session_id": st.session_state.session_id,
        "sender": "user",
        "sender_name": "User"
    },
    "ChatOutput-yudoU": {
        "session_id": st.session_state.session_id,
        "sender": "assistant",
        "sender_name": "Assistant"
    },
    "Memory-ZNCLd": {
        "session_id": st.session_state.session_id
    }
}

def extract_message_from_response(response):
    """Extract the actual message text from the Langflow response"""
    try:
        if isinstance(response, list) and len(response) > 0:
            first_response = response[0]
            if hasattr(first_response, 'outputs'):
                outputs = first_response.outputs
                if isinstance(outputs, list) and len(outputs) > 0:
                    results = outputs[0].results
                    if isinstance(results, dict) and 'message' in results:
                        message = results['message']
                        if isinstance(message, dict) and 'text' in message:
                            return message['text']
                        elif hasattr(message, 'text'):
                            return message.text
                        else:
                            return str(message)
            return str(first_response)
    except Exception as e:
        st.error(f"Error extracting message: {str(e)}")
    return str(response)

# Load the Langflow configuration
try:
    with open("ohochatflow.json", "r") as f:
        flow_config = json.load(f)
except Exception as e:
    st.error(f"Error loading flow configuration: {str(e)}")
    flow_config = {}

# Display chat title
st.title("💬 Chat Assistant")
st.markdown("---")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display assistant thinking message
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Prepare the input with session information
                chat_input = create_chat_input(prompt)
                
                # Run the Langflow model
                result = run_flow_from_json(
                    flow="ohochatflow.json",
                    input_value=chat_input,
                    tweaks=TWEAKS
                )
                
                # Extract the actual message from the response
                message = extract_message_from_response(result)
                
                # Display the response
                st.markdown(message)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": message})
                
            except Exception as e:
                error_message = f"An error occurred: {str(e)}"
                st.error(error_message)
                st.write("Full error:", e)

# Add a clear button to reset the chat
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# Debug section
if st.checkbox("Show debug info"):
    st.write("Session ID:", st.session_state.session_id)
    st.write("Chat Input Structure:", create_chat_input("example"))
    st.write("TWEAKS configuration:", TWEAKS)
    st.write("Last raw response:", result if 'result' in locals() else "No response yet")
    st.write("Flow Config:", flow_config)