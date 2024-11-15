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

# Define chat configuration
CHAT_CONFIG = {
    "session_id": st.session_state.session_id,
    "sender": "user",
    "sender_name": "User"
}

# Define the tweaks dictionary
TWEAKS = {
    "File-5WyjM": {},
    "SplitText-M5sZ2": {},
    "Pinecone-Ia2GC": {},
    "OpenAIEmbeddings-pmhCH": {},
    "ChatInput-dtNrJ": CHAT_CONFIG,
    "Pinecone-Ki9ox": {},
    "OpenAIEmbeddings-aKxV5": {},
    "ParseData-XV7R7": {},
    "Prompt-y8lI9": {},
    "Memory-ZNCLd": {
        "session_id": st.session_state.session_id
    },
    "OpenAIModel-EiWSb": {},
    "ChatOutput-yudoU": {
        **CHAT_CONFIG,
        "sender": "assistant",
        "sender_name": "Assistant"
    },
    "File-a7Evd": {},
    "File-7CouN": {},
    "File-UFmKb": {},
    "File-GPZCY": {}
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
                # Run the Langflow model with just the prompt string
                result = run_flow_from_json(
                    flow="ohochatflow.json",
                    input_value=prompt,  # Just pass the prompt string directly
                    session_id=st.session_state.session_id,
                    fallback_to_env_vars=True,
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

# Debug section with environment detection
is_local = not os.environ.get('STREAMLIT_DEPLOYED', False)
if st.checkbox("Show debug info"):
    st.write("Session ID:", st.session_state.session_id)
    st.write("Environment:", "Local" if is_local else "Deployed")
    st.write("TWEAKS configuration:", TWEAKS)
    st.write("Last raw response:", result if 'result' in locals() else "No response yet")