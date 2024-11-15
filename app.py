import streamlit as st
from langflow.load import run_flow_from_json
import json
import uuid  # Add this import for generating unique IDs

# Configure Streamlit page
st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="💬",
    layout="centered"
)

# Initialize session state for chat history and session ID if they don't exist
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Define the tweaks dictionary with chat configuration
TWEAKS = {
    "File-5WyjM": {},
    "SplitText-M5sZ2": {},
    "Pinecone-Ia2GC": {},
    "OpenAIEmbeddings-pmhCH": {},
    "ChatInput-dtNrJ": {
        "session_id": st.session_state.session_id,
        "sender": "user",
        "sender_name": "User"
    },
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
    "File-GPZCY": {}
}

def extract_message_from_response(response):
    """Extract the actual message text from the Langflow response"""
    try:
        if isinstance(response, list) and len(response) > 0:
            if hasattr(response[0], 'outputs'):
                message_data = response[0].outputs[0].results['message']
                if hasattr(message_data, 'text'):
                    return message_data.text
                elif isinstance(message_data, dict) and 'text' in message_data:
                    return message_data['text']
                else:
                    return str(message_data)
    except Exception as e:
        st.error(f"Error extracting message: {str(e)}")
    return str(response)  # Return full response as fallback

# Display chat title
st.title("💬 Oho Chat Assistant")
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
                # Run the Langflow model
                result = run_flow_from_json(
                    flow="ohochatflow.json",
                    input_value=prompt,
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

# Add a clear button to reset the chat
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# Optional: Add a debug section to see the raw response
if st.checkbox("Show debug info"):
    st.write("Session ID:", st.session_state.session_id)
    st.write("Last raw response:", result if 'result' in locals() else "No response yet")