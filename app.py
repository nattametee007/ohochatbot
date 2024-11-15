import streamlit as st
from langflow.load import run_flow_from_json
import json
import uuid

# Configure Streamlit page
st.set_page_config(
    page_title="Oho AI Chat Assistant",
    page_icon="💬",
    layout="centered"
)

# Initialize session state for chat history, memory, and session ID
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "memory" not in st.session_state:
    st.session_state.memory = []

# Define the tweaks dictionary with chat configuration and memory
TWEAKS = {
    "ChatInput-dtNrJ": {
        "session_id": st.session_state.session_id,
        "sender": "user",
        "sender_name": "User"
    },
    "Memory-ZNCLd": {
        "session_id": st.session_state.session_id,
        "memory": st.session_state.memory
    },
    "File-5WyjM": {},
    "SplitText-M5sZ2": {},
    "Pinecone-Ia2GC": {},
    "OpenAIEmbeddings-pmhCH": {},
    "Pinecone-Ki9ox": {},
    "OpenAIEmbeddings-aKxV5": {},
    "ParseData-XV7R7": {},
    "Prompt-y8lI9": {},
    "OpenAIModel-EiWSb": {},
    "ChatOutput-yudoU": {},
    "File-a7Evd": {},
    "File-7CouN": {},
    "File-UFmKb": {},
    "File-GPZCY": {}
}

def extract_message_from_response(response):
    """Extract the actual message text from the Langflow response and update memory"""
    try:
        if isinstance(response, list) and len(response) > 0:
            # Extract message
            if hasattr(response[0], 'outputs'):
                message_data = response[0].outputs[0].results['message']
                
                # Update memory if present in response
                if 'memory' in response[0].outputs[0].results:
                    st.session_state.memory = response[0].outputs[0].results['memory']
                
                # Extract message text
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
st.title("💬 Oho AI Chat Assistant")
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
                # Update tweaks with current memory
                TWEAKS["Memory-ZNCLd"]["memory"] = st.session_state.memory
                
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
    st.session_state.memory = []  # Also clear the memory
    st.rerun()

# Debug section
if st.checkbox("Show debug info"):
    st.write("Session ID:", st.session_state.session_id)
    st.write("Memory:", st.session_state.memory)
    st.write("Last raw response:", result if 'result' in locals() else "No response yet")