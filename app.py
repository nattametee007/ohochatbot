import streamlit as st
from langflow.load import run_flow_from_json
import json

# Configure Streamlit page
st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="💬",
    layout="centered"
)

# Initialize session state for chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Define the tweaks dictionary
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
    "File-GPZCY": {}
}

def extract_message_from_response(response):
    """Extract the actual message text from the Langflow response"""
    try:
        # The message is nested in the response structure
        # Access the first message from the outputs
        if hasattr(response[0], 'outputs'):
            message_data = response[0].outputs[0].results['message']
            if hasattr(message_data, 'text'):
                return message_data.text
            elif isinstance(message_data, dict) and 'text' in message_data:
                return message_data['text']
    except Exception as e:
        st.error(f"Error extracting message: {str(e)}")
        return str(response)  # Return full response as fallback
    return str(response)  # Return full response as fallback

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
                # Run the Langflow model
                result = run_flow_from_json(
                    flow="ohochat.json",
                    input_value=prompt,
                    session_id=str(st.session_state),
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
    st.rerun()  # Updated from experimental_rerun() to rerun()

# Optional: Add a debug section to see the raw response
if st.checkbox("Show debug info"):
    st.write("Last raw response:", result if 'result' in locals() else "No response yet")