import streamlit as st
from langflow.load import run_flow_from_json
import os
from typing import Dict, Any, Tuple
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Validate required environment variables
def validate_env_vars():
    """Validate that required environment variables are set."""
    required_vars = ['OPENAI_API_KEY', 'PINECONE_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        st.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        st.info("Please set these variables in your .env file or in your Streamlit deployment settings.")
        return False
    return True

def format_chat_history(messages: list) -> str:
    """Format chat history into a string."""
    formatted_history = []
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted_history.append(f"{role}: {msg['content']}")
    return "\n".join(formatted_history[-6:])  # Keep last 6 messages for context


TWEAKS = {
    "ChatInput-8k4C7": {
        "background_color": "",
        "chat_icon": "",
        "files": "",
        "input_value": "",
        "sender": "User",
        "sender_name": "User",
        "session_id": "",
        "should_store_message": True,
        "text_color": ""
    },
    "ParseData-WAHhe": {
        "sep": "\n",
        "template": "{text}"
    },
    "Prompt-kHdIO": {
        "context": "",
        "question": "",
        "template": "{context} \n--- \nPrevious conversation:\n{memory}\n\nHi there! I'm {name}, a friendly service assistant ({gender}). I really enjoy helping people and having natural conversations! \n\nGiven what I know from the context above and our previous conversation, I'll do my best to help you with your question. I aim to keep our chat warm and casual - just like talking to a friend who's knowledgeable and eager to help.\n\nQuestion: {question}\n\nLet me share my thoughts on that...\n[Answer in a conversational, friendly tone while maintaining professionalism and referring back to previous conversation when relevant]\n\nIs there anything else you'd like me to clarify? I'm happy to explain further or approach this from a different angle if that would be helpful!",
        "memory": "",
        "name": "Oho",
        "gender": "female"
    },
    "OpenAIModel-bJOaR": {
        "api_key": os.getenv('OPENAI_API_KEY'),
        "input_value": "",
        "json_mode": False,
        "max_tokens": None,
        "model_kwargs": {},
        "model_name": "gpt-4o-mini",
        "openai_api_base": "",
        "output_schema": {},
        "seed": 1,
        "stream": True,
        "system_message": "",
        "temperature": 0.4
    },
    "ChatOutput-QBSG8": {
        "background_color": "",
        "chat_icon": "",
        "data_template": "{text}",
        "input_value": "",
        "sender": "Machine",
        "sender_name": "AI",
        "session_id": "",
        "should_store_message": True,
        "text_color": ""
    },
    "OpenAIEmbeddings-2xWVE": {
        "chunk_size": 1000,
        "client": "",
        "default_headers": {},
        "default_query": {},
        "deployment": "",
        "dimensions": None,
        "embedding_ctx_length": 1536,
        "max_retries": 3,
        "model": "text-embedding-3-small",
        "model_kwargs": {},
        "openai_api_base": "",
        "openai_api_key": os.getenv('OPENAI_API_KEY'),
        "openai_api_type": "",
        "openai_api_version": "",
        "openai_organization": "",
        "openai_proxy": "",
        "request_timeout": None,
        "show_progress_bar": False,
        "skip_empty": False,
        "tiktoken_enable": True,
        "tiktoken_model_name": ""
    },
    "Pinecone-invrX": {
        "distance_strategy": "Cosine",
        "index_name": "ohotest",
        "namespace": "ohotest",
        "number_of_results": 4,
        "pinecone_api_key": os.getenv('PINECONE_API_KEY'),
        "search_query": "",
        "text_key": "text"
    }
}
def load_flow_file(file_path: str) -> Dict[str, Any]:
    """Load the flow JSON file."""
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        st.error(f"Flow file not found: {file_path}")
        return None
    except json.JSONDecodeError:
        st.error("Invalid JSON file")
        return None

def extract_message_data(result) -> Tuple[str, str, str, str]:
    """
    Extract message, session_id, sender, and sender_name from the LangFlow response.
    Returns: (message, session_id, sender, sender_name)
    """
    try:
        first_output = result[0]
        if hasattr(first_output, 'outputs') and first_output.outputs:
            output = first_output.outputs[0]
            if hasattr(output, 'results') and hasattr(output.results, 'get'):
                message_data = output.results.get('message', {})
                if hasattr(message_data, 'data'):
                    data = message_data.data
                elif isinstance(message_data, dict):
                    data = message_data
                else:
                    data = {'text': str(message_data)}
                
                return (
                    data.get('text', ''),
                    data.get('session_id', ''),
                    data.get('sender', ''),
                    data.get('sender_name', '')
                )
        
        return ('No message found', '', '', '')
    except Exception as e:
        st.error(f"Error extracting message data: {str(e)}")
        return ('Error processing response', '', '', '')

def process_message(message: str, flow_data: Dict[str, Any], chat_history: str) -> Tuple[str, str, str, str]:
    """Process the message using LangFlow."""
    try:
        # Update TWEAKS with chat history
        updated_tweaks = TWEAKS.copy()
        updated_tweaks["Prompt-kHdIO"]["memory"] = chat_history
        
        result = run_flow_from_json(
            flow=flow_data,
            input_value=message,
            session_id=st.session_state.get('session_id', ''),
            fallback_to_env_vars=True,
            tweaks=updated_tweaks
        )
        return extract_message_data(result)
    except Exception as e:
        st.error(f"Error processing message: {str(e)}")
        return None

def main():
    st.title("Oho AI Chat Interface")
    
    # Validate environment variables before proceeding
    if not validate_env_vars():
        st.stop()
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'session_id' not in st.session_state:
        st.session_state.session_id = os.urandom(16).hex()
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    
    # Load the flow file
    flow_data = load_flow_file("rag1.json")
    if not flow_data:
        st.stop()

    # Create a container for messages
    chat_container = st.container()
    
    with chat_container:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat input
    if not st.session_state.processing:
        if prompt := st.chat_input("What would you like to know?"):
            st.session_state.processing = True
            
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Get chat history
            chat_history = format_chat_history(st.session_state.messages[:-1])
            
            # Process the message and get response
            message, _, _, _ = process_message(prompt, flow_data, chat_history)
            
            if message:
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": message
                })
            
            st.session_state.processing = False
            st.rerun()

if __name__ == "__main__":
    main()