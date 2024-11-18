import streamlit as st
from langflow.load import run_flow_from_json
import os
from typing import Dict, Any, Tuple
import json
from dotenv import load_dotenv
from functools import lru_cache
import time

# Load environment variables
load_dotenv()

# Cache for API rate limiting
class RateLimiter:
    def __init__(self, calls_per_minute=60):
        self.calls_per_minute = calls_per_minute
        self.calls = []
        
    def can_call(self) -> bool:
        current_time = time.time()
        minute_ago = current_time - 60
        
        # Remove calls older than 1 minute
        self.calls = [call_time for call_time in self.calls if call_time > minute_ago]
        
        if len(self.calls) < self.calls_per_minute:
            self.calls.append(current_time)
            return True
        return False

# Initialize rate limiter
rate_limiter = RateLimiter(calls_per_minute=60)  # Adjust limit as needed

@st.cache_data(ttl=3600)  # Cache for 1 hour
def validate_env_vars() -> bool:
    """Validate that required environment variables are set."""
    required_vars = ['OPENAI_API_KEY', 'PINECONE_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        st.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        st.info("Please set these variables in your .env file or in your Streamlit deployment settings.")
        return False
    return True

@st.cache_data(ttl=60)  # Cache for 1 minute
def format_chat_history(messages: tuple) -> str:  # Changed to tuple for caching
    """Format chat history into a string."""
    formatted_history = []
    for msg in messages:
        role = "User" if msg[0] == "user" else "Assistant"
        formatted_history.append(f"{role}: {msg[1]}")
    return "\n".join(formatted_history[-6:])  # Keep last 6 messages for context

@st.cache_resource  # Cache the flow data permanently
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

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_tweaks(chat_history: str) -> Dict:
    """Get TWEAKS configuration with updated chat history."""
    base_tweaks = {
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
            "memory": chat_history,
            "name": "Oho",
            "gender": "female"
        },
        "OpenAIModel-bJOaR": {
            "api_key": os.getenv('OPENAI_API_KEY'),
            "model_name": "gpt-4o-mini",
            "temperature": 0.4,
            "stream": True
        },
        "ChatOutput-QBSG8": {
            "sender": "Machine",
            "sender_name": "AI",
            "should_store_message": True
        },
        "OpenAIEmbeddings-2xWVE": {
            "model": "text-embedding-3-small",
            "openai_api_key": os.getenv('OPENAI_API_KEY')
        },
        "Pinecone-invrX": {
            "distance_strategy": "Cosine",
            "index_name": "ohotest",
            "namespace": "ohotest",
            "number_of_results": 4,
            "pinecone_api_key": os.getenv('PINECONE_API_KEY')
        }
    }
    return base_tweaks

@lru_cache(maxsize=100)  # Cache recent message processing results
def process_message(message: str, chat_history: str, session_id: str) -> Tuple[str, str, str, str]:
    """Process the message using LangFlow with caching."""
    try:
        # Check rate limiting
        if not rate_limiter.can_call():
            time.sleep(1)  # Wait if rate limit exceeded
            
        flow_data = load_flow_file("rag1.json")
        tweaks = get_tweaks(chat_history)
        
        result = run_flow_from_json(
            flow=flow_data,
            input_value=message,
            session_id=session_id,
            fallback_to_env_vars=True,
            tweaks=tweaks
        )
        return extract_message_data(result)
    except Exception as e:
        st.error(f"Error processing message: {str(e)}")
        return None

def extract_message_data(result) -> Tuple[str, str, str, str]:
    """Extract message data from the LangFlow response."""
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
    
    # Create a container for messages
    chat_container = st.container()
    
    # Chat input - Get user input first
    if not st.session_state.processing:
        prompt = st.chat_input("What would you like to know?")
        if prompt:
            st.session_state.processing = True
            st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display all messages including the new user message
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # If we're processing, show the thinking indicator
        if st.session_state.processing and st.session_state.messages:
            # Get the last user message
            last_message = st.session_state.messages[-1]
            if last_message["role"] == "user":
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        # Convert messages to tuple for caching
                        messages_tuple = tuple((msg["role"], msg["content"]) 
                                            for msg in st.session_state.messages[:-1])
                        chat_history = format_chat_history(messages_tuple)
                        
                        # Process the message and get response
                        message, _, _, _ = process_message(
                            last_message["content"], 
                            chat_history, 
                            st.session_state.session_id
                        )
                        
                        if message:
                            # Add assistant response to chat history
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": message
                            })
                        
                        # Reset processing flag
                        st.session_state.processing = False
                        st.rerun()

if __name__ == "__main__":
    main()