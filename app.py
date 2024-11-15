# app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langflow.load import run_flow_from_json
import uvicorn
import json
import os
from typing import Optional
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Oho Chat API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = ""

class ChatResponse(BaseModel):
    response: str

# Load flow configuration
try:
    with open("ohochatflow.json", "r") as f:
        flow_config = json.load(f)
        logger.info("Successfully loaded flow configuration")
except FileNotFoundError:
    logger.error("Flow configuration file 'ohochatflow.json' not found!")
    raise Exception("Flow configuration file 'ohochatflow.json' not found!")

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

def extract_message_from_response(response_obj):
    """Extract the actual message text from the Langflow response object"""
    try:
        # Convert response object to string if it's not already
        response_str = str(response_obj)
        
        # Try to find the message content directly
        if isinstance(response_obj, (dict, str)):
            # If it's a dictionary, try to get the message directly
            if isinstance(response_obj, dict):
                if 'text' in response_obj:
                    return response_obj['text']
                if 'message' in response_obj:
                    return response_obj['message']

        # If the above fails, try to extract using regex
        message_match = re.search(r"text='([^']*)'", response_str)
        if message_match:
            return message_match.group(1)
            
        # If regex fails, try to find any Thai text in the response
        thai_text_match = re.search(r'[ก-๙]+[^"\']*[ก-๙]+', response_str)
        if thai_text_match:
            return thai_text_match.group(0)

        # If all else fails, return a default message
        return "ขออภัยค่ะ ไม่สามารถประมวลผลข้อความได้ในขณะนี้"

    except Exception as e:
        logger.error(f"Error extracting message: {str(e)}")
        return "ขออภัยค่ะ เกิดข้อผิดพลาดในการประมวลผล"

@app.get("/health")
def health_check():
    return {"status": "healthy"}

def run_flow_async(message: str, session_id: str):
    """Run the flow in a synchronous context"""
    try:
        return run_flow_from_json(
            flow="ohochatflow.json",
            input_value=message,
            session_id=session_id,
            fallback_to_env_vars=True,
            tweaks=TWEAKS
        )
    except Exception as e:
        logger.error(f"Error in run_flow_async: {str(e)}")
        raise e

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        logger.info(f"Received chat request with message: {request.message}")
        
        # Run the flow synchronously
        result = run_flow_async(request.message, request.session_id)
        
        # Extract the actual message from the complex response
        clean_response = extract_message_from_response(result)
        
        logger.info(f"Processed response: {clean_response}")
        return ChatResponse(response=clean_response)
    
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        loop="asyncio",
        reload=True
    )