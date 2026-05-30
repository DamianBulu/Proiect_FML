import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from ChatService import ChatService

app = FastAPI(title="LM Studio LangChain Backend")


class ChatHistoryMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    prompt: str
    chat_history: Optional[list[ChatHistoryMessage]] = None


chatService = ChatService()


@app.post("/generate-stream")
def stream_chat(payload: ChatRequest):
    """
    FastAPI endpoint that streams text back to the LM Studio Plugin.
    Accepts the latest prompt and the full chat history from the bridge.
    """

    # Convert Pydantic models to plain dicts for the LangGraph state
    history = [msg.model_dump() for msg in payload.chat_history] if payload.chat_history else []

    def text_generator():
        # Get the response from the LangGraph pipeline
        chunks = chatService.getResponse(payload.prompt, chat_history=history)

        for chunk in chunks:
            # Extract just the raw text string from the AIMessageChunk object
            if hasattr(chunk, 'content'):
                yield chunk.content
            else:
                yield str(chunk)

    # Hand the clean string generator over to FastAPI
    return StreamingResponse(
        text_generator(),
        media_type="text/plain"
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)