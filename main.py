import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from chatService import ChatService

app = FastAPI(title="LM Studio LangChain Backend")


class ChatRequest(BaseModel):
    prompt: str


chatService = ChatService()


@app.post("/generate-stream")
def stream_chat(payload: ChatRequest):
    """
    FastAPI endpoint that streams text back to the LM Studio Plugin.
    """

    def text_generator():
        # Get the token stream from LangChain
        chunks = chatService.getResponse(payload.prompt)

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