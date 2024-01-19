from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.staticfiles import StaticFiles

import base64
from operator import itemgetter
from pathlib import Path

from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableMap
from openai import OpenAI

app = FastAPI()

origins = [
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Data(BaseModel):
    input: str
    chat_history: List[str]


@app.post("/api/complete")
async def get(payload: Data):
    print("got " + str(payload.input))
    result = get_response(payload.input, payload.chat_history)
    return {"data": result["data"], "text": result["text"]}


client = OpenAI()
chat = ChatOpenAI(model="gpt-4")


def text_to_speech(input):
    print("got response from llm: " + str(input))
    speech_file_path = Path(__file__).parent / "speech.mp3"
    response = client.audio.speech.create(
        model="tts-1",
        voice="echo",
        input=input,
    )
    print("speech done: " + str(input))

    response.stream_to_file(speech_file_path)
    with open(speech_file_path, "rb") as f:
        data = f.read()
        return base64.b64encode(data).decode()


REPHRASE_TEMPLATE = """\
Given the following conversation and a follow up question, convert the conversation to fact that you know.

Chat History:
{chat_history}
"""


def get_response(input_text, chat_history):
    llm = ChatOpenAI(model="gpt-4")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """
      Your name is bro. the user wants to talk with a friend. Be kind and try to help them. Use short phrases to respond to them. Don't annoy them.
      Also answer in the same exact language. Use a super friendly and encouraging tone. Be concise and to the point. Don't be too formal. Talk like a teenager.
      Your answer will be converted to audio, users has no visibility to your answer, so don't use code examples or things that requires visual context.
      This is the context of the current conversation, Just continue the conversation from here answering the human question.:
       
      {context} 
      """),
            ("human", "{question}"),
        ]
    )
    result = (RunnableMap(
        {"context": itemgetter("chat_history"),
         "question": itemgetter("question")}) | prompt | llm | StrOutputParser()).invoke(
        {"question": input_text, "chat_history": "\n".join(chat_history)})
    return {"data": text_to_speech(result), "text": result}




if __name__ == "__main__":
    import uvicorn

    app.mount('/', StaticFiles(directory=Path(__file__).parent.parent.joinpath("frontend"), html=True))
    uvicorn.run(app, host="0.0.0.0", port=8000)
