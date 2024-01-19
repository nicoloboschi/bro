# bro

Experimental project to create a voice to voice chatbot using speech recognition and OpenAI products.

```
export OPENAI_API_KEY=...
poetry install
poetry run python -m app
```

or Docker:
```
poetry self add poetry-dockerize-plugin
poetry dockerize

export OPENAI_API_KEY=...
docker run --rm -e OPENAI_API_KEY=$OPENAI_API_KEY -p 8000:8000 -it bro:latest
```


