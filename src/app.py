import os
from urllib.parse import urljoin
import json
import openai
import replicate
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from transformers import pipeline
from vectors import embeddings, index
app = FastAPI()

openai.api_key = os.environ['OPENAI_API_KEY']

# model for coherence evalation
coherence_evaluator = pipeline("text-classification", model="cointegrated/roberta-large-cola-krishna2020")

html_content = """
<!DOCTYPE html>
<html>
    <head>
        <title>LLM Model Evaluation</title>
    </head>
    <body>
        <h1>LLM Model Evaluation</h1>
        <form id="form">
            <label for="prompt">Enter your prompt:</label><br><br>
            <input type="text" id="prompt" name="prompt"><br><br>
            <input type="submit" value="Submit">
        </form>
        <div id="responses"></div>
        <script>
            const form = document.getElementById('form');
            form.addEventListener('submit', async (event) => {
                event.preventDefault();
                const prompt = document.getElementById('prompt').value;
                const responseDiv = document.getElementById('responses');
                responseDiv.innerHTML = '';
                const socket = new WebSocket('ws://localhost:8000/ws');
                socket.onopen = () => {
                    socket.send(JSON.stringify({ prompt }));
                };
                socket.onmessage = (event) => {
                    const message = JSON.parse(event.data);
                    responseDiv.innerHTML += `<p>${message.model}: ${message.response}</p>`;
                };
            });
        </script>
    </body>
</html>
"""

@app.get("/")
async def get():
    return HTMLResponse(html_content)

#Implement Websockets or Server-Sent Events (SSE) to enable real-time streaming ofresponses to the frontend.
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        data = json.loads(data)
        prompt = data['prompt']
        
        # Query Pinecone vector database
        search_results = index.query(
            vector=embeddings.embed_query(prompt),
            top_k=5
        )
        
 
        relevant_chunks = [result['metadata']['text'] for result in search_results['matches']]
        combined_prompt = "\n".join(relevant_chunks) + "\n" + prompt

        # LLMs 
        models = {
            "gpt-3.5-turbo": lambda p: openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": p}]),
            "gpt-4": lambda p: openai.ChatCompletion.create(model="gpt-4", messages=[{"role": "user", "content": p}]),
            "Llama-2-70b-chat": lambda p: replicate.run("replicate/llama-2-70b-chat:latest", input={"prompt": p}),
            "Falcon-40b-instruct": lambda p: replicate.run("joehoover/falcon-40b-instruct:latest", input={"prompt": p})
        }
        
        responses = {}
        
        # Query each model
        for model_name, model_func in models.items():
            response = model_func(combined_prompt)
            if model_name in ["gpt-3.5-turbo", "gpt-4"]:
                response_text = response.choices[0].message['content']
            else:
                response_text = response['output']
            responses[model_name] = response_text
            await websocket.send_text(json.dumps({"model": model_name, "response": response_text}))
        
        # compare and evaluate the generated outputs to find thebest-performing LLM for the given user input.
        best_model = evaluate_responses(responses, relevant_chunks)
        await websocket.send_text(json.dumps({"model": "Best Model", "response": best_model}))

def evaluate_responses(responses, relevant_chunks):
    def relevance_score(response):
        # count the number of relevant chunk words in the response
        response_words = set(response.split())
        return sum(any(word in response_words for word in chunk.split()) for chunk in relevant_chunks)

    def coherence_score(response):
        # evaluate coherence
        result = coherence_evaluator(response)
        return result[0]['score'] if result[0]['label'] == 'acceptable' else 0

    def length_score(response):
        # Length score: penalize too short (<50) or too long (>500) responses
        length = len(response.split())
        if length < 50 or length > 500:
            return 0
        return length

    scores = {}
    for model, response in responses.items():
        relevance = relevance_score(response)
        coherence = coherence_score(response)
        length = length_score(response)
        total_score = relevance + coherence + length
        scores[model] = total_score
