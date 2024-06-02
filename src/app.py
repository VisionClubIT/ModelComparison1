import os
import json
import openai
import replicate
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from vectors import embeddings, index

app = FastAPI()

openai.api_key = os.environ['OPENAI_API_KEY']

# frontend
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
            embeddings.embed_query(prompt),
            top_k=5,
            namespace="default"
        )
        
        # Extract relevant chunks
        relevant_chunks = [result['metadata']['text'] for result in search_results['matches']]
        combined_prompt = "\n".join(relevant_chunks) + "\n" + prompt

        #  LLMs
        models = {
            "gpt-3.5-turbo": lambda p: openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": p}]),
            "gpt-4": lambda p: openai.ChatCompletion.create(model="gpt-4", messages=[{"role": "user", "content": p}]),
            "Llama-2-70b-chat": lambda p: replicate.run("replicate/llama-2-70b-chat:latest", input={"prompt": p}),
            "Falcon-40b-instruct": lambda p: replicate.run("joehoover/falcon-40b-instruct:latest", input={"prompt": p})
        }
        
        # provides theresponse of the LLMs.
        for model_name, model_func in models.items():
            response = model_func(combined_prompt)
            if model_name in ["gpt-3.5-turbo", "gpt-4"]:
                response_text = response.choices[0].message['content']
            else:
                response_text = response['output']
            await websocket.send_text(json.dumps({"model": model_name, "response": response_text}))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
