from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn


app = FastAPI()

from multi_model_analyzer import generate

@app.post("/generate")
async def generate_reference(request: Request):
    try:
        form_data = await request.json()
        student_name = form_data.get('studentName')
        result = generate(NAME=student_name)
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read())

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


