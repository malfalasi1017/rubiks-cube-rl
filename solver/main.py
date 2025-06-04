from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class SolverRequest(BaseModel):
    cube_state: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Rubik's Cube Solver API"}

@app.post("/solve")
def solve_cube(request: SolverRequest):
    # Dummy implementation for solving the cube
    return {
        "solution": f"Solving cube with state: {request.cube_state}"
    }