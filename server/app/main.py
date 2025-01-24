from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path


from app.functions import get_states, get_cities
from app.recommendation import get_recommendations


app = FastAPI()

# Monta la carpeta estática
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Ruta para servir el archivo HTML
@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_path = Path(__file__).parent / "static" / "index.html"
    return html_path.read_text()

@app.get("/states")
async def states():
    """
    Devuelve una lista de estados con sus IDs y nombres.
    """
    try:
        states_list = get_states()
        return {"states": states_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cities/{id_state}")
async def cities(id_state: int):
    """
    Devuelve una lista de ciudades para un estado dado.
    """
    try:
        cities_list = get_cities(id_state)
        return {"cities": cities_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# Modelo para la validación del formulario
class FormData(BaseModel):
    name: str
    state: int
    city: int

@app.post("/recommend")
async def recommend(data: FormData):
    """
    Maneja el formulario y devuelve recomendaciones.
    """
    try:
        recommendations = get_recommendations(data.state, data.city)
        return {"recommendations": recommendations}
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# Definir modelo de entrada para la solicitud
class RecommendationRequest(BaseModel):
    keyword: str
    city: Optional[str] = None
    avg_rating: Optional[float] = None
    business_name: Optional[str] = None
    region: Optional[str] = None
    
# Endpoint para obtener recomendaciones
@app.post("/recommendations")
def recommend(request: RecommendationRequest):
    #svd = TruncatedSVD(n_components=100)
    try:
        recommendations = get_recommendations(
            keyword=request.keyword,
            city=request.city
        )
        return {"recommendations": recommendations}
    except Exception as e:
        return {"error": str(e)}