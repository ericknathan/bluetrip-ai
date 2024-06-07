
## Importações
import io

from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from typing import Annotated

from fastapi import FastAPI, File
from fastapi.middleware.cors import CORSMiddleware

import uvicorn

## Carregamento do modelo gerado pelo Teachable Machine e processo de Labelling
model = load_model("./model/keras_model.h5", compile=False)
class_names = open("./model/labels.txt", "r", encoding="utf8").readlines()

## Instanciando o FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

## Criação das rotas para conexão com o front-end
### Rota de status para verificar se a aplicação está sendo executada
#### [GET] /status
@app.get("/status")
async def app_status():
    return 'ok'

### Rota para predição de imagem enviada no corpo da requisição
#### [POST] /identify-specie - file: File
@app.post("/identify-specie")
async def identify_specie(file: Annotated[bytes, File()]):
    image = Image.open(io.BytesIO(file)).convert("RGB")

    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index][2:].strip()
    confidence_score = prediction[0][index]

    return {
        "name": class_name,
        "score": round(confidence_score * 100, 3),
        "type": "Tubarão" if class_name.startswith("Tubarão") else "Peixe"
    }
