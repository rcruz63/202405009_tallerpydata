import openai
import numpy as np
from functools import lru_cache
from dotenv import load_dotenv
import os
import sys
# import time
from chunks import chunks
from names import names


def main(query):

    response = chatbot_rag(query)
    print(response.choices[0].message.content)


def init(query):
    global client, _cache, embeddings

    load_dotenv()
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        raise ValueError("La clave API de OpenAI no está configurada en el archivo .env")

    client = openai.OpenAI(api_key=openai_api_key)

    # Cache para las respuestas
    _cache = {}
    # star_time = time.time()
    embeddings = [get_embedding(chunk) for chunk in chunks]
    # end_time = time.time()
    # print(f"Embeddings took {end_time - star_time:.2f} seconds")


# Función para crear claves de caché
def make_cache_key(kwargs):
    def convert_to_hashable(value):
        if isinstance(value, dict):
            return frozenset((k, convert_to_hashable(v)) for k, v in sorted(value.items()))
        elif isinstance(value, list):
            return tuple(convert_to_hashable(v) for v in value)
        elif isinstance(value, set):
            return frozenset(convert_to_hashable(v) for v in value)
        else:
            return value

    return frozenset((k, convert_to_hashable(v)) for k, v in sorted(kwargs.items()))


# Función que interactúa con la API de OpenAI y utiliza la caché
def llm(**kwargs):
    key = make_cache_key(kwargs)
    try:
        return _cache[key]
    except KeyError:
        _cache[key] = result = client.chat.completions.create(**kwargs)
        return result


# Decorador para la caché LRU
@lru_cache(3000)
def get_embedding(text, model="text-embedding-3-large"):
    client = openai.OpenAI()
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


# Función para calcular la similitud coseno
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# Función para consultar la base de datos
def query_bbdd(query, top_n=5):
    # start_time = time.time()
    query_embedding = get_embedding(query)

    # Calcular la similitud coseno entre la consulta y cada embedding de los chunks
    similarities = [cosine_similarity(query_embedding, embedding) for embedding in embeddings]

    # Obtener los índices de los top-n chunks más similares
    top_indices = np.argsort(similarities)[-top_n:]

    # Recuperar los top-n chunks más similares junto con su similitud coseno
    top_chunks = [(names[i], chunks[i], similarities[i]) for i in top_indices]
    # end_time = time.time()
    # print(f"query_bbdd took {end_time - start_time:.2f} seconds")

    return top_chunks


def chatbot_rag(pregunta):
    init(pregunta)
    # start_time = time.time()
    results = query_bbdd(pregunta)
    if len(results) == 0:
        print("Unable to find matching results.")
        return

    context = "\n\n---\n\n".join([doc for _name, doc, _score in results])
    messages = [
        {"role": "system", "content": f"""
Eres un chatbot experto en la biblioteca estándar de Python.

Debes ayudar a explicar código Python, comentar código Python, corregir código Python o desarrollar código Python.
NO debes contestar ninguna pregunta NO relacionada con Python.
Si te hacen alguna pregunta no relacionada con Python DEBES rechazar responder amablemente y hacer un chiste.
Antes de rechazar una pregunta asegurate que no está relacionada con Python. 
Por ejemplo pandas es una librería de Python para análisis de datos.

Si tienes que insertar código python en los ejemplos DEBES enmarcarlo asi:

```python
<<El codigo a insertar (no incluir los simbolos << >>)>>
```


Utiliza la siguiente base de conocimientos para responder a la pregunta del usuario:

Base de conocimientos: {context}

"""},
        {"role": "user", "content": pregunta}
    ]
    # print(prompt)

    response_text = llm(max_tokens=3000, model="gpt-4o", messages=messages)
    # end_time = time.time()
    # print(f"chatbot_rag took {end_time - start_time:.2f} seconds")

    # print(response_text)
    return response_text


if __name__ == "__main__":

    # Ejemplo de uso del chatbot
    query = sys.argv[1]

    main(query)
