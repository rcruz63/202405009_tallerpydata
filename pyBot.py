import openai
import numpy as np
import inspect
from functools import lru_cache
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os
import time


def main():
    global client
    global openai_api_key
    global _cache
    global urls
    global chunks
    global names

    load_dotenv()
    openai_api_key = os.getenv('OPENAI_API_KEY')

    client = openai.OpenAI()

    # Cache para las respuestas
    _cache = {}

    # URLs de las que extraer contenido
    urls = {
        "Definir una Función": "https://docs.python.org/3/tutorial/controlflow.html#defining-functions",
        "Listas en Python": "https://docs.python.org/3/tutorial/introduction.html#lists",
        "Tuplas y Secuencias": "https://docs.python.org/3/tutorial/datastructures.html#tuples-and-sequences",
        "Diccionarios en Python": "https://docs.python.org/3/tutorial/datastructures.html#dictionaries",
        "Sets": "https://docs.python.org/3/tutorial/datastructures.html#sets",
        "Estructuras de Control de Flujo": "https://docs.python.org/3/tutorial/controlflow.html",
        "Sentencias Condicionales": "https://docs.python.org/3/tutorial/controlflow.html#if-statements",
        "Bucles While": "https://docs.python.org/3/tutorial/introduction.html#first-steps-towards-programming",
        "Bucles For": "https://docs.python.org/3/tutorial/controlflow.html#for-statements",
        "Funciones Lambda": "https://docs.python.org/3/tutorial/controlflow.html#lambda-expressions",
        "Manejo de Excepciones": "https://docs.python.org/3/tutorial/errors.html",
        "Clases y Objetos": "https://docs.python.org/3/tutorial/classes.html",
        "Herencia en Python": "https://docs.python.org/3/tutorial/classes.html#inheritance",
        "Módulos en Python": "https://docs.python.org/3/tutorial/modules.html",
        "Entrada y Salida de Archivos": "https://docs.python.org/3/tutorial/inputoutput.html",
        "Manejo de Argumentos de Línea de Comandos": "https://docs.python.org/3/library/argparse.html",
        "Expresiones Regulares": "https://docs.python.org/3/library/re.html",
        "Gestión de Fechas y Horas": "https://docs.python.org/3/library/datetime.html",
        "Operaciones de Sistema": "https://docs.python.org/3/library/os.html",
        "Interactuando con el Sistema Operativo": "https://docs.python.org/3/library/os.path.html"
    }

    # Extraer el contenido de las URLs y calcular sus embeddings
    names = list(urls.keys())
    chunks = [extract_text_from_url(url) for url in urls.values()]

    with open('chunks.py', 'w') as file:
        file.write("chunks = [\n")
        for chunk in chunks:
            file.write(f"   \"{str(chunk)}\",\n")
        file.write("]\n\n")
    with open('names.py', 'w') as file:
        file.write("names = [\n")
        for name in names:
            file.write(f"   \"{str(name)}\",\n")
        file.write("]\n\n")


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


# Función para obtener funciones de un módulo
def get_module_functions(module_name):
    module = __import__(module_name)
    functions = []

    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj) or inspect.isbuiltin(obj):
            ds = inspect.getdoc(obj)
            function_info = {
                "name": name,
                "docstring": ds
            }
            if ds:
                functions.append(function_info)

    return functions


# Función para generar resúmenes utilizando OpenAI
def generate_summary(text, max_tokens=3000):
    prompt = f"Resumir el siguiente texto:\n{text}\n\nResumen:"
    start_time = time.time()
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",  # Selecciona el modelo adecuado
        prompt=prompt,
        max_tokens=max_tokens
    )
    end_time = time.time()
    print(f"generate_summary took {end_time - start_time:.2f} seconds")
    summary = response.choices[0].text.strip()
    return summary


# Función para calcular la similitud coseno
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# Función para extraer contenido de una URL
def extract_text_from_url(url):
    start_time = time.time()
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.get_text()
    summary = generate_summary(text[:3000], max_tokens=300)
    # paragraphs = soup.find_all('p')
    # return ' '.join([para.get_text() for para in paragraphs])
    end_time = time.time()
    print(f"extract_text_from_url took {end_time - start_time:.2f} seconds")
    return summary.replace('"', "'")


if __name__ == "__main__":
    main()
