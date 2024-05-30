import openai
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os


def main():
    load_dotenv()
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        raise ValueError("La clave API de OpenAI no está configurada en el archivo .env")

    openai.api_key = openai_api_key

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

    names, chunks = extract_and_summarize_urls(urls)

    save_to_file('chunks.py', 'chunks', chunks)
    save_to_file('names.py', 'names', names)


def extract_and_summarize_urls(urls):
    """Extrae el contenido de las URLs y genera resúmenes."""
    names = list(urls.keys())
    chunks = [extract_text_from_url(url) for url in urls.values()]
    return names, chunks


def save_to_file(filename, variable_name, data):
    """Guarda los datos en un archivo."""
    with open(filename, 'w') as file:
        file.write(f"{variable_name} = [\n")
        for item in data:
            file.write(f"   {repr(item)},\n")
        file.write("]\n")


def generate_summary(text, max_tokens=300):
    """Genera un resumen utilizando OpenAI."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"""Resumir el siguiente texto.
El resumen generado debe ser un string de python evitando comillas dobles y cualquier caracter especial.
El texto debe ser generado sin indicar que es un resumen, debe estar escrito como si fuera un articulo explicativo.
El texto debe evitar utilizar la palabra "resumen" o "resumir" para evitar redundancia.


Texto:
{text}


Resumen:
"""}
    ]

    client = openai.OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=max_tokens
    )
    summary = response.choices[0].message.content.strip().replace("{", "\\{").replace("}", "\\}")
    return summary


def extract_text_from_url(url):
    """Extrae el contenido de una URL y genera un resumen."""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.get_text()
    summary = generate_summary(text[:3000], max_tokens=300)
    return summary.replace('"', "'")


if __name__ == "__main__":
    main()
