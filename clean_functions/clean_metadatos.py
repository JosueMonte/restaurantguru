import os
import json
import pandas as pd
import re

def process_metadata_files(input_dir, output_file):
    """
    Procesa archivos JSON de metadatos para extraer información de restaurantes.

    Args:
        input_dir (str): Ruta de la carpeta que contiene los archivos JSON.
        output_file (str): Ruta del archivo de salida en formato Parquet.

    Returns:
        None: Genera un archivo .parquet con los datos procesados.
    """

    # Inicializamos una lista para almacenar los datos procesados
    processed_data = []

    # Función para extraer componentes de la dirección
    def extract_address_components(address, name):
        try:
            # Eliminar el nombre del lugar de la dirección
            address = address.replace(name + ", ", "")
            # Extraer partes de la dirección usando una expresión regular
            match = re.match(r"^(.*),\s*(.*),\s*(\w{2})\s*(\d{5})$", address)
            if match:
                street, city, state, postal_code = match.groups()
                return street, city, state, postal_code
        except Exception:
            pass
        return None, None, None, None

    # Función para leer archivos .json con manejo de errores
    def load_json_file(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)  # Intentar cargar como un único JSON
            except json.JSONDecodeError:
                # Intentar cargar línea por línea si el JSON está separado
                f.seek(0)  # Reiniciar el cursor del archivo
                return [json.loads(line) for line in f if line.strip()]  # Cargar línea por línea

    # Recorremos todos los archivos .json en la carpeta
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(input_dir, file_name)
            raw_data = load_json_file(file_path)

            # Verificar si raw_data es una lista o un diccionario
            if isinstance(raw_data, dict):
                raw_data = [raw_data]  # Convertir a lista si es un único diccionario

            for data in raw_data:
                # Filtrar por categorías que incluyan la palabra "restaurant"
                categories = data.get("category", [])
                if categories is None:
                    categories = []  # Convertimos None a una lista vacía

                if any("restaurant" in category.lower() for category in categories):
                    # Extraer componentes de la dirección
                    street, city, state, postal_code = extract_address_components(data.get("address", ""), data.get("name", ""))

                    # Agregar datos procesados a la lista
                    processed_data.append({
                        "gmap_id": data.get("gmap_id"),
                        "name": data.get("name"),
                        "address": street,
                        "city": city,
                        "state": state,
                        "postal_code": postal_code,
                        "latitude": data.get("latitude"),
                        "longitude": data.get("longitude"),
                        "stars": data.get("avg_rating"),
                        "review_count": data.get("num_of_reviews"),
                        "categories": categories,
                        "hours": data.get("hours")
                    })

    # Convertimos la lista de datos procesados en un DataFrame
    df = pd.DataFrame(processed_data)

    # Exportamos el DataFrame a un archivo .parquet
    df.to_parquet(output_file, index=False)
    print(f"Archivo exportado exitosamente en {output_file}")

# Ejemplo de uso
if __name__ == "__main__":
    input_directory = "../Datasets/Google/metadata-sitios"
    output_path = "../Data_cleaned/metadatosgoogle/filtered_restaurants.parquet"
    process_metadata_files(input_directory, output_path)
