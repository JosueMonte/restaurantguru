import os
import json
import re
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from typing import Dict, List, Tuple
##
class LocalGeocoder:
    def __init__(self, csv_path='clean_functions/uscities.csv'):
        # Cargar y preprocesar el dataframe una vez
        self.cities_df = pd.read_csv(csv_path)
        self.cities_df[['lat_rad', 'lng_rad']] = np.deg2rad(self.cities_df[['lat', 'lng']])
        self.kdtree = cKDTree(self.cities_df[['lat_rad', 'lng_rad']].values)

    def find_city(self, zip_code, lat, lon, zip_radius=5):
        query_point = np.deg2rad([[lat, lon]])
        zip_str = str(zip_code)[:zip_radius]
        possible_matches = self.cities_df[
            self.cities_df['zips'].astype(str).str.startswith(zip_str)
        ]
        
        if possible_matches.empty:
            return None
        
        tree = cKDTree(possible_matches[['lat_rad', 'lng_rad']].values)
        distance, index = tree.query(query_point, k=1)
        
        return possible_matches.iloc[index]['city'] if len(index) > 0 else None

def process_metadata_files(input_dir, output_file, df_info_states):
    processed_data = []

    def extract_address_components(address: str, name: str, city: str, state_id: str, zips: str) -> Tuple[str, str, str, str]:
        try:
            # Remove name from the address for cleaner processing
            address = address.replace(name + ", ", "")
            # Check for full format: street, city, state, postal code, country
            match_full = re.match(r"^(.*),\s*(.*),\s*(\w{2})\s*(\d{5}),\s*(.*)$", address)
            if match_full:
                street, city, state, postal_code, country = match_full.groups()
                return street, city, state, postal_code

            # Check for shorter format: street, city, state, postal code
            match_short = re.match(r"^(.*),\s*(.*),\s*(\w{2})\s*(\d{5})$", address)
            if match_short:
                street, city, state, postal_code = match_short.groups()
                return street, city, state, postal_code

        except Exception:
            pass
            # Return fallback values if parsing fails
        return address, city, state_id, zips[:5]

    def load_json_file(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                f.seek(0)
                return [json.loads(line) for line in f if line.strip()]

    for file_name in os.listdir(input_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(input_dir, file_name)
            raw_data = load_json_file(file_path)

            if isinstance(raw_data, dict):
                raw_data = [raw_data]

            for data in raw_data:
                categories = data.get("category", []) or []
                
                if any("restaurant" in category.lower() for category in categories):
                    street, city, state, postal_code = extract_address_components(
                        data.get("address", ""), 
                        data.get("name", ""), 
                        data.get("city", ""), 
                        data.get("state_id", ""), 
                        data.get("zips", "")
                    )
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

    df = pd.DataFrame(processed_data)

    # Uso del geocoder para completar ciudades
    geocoder = LocalGeocoder()
    df['city'] = df.apply(
        lambda row: geocoder.find_city(row['postal_code'], row['latitude'], row['longitude']), 
        axis=1
    )

    df.dropna(subset=['city'], inplace=True)

    # Solución al error de tipo de datos en la columna 'city'
    df['city'] = df['city'].apply(lambda x: str(x) if x is not None else None)

    # Exportar a parquet
    df.to_parquet(output_file, index=False)
    print(f"Archivo exportado exitosamente en {output_file}")


def clean_and_merge_reviews(folder_path, output_folder, gmap_id_list):
    # Crear carpeta de salida si no existe
    os.makedirs(output_folder, exist_ok=True)

    # Iterar sobre las carpetas (estados)
    for state_folder in os.listdir(folder_path):
        state_path = os.path.join(folder_path, state_folder)
        if os.path.isdir(state_path) and state_folder.startswith("review-"):
            
            state_name = state_folder.replace("review-", "")

            state_data = []

            # Iterar sobre los archivos JSON dentro de cada carpeta de estado
            for file_name in os.listdir(state_path):
                if file_name.endswith(".json"):
                    file_path = os.path.join(state_path, file_name)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        try:
                            for line in f:  # Leer línea por línea para manejar JSON con múltiples objetos
                                state_data.append(json.loads(line))
                        except json.JSONDecodeError:
                            print(f"Error al decodificar el archivo: {file_path}")

            # Crear un DataFrame para los datos de este estado
            state_df = pd.DataFrame(state_data)

            # Filtrar por gmap_id
            state_df = state_df[state_df['gmap_id'].isin(gmap_id_list)]

            # Convertir columnas con listas o diccionarios en cadenas JSON para que sean hashables
            for col in state_df.columns:
                if state_df[col].apply(lambda x: isinstance(x, (list, dict))).any():
                    state_df[col] = state_df[col].apply(json.dumps)

            # Eliminar duplicados solo si toda la fila es duplicada
            initial_rows = state_df.shape[0]
            state_df.drop_duplicates(inplace=True)
            duplicates_removed = initial_rows - state_df.shape[0]
            
            # Eliminar valores nulos en las columnas clave
            initial_rows = state_df.shape[0]
            state_df.dropna(subset=['user_id', 'name', 'time', 'rating', 'gmap_id'], inplace=True)
            nulls_removed = initial_rows - state_df.shape[0]

            # Informar de los datos eliminados
            print(f"En el estado {state_name} se eliminaron {duplicates_removed} duplicados y {nulls_removed} valores nulos.")

            # Guardar los datos del estado en un archivo individual
            state_output_path = os.path.join(output_folder, f"{state_name}_cleaned.parquet")
            state_df.to_parquet(state_output_path, index=False)
            print(f"Datos del estado {state_name} guardados en {state_output_path}")

    # Unir todos los archivos parquet generados
    all_files = [os.path.join(output_folder, file) for file in os.listdir(output_folder) if file.endswith(".parquet")]
    final_df = pd.concat([pd.read_parquet(file) for file in all_files], ignore_index=True).drop_duplicates(inplace=True)

    # Mostrar la cantidad total de filas
    print(f"Dataset final: {final_df.shape[0]} filas después de limpiar y unir.")
    return final_df

# Ejemplo de uso
if __name__ == "__main__":
    input_directory = "../Datasets/Google/metadata-sitios"
    output_path = "../Data_cleaned/metadatosgoogle/filtered_restaurants.parquet"
    process_metadata_files(input_directory, output_path)
