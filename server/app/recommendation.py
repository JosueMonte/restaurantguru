from google.cloud import bigquery
from google.cloud import storage
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io
import logging

# Configuración del logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializa los clientes
bigquery_client = bigquery.Client()
#client = InferenceAPIClient("meta-llama/Llama-2-7b-chat-hf", token="")

credentials, project = bigquery_client._credentials, bigquery_client.project
storage_client = storage.Client(credentials=credentials)

# Define el bucket y el archivo del dataset
BUCKET_NAME = "datosfuentes"
FILE_NAME = "tablas/modelo3.parquet"

_cached_dataset = None
# Carga el dataset desde Google Cloud Storage
def load_dataset():

    global _cached_dataset
    if _cached_dataset is not None:
        logger.info("Usando el dataset cacheado en memoria.")
        return _cached_dataset

    try:
        logger.info(f"Cargando el dataset desde {BUCKET_NAME}/{FILE_NAME}")
        bucket = storage_client.get_bucket(BUCKET_NAME)
        blob = bucket.blob(FILE_NAME)
        file_contents = blob.download_as_string()
        df = pd.read_parquet(io.BytesIO(file_contents))

        # Preprocesamiento inicial
        logger.info("Preprocesando el dataset...")
        df = df.drop_duplicates(subset=['id_business', 'business_name', 'city_name', 'business_address'])
        df = df[(df['date'].dt.year == 2021)]
        df = df[(df['avg_rating'] >= 4)]
        df['description'] = df.get('description', pd.Series()).fillna('')
        logger.info(f"Dataset cargado y preprocesado: {df.shape[0]} filas, {df.shape[1]} columnas")

        _cached_dataset = df
        return _cached_dataset
    
    except Exception as e:
        logger.error(f"Error al cargar el dataset: {e}")
        raise RuntimeError(f"Error al cargar el dataset: {e}")

# def generate_description_with_llama(reviews, business_description, business_name):
#     """
#     Genera una descripción breve y atractiva de un restaurante utilizando LLaMA a través de la biblioteca text-generation.
#     """
#     if not business_name.strip():
#         logging.warning("Nombre del negocio vacío. No se puede generar una descripción personalizada.")
#         business_name = "Este restaurante"

#     # Crear el prompt optimizado
#     prompt = (
#         f"Crea una descripción breve y atractiva para un restaurante llamado '{business_name}'. "
#         f"Incluye información clave basada en esta descripción del restaurante y las siguientes opiniones de los clientes: "
#         f"\n\nDescripción del restaurante: {business_description}."
#         f"\n\nOpiniones de los clientes: {reviews}."
#         f"\n\nLa descripción debe ser profesional y atractiva."
#     ).strip()

#     try:
#         logging.info(f"Generando descripción con LLaMA para {business_name}...")

#         response = client.generate(
#             prompt,
#             max_new_tokens=200,
#             temperature=0.7,
#             top_p=0.95,
#         )

#         return response.generated_text
#     except Exception as e:
#         logging.error(f"Error al generar descripción con LLaMA: {e}")
#         return "Descripción no disponible"
    
# Obtiene reseñas desde BigQuery
def fetch_reviews_and_coordinates(restaurant_id):
    """
    Obtiene las reseñas y las coordenadas de un restaurante específico desde BigQuery.
    Convierte los tipos de datos a tipos nativos de Python para evitar errores de serialización.
    """
    try:
        logger.info(f"Consultando reseñas y coordenadas para id_business={restaurant_id}")
        query = f"""
            SELECT b.text AS review_text, c.latitude, c.longitude
            FROM `negocios_comida.reviews` b
            JOIN `negocios_comida.business` c ON b.id_business = c.id
            WHERE b.text is not null AND b.rating >= 4 and b.id_business = {restaurant_id} LIMIT 5
        """
        # job_config = bigquery.QueryJobConfig(
        #     query_parameters=[bigquery.ScalarQueryParameter("restaurant_id", "INT64", restaurant_id)]
        # )
        # query_job = bigquery_client.query(query, job_config=job_config)
        query_job = bigquery_client.query(query)
        reviews = []
        latitude, longitude = None, None

        for row in query_job:
            reviews.append(str(row.review_text))
        latitude = float(row.latitude) if row.latitude is not None else None
        longitude = float(row.longitude) if row.longitude is not None else None

        logger.info(f"Reseñas obtenidas: {len(reviews)}, Coordenadas: ({latitude}, {longitude})")
        return " ".join(reviews[:5]), latitude, longitude
    except Exception as e:
        logger.error(f"Error al obtener reseñas y coordenadas: {e}")
        raise RuntimeError(f"Error al obtener reseñas y coordenadas: {e}")


# Normaliza un DataFrame para que sea serializable en JSON
def normalize_dataframe(df):
    try:
        logger.info("Normalizando DataFrame para serialización JSON...")
        for col in df.select_dtypes(include=['int64', 'float64']).columns:
            df[col] = df[col].astype('object')
        return df
    except Exception as e:
        logger.error(f"Error al normalizar el DataFrame: {e}")
        raise RuntimeError(f"Error al normalizar el DataFrame: {e}")

# Función principal para obtener recomendaciones
def get_recommendations(keyword, city=None, avg_rating=None, business_name=None, region=None):
    print("keyword:",keyword)
    print("city:",city)
    try:
        # Carga el dataset
        df = load_dataset()

        # Filtra el dataframe según los parámetros proporcionados
        logger.info("Filtrando el DataFrame según los parámetros proporcionados...")
        filtered_df = df.copy()
        if city:
            filtered_df = filtered_df[filtered_df['city_name'] == city]
        if avg_rating:
            filtered_df = filtered_df[filtered_df['avg_rating'] >= avg_rating]
        if business_name:
            filtered_df = filtered_df[filtered_df['business_name'].str.contains(business_name, case=False)]
        if region:
            filtered_df = filtered_df[filtered_df['region'].str.contains(region, case=False)]

        if filtered_df.empty:
            logger.warning("El DataFrame filtrado está vacío. No se encontraron resultados.")
            return []

        # Vectorización para el modelo de recomendación
        logger.info("Vectorizando descripciones...")
        vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(filtered_df['description'].fillna(''))

        # Calcular similitudes
        keyword_tfidf = vectorizer.transform([keyword])
        similarity_scores = cosine_similarity(keyword_tfidf, tfidf_matrix).flatten()
        top_indices = similarity_scores.argsort()[-3:][::-1]
        top_scores = similarity_scores[top_indices]

        # Construir recomendaciones
        recommendations = []
        for idx, score in zip(top_indices, top_scores):
            business = filtered_df.iloc[idx]
            reviews, latitude, longitude = fetch_reviews_and_coordinates(business['id_business'])
            #description = generate_description_with_llama(reviews, business['description'], business['business_name'])
            recommendations.append({
                "id_business": business['id_business'],
                "business_name": business['business_name'],
                "category_name": business['category_name'],
                "city_name": business['city_name'],
                "business_address": business['business_address'],
                "latitude": latitude,
                "longitude": longitude,
                "avg_rating": business['avg_rating'],
                "region": business['region'],
                "hours": business['hours'],
                "similarity": f"{round(score * 100)}%",
                #"description": description
            })

        # Normalizar y retornar resultados
        recommendations_df = pd.DataFrame(recommendations)
        logger.info(f"Recomendaciones generadas: {recommendations_df.shape[0]} filas")
        recommendations_df = normalize_dataframe(recommendations_df)
        return recommendations_df.to_dict(orient='records')

    except Exception as e:
        logger.error(f"Error al obtener recomendaciones: {e}")
        raise RuntimeError(f"Error al obtener recomendaciones: {e}")
