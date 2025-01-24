from google.cloud import bigquery
#from huggingface_hub import InferenceClient

# Inicializa el cliente de BigQuery
client = bigquery.Client()

# Inicializa el cliente de Hugging Face
#huggingface_client = InferenceClient(model="tiiuae/falcon-7b-instruct", token="")


def get_states():
    """
    Consulta para obtener los estados disponibles.
    """
    query = """
        SELECT id, state_name
        FROM `proyectrestaurant-447114.negocios_comida.states`
        ORDER BY state_name
    """
    query_job = client.query(query)
    # Devuelve los resultados como una lista de diccionarios
    return [{"id": row.id, "name": row.state_name} for row in query_job]

def get_cities(id_state):
    """
    Consulta para obtener las ciudades de un estado específico.
    """
    query = """
        SELECT city_name
        FROM `proyectrestaurant-447114.negocios_comida.cities`
        WHERE id_state = @id_state
        GROUP BY city_name
        ORDER BY city_name
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("id_state", "INT64", id_state)
        ]
    )
    query_job = client.query(query, job_config=job_config)
    # Devuelve los resultados como una lista de diccionarios
    return [{"name": row.city_name} for row in query_job]

def fetch_reviews(restaurant_id):
    """
    Obtiene las reseñas de un restaurante específico desde BigQuery.
    """
    query = f"""
        SELECT text
        FROM `negocios_comida.reviews`
        WHERE id_business = @restaurant_id
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("restaurant_id", "INT64", restaurant_id)]
    )
    query_job = client.query(query, job_config=job_config)
    reviews = [row.text for row in query_job if row.text is not None]
    return " ".join(reviews[:5])  # Limita a las primeras 5 reseñas

# def generate_description_with_falcon(reviews):
#     """
#     Usa Hugging Face Falcon para generar una descripción basada en las reseñas.
#     """
#     if not reviews.strip():  # Verifica si las reseñas están vacías
#         return "No hay suficiente información para generar una descripción."

#     try:
#         response = huggingface_client.text_generation(reviews, max_new_tokens=200, temperature=0.7)
#         return response
#     except Exception as e:
#         print(f"Error con Hugging Face Falcon: {e}")
#         return "Descripción no disponible"

def get_recommendations(state, city):
    """
    Genera recomendaciones de restaurantes con sus descripciones.
    """
    try:
        # Simula IDs de restaurantes recomendados
        recommended_ids = [1, 2, 6]

        # Query para información básica de los restaurantes
        query_business = f"""
            SELECT b.id, b.business_name, b.address, b.latitude, b.longitude, c.city_name
            FROM `negocios_comida.business` b
            LEFT JOIN `negocios_comida.cities` c ON c.id = b.id_city
            WHERE b.id IN UNNEST(@ids)
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ArrayQueryParameter("ids", "INT64", recommended_ids)]
        )
        business_result = client.query(query_business, job_config=job_config).result()

        # Construye la respuesta
        recommendations = []
        for row in business_result:
            reviews = fetch_reviews(row.id)
            #description = generate_description_with_falcon(reviews)
            recommendations.append({
                "id": row.id,
                "name": row.business_name,
                "address": row.address,
                "latitude": row.latitude,
                "longitude": row.longitude,
                "city": row.city_name,
                #"description": description
            })

        return recommendations

    except Exception as e:
        raise RuntimeError(f"Error al obtener recomendaciones: {e}")