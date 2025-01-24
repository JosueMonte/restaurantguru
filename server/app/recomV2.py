# 1. Importar libreria para el modelo de ML
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ndcg_score
from sklearn.metrics import ndcg_score
from sklearn.model_selection import KFold
from google.cloud import bigquery

# 2. Importar libreria para vincular con Big Query
# Esta parte del código lo dejo para cuando te vincules a Big Query. Yo voy a usar un modelo 3 que te voy a pasar
client = bigquery.Client()
#proyectrestaurant-447114.negocios_comida.mlasentiment
query = """
    SELECT * FROM `proyectrestaurant-447114.negocios_comida.mlasentiment`
    
"""
df_ml_4_1 = client.query(query).to_dataframe()

# 3. Descargar el dataset (yo voy a trabajar en local, pero esto habría que cambiarlo para que se vincule directamente con Big Query)
#df_ml_4_1 = pd.read_parquet('modelo4.parquet')

# 4. Funcion 1:


# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')

# Fit and transform the data
tfidf_matrix = vectorizer.fit_transform(df_ml_4_1['description'])


def get_recommendations2(keyword, city=None, avg_rating=None, business_name=None, region=None, aspect=None, df=df_ml_4_1, vectorizer=vectorizer, svd=None):
    filtered_df = df.copy()
    if city:
        filtered_df = filtered_df[filtered_df['city_name'] == city]
    if avg_rating:
        filtered_df = filtered_df[filtered_df['avg_rating'] >= avg_rating]
    if business_name:
        filtered_df = filtered_df[filtered_df['business_name'].str.contains(
            business_name, case=False)]
    if region:
        filtered_df = filtered_df[filtered_df['region'].str.contains(
            region, case=False)]
    if aspect:
        filtered_df = filtered_df[filtered_df[aspect] == 'POSITIVE']

    if filtered_df.empty:
        return pd.DataFrame({'business_name': [], 'similarity': [], 'city_name': [], 'avg_rating': [], 'region': [], 'weighted_rating': []})

    filtered_df = filtered_df.drop_duplicates(subset=['id_business'])
    filtered_df['description'] = filtered_df['description'].fillna('')
    tfidf_matrix_filtered = vectorizer.transform(filtered_df['description'])

    n_features = tfidf_matrix_filtered.shape[1]
    n_components = min(100, n_features)
    if svd:
        svd.n_components = n_components
        tfidf_matrix_filtered = svd.fit_transform(tfidf_matrix_filtered)

    keyword_tfidf = vectorizer.transform([keyword])
    if svd:
        keyword_tfidf = svd.transform(keyword_tfidf)

    keyword_sim = cosine_similarity(keyword_tfidf, tfidf_matrix_filtered)
    sim_scores = list(enumerate(keyword_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    filtered_scores = [score for score in sim_scores if score[1] > 0.1]

    top_indices = [i[0] for i in filtered_scores[:3]]
    scores = [i[1] for i in filtered_scores[:3]]
    similarity_percentages = [f"{round(score * 100)}%" for score in scores]

    recommendations = pd.DataFrame({
        'business_name': filtered_df['business_name'].iloc[top_indices].values,
        'category_name': filtered_df['category_name'].iloc[top_indices].values,
        'similarity': similarity_percentages,
        'city_name': filtered_df['city_name'].iloc[top_indices].values,
        'business_address': filtered_df['business_address'].iloc[top_indices].values,
        'avg_rating': filtered_df['avg_rating'].iloc[top_indices].values,
        'region': filtered_df['region'].iloc[top_indices].values,
        'hours': filtered_df['hours'].iloc[top_indices].values,
        f'{aspect}': filtered_df[f'{aspect}'].iloc[top_indices].values,
        # Agregar la columna weighted_rating
        'weighted_rating': filtered_df['weighted_rating'].iloc[top_indices].values
    })

    # Ordenar por weighted_rating de forma descendente
    recommendations = recommendations.sort_values(
        by='weighted_rating', ascending=False)

    # Dropear la columna weighted_rating antes de mostrar
    recommendations = recommendations.drop(columns=['weighted_rating'])

    return recommendations.to_dict(orient='records')


# Ejemplo de llamada a la función con análisis de aspecto
# recommended = get_recommendations2("rice", aspect='price_sentiment')
# print(f"Recommended Restaurants:{recommended}")

