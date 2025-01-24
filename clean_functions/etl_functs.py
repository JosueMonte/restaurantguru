import pandas as pd
import numpy as np
from clean_functions.merges_tk import *
from nltk.corpus import stopwords
import nltk

#
class DataPipeline:
    def __init__(self, yelp_path, google_path, yelp_reviews_path, google_reviews_path):
        self.df_yelp = pd.read_parquet(yelp_path)
        self.df_google = pd.read_parquet(google_path)
        self.df_rev_yelp = pd.read_parquet(yelp_reviews_path).drop(columns=["useful", "funny", "cool"]).rename(columns={'user_id': 'id_user'})
        self.df_rev_goo = pd.read_parquet(google_reviews_path).drop_duplicates(
            subset=['user_id', 'name', 'time', 'rating', 'text', 'pics', 'resp', 'gmap_id']
        ).rename(columns={'user_id': 'id_user'})
        self.cleaner = RestaurantDataCleaner()
        self.matcher = FuzzyMatcher()
        self.merger = DataMerger(self.cleaner, self.matcher)
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))

    def merge_data(self):
        result = self.merger.merge_datasets(self.df_google, self.df_yelp)
        # result.rename(columns={'gmap_id': 'id_G'}, inplace=True)
        result['id'] = range(1, len(result) + 1)
        return result
    
    def _normalize_hours(self, value):
        """
        Normaliza los horarios a un formato estándar de diccionario
        """
        if isinstance(value, np.ndarray):
            return {day: self._format_hours(hours) for day, hours in value}
        elif isinstance(value, dict):
            return {day: hours for day, hours in value.items()}
        elif isinstance(value, list):
            return {}  # Maneja el caso de lista vacía
        return None

    def _format_hours(self, hours_str):
        """
        Formatea las cadenas de horario a un formato estándar
        """
        if isinstance(hours_str, str):
            if 'Open 24 hours' in hours_str or 'Open 24 h' in hours_str:
                return '0:0-0:0'
        return hours_str

    def _normalize_hours_column(self, result):
        """
        Aplica la normalización a la columna 'hours' del DataFrame
        """
        if 'hours' in result.columns:
            result['hours'] = result['hours'].apply(self._normalize_hours)
        return result

    def process_reviews(self, result):
        ids_google = result[['id', 'id_G']].dropna().drop_duplicates()
        ids_yelp = result[['id', 'id_Y']].dropna().drop_duplicates()
        ids_google.rename(columns={"id": "id_business"}, inplace=True)
        ids_yelp.rename(columns={"id": "id_business"}, inplace=True)
        # Formateamos el dataset de reviews de Yelp
        self.df_rev_yelp['origin'] = 'Y'
        self.df_rev_yelp.rename(columns={"stars_y": "rating", "business_id": "id_Y"}, inplace=True)

        # Formateamos el dataset de reviews de Google
        self.df_rev_goo['time'] = pd.to_datetime(self.df_rev_goo['time'], unit='ms')
        self.df_rev_goo['origin'] = 'G'
        self.df_rev_goo['review_id'] = range(1, len(self.df_rev_goo) + 1)
        self.df_rev_goo.drop(columns=["name", "pics", "resp"], inplace=True)
        self.df_rev_goo.rename(columns={"time": "date"}, inplace=True)

        df_goo_merged = pd.merge(self.df_rev_goo, ids_google, left_on='gmap_id', right_on='id_G', how='inner')
        df_goo_merged['review_id'] = range(1, len(df_goo_merged) + 1)
        df_yelp_merged = pd.merge(self.df_rev_yelp, ids_yelp, on='id_Y', how='inner')
        
        df_rev = pd.concat([df_yelp_merged, df_goo_merged], ignore_index=True)
        df_rev= df_rev.rename(columns={'review_id':'id'})
        df_rev['id_business'] = df_rev['id_business'].astype(str)
        df_rev = df_rev[df_rev['id_business'] != 'nan']
        df_rev['id_business'] = df_rev['id_business'].astype(int) #### Revisar!
        df_rev = self.process_business_ids(df_rev)
        df_rev['id'] = df_rev['id'].astype(str)
        df_rev = df_rev.drop(columns=['gmap_id'])
        df_rev['date'] = df_rev['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df_rev.to_parquet("ETL/reviews.parquet", index=False)
        return df_rev


    def process_business_ids(self, df):
        df = df.copy()
        df.loc[(df['id_business'].isna()) & (df['id_Y'].notna()), 'id_business'] = df['id_Y']
        df.loc[(df['id_business'].isna()) & (df['id_G'].notna()), 'id_business'] = df['id_G']
        return df.drop(columns=['id_G', 'id_Y'])

    def create_states_and_cities(self, result):
        cities_df = pd.read_csv("clean_functions/uscities.csv")
        state_mapping = cities_df[["state_id", "state_name"]].drop_duplicates()
        states = result[["state"]].drop_duplicates().reset_index(drop=True)
        states["id"] = range(1, len(states) + 1)
        states.rename(columns={"state": "state_code"}, inplace=True)
        states = states.merge(
        state_mapping,
        left_on="state_code",
        right_on="state_id",
        how="left"
    )
        states = states[["id", "state_code", "state_name"]]
        states.to_parquet("ETL/states.parquet", index=False)

        cities = result[["city", "postal_code", "state"]].drop_duplicates().reset_index(drop=True)
        cities = cities.merge(states, left_on="state", right_on="state_code").drop(columns=["state_code", "state", "state_name"])
        cities.rename(columns={"id": "id_state", "city": "city_name"}, inplace=True)
        cities["id"] = range(1, len(cities) + 1)
        cities.to_parquet("ETL/cities.parquet", index=False)

        return states, cities

    def clean_category(self, cat):
        return ' '.join([word for word in cat.split() if word.lower() not in self.stop_words])

    def create_business_tables(self, result, cities_df):
        # Filtrar categorías que contienen la palabra "restaurant"
        filtered_categories = [
            self.clean_category(cat) for sublist in result["categories"] for cat in sublist
            if "restaurant" in self.clean_category(cat).lower()
        ]

        # Crear el DataFrame de categorías
        categories = pd.DataFrame(
            filtered_categories,
            columns=["category_name"]
        ).drop_duplicates().reset_index(drop=True)
        categories["id_category"] = range(1, len(categories) + 1)
        categories.to_parquet("ETL/categories.parquet", index=False)

        # Crear el DataFrame de negocios
        business = result[["id", "id_G", "id_Y", "name",'stars', "city", "postal_code", "latitude", "longitude", 'address', 'hours']].copy()
        business = business.merge(
            cities_df[["postal_code", "city_name", "id"]],
            left_on=["postal_code", "city"],
            right_on=["postal_code", "city_name"],
            how="left"
        ).rename(columns={"id_x": "id", "id_y": "id_city", "name": "business_name"}).drop(columns=["city_name", "city", "postal_code"])
        business = self._normalize_hours_column(business)
        business.drop_duplicates(subset=['id_G','id_Y', 'business_name']).to_parquet("ETL/business.parquet", index=False)

        # Crear el DataFrame de relaciones negocio-categoría
        business_categories_rows = []
        for idx, row in result.iterrows():
            business_id = row["id"]
            for category in row["categories"]:
                cleaned_category = self.clean_category(category)
                if "restaurant" in cleaned_category.lower():
                    category_id = categories[categories["category_name"] == cleaned_category]["id_category"].iloc[0]
                    business_categories_rows.append({
                        "id_business": business_id,
                        "id_category": category_id ## Revisar!!
                    })
        categories = categories.rename(columns={'id_category': 'id'}) ## Revisar!!
        business_categories = pd.DataFrame(business_categories_rows)
        business_categories["id"] = range(1, len(business_categories) + 1)
        business_categories.to_parquet("ETL/business_categories.parquet", index=False)

        return business, categories, business_categories


    def create_users_table(self, df_rev):
        users = df_rev[["id_user"]].drop_duplicates()
        users = pd.DataFrame(users).rename(columns={'id_user': 'id'})
        users.to_parquet("ETL/users.parquet", index=False)
        return users
