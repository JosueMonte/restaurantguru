# Diccionario de datos

## Yelp y Google Maps 
Consideraciones:
* Las variables seleccionadas son estimativas. Hay que probar y revisar.
* Aún resta cruzar los datasets de Yelp y Google Maps pero la idea es compartir las tablas con ambos datos.

### business.pkl
El archivo `business.pkl` contiene un conjunto de datos sobre negocios y sus características clave. Incluye tales atributos:
* id_business
* name
* address
* id_city
* id_business_category
* latitude
* longitude
* postal_code
* rating
* review_count
  
### checkin.json
El archivo `checkin.json` contiene datos sobre los registros (check-ins) realizados en los negocios, proporcionando información sobre la frecuencia y los patrones de visitas de los usuarios. Incluye tales atributos:
* id_business
* date

### review.json
El archivo `review.json` contiene un conjunto de datos detallado sobre las reseñas realizadas por los usuarios para diversos negocios. Incluye tales atributos:
* id_review
* id_user
* id_business
* rating
* text
* date
* useful
* funny
* cool

### tip.json
El archivo `tip.json` contiene información sobre los consejos (tips) proporcionados por los usuarios de Yelp para diversos negocios. Incluye tales atributos:
* id_review
* id_user
* id_business
* text
* date
* compliment_count

### user.parquet
* id_user
* name
* review_count
* elite
* friends
* fans
* useful
* funny
* cool
