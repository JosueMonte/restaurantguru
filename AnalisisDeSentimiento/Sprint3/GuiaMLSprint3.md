# EDA/ FEAUTURE SELECTION

# 1. Realizar un EDA profundo

El Análisis Exploratorio de Datos (EDA) es la base para entender los datos y tomar decisiones informadas. Implica:

* **Comprender los datos**: Analizar el contenido y formato de las columnas, las dimensiones del dataset y la calidad de los datos.
  * Identificar valores faltantes, duplicados o datos atípicos.
  * Verificar el tipo de datos en cada columna (numérico, categórico, etc.).

* **Analizar la distribución de las variables**: Usar visualizaciones para estudiar la distribución de datos y su comportamiento (por ejemplo, histogramas, boxplots).

* **Identificar relaciones entre variables**:
  * Calcular correlaciones entre variables numéricas.
  * Analizar interacciones entre variables categóricas y numéricas (por ejemplo, gráficas de barras, heatmaps).

* **Detectar patrones y tendencias**: Observar si hay patrones temporales, espaciales o tendencias que puedan ser útiles para el modelo.

**Objetivo del EDA:** Identificar qué variables podrían ser útiles para el modelo y comprender cómo las características afectan al objetivo que deseas predecir.

# 2. Selección de features

A partir del EDA, se deberá elegir las características más relevantes para el modelo. Pasos:

* **Eliminar características irrelevantes**:
  * Variables redundantes (por alta correlación).
  * Variables con baja varianza (que no aportan información).
  * Variables que no están relacionadas con el objetivo del modelo.

* **Transformar características si es necesario**:
  * Normalización/estandarización de variables numéricas.
  * Codificación de variables categóricas (por ejemplo, One-Hot Encoding, Label Encoding).

* **Incluir interacciones o nuevas características**:
  * Crear nuevas variables si se identificaron patrones o relaciones útiles.

**Resultado esperado:** Un conjunto de features limpio, relevante y preparado para alimentar al modelo de Machine Learning.

# 3. Fundamentar la elección del modelo

Se debe elegir un modelo de Machine Learning basado en lo observado en el EDA y en la naturaleza de los datos.

* **Criterios para elegir el modelo**:
  * **Tipo de problema**: Clasificación, regresión, clustering, etc.
  * **Distribución de datos**: Si las variables objetivo están desbalanceadas, se puede elegir modelos robustos frente a este problema o aplicar técnicas como oversampling/undersampling.
  * **Dimensionalidad del dataset**: Si hay muchas features, tal vez sea mejor un modelo que maneje bien datos de alta dimensionalidad (p. ej., Random Forest, XGBoost).
  * **Tamaño del dataset**: Algunos modelos, como redes neuronales, requieren datasets grandes para un buen desempeño.

**Ejemplo de fundamentación:** Si, a partir del EDA, se identifica un problema de clasificación con un dataset de tamaño moderado y no muchas features, podrías justificar el uso de un modelo como Random Forest por su capacidad para manejar datos con relaciones no lineales y su robustez frente a datos ruidosos.

# 4. Desarrollo basado en el EDA

Finalmente, se debe asegurar que todo el proceso (desde la limpieza de datos hasta la selección del modelo) está fundamentado en las conclusiones obtenidas en el EDA. Esto incluye:

* Explicar por qué se eliminó o transformó ciertas variables.
* Justificar las técnicas de preprocesamiento usadas.
* Explicar cómo el EDA influyó en la elección del modelo.

**Resultado esperado:** Un modelo bien fundamentado, con datos preparados y características seleccionadas que maximicen el desempeño del algoritmo.


# MODELO MACHINE LEARNING

# 1. Modelo funcional

Esto implica que el modelo debe estar operativo y capaz de realizar predicciones. Para llegar a esta etapa:

* **Entrenar el modelo inicial**: Usa los datos preparados (tras el EDA y la selección de features) para entrenar un modelo básico.

* **Verificar funcionalidad**: Asegúrate de que el modelo procesa los datos correctamente y produce salidas coherentes (predicciones).

* **Pipeline completo**: Si es necesario, implementa un pipeline que incluya preprocesamiento, transformación de datos y predicción para que el modelo sea completamente funcional.

**Ejemplo**: Entrenar un modelo de regresión lineal o Random Forest con hiperparámetros predeterminados como punto de partida.

# 2. Ajuste de parámetros (hiperparámetros)

Este paso se refiere a optimizar el modelo ajustando sus hiperparámetros (valores que configuran el modelo y afectan su rendimiento). Pasos:

* **Definir hiperparámetros a ajustar**: Seleccionar los parámetros clave para el modelo. Por ejemplo:
  * Árboles de decisión: Profundidad máxima, número mínimo de muestras por hoja.
  * Random Forest: Número de estimadores, profundidad máxima.
  * Redes neuronales: Cantidad de capas, tasa de aprendizaje, funciones de activación.

* **Elegir una técnica de optimización**:
  * **Grid Search**: Explora combinaciones de hiperparámetros dentro de un rango definido.
  * **Random Search**: Busca combinaciones al azar dentro del espacio de hiperparámetros.
  * **Bayesian Optimization o algoritmos avanzados**: Métodos más sofisticados para problemas complejos.

* **Validación cruzada (cross-validation)**: Divide los datos en varios subconjuntos para garantizar que los resultados sean robustos y no dependan de una única división de los datos.

**Resultado esperado**: Un conjunto de hiperparámetros optimizados que mejoren el rendimiento del modelo.

# 3. Métricas objetivo

Esto implica medir el desempeño del modelo usando métricas adecuadas al problema y asegurarte de que los resultados sean satisfactorios.

* **Seleccionar métricas adecuadas**:
  * **Regresión**: Error Cuadrático Medio (MSE), Error Absoluto Medio (MAE), R².
  * **Clasificación**: Precisión, Recall, F1-Score, AUC-ROC.
  * **Modelos desbalanceados**: Usar métricas como el F1-Score o el AUC-ROC en lugar de la precisión.

* **Evaluar resultados**:
  * Verifica si los valores de las métricas cumplen con los objetivos definidos previamente.
  * Identifica posibles áreas de mejora (por ejemplo, bajo rendimiento en clases minoritarias).

**Ejemplo**: Si el objetivo es clasificar correctamente opiniones positivas y negativas, se podría usar la métrica F1-Score para garantizar un equilibrio entre precisión y recall.

# 4. Arrojando resultados acordes

El modelo debe ser capaz de generar resultados que estén alineados con las expectativas o requisitos del proyecto. Esto implica:

* **Interpretar los resultados**: Analizar las métricas obtenidas para comprender el desempeño del modelo.

* **Realizar ajustes finales**: Si las métricas no son satisfactorias, se podría:
  * Probar otro modelo.
  * Modificar el preprocesamiento o la selección de features.
  * Ajustar nuevamente los hiperparámetros.

* **Comparar con un baseline**: Asegúrar de que el modelo supera una línea base (baseline), como un modelo simple o resultados aleatorios.

**Resultado esperado**: Un modelo optimizado que genera predicciones útiles y satisfactorias según las métricas objetivo, listo para ser implementado o utilizado en producción.

# MODELO ML EN PRODUCCION

# 1. Modelo de Machine Learning deployado en la nube

Esto significa que el modelo debe estar alojado en un servidor o servicio en la nube para que sea accesible de manera remota. Pasos:

* **Seleccionar un servicio de nube**: Usar plataformas como AWS, Google Cloud, Azure o servicios más simples como Heroku o Render.

* **Empaquetar el modelo**:
  * Guardar el modelo entrenado en un formato adecuado (por ejemplo, `.pkl` o `.joblib` para modelos scikit-learn).
  * Crear un script o aplicación que permita cargar el modelo y procesar nuevas entradas.

* **Subir el modelo a la nube**:
  * Configurar un servidor o servicio de hosting para alojar el modelo.
  * Usar frameworks como Flask, FastAPI o Django para crear una API que permita interactuar con el modelo.

# 2. Acceso a través de una interfaz gráfica (tipo Streamlit)

Streamlit es una herramienta que permite construir aplicaciones web de manera sencilla para interactuar con el modelo. Pasos:

* **Desarrollar la interfaz**:
  * Crear un archivo Python que use Streamlit para construir la UI (por ejemplo, un formulario donde el usuario ingrese datos y reciba predicciones).

* **Conectar la interfaz al modelo**:
  * Cargar el modelo dentro de la aplicación Streamlit o realizar llamadas a la API en la nube para obtener predicciones.

* **Deployar la app Streamlit**:
  * Usar servicios como Streamlit Cloud o un servidor en la nube para que otros usuarios puedan acceder a la interfaz.

**Resultado esperado**: Una aplicación web donde los usuarios pueden interactuar con el modelo de forma sencilla.

# 3. Acceso mediante llamados a un endpoint en la nube

Un endpoint es una URL que permite a otros sistemas interactuar con el modelo a través de solicitudes HTTP (por ejemplo, `POST` o `GET`). Pasos:

* **Crear una API**: Usar frameworks como Flask o FastAPI para construir una API que:
  * Reciba datos en formato JSON (o cualquier formato estándar).
  * Procesa los datos y genera una predicción.
  * Devuelve la predicción como respuesta.

* **Hostear la API en la nube**:
  * Subir la aplicación de API a un servicio como AWS Lambda, Google Cloud Run, o Heroku.
  * Generar un endpoint público.

**Resultado esperado**: Otros sistemas pueden enviar datos al endpoint y recibir predicciones automáticamente.

# 4. Salida del modelo consumida por otro componente del proyecto

En este caso, el modelo no es accedido directamente por los usuarios, sino que forma parte de un sistema más amplio. Ejemplos:

* **Dashboards**: Las predicciones se usan para alimentar visualizaciones en un tablero interactivo (por ejemplo, con Power BI o Tableau).

* **Sistemas de recomendación**: Las predicciones se integran en una plataforma que sugiere productos o servicios.

* **Automatización**: Las salidas del modelo son utilizadas para tomar decisiones en sistemas automatizados, como enviar notificaciones a los usuarios o ajustar precios dinámicamente.

**Resultado esperado**: El modelo opera de manera silenciosa pero efectiva como un componente en un flujo más amplio.

# Elección de método según el caso

* Si el modelo debe ser usado directamente por usuarios: Usa **Streamlit** para construir una interfaz amigable.

* Si el modelo debe integrarse en otros sistemas: Crea una **API con un endpoint en la nube**.

* Si el modelo es parte de un sistema mayor: Asegúrate de que la salida sea compatible con el sistema final (por ejemplo, en formato JSON o CSV).

**Conclusión**: Al finalizar, el modelo estará disponible y funcional, ya sea para ser usado directamente, llamado desde otros sistemas, o como parte de una solución más amplia.