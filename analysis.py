# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import re
import nltk
import pandas as pd
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("twitter.csv")

#Seleccionamos las columnas necesarias
columnas = [
    "css-901oao 9",
    #"css-4rbku5 2",
    "css-901oao 10",
    "css-901oao 15",
    "css-4rbku5 3",
    "css-901oao 15",
    "css-4rbku5 4",
    "css-4rbku5 5",
    "css-4rbku5 6",
    "css-901oao 16"
]
tweets = df[columnas].copy()
# Reemplazar NaN con una cadena vacía
tweets.fillna("", inplace=True)
tweets["texto"] = tweets.apply(lambda row: ' '.join(row), axis=1)
#tweets['Texto_Unido'] = tweets.sum(axis=1)
text = pd.DataFrame(tweets.texto)
text['texto'] = text['texto'].apply(lambda x: re.sub(r'[^\w\s]', '', x.lower()))
# Tokenización
text['tokens'] = text['texto'].apply(word_tokenize)
# Análisis de polaridad
###Diccionarios en español para el análisis de polaridad
nltk.download('punkt')
from nltk.corpus import stopwords
# Obtención de listado de stopwords del inglés
stop_words = list(stopwords.words('spanish'))
# Se añade la stoprword: amp, ax, ex
stop_words.extend(("q", "d", "van", "si", "pa"))
print(stop_words[:10])

# Filtrado para excluir stopwords
# ==============================================================================
tweets_tidy = text[~(text["texto"].isin(stop_words))]

##########NUBE DE PALABRAS
def preprocess_text(text):
    words = text # Tokenizar y convertir a minúsculas
    words = [word for word in words if word.isalpha()]  # Eliminar caracteres no alfabéticos
    words = [word for word in words if word not in stop_words]  # Eliminar palabras vacías
    return words

preprocessed_text = tweets_tidy["tokens"].apply(preprocess_text)

def get_polarity(text):
  analysis = TextBlob(text)
  result = analysis.translate(from_lang = 'es', to = 'en').sentiment.polarity
  return result

tweets_tidy['polarity'] = tweets_tidy['texto'].apply(get_polarity)


# Unimos todas las listas de palabras en una sola lista
all_words = [word for sublist in preprocessed_text.values for word in sublist]

# Creamos un DataFrame con las palabras y sus frecuencias
word_counts = pd.Series(all_words).value_counts()

# Tomamos las palabras más comunes (puedes ajustar este valor según tus necesidades)
top_words = word_counts.head(15)

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# Función para obtener el sentimiento de un texto
def obtener_sentimiento(texto):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(texto)["compound"]

# Aplicar la función a la columna "texto" y crear una nueva columna "sentimiento"
text["sentimiento"] = text["texto"].apply(obtener_sentimiento)

# Mostrar el resultado
print(text[["texto", "sentimiento"]])


##################GRAFICA#############################
import plotly.graph_objects as go
import plotly.offline as ply
from plotly.offline import plot

# Creamos la gráfica de barras
fig = go.Figure(data=[go.Bar(x=top_words.index, y=top_words.values)])

# Personalizamos el diseño de la gráfica
fig.update_layout(
    title='Conteo de las 10 palabras más frecuentes',
    xaxis_title='Palabra',
    yaxis_title='Frecuencia'
)
fig.write_html("bar.html")
# Mostramos la gráfica
plot(fig)

####################################
# Contar la cantidad de sentimientos negativos, neutros y positivos
negativos = text[text["sentimiento"] < 0].count().values[0]
neutros = text[text["sentimiento"] == 0].count().values[0]
positivos = text[text["sentimiento"] > 0].count().values[0]

# Crear la gráfica de pastel
labels = ["Negativos", "Neutros", "Positivos"]
values = [negativos, neutros, positivos]

fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

fig.write_html("pie.html")
# Mostrar la gráfica
plot(fig)
