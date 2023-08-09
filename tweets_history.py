# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


# Tratamiento de datos

import numpy as np
import pandas as pd
import string
import re
from datetime import datetime, timedelta
# Gráficos

import matplotlib.pyplot as plt
from matplotlib import style
#style.use('ggplot') or plt.style.use('ggplot')

# Preprocesado y modelado

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords

# Configuración warnings
import warnings
warnings.filterwarnings('ignore')
#%%
#Lectura de Datos
df = pd.read_csv("tweets_history.csv")
#%%
#Distribución Temporal de los Tweets


ts = df[['css-4rbku5']]
ts.rename(columns={'css-4rbku5':'fecha'}, inplace=True)
ts = ts.dropna()

# Utilizamos una expresión regular para identificar las fechas que son diferentes a números y que comienzan con "#" o alguna letra
ts_cleaned = ts[~ts['fecha'].str.match(r'^[^0-9]|\#')]
#%%

from datetime import datetime

def obtener_fecha(string_fecha):
    # Diccionario para mapear los nombres abreviados de los meses a sus números correspondientes
    meses_dict = {
        'ene.': '01', 'feb.': '02', 'mar.': '03', 'abr.': '04', 'may.': '05', 'jun.': '06',
        'jul.': '07', 'ago.': '08', 'sept.': '09', 'oct.': '10', 'nov.': '11', 'dic.': '12'
    }

    try:
        # Verificar si el string está en formato 'Xh' y obtener la hora si es el caso
        if string_fecha.endswith('h'):
            hora = int(string_fecha.replace('h', ''))
            fecha_actual = datetime.now().replace(hour=hora, minute=0, second=0, microsecond=0)
        else:
            # Verificar si el string contiene el año
            if len(string_fecha.split()) == 3:
                dia, mes_abreviado, anio = string_fecha.strip().split(' ')
            else:
                dia, mes_abreviado = string_fecha.strip().split(' ')
                anio = datetime.now().year

            # Obtener el número de mes del diccionario
            mes_numero = meses_dict[mes_abreviado]

            # Crear el objeto de fecha con el formato 'YYYY-MM-DD'
            fecha_actual = datetime.strptime(f"{anio}-{mes_numero}-{dia.zfill(2)}", "%Y-%m-%d")

        return fecha_actual.date()
    except (KeyError, ValueError):
        # Si ocurre un error en el formato o el mes no está en el diccionario, retornar None
        return None

ts_cleaned = ts_cleaned['fecha'].apply(obtener_fecha)
# Convertimos la serie 'fecha' al tipo de dato datetime
ts_cleaned = pd.DataFrame(ts_cleaned)
ts_cleaned['fecha'] = pd.to_datetime(ts_cleaned['fecha'])

# Agrupar por mes y contar el número de días por mes
ts_cleaned = ts_cleaned.groupby(ts_cleaned['fecha'].dt.to_period('M')).size().reset_index(name='tweets')

import plotly.express as px
import plotly.offline as ply
from plotly.offline import plot


# Convertir la columna 'fecha' a tipo string
ts_cleaned['fecha'] = ts_cleaned['fecha'].astype(str)

fig = px.line(ts_cleaned, x='fecha', y='tweets')
fig.write_html("history.html")
plot(fig)

ts_cleaned.describe()

###########################################################################################################
#%%
dff = pd.read_csv("harfuch.csv")
dfff = pd.read_csv("omar_garcia_herfuch.csv")