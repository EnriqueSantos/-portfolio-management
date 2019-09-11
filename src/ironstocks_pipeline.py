# -*- coding: utf-8 -*-
import numpy as np  # Librería para cálculos númericos optimizados. 
import pandas as pd # Librería para manejo de estructuras de datos.

import requests     # Librería para solicitudes http/s
import json         # Manejo de archivos json.
import matplotlib.pyplot as plt # Librería de visualización 
import seaborn as sns           # Otra librería de visualización
import re           # Manejo de expresiones regulares.

import cvxopt as opt # Librería para procesos de optimización.
from cvxopt import blas, solvers # Herramientas de optimización.

import random

from pandas.plotting import register_matplotlib_converters # Manejo de datetime-stamps como etiquetas del eje 
register_matplotlib_converters()

# Para limitar un poco el universo de stocks disponible  >10,000, se filtraron stocks con información
# disponible sobre el sector y la industria a la que pertenecen, capitalización en el mercado y volumen.
available_stocks = pd.read_csv('../data/available_stocks.csv')

# Peticiones a la API Alpha Vantage.

#Cargamos el token para el API
tokenA = open("../tokenA.txt", "r").read()

def request_dataA(stock,freq = 'Daily',period = 100):
    """
    stock: Nombre del intrumento listado en algún exchange americano. Conocido como Ticker o Symbol.
    freq: Timeframe sobre el que se solicitaran los datos. ['Daily','Weekly','Monthly'], por el momento
          no se solicitan intradaía. Default: 'Daily'
    period: Cantidad de datos que se solicitan. Default: 100
    """
    if stock not in available_stocks['Stock'].tolist():
        print('El stock "',stock,'" no se encuentra disponible, se sustituira por un stock aleatorio')
        stock = available_stocks.iloc[random.randint(0, len(available_stocks))]
        print('Ahora tenemos el stock',stock)
    # Hacemos distinción en datos diarios, semanales y mensuales.
    large_timeframe = ['daily','weekly','monthly']
    if freq.lower() in large_timeframe:
        petition = 'https://www.alphavantage.co/query?function=TIME_SERIES_{}&symbol={}&apikey={}'.format(freq.upper(),stock,tokenA)
    else:
        petition = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={}&interval={}&apikey={}'.format(stock,freq.lower(),tokenA)
    data = requests.get(petition).json()
    stock = pd.DataFrame.from_dict(data['Time Series ({})'.format(freq.capitalize())]).T
    return stock