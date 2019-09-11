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
    Realiza una petición al API de Alpha Vantage.
    stock: Nombre del intrumento listado en algún exchange americano. Conocido como Ticker o Symbol.
    freq: Timeframe sobre el que se solicitaran los datos. ['Daily','Weekly','Monthly'], por el momento
          no se solicitan intradaía. Default: 'Daily'
    period: Cantidad de datos que se solicitan. Default: 100
    """
    rm_s = stock
    if stock not in available_stocks['Stock'].tolist():
        print('El stock "',stock,'" no se encuentra disponible, se sustituira por un stock aleatorio')
        stock = available_stocks.iloc[random.randint(0, len(available_stocks))][0]
        rm_s = stock
        print('Ahora tenemos el stock',stock)
    # Hacemos distinción en datos diarios, semanales y mensuales.
    large_timeframe = ['daily','weekly','monthly']
    if freq.lower() in large_timeframe:
        petition = 'https://www.alphavantage.co/query?function=TIME_SERIES_{}&symbol={}&apikey={}'.format(freq.upper(),stock,tokenA)
    else:
        petition = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={}&interval={}&apikey={}'.format(stock,freq.lower(),tokenA)
    data = requests.get(petition).json()
    stock = pd.DataFrame.from_dict(data['Time Series ({})'.format(freq.capitalize())]).T
    return (stock,rm_s)

def stack_stocks(list_stocks,freq_selected = 'Daily',period_work = 100):
    """
    Apila información de una lista de stocks.
    list_stocks: Una lista de stocks para construir el portafolio.
    freq_selected: Timeframe de la información.
    period_work: Número de datos por stock.
    """
    data = {}
    for stock in list_stocks:
        data_stock,n_stock = request_dataA(stock,freq_selected,period_work)
        # Creamos un diccionario cuya 'key' sea el nombre del stock y el value un dataframe
        # que contenga su información más relevante en precio.
        data[n_stock] = data_stock
        # Constuimos un Multiindex DataFrame
    return pd.concat(data)

def get_returns_stack(stack_stocks):
    tickers = stack_stocks.index.levels[0].tolist()
    returns_stocks = pd.DataFrame(columns=tickers)
    for stock in tickers:
        returns_stocks[stock] = stack_stocks.loc[stock]['4. close'].sort_index().apply(float).pct_change()
    # Eliminamos la primera fila de los retornos
    returns_stocks.drop(returns_stocks.index[0],inplace=True)
    return returns_stocks

def visualize_returns(stack_stocks,returns_stack):
    tickers = stack_stocks.index.levels[0].tolist()
    plt.subplots(1,1,figsize=[14,9])
    plt.title('Stocks returns')
    plt.xlabel('Date')
    plt.ylabel('Return')
    for sub in range(returns_stack.shape[1]):
        plt.plot(pd.to_datetime(stack_stocks.loc[tickers[sub]].index[1:]),returns_stack[tickers[sub]])

def rand_weights(n):
    ''' Produces n random weights that sum to 1 '''
    k = np.random.rand(n)
    return k / sum(k)

def random_portfolio(returns):
    ''' 
    Returns the mean and standard deviation of returns for a random portfolio
    '''
    return_vec = np.array(returns.T)
    p = np.asmatrix(np.mean(return_vec, axis=1))
    w = np.asmatrix(rand_weights(return_vec.shape[0]))
    C = np.asmatrix(np.cov(return_vec))
    
    mu = w * p.T
    sigma = np.sqrt(w * C * w.T)
    
    # This recursion reduces outliers to keep plots pretty
    if sigma > 2.0:
        return random_portfolio(returns)
    return mu, sigma

def visualize_portfolios(stds,means):
    plt.subplots(1,1,figsize=[14,9])
    plt.plot(stds, means, 'o', markersize=5)
    plt.xlabel('Risk')
    plt.ylabel('Returns')
    plt.title('Mean and standard deviation of returns of randomly generated portfolios');

def optimal_portfolio(returns_c):
    returns = np.array(returns_c.T)
    n = len(returns)
    returns = np.asmatrix(returns)
    
    N = 100
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]
    
    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns))
    pbar = opt.matrix(np.mean(returns, axis=1))
    
    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    
    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] 
                  for mu in mus]
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    return np.asarray(wt), returns, risks

def visualize_optimals(stds,means,risks,returns):
    plt.subplots(1,1,figsize=[12,8])
    plt.plot(stds, means, 'o')
    plt.ylabel('Returns')
    plt.xlabel('Risk')
    plt.plot(risks, returns, 'y-o');
    
def show_w(weights,tickers):
    weight_stock = pd.DataFrame(columns=['stock','weight'])
    weight_stock['weight'] = pd.Series(weights[:,0]*100)
    weight_stock['stock'] = tickers
    return weight_stock