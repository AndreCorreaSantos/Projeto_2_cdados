import ta
import math
from sklearn.linear_model import LogisticRegression


def get_x_y(df): #funcao que recebe o dataframe e retorna x -uma lista de listas que contem todos os features para cada row do dataframe. E retorna y - uma lista com o target para cada row
    change = [0]
    x = []
    for i in range(0,len(df)):
        x.append([df.iloc[i].MACD,df.iloc[i].RSI,df.iloc[i].SO,df.iloc[i]["12_SMA"],df.iloc[i]["12_EMA"],df.iloc[i].ROC,df.iloc[i]["tradecount"]])
        if i != 0:
            diff = df.iloc[i].close- df.iloc[i-1].close
            if diff >= 0 :
                change.append(1)
            if diff < 0:
                change.append(0)

    return x,change

def calculate_indicators(df):
    #calculando os indicadores e colocando na base de dados que a funcao recebe

    #1 - MACD 
    df["MACD"] = ta.trend.macd(close=df.close) # Nas primeiras 26 linhas MACD retorna nan, pois precisa de 26 elementos.

    #2 - RSI lembrar de ajustar window para melhorar resultados. Lembrar que nas primeiras 14 linhas RSI = nan, pois ele precisa de 14 elementos anteriores.
    RSI_window = 14
    df["RSI"] = ta.momentum.rsi(df.close,window=RSI_window)


    #3 - stochastic oscillator (SO)
    SO_window = 14
    df["SO"] = ta.momentum.stoch(close=df.close,high=df.high,low=df.low,window=SO_window)

    #3 - Simple moving average (SMA)

    df["12_SMA"] = ta.trend.sma_indicator(close=df.close,window=12)
    df["26_SMA"] = ta.trend.sma_indicator(close=df.close,window=26)

    #4 - Exponential moving average (EMA)

    df["12_EMA"] = ta.trend.ema_indicator(close=df.close,window=12)
    df["26_EMA"] = ta.trend.ema_indicator(close=df.close,window=26)

    #5 - Rate of change (ROC)
    ROC_window = 24 #definir window direito (minima ideia qual window tem que meter aqui)
    df["ROC"] = ta.momentum.roc(close=df.close,window=ROC_window)
    
    return

def calcula_p(inputs,coeficientes): #funcao que recebe valores de features e coeficientes de regressao e calcula o target estimado (entre 0 e 1)
    soma = 0
    for inp in range(0,len(inputs)):
        soma += coeficientes[inp]*inputs[inp]
    return math.exp(soma)/(1+math.exp(soma))

def calc_coefs(x_train,y_train): #funcao que recebe uma lista com os features para cada row (x_train) e uma lista com os targets para cada row (y_train)
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train,y_train)
    coeficientes = list(model.coef_[0])
    return coeficientes



#6 - indicadores de volume, mineração de btc e remuneração dos mineradores







