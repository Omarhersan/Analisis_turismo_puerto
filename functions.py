import warnings; 
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import prophet
from sklearn.metrics import mean_absolute_percentage_error
from typing import List


colors = sns.color_palette('tab10')

def time_series_forecasting(df: pd.DataFrame, 
                            period_to_forecast: int,
                            seasonality_mode: str, 
                            modelo : str) -> pd.DataFrame:
    
    '''
    Given a dataframe, creates a prophet model for each of the columns of the dataframe
    
    Parameters: 
    - df: DataFrame pandas object
    - perido_to_forecast: Period to forecast in months
    - seasonality_mode: additive or multiplicative
    
    
    Returns:
    Error
    '''
    
    
    #df = np.log(df)
    features = [c for c in df.columns if df[c].dtype != 'datetime64']
    forecast_error = {}
    for feature in features:
        
        # Building the time series dataframe
        dataframe = pd.DataFrame()
        dataframe['y'] = df[feature]
        dataframe['ds'] = df.index
        
        #Train_test_split
        # Usaremos los datos hasta 2020 para el train y los datos de 2021 pora el test
        train = dataframe.where(dataframe['ds']<pd.to_datetime('2021-01-01')).dropna()
        test = dataframe.where(dataframe['ds'] >= pd.to_datetime('2021-01-01')).dropna()        
        
        # Training the model
        m = prophet.Prophet(seasonality_mode= seasonality_mode)
        m.fit(train)
        future = m.make_future_dataframe(periods = period_to_forecast, freq = 'MS')
        prediction = m.predict(future)
        
        # Evaluating the model
        y_pred = prediction['yhat'][-12:]
        mape = mean_absolute_percentage_error(test['y'], y_pred)
        forecast_error[feature] = mape
        
        # Future prediction
        future_2 = m.make_future_dataframe(periods = 24, freq = 'MS')
        future_2.where(future_2['ds'] >= pd.to_datetime('2023-01-01'))
        prediction_2 = m.predict(future_2)
        prediction_2.index = future_2['ds']
        prediction_2 = prediction_2['yhat']
        
        # Graphs
        plot_models(test, seasonality_mode, m , feature, prediction, prediction_2)
       
        
    # Errores
    error = pd.DataFrame([forecast_error], index = [modelo]).transpose()
        
    return error



def plot_models(test: pd.DataFrame, 
                seasonality_mode :str ,
                m: prophet.Prophet,
                feature: str,
                prediction: pd.DataFrame,
                prediction_2: pd.DataFrame) -> None:
    
    # Graphs
    fig,axs = plt.subplots(figsize=(10, 6))
    axs.set_title(f'{feature} forecast_{seasonality_mode}')
    axs.plot(test['y'], color = 'red')
    axs.plot(prediction_2, color = 'green')
    fig = m.plot(prediction, ax = axs)

    fig2 = m.plot_components(prediction)

    
    return None

def create_models(df: pd.DataFrame, modelos : List[str],
                  seasonality_mode: List[str] = ['additive', 'multiplicative'] , 
                  period_to_forecast: int = 12) -> pd.DataFrame:
    
    '''
    Generates models for time series analysis in meta's prophet.
    Parameters:
    - df: Dataframe
    - modelos: List of the names of the models to evaluate
    - seasonality_mode: defaults to the 2 possible seasonality modes
    - period_to_forecast: defaults to 12 months
    
    returns:
    - frame: A pandas dataframe with the model name and the mape of each feature evaluated
    
    
    '''
    
    frame = pd.DataFrame()
    for modo in seasonality_mode:
        
        for modelo in modelos:

            resultados_modelo = time_series_forecasting(df,period_to_forecast, 
                                                        seasonality_mode = modo, 
                                                        modelo = f'{modelo}_{modo}')
            frame = pd.concat([frame, resultados_modelo], axis = 1)

    
    return frame
    
    
    
    
    