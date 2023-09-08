import mysql.connector
import csv
import pandas as pd
import json
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose as sd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, mean_squared_log_error
from sklearn.model_selection import train_test_split as split
import warnings
import itertools
import string
import secrets
from scipy import stats 
from statsmodels.graphics.gofplots import qqplot as qq
warnings.filterwarnings("ignore")

class Arima_exp:
    



    def givePredictions(self, company_Name):
        conn = mysql.connector.connect(
        host ="localhost",
        database = "companydb",
        user = "sanjula" ,
        password = "1234"
        )
        query = "SELECT * FROM company_records"

        df = pd.read_sql(query, conn)

        grouped_DF = df.groupby('Refno')
        selected_Df = grouped_DF.get_group(company_Name)


        

        if len(selected_Df) >= 15:
            selected_Df['Year_End'] = pd.to_datetime(selected_Df['Year_End'], format='%d/%m/%Y')
            selected_Df = selected_Df.set_index('Year_End')

            df_resampled = selected_Df.groupby(pd.Grouper(freq='Y')).sum()
            df_resampled = df_resampled.reset_index()

            selected_Df = df_resampled
            selected_Df = selected_Df.set_index('Year_End')
            selected_Df = selected_Df.drop('Refno', axis=1)
            selected_Df = selected_Df.drop('AudType', axis=1)
            selected_Df = selected_Df.drop('PrjCat', axis=1)

            selected_Df = selected_Df.mask(selected_Df == 0, selected_Df.mean(),axis=1)
            
            selected_Df['Current_Ratio'] = (selected_Df['Cur_Assets'] / selected_Df['Cur_Liab'])
            selected_Df['Quick_Ratio'] = ((selected_Df['Cur_Assets']) -(selected_Df['Stock_Invent']) / selected_Df['Cur_Liab'])
            selected_Df['Net_pro_Rate'] = (selected_Df['Pro_Aft_Tax'] / selected_Df['Turnover'])*100
            selected_Df['Debt_to_Assets'] = ((selected_Df['NC_Liab'])+(selected_Df['Cur_Liab']) / selected_Df['Tot_Assets'])
            selected_Df['Assets_Turnover'] = (selected_Df['Turnover'] / selected_Df['Tot_Assets'])
            

            df_NetProfit= pd.DataFrame(selected_Df.iloc[:,13])
            df_CR = pd.DataFrame(selected_Df.iloc[:,-5])
            df_QR = pd.DataFrame(selected_Df.iloc[:,-4])
            df_DebtToAssets = pd.DataFrame(selected_Df.iloc[:,-2])
            df_AT = pd.DataFrame(selected_Df.iloc[:,-1])


            x = selected_Df.index[-2]
            output = x.strftime('%Y-%m-%d')

            def predict_Arima(df):
                train = df.loc[df.index <=  output ]
                test = df.loc[df.index >= output]
                return test,train
            
            test, train =predict_Arima(df_NetProfit)
            test_CR, train_CR = predict_Arima(df_CR)
            test_QR, train_QR = predict_Arima(df_QR)
            test_AT, train_AT = predict_Arima(df_AT)
            test_DA, train_DA = predict_Arima(df_DebtToAssets)

            


            def select_BestOrder(orders,train_data,test_data):
                best_order = None
                best_rmse = float('inf') 
                best_aic = float('inf') 
                
                for order in orders:
                    try:
                        model = ARIMA(np.asarray(train_data).astype(float), order = order)
                        model_fit = model.fit()
                        forecast = model_fit.forecast(steps=len(test_data))  
                        predictions= forecast
                        rmse = np.sqrt(mean_squared_error(test_data, predictions))
                        aic = model_fit.aic
                        #print("Order -",order," RMSE:", rmse," AIC:", aic)
                        if rmse < best_rmse :
                            best_order = order
                            best_rmse = rmse
                            best_aic = aic
                    except:
                        continue
                print(best_order, best_rmse,best_aic)
                return best_order
            #


            def ARIMA_Imple(train,test):
                
                p_range = range(3)  
                d_range = range(2)  
                q_range = range(3) 

                order_series = []
                    
                for p, d, q in itertools.product(p_range, d_range, q_range):
                    order_series.append((p, d, q))
                    

                # Generate the order series
                orders = order_series



                np.asarray(train.iloc[:,0])
                order =select_BestOrder(orders,train,test)
                
                model = ARIMA(np.asarray(train).astype(float), order= order)  
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=len(test))  
                predictions=forecast
                forecast = pd.DataFrame(forecast, index=test.index, columns=['Forecast'])
                
                rmse = np.sqrt(mean_squared_error(test, predictions))
                
                
                last_index = test.index[0]
                forecast_values = model_fit.forecast(steps=5)

                forecast_index = pd.date_range(start=last_index , periods=5, freq='Y')

                # Create a dataframe with the index and forecasted values
                forecast_df = pd.DataFrame({'Forecast': forecast_values})
                forecast_df.index= forecast_index

                # Print the forecast dataframe
                forecast_df.iloc[0, 0]= test.iloc[0,0]

                combined_data = pd.concat([train, forecast_df], axis=0)
                median_values = train.median()
                median_value_for_column = median_values.iloc[0]
                print(median_value_for_column)

                combined_data = combined_data.drop('Forecast', axis=1)
                combined_data['values'] = median_value_for_column
                combined_data = combined_data.drop(combined_data.columns[0], axis=1)

                y_upper = median_value_for_column + median_value_for_column * 0.4
                y_lower = median_value_for_column - median_value_for_column * 0.4


                plt.figure(figsize=(10, 6))
                plt.plot(train.index, train.iloc[:,0], label='Training')
                plt.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast')
                plt.plot(test.index, test.iloc[:,0], label='Actual')



                plt.xlabel('Time')
                plt.ylabel(train.columns[0])
                plt.legend()
                plt.title('ARIMA Forecast Of '+train.columns[0])

                
                plot_data = {
                    'train_data': {
                        'index': train.index.astype(str).tolist(),
                        'values': train.iloc[:, 0].values.tolist()
                    },
                    'forecast_data': {
                        'index': forecast_df.index.astype(str).tolist(),
                        'values': forecast_df['Forecast'].values.tolist()
                    },
                    'test_data': {
                        'index': test.index.astype(str).tolist(),
                        'values': test.iloc[:, 0].values.tolist()
                    },
                    'combined_data':{
                        'index': combined_data.index.astype(str).tolist(),
                        'values': combined_data['values'].values.tolist()
                    }
                }

                # Save the plot data to a JSON file
                #with open('json_predicts\\'+ company_Name +train.columns[0]+'.json', 'w') as file:
                #    json.dump(plot_data, file)
                

                #plt.savefig('prediction_plots\\'+train.columns[0]+'.PNG')
                print("\n RMSE "+train.columns[0]+" =",rmse)
                print("\n")
                print(predictions)
                return plot_data

            net_profit_plot = ARIMA_Imple(train,test)
            cur_rate_plot = ARIMA_Imple(train_CR,test_CR)
            quick_ratio_plot = ARIMA_Imple(train_QR,test_QR)
            AT_plot = ARIMA_Imple(train_AT,test_AT)
            debt_Asset_plot= ARIMA_Imple(train_DA,test_DA)

            """combined_data = {}

            combined_data['net_profit_plot'] = json.loads(net_profit_plot)
            combined_data['cur_rate_plot'] = json.loads(cur_rate_plot)
            combined_data['quick_ratio_plot'] = json.loads(quick_ratio_plot)
            combined_data['AT_plot'] = json.loads(AT_plot)
            combined_data['debt_Asset_plot'] = json.loads(debt_Asset_plot)

            # Convert the dictionary to JSON string
            combined_json = json.dumps(combined_data)

            x=(''.join(secrets.choice(string.ascii_uppercase + string.ascii_lowercase) for i in range(7)))

            # Save the plot data to a JSON file
            with open(x+'.json', 'w') as file:
                json.dump(combined_json, file)"""


            return net_profit_plot,cur_rate_plot, quick_ratio_plot,AT_plot,debt_Asset_plot

        else:
            net_profit_plot = None
            cur_rate_plot = None
            quick_ratio_plot = None
            AT_plot = None
            debt_Asset_plot= None

            
            return net_profit_plot,cur_rate_plot, quick_ratio_plot,AT_plot,debt_Asset_plot
        

    
            

    

