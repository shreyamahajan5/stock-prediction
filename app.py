
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the trained LSTM model
model = keras.models.load_model('model.h5')  # Replace with your model file

st.title('Stock Price Prediction App')

# Load historical stock data from the CSV file
def load_data():
    df = pd.read_csv("/Users/shreyamahajan/Desktop/hande/MSFT.csv", encoding='utf-8')
    return df

df_train = load_data()

if df_train is not None:
    df = load_data()
    
    # Data preprocessing: assuming columns 'Date', 'Close', and 'Volume'
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Predict future stock prices
    num_days = st.slider("Number of future days to predict", 1, 30, 7)
    last_date = df.index[-1]
   
    # Generate future dates without 'closed' argument
    future_dates = [last_date + pd.DateOffset(days=i) for i in range(num_days + 1)]

    predicted_prices = []
    true_prices = []  # Store true prices for calculating metrics
    for i in range(num_days):
        input_data = df['Close'].values[-3:]  # Use a window of size 3
        input_data = np.reshape(input_data, (1, -1, 1))
        prediction = model.predict(input_data)
        predicted_prices.append(prediction[0][0])
        
        # Retrieve the true price for the next day
        true_price = df['Close'][-num_days + i + 1]
        true_prices.append(true_price)

        # Add the predicted value to the DataFrame
        new_row = pd.DataFrame({'Close': [prediction[0][0]]}, index=[df.index[-1] + pd.DateOffset(days=1)])
        df = pd.concat([df, new_row])

    # Create a DateTimeIndex for future dates and concatenate it with the existing index
    future_date_index = pd.DatetimeIndex(future_dates[1:])
    df = df.reindex(df.index.union(future_date_index))

    # Calculate evaluation metrics
    mae = mean_absolute_error(true_prices, predicted_prices)
    mse = mean_squared_error(true_prices, predicted_prices)
    rmse = np.sqrt(mse)

    st.write("Predicted prices:")
    st.line_chart(df['Close'].tail(num_days * 2))  # Display the predicted prices chart
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    # Historical vs. Predicted Comparison



    
    # st.subheader("Historical vs. Predicted Comparison")
    # # Calculate the length of the data to be displayed
    # data_length = num_days * 2

    # # Ensure that "Historical" and "Predicted" have the same length as data_length
    # # Slice the arrays to match the specified length
    # comparison_data = pd.DataFrame({
    #     'Historical': df['Close'][-data_length:], 
    #     'Predicted': predicted_prices[-data_length:]
    # })

    # st.line_chart(comparison_data)


    # Concatenate the predicted prices into a Pandas Series
    predicted_prices_series = pd.Series(predicted_prices, index=future_date_index)
    st.write("Predicted Prices (Series):")
    st.line_chart(predicted_prices_series)
