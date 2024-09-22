# Time Series Forecating
Bitcoin Price Prediction with Time Series Forecasting and LSTM
Overview
This project uses a Bidirectional LSTM (Long Short-Term Memory) model to predict Bitcoin prices based on historical data. The dataset consists of 1-minute interval data from 2012 to 2020, resampled into hourly intervals. The model aims to capture patterns in the data and forecast the price of Bitcoin using time series forecasting techniques.

# Table of Contents
Project Structure
Getting Started
Data Preprocessing
Model Architecture
Training the Model
Results
Conclusion
Dependencies
Usage
License

# Project Structure
bash
Copy code
├── preprocess_data.py        # Preprocess the raw dataset
├── forecast_data.py          # Main file for model training and evaluation
├── bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv # Dataset (or link to dataset)
└── README.md                 # Project documentation
Getting Started
Clone the repository:

bash
Copy code
git clone https://github.com/username/repo.git
cd repo
Install required dependencies:

bash
Copy code
pip install -r requirements.txt
Download the dataset:
If the dataset is not included in this repository, download it from Kaggle's Bitcoin Dataset or any relevant source.

Run the preprocessing script:

bash
Copy code
python preprocess_data.py
Data Preprocessing
The preprocessing script (preprocess_data.py) performs the following steps:

Cleans and removes NaN values.
Reformats timestamps and sets them as the index.
Resamples the data into hourly intervals (Open, Close, Volume).
Splits the dataset into training and validation sets.
Standardizes the data for better performance during training.
Model Architecture
The model is a Bidirectional LSTM with 64 units, designed to capture both past and future trends in the Bitcoin price data. It consists of:

A Bidirectional LSTM layer to process sequential data in both directions.
A Dense output layer to predict the future price.
Training the Model
To train the model, run the following command:

bash
Copy code
python forecast_data.py
The training process includes:

Defining the model with Adam optimizer and Mean Squared Error (MSE) as the loss function.
Training for 10 epochs with validation to monitor performance.
Visualizing the loss and prediction results after training.
Results
The model captures the general trend of Bitcoin price movements. The project includes:

A plot of training/validation loss across epochs.
Predictions vs. actual prices for 24-hour windows.
Conclusion
While this Bidirectional LSTM model shows promise, predicting Bitcoin prices remains challenging due to market volatility. This model serves as a foundation for further exploration, with potential improvements in feature selection and tuning.

Dependencies
Python 3.x
TensorFlow
NumPy
Pandas
Matplotlib
Install all dependencies using the command:

bash
Copy code
pip install -r requirements.txt
Usage
Preprocess the dataset using preprocess_data.py.
Train and evaluate the model using forecast_data.py.
Visualize the results and predictions.