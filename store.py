import numpy as np
import pandas as pd

# reading data from source
dataset = pd.read_csv('./dataset.csv')


# function to scale the features using z-score normalization
def standardize_with_z_score(data):
    mean = np.mean(data)
    standard_deviation = np.std(data)
    for index, xi in enumerate(data):
        data[index] = (xi - mean) / standard_deviation
    return data


class EconomicIndicators:
    def __init__(self, data):
        self.year = data["Year"]
        self.gdp = data["GDP (in billions USD)"]
        self.inflation_rate = data["Inflation Rate (%)"]
        self.unemployment_rate = data["Unemployment Rate (%)"]
        self.fiscal_deficit = data["Fiscal Deficit (in billions USD)"]
        self.exchange_rate = data["Exchange Rate (USD to INR)"]
        self.stock_price = data["Nifty 50 (NSEI)"]


# creating a list of EconomicIndicators object
records = list(map(lambda data: EconomicIndicators(data), dataset.to_dict("records")))

years = np.array(list(dataset.get("Year")))

# scaling the features using z-score normalization
gdp = standardize_with_z_score(np.array(list(dataset.get("GDP (in billions USD)"))))
inflation_rate = standardize_with_z_score(np.array(list(dataset.get("Inflation Rate (%)"))))
unemployment_rate = standardize_with_z_score(np.array(list(dataset.get("Unemployment Rate (%)"))))
fiscal_deficit = standardize_with_z_score(np.array(list(dataset.get("Fiscal Deficit (in billions USD)"))))
exchange_rate = standardize_with_z_score(np.array(list(dataset.get("Exchange Rate (USD to INR)"))))
stock_price = np.array(list(dataset.get("Nifty 50 (NSEI)")))
