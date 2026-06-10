import pandas as pd
import numpy as np


def generate_energy_dataset():
    dates = pd.date_range(start='2024-01-01', periods=365)

    demand = np.random.randint(40, 120, size=365)

    prices = demand * 1.25 + np.random.normal(0, 5, size=365)

    df = pd.DataFrame({
        'date': dates,
        'demand': demand,
        'price': prices
    })

    return df