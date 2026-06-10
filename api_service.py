import requests
import pandas as pd
from config import OPENWEATHER_API_KEY, CITY, EIA_API_KEY

def fetch_weather_data():
    try:
        url = (
            f"https://api.openweathermap.org/data/2.5/weather?q={CITY}"
            f"&appid={OPENWEATHER_API_KEY}&units=metric"
        )

        response = requests.get(url)
        data = response.json()

        temperature = data['main']['temp']
        condition = data['weather'][0]['main']

        return temperature, condition

    except Exception:
        return 30, "Clear"

def fetch_weather_forecast():

    url = (
        f"https://api.openweathermap.org/data/2.5/forecast?q={CITY}"
        f"&appid={OPENWEATHER_API_KEY}&units=metric"
    )

    response = requests.get(url)

    data = response.json()

    forecast_list = data['list']

    weather_data = []

    for item in forecast_list[:10]:

        weather_data.append({
            "datetime": item['dt_txt'],
            "temperature": item['main']['temp'],
            "humidity": item['main']['humidity'],
            "condition": item['weather'][0]['main']
        })

    return weather_data

def fetch_real_energy_prices():

    url = (
        f"https://api.eia.gov/v2/electricity/"
        f"retail-sales/data/?api_key={EIA_API_KEY}"
        f"&frequency=monthly"
        f"&data[0]=price"
        f"&sort[0][column]=period"
        f"&sort[0][direction]=desc"
        f"&offset=0&length=500"
    )

    response = requests.get(url)

    data = response.json()

    records = data['response']['data']

    df = pd.DataFrame(records)
    
    # print("Raw rows:", len(df))
    # print(df["period"].nunique())
    # print(df["period"].sort_values().unique()[:10])
    # print(df["period"].sort_values().unique()[-10:])
    
    df = df[df["sectorName"] == "all sectors"]
    
    df["price"] = pd.to_numeric(
        df["price"],
        errors="coerce"
    )

    df["period"] = pd.to_datetime(
        df["period"],
        format="%Y-%m"
    )

    df = (
        df.groupby("period", as_index=False)
        .agg({"price":"mean"})
    )

    return df