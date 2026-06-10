import os
from dotenv import load_dotenv

load_dotenv()

CITY = "Bhopal"
DATABASE_NAME = "energy_trading.db"

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
EIA_API_KEY = os.getenv("EIA_API_KEY")

# print(f"Loaded API Keys: OPENWEATHER_API_KEY={OPENWEATHER_API_KEY}, EIA_API_KEY={EIA_API_KEY}")