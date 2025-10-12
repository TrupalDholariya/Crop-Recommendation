# Smart Crop Advisory System - Sprint 2 Module
# This script integrates a live weather API and a rule-based engine to provide timely advice.

import requests
from datetime import datetime

class WeatherService:
    """
    Handles all interactions with the weather API (OpenWeatherMap).
    """
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5/weather?"

    def get_weather(self, lat, lon):
        """
        Fetches real-time weather data for a given latitude and longitude.
        """
        if not self.api_key:
            return None, "API key is missing. Please get one from OpenWeatherMap."

        url = f"{self.base_url}lat={lat}&lon={lon}&appid={self.api_key}&units=metric"
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raises an exception for bad status codes (4xx or 5xx)
            data = response.json()
            
            # Extracting the most relevant information
            weather_data = {
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'description': data['weather'][0]['description'].capitalize(),
                'wind_speed': data['wind']['speed'],
                'city': data['name'],
                'icon': data['weather'][0]['icon']
            }
            return weather_data, None
        except requests.exceptions.HTTPError as http_err:
            if response.status_code == 401:
                return None, "API Error: Unauthorized. Please check your API key."
            return None, f"API Error: HTTP error occurred: {http_err}"
        except Exception as err:
            return None, f"An unexpected error occurred: {err}"


class CropAdvisoryEngine:
    """
    Generates rule-based advice based on location, date, and weather.
    """
    def __init__(self, weather_data, location="Ahmedabad"):
        self.weather = weather_data
        self.location = location
        self.current_date = datetime.now()

    def get_crop_calendar_advice(self):
        """
        Provides general advice based on the month for the Ahmedabad region.
        """
        month = self.current_date.month
        advice = []
        
        # This is a simplified crop calendar for Gujarat.
        if month in [6, 7]: # June-July: Kharif Sowing Season
            advice.append("This is the peak sowing season for Kharif crops like Cotton, Groundnut, and Paddy.")
            advice.append("Ensure proper field preparation and use certified seeds for best results.")
        elif month in [8, 9]: # August-September: Kharif Growing Season
            advice.append("Kharif crops are in the vegetative growth stage.")
            advice.append("Crucial time for weeding and pest management. Regularly monitor your fields.")
            advice.append("Apply top-dressing of fertilizer if not already done.")
        elif month in [10, 11]: # October-November: Kharif Harvest & Rabi Sowing
            advice.append("Time to harvest mature Kharif crops.")
            advice.append("Prepare fields for Rabi crops like Wheat, Mustard, and Cumin.")
        elif month in [12, 1, 2]: # December-February: Rabi Growing Season
            advice.append("Rabi crops are in their main growth period.")
            advice.append("Manage irrigation carefully to avoid water stress during the winter.")
            advice.append("Protect crops from frost if low temperatures are forecast.")
        else:
            advice.append("This is the off-season or Zaid season. Good time for land preparation and soil testing.")

        return advice

    def get_weather_based_alerts(self):
        """
        Generates immediate, actionable alerts based on current weather conditions.
        
        """
        alerts = []
        temp = self.weather['temperature']
        humidity = self.weather['humidity']
        description = self.weather['description'].lower()

        # Temperature-based alerts
        if temp > 38:
            alerts.append("HEATWAVE ALERT: High temperatures can cause water stress. Consider irrigating your fields during early morning or late evening.")
        elif temp < 10:
            alerts.append("COLD WEATHER ALERT: Low temperatures can lead to frost. Consider light irrigation or creating smoke to protect sensitive crops.")

        # Rain-based alerts
        if 'rain' in description:
            alerts.append("RAIN ALERT: Postpone irrigation. If heavy rain is forecast, ensure proper field drainage to prevent waterlogging.")
        
        # Humidity-based alerts
        if humidity > 85:
            alerts.append("HIGH HUMIDITY ALERT: High humidity increases the risk of fungal diseases like blight and mildew. Proactively inspect your crops.")
            
        # Pest alert (based on conditions favorable for pests)
        if temp > 25 and humidity > 70:
            alerts.append("PEST WATCH: Warm and humid conditions are favorable for pests like aphids and whiteflies. Regular monitoring is advised.")
        
        if not alerts:
            alerts.append("Weather is currently favorable. Continue with normal farm operations.")

        return alerts

def main():
    """
    Main function to run the simulation.
    """
    print("--- Sprint 2: Weather & Advisory Module ---")

    # --- Step 1: Get Your API Key ---
    # Go to https://openweathermap.org/ and sign up for a free API key.
    OPENWEATHER_API_KEY = "13513112bbbf5d4532bae6db40a11e94"  # <-- IMPORTANT: REPLACE WITH YOUR ACTUAL KEY

    # Location for Ahmedabad, Gujarat
    AHMEDABAD_LAT = 23.0225
    AHMEDABAD_LON = 72.5714
    
    # --- Step 2: Fetch Weather Data ---
    print(f"\nFetching real-time weather for Ahmedabad...")
    weather_service = WeatherService(OPENWEATHER_API_KEY)
    weather, error = weather_service.get_weather(AHMEDABAD_LAT, AHMEDABAD_LON)

    if error:
        print(f"Error: {error}")
        return

    print("Weather data fetched successfully!")
    print(f"  City: {weather['city']}")
    print(f"  Temperature: {weather['temperature']}°C")
    print(f"  Humidity: {weather['humidity']}%")
    print(f"  Condition: {weather['description']}")

    # --- Step 3: Generate Advisory ---
    advisory_engine = CropAdvisoryEngine(weather)

    print("\n--- [1] CROP CALENDAR ADVISORY ---")
    calendar_advice = advisory_engine.get_crop_calendar_advice()
    for item in calendar_advice:
        print(f"- {item}")

    print("\n--- [2] REAL-TIME WEATHER ALERTS ---")
    weather_alerts = advisory_engine.get_weather_based_alerts()
    for alert in weather_alerts:
        print(f"- {alert}")


if __name__ == '__main__':
    main()