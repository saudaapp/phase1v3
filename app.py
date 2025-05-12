import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
# Matplotlib is imported but not explicitly used in the final charts; can be removed if not needed elsewhere.
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
from datetime import datetime, timedelta
import requests
import base64
from PIL import Image, ImageDraw, ImageFont # Ensure ImageDraw and ImageFont are imported for the fallback
import io # Imported but not explicitly used; PIL.Image.open handles file paths and byte streams.
import yfinance as yf
import random
# ML models are imported but not used in the current version of the script.
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import StandardScaler
# from statsmodels.tsa.seasonal import seasonal_decompose
# from statsmodels.tsa.arima.model import ARIMA
import time
# JSON imported but not explicitly used.
# import json
from typing import Dict, List # Optional was removed as it was not used

# Set page configuration
st.set_page_config(
    page_title="Sauda Food Insights LLC",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define color scheme based on logo
PRIMARY_COLOR = "#2a5d4c"  # Dark green from logo
SECONDARY_COLOR = "#8bc34a"  # Light green from logo
ACCENT_COLOR = "#4fc3f7"  # Light blue from logo
BG_COLOR = "#f9f8e8"  # Light cream background

# Custom CSS
st.markdown(f"""
<style>
    .reportview-container .main .block-container {{
        padding-top: 1rem;
        padding-bottom: 1rem;
    }}
    .stApp {{
        background-color: {BG_COLOR};
    }}
    h1, h2, h3 {{
        color: {PRIMARY_COLOR};
    }}
    .stButton>button {{
        background-color: {SECONDARY_COLOR};
        color: white;
        border-radius: 5px;
    }}
    .stButton>button:hover {{
        background-color: {PRIMARY_COLOR};
    }}
    .stSelectbox label, .stMultiselect label, .stRadio label {{ /* Added stRadio */
        color: {PRIMARY_COLOR};
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: white;
        color: {PRIMARY_COLOR};
        border-radius: 4px 4px 0 0;
        border: 1px solid #ddd;
        border-bottom: none;
        padding: 10px 16px;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {SECONDARY_COLOR};
        color: white;
    }}
    .analysis-section {{
        margin-bottom: 20px;
        padding: 15px; /* Increased padding */
        background-color: #ffffff; /* Changed to white for better contrast with BG_COLOR */
        border-radius: 8px; /* Smoother radius */
        border: 1px solid #e0e0e0; /* Light border */
        box-shadow: 0 2px 4px rgba(0,0,0,0.05); /* Subtle shadow */
    }}
    .download-button {{
        display: inline-block;
        padding: 10px 20px;
        background-color: {SECONDARY_COLOR};
        color: white !important; /* Ensured text color is white */
        text-decoration: none;
        border-radius: 5px;
        margin-top: 10px; /* Added margin */
    }}
    .download-button:hover {{
        background-color: {PRIMARY_COLOR};
        color: white !important; /* Ensured text color is white */
    }}
    @media (max-width: 768px) {{
        .stSidebar {{
            width: 100% !important;
        }}
        .chart-container {{ /* This class was in your original CSS but not applied */
            height: auto !important;
        }}
        /* Make Plotly charts responsive */
        .plotly-chart {{
            height: auto !important;
            width: 100% !important;
        }}
    }}
</style>
""", unsafe_allow_html=True)

# Load and display logo
logo_path = "IMG_3036.png"  # Ensure this path is correct relative to your app.py
try:
    logo = Image.open(logo_path)
    # Use columns for better centering and control
    st.columns(3)[1].image(logo, width=300) # Simplified column usage for centering
except FileNotFoundError:
    st.title("Sauda Food Insights LLC")
    st.caption("Food Insights Platform - Logo not found")
except Exception as e:
    st.error(f"Could not load logo: {e}")
    st.title("Sauda Food Insights LLC")
    st.caption("Food Insights Platform")


# Function to generate simulated data (helper for weather/crop fallback)
def generate_simulated_timeseries_data(dates, base_val, sin_amp, lin_trend_factor, noise_std_dev, seed_offset=0):
    np.random.seed(sum(ord(c) for c in str(dates[0])) + seed_offset) # Basic seed
    sin_wave = sin_amp * np.sin(np.linspace(0, 4 * np.pi, len(dates)))
    linear_trend = np.linspace(0, base_val * lin_trend_factor, len(dates))
    noise = np.random.normal(0, noise_std_dev, len(dates))
    return np.maximum(0, base_val + sin_wave + linear_trend + noise)

# Function to get available commodities
@st.cache_data(ttl=3600) # Cache for 1 hour
def get_available_commodities():
    # Base commodities with Yahoo Finance tickers
    base_commodities_yf = {
        "ZW=F": "Wheat", "ZC=F": "Corn", "ZS=F": "Soybeans", "ZM=F": "Soybean Meal",
        "ZL=F": "Soybean Oil", "ZO=F": "Oats", "ZR=F": "Rice", "KE=F": "KC Wheat",
        "JO=F": "Orange Juice", "CC=F": "Cocoa", "KC=F": "Coffee", "SB=F": "Sugar",
        "LE=F": "Live Cattle", "GF=F": "Feeder Cattle", "HE=F": "Lean Hogs",
        "CT=F": "Cotton", "LBS=F": "Lumber", "DC=F": "Class III Milk"
    }
    # Commodities without reliable public tickers (will use simulated price data)
    base_commodities_other = {
        "PEPPER_custom": "Black Pepper", "CINNAMON_custom": "Cinnamon",
        "ALMOND_custom": "Almonds", "CASHEW_custom": "Cashews",
        "SALMON_custom": "Salmon", "SHRIMP_custom": "Shrimp",
        # Add more fruits/vegetables here with a "_custom" suffix or similar
        "APPLE_custom": "Apples", "BANANA_custom": "Bananas", "TOMATO_custom": "Tomatoes"
    }

    valid_commodities = {}
    # Validate Yahoo Finance tickers
    for ticker, name in base_commodities_yf.items():
        try:
            stock_info = yf.Ticker(ticker).info
            if stock_info and stock_info.get('regularMarketPrice') is not None:
                valid_commodities[ticker] = name
            else:
                # If yfinance doesn't find it, treat as custom for simulated data
                st.sidebar.caption(f"Note: Data for {name} ({ticker}) will be simulated as live data wasn't immediately available.")
                valid_commodities[ticker + "_simulated"] = f"{name} (Simulated Price)"
        except Exception: # Broad exception for network issues or invalid tickers
            st.sidebar.caption(f"Note: Data for {name} ({ticker}) will be simulated due to fetch error.")
            valid_commodities[ticker + "_simulated"] = f"{name} (Simulated Price)"

    # Add other commodities that will use simulated price data by default
    for ticker, name in base_commodities_other.items():
        valid_commodities[ticker] = f"{name} (Simulated Price)"

    # Custom ticker input
    custom_ticker_input = st.sidebar.text_input("Enter Custom Commodity Ticker (Yahoo Finance)", "")
    st.sidebar.caption("Tip: For live data, use Yahoo Finance tickers (e.g., 'NG=F' for Natural Gas). Otherwise, limited simulated data may be shown.")
    if custom_ticker_input:
        custom_ticker_key = custom_ticker_input.strip().upper()
        if custom_ticker_key and custom_ticker_key not in valid_commodities:
            try:
                info = yf.Ticker(custom_ticker_key).info
                if info and info.get('regularMarketPrice') is not None:
                    short_name = info.get('shortName', custom_ticker_key)
                    valid_commodities[custom_ticker_key] = short_name
                    st.sidebar.success(f"Added custom commodity: {short_name} ({custom_ticker_key})")
                else:
                    st.sidebar.warning(f"Could not fetch live data for {custom_ticker_key}. It will use simulated price data.")
                    valid_commodities[custom_ticker_key + "_simulated"] = f"{custom_ticker_key} (Custom, Simulated Price)"
            except Exception:
                st.sidebar.warning(f"Invalid or unavailable ticker: {custom_ticker_key}. It will use simulated price data.")
                valid_commodities[custom_ticker_key + "_simulated"] = f"{custom_ticker_key} (Custom, Simulated Price)"
    return valid_commodities

# Function to get price data
@st.cache_data(ttl=3600)
def get_price_data(ticker: str, commodity_name: str, period: str = "5y") -> pd.DataFrame:
    if "_custom" in ticker or "_simulated" in ticker:
        st.info(f"Displaying simulated price data for {commodity_name}.")
        end_date = datetime.now()
        try:
            years = int(period[0]) if period and period[0].isdigit() else 5 # default to 5 years if period is unusual
        except:
            years = 5
        start_date = end_date - pd.DateOffset(years=years)
        dates = pd.date_range(start=start_date, end=end_date, freq='B') # Business days
        if dates.empty: # Handle case where date range might be invalid leading to empty dates
             return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])


        # Simulate price data
        seed = sum(ord(c) for c in ticker)
        np.random.seed(seed)
        price_base = 100 + (seed % 200) # Vary base price
        prices = price_base + np.random.randn(len(dates)).cumsum() * 0.5 + np.sin(np.linspace(0, 20*np.pi, len(dates))) * 5
        prices = np.maximum(prices, 10) # Ensure positive prices
        data = pd.DataFrame(prices, index=dates, columns=['Close'])
        data['Open'] = data['Close'] - np.random.rand(len(dates)) * 2
        data['High'] = data['Close'] + np.random.rand(len(dates)) * 2
        data['Low'] = data['Open'] - np.random.rand(len(dates)) * 2
        data['Volume'] = np.random.randint(1000, 10000, size=len(dates))
        return data

    # Attempt to fetch from Yahoo Finance
    max_retries = 2 # Reduced retries for faster UI
    for attempt in range(max_retries):
        try:
            with st.spinner(f"Fetching price data for {commodity_name} ({ticker})... Attempt {attempt + 1}"):
                data = yf.download(ticker, period=period, timeout=10, auto_adjust=True, back_adjust=True) # Added auto_adjust
            if not data.empty:
                return data
            # Do not raise exception here if empty, let it fall through to fallback logic if max_retries reached
            st.caption(f"Attempt {attempt + 1} for {ticker} returned empty data.")
        except Exception as e:
            st.caption(f"Attempt {attempt + 1} for {ticker} failed: {e}") # More subtle warning during retries
            if attempt == max_retries - 1:
                st.warning(f"Error fetching live price data for {commodity_name} ({ticker}) after {max_retries} attempts. Displaying simulated data instead.")
                return get_price_data(ticker + "_simulated", commodity_name, period) # Call itself with simulated flag
            time.sleep(1 + attempt) # Progressive backoff

    # If all attempts failed and did not fall into exception's simulated path (e.g. yf.download returned empty consistently)
    st.warning(f"All attempts to fetch live price data for {commodity_name} ({ticker}) failed or returned empty. Displaying simulated data.")
    return get_price_data(ticker + "_simulated", commodity_name, period)


# Function to get weather data
@st.cache_data(ttl=3600)
def get_weather_data(region: str, days_to_fetch_actual_weather: int = 5) -> pd.DataFrame:
    api_key = None
    try:
        api_key = st.secrets.get("OPENWEATHERMAP_API_KEY")
        if not api_key:
            st.warning("OpenWeatherMap API key not found or empty in secrets. Using simulated weather data.")
    except KeyError:
        st.warning("OpenWeatherMap API key setting not found in secrets. Using simulated weather data.")
    except Exception as e:
        st.warning(f"Could not retrieve OpenWeatherMap API key due to: {e}. Using simulated weather data.")
        api_key = None


    region_coords = {
        "Asia": (35.8617, 104.1954), "Africa": (8.7832, 34.5085),
        "South America": (-14.2350, -51.9253), "North America": (37.0902, -95.7129),
        "Europe": (54.5260, 15.2551), "Middle East": (29.3117, 47.4818),
        "Oceania": (-25.2744, 133.7751)
    }
    lat, lon = region_coords.get(region, (0, 0))

    simulated_fallback_needed = True
    full_period_simulated = False # Flag to indicate if the entire 730-day period will be simulated initially

    if api_key and lat != 0 and lon != 0:
        st.info(f"Attempting to fetch actual weather data for {region} for the last {days_to_fetch_actual_weather} available days...")
        weather_data_list = []

        for i in range(days_to_fetch_actual_weather, 0, -1): # Go from N days ago to yesterday
            fetch_date = datetime.now() - timedelta(days=i)
            fetch_timestamp = int(fetch_date.timestamp())
            try:
                params = {
                    "lat": lat, "lon": lon, "dt": fetch_timestamp,
                    "appid": api_key, "units": "metric"
                }
                # Using One Call API. Ensure your key supports this version/endpoint.
                # The URL might be /data/2.5/onecall/timemachine or /data/3.0/onecall/timemachine
                # Check OpenWeatherMap documentation for the correct endpoint for your API key version.
                # For example, One Call API 3.0: https://openweathermap.org/api/one-call-3
                response = requests.get("https://api.openweathermap.org/data/2.5/onecall/timemachine", params=params, timeout=15)
                response.raise_for_status()
                data = response.json()

                # Structure for "onecall/timemachine" can be data -> hourly or data -> data[0] -> hourly
                hourly_data = data.get("hourly", [])
                if not hourly_data and "data" in data and isinstance(data["data"], list) and len(data["data"]) > 0:
                    hourly_data = data["data"][0].get("hourly", [])


                if hourly_data:
                    daily_temps = [h["temp"] for h in hourly_data if "temp" in h]
                    daily_rain_list = [h.get("rain", {}).get("1h", 0) if isinstance(h.get("rain"), dict) else h.get("rain", 0) if isinstance(h.get("rain"), (int, float)) else 0 for h in hourly_data]


                    avg_temp = np.mean(daily_temps) if daily_temps else np.nan
                    total_rain = sum(daily_rain_list) if daily_rain_list else 0.0

                    weather_data_list.append({
                        "Date": pd.to_datetime(fetch_date.date()),
                        "Temperature": avg_temp,
                        "Rainfall": total_rain
                    })
                else:
                     weather_data_list.append({
                        "Date": pd.to_datetime(fetch_date.date()),
                        "Temperature": np.nan,
                        "Rainfall": np.nan
                    })
                time.sleep(0.1)

            except requests.exceptions.HTTPError as e_http:
                if e_http.response.status_code == 401:
                    st.error("OpenWeatherMap API Key is invalid or unauthorized. Please check your secrets. Weather data will be simulated.")
                    api_key = None
                    full_period_simulated = True # Entire period must be simulated
                    break # Stop trying with bad key
                st.caption(f"Weather API HTTP error for {fetch_date.strftime('%Y-%m-%d')}: {e_http}.")
            except Exception as e:
                st.caption(f"Error fetching or processing weather for {fetch_date.strftime('%Y-%m-%d')}: {e}.")

        if full_period_simulated: # If API key was bad, force full simulation
             simulated_fallback_needed = True
        elif weather_data_list:
            df_weather = pd.DataFrame(weather_data_list)
            df_weather.dropna(subset=['Temperature'], inplace=True)
            if not df_weather.empty:
                st.success(f"Successfully fetched {len(df_weather)} days of actual weather data for {region}.")
                simulated_fallback_needed = False # We have some actual data

                if len(df_weather) < 730: # If actual data is less than 2 years
                    st.info(f"Actual weather data is for the last {len(df_weather)} days. Older data in the 2-year period will use simulation for consistent charting.")
                    sim_start_date = datetime.now() - timedelta(days=730)
                    # Ensure sim_end_date is before the first date of actual data
                    sim_end_date = df_weather['Date'].min() - timedelta(days=1) if not df_weather.empty else datetime.now() - timedelta(days=len(df_weather)+1)

                    if sim_start_date < sim_end_date:
                        sim_dates = pd.date_range(start=sim_start_date, end=sim_end_date, freq='D')
                        if not sim_dates.empty:
                            sim_temp = generate_simulated_timeseries_data(sim_dates, 20, 10, 0.01, 2, seed_offset=sum(ord(c) for c in region))
                            sim_rain = generate_simulated_timeseries_data(sim_dates, 50, 30, 0.005, 10, seed_offset=sum(ord(c) for c in region)+1)
                            df_sim_weather = pd.DataFrame({"Date": sim_dates, "Temperature": sim_temp, "Rainfall": sim_rain})
                            df_weather = pd.concat([df_sim_weather, df_weather], ignore_index=True)
                        else: # No simulation period needed if sim_dates is empty
                            pass
                    df_weather = df_weather.sort_values(by="Date").reset_index(drop=True)
                return df_weather
            else: # No valid actual data was processed
                simulated_fallback_needed = True
        else: # No data points collected, even if API key was present
             simulated_fallback_needed = True


    if simulated_fallback_needed:
        if not full_period_simulated: # Avoid double message if already warned about API key
            st.warning(f"Using simulated weather data for {region} for the 730-day period.")
        end_date_sim = datetime.now()
        start_date_sim = end_date_sim - timedelta(days=730)
        dates_sim = pd.date_range(start=start_date_sim, end=end_date_sim, freq='D')
        if dates_sim.empty:
            return pd.DataFrame(columns=['Date', 'Temperature', 'Rainfall'])
        temp_sim = generate_simulated_timeseries_data(dates_sim, 20, 10, 0.01, 2, seed_offset=sum(ord(c) for c in region))
        rain_sim = generate_simulated_timeseries_data(dates_sim, 50, 30, 0.005, 10, seed_offset=sum(ord(c) for c in region)+1)
        return pd.DataFrame({"Date": dates_sim, "Temperature": temp_sim, "Rainfall": rain_sim})


# Function to get crop health data (currently simulated based on NASA POWER API idea or full simulation)
@st.cache_data(ttl=3600)
def get_crop_health_data(region: str, commodity: str) -> pd.DataFrame:
    st.info(f"Crop health indicators for {region} ({commodity}) are currently model-derived estimates or simulations.")

    api_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    region_coords = {
        "Asia": (35.8617, 104.1954), "Africa": (8.7832, 34.5085),
        "South America": (-14.2350, -51.9253), "North America": (37.0902, -95.7129),
        "Europe": (54.5260, 15.2551), "Middle East": (29.3117, 47.4818),
        "Oceania": (-25.2744, 133.7751)
    }
    lat, lon = region_coords.get(region, (0, 0))

    end_date_dt = datetime.now()
    start_date_dt = end_date_dt - timedelta(days=730)
    dates_daily = pd.date_range(start=start_date_dt, end=end_date_dt, freq='D')
    if dates_daily.empty:
        st.warning(f"Date range for crop health in {region} is invalid. Using full simulation.")
        # Fall through to full simulation logic at the end of the function.
    else: # Proceed if dates_daily is not empty
        try:
            params = {
                "latitude": lat, "longitude": lon,
                "start": start_date_dt.strftime("%Y%m%d"),
                "end": end_date_dt.strftime("%Y%m%d"),
                "community": "ag",
                "parameters": "T2M,PRECTOTCORR,ALLSKY_SFC_SW_DWN",
                "format": "JSON", "user": "anonymous"
            }
            with st.spinner(f"Fetching agro-meteorological data from NASA POWER for {region}..."):
                response = requests.get(api_url, params=params, timeout=30)
            response.raise_for_status()
            data_nasa = response.json()

            temp_data = data_nasa.get("properties", {}).get("parameter", {}).get("T2M", {})
            precip_data = data_nasa.get("properties", {}).get("parameter", {}).get("PRECTOTCORR", {})

            df_nasa = pd.DataFrame(index=dates_daily) # Ensure df_nasa is initialized before assignment
            df_nasa['Temperature'] = [temp_data.get(d.strftime("%Y%m%d"), np.nan) for d in dates_daily]
            df_nasa['Precipitation'] = [precip_data.get(d.strftime("%Y%m%d"), np.nan) for d in dates_daily]

            df_nasa.interpolate(method='linear', limit_direction='both', inplace=True)
            df_nasa.bfill(inplace=True) # FIX: Use new method
            df_nasa.ffill(inplace=True) # FIX: Use new method

            if df_nasa.isnull().any().any():
                raise ValueError("NASA POWER data has persistent NaNs after processing.")

            temp_norm = (df_nasa['Temperature'] - df_nasa['Temperature'].mean()) / (df_nasa['Temperature'].std() + 1e-6)
            precip_norm = (df_nasa['Precipitation'] - df_nasa['Precipitation'].mean()) / (df_nasa['Precipitation'].std() + 1e-6)

            ndvi_seasonal = 0.2 * np.sin(np.linspace(0, 4 * np.pi, len(dates_daily)))
            ndvi = 0.5 + ndvi_seasonal + 0.1 * temp_norm * (1 - np.abs(temp_norm)) - 0.05 * precip_norm**2
            ndvi = np.clip(ndvi, 0.1, 0.9)

            soil_moisture_seasonal = 0.1 * np.sin(np.linspace(0, 4 * np.pi, len(dates_daily)) + np.pi/2)
            soil_moisture = 0.4 + soil_moisture_seasonal + 0.2 * precip_norm - 0.1 * temp_norm
            soil_moisture = np.clip(soil_moisture, 0.1, 0.8)

            crop_stress_temp = 20 * np.abs(temp_norm)**1.5
            crop_stress_precip = 20 * (-precip_norm + 0.5)
            crop_stress = crop_stress_temp + crop_stress_precip + (0.5 - soil_moisture) * 30
            crop_stress = np.clip(crop_stress, 5, 95)

            st.success(f"Generated crop health indicators for {region} using NASA POWER data.")
            return pd.DataFrame({
                "Date": dates_daily,
                "NDVI_Estimated": ndvi,
                "Soil_Moisture_Estimated": soil_moisture,
                "Crop_Stress_Estimated": crop_stress
            })

        except Exception as e:
            st.warning(f"Crop health API (NASA POWER) error or processing issue for {region}: {e}. Using fully simulated crop health data.")
            # Fall through to full simulation logic if dates_daily was initially empty or exception occurred

    # Full simulation if NASA POWER fails OR if dates_daily was initially empty
    # Re-generate dates_daily here if it was empty to ensure simulation runs
    if 'dates_daily' not in locals() or dates_daily.empty:
        end_date_dt_sim = datetime.now()
        start_date_dt_sim = end_date_dt_sim - timedelta(days=730)
        dates_daily = pd.date_range(start=start_date_dt_sim, end=end_date_dt_sim, freq='D')
        if dates_daily.empty: # Final fallback if date generation is problematic
             return pd.DataFrame(columns=['Date', 'NDVI_Estimated', 'Soil_Moisture_Estimated', 'Crop_Stress_Estimated'])


    seed = sum(ord(c) for c in region) + sum(ord(c) for c in commodity)
    ndvi_sim = generate_simulated_timeseries_data(dates_daily, 0.5, 0.2, 0.005, 0.05, seed_offset=seed)
    soil_moisture_sim = generate_simulated_timeseries_data(dates_daily, 0.3, 0.1, -0.002, 0.03, seed_offset=seed+1)
    crop_stress_sim = generate_simulated_timeseries_data(dates_daily, 30, 15, 0.01, 5, seed_offset=seed+2)

    return pd.DataFrame({
        "Date": dates_daily,
        "NDVI_Estimated": np.clip(ndvi_sim, 0.1, 0.9),
        "Soil_Moisture_Estimated": np.clip(soil_moisture_sim, 0.1, 0.8),
        "Crop_Stress_Estimated": np.clip(crop_stress_sim, 5, 95)
    })


# Function to get trade flow data (currently simulated)
@st.cache_data(ttl=3600)
def get_trade_flow_data(commodity: str, origin: str, destination: str) -> pd.DataFrame:
    st.info(f"Trade flow data between {origin} and {destination} for {commodity} is currently simulated.")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    dates = pd.date_range(start=start_date, end=end_date, freq='M')
    if dates.empty : return pd.DataFrame(columns=['Date', 'Volume_Simulated', 'Price_Simulated'])


    seed = sum(ord(c) for c in commodity) + sum(ord(c) for c in origin) + sum(ord(c) for c in destination)
    base_volume = 1000 + (seed % 5000)
    volume_sim = generate_simulated_timeseries_data(dates, base_volume, base_volume * 0.3, 0.1, base_volume * 0.1, seed_offset=seed)

    base_price = 100 + (seed % 300)
    price_sim = generate_simulated_timeseries_data(dates, base_price, base_price * 0.2, 0.05, base_price * 0.1, seed_offset=seed+1)

    return pd.DataFrame({"Date": dates, "Volume_Simulated": volume_sim, "Price_Simulated": price_sim})

# Function to generate market opportunities (simplified)
def generate_market_opportunities(commodity_name: str, selected_region: str, user_type: str) -> List[Dict]:
    regions_map = {
        "Asia": ["China", "India", "Vietnam", "Indonesia", "Thailand"],
        "Africa": ["Nigeria", "Egypt", "South Africa", "Kenya", "Ethiopia"],
        "South America": ["Brazil", "Argentina", "Colombia", "Chile", "Peru"],
        "North America": ["USA", "Canada", "Mexico"],
        "Europe": ["Germany", "France", "UK", "Italy", "Spain", "Netherlands"],
        "Middle East": ["UAE", "Saudi Arabia", "Turkey", "Israel"],
        "Oceania": ["Australia", "New Zealand"]
    }
    if selected_region not in regions_map:
        st.error(f"Selected region '{selected_region}' not in regions_map. Defaulting opportunity generation.")
        return [{"title": "Error generating opportunities", "description": "Invalid region selected.", "rationale": "-", "potential_impact": "-", "implementation_timeline": "-", "risk_level": "High", "contacts": []}]

    other_regions = [r for r in regions_map.keys() if r != selected_region]
    if not other_regions: other_regions = list(regions_map.keys()) # Fallback if only one region defined

    num_opportunities = min(2, len(other_regions))
    if num_opportunities == 0 and len(other_regions) > 0: num_opportunities = 1 # Ensure at least one if possible
    elif len(other_regions) == 0: return [] # No other regions to pick from

    diversification_target_regions = random.sample(other_regions, num_opportunities)
    opportunities = []

    for target_region_name in diversification_target_regions:
        if not regions_map.get(target_region_name): continue
        target_country = random.choice(regions_map[target_region_name])
        opp_data = {
            "rationale_options_buyer": [
                f"Recent reports indicate favorable harvest conditions for {commodity_name} in {target_region_name}.",
                f"{target_country} is increasing its export focus on {commodity_name}.",
                f"Potential for lower sourcing costs or unique varieties from {target_country}."
            ],
            "rationale_options_seller": [
                f"Growing demand for {commodity_name} observed in {target_country}.",
                f"Market analysis suggests a supply gap for {commodity_name} in {target_region_name}.",
                f"Favorable import tariffs or consumer preferences in {target_country}."
            ]
        }
        if user_type == "Buyer":
            opp = {
                "title": f"Source {commodity_name} from {target_country} ({target_region_name})",
                "description": (f"Explore {target_country} in {target_region_name} as a potential new sourcing origin "
                                f"for {commodity_name} to diversify your supply chain for the {selected_region} market."),
                "rationale": random.choice(opp_data["rationale_options_buyer"]),
                "potential_impact": f"Estimated {random.randint(3, 15)}% cost saving or supply stability.",
                "implementation_timeline": f"{random.randint(2, 5)} months for initial sourcing.",
                "risk_level": random.choice(["Low", "Medium"]),
                "contacts": generate_contacts(target_country, commodity_name, 2),
                "target_region_for_data": target_region_name
            }
        else:  # Seller
            opp = {
                "title": f"Export {commodity_name} to {target_country} ({target_region_name})",
                "description": (f"Explore {target_country} in {target_region_name} as a potential new export market "
                                f"for {commodity_name} produced in {selected_region}."),
                "rationale": random.choice(opp_data["rationale_options_seller"]),
                "potential_impact": f"Potential {random.randint(5, 20)}% revenue increase.",
                "implementation_timeline": f"{random.randint(3, 7)} months for market entry.",
                "risk_level": random.choice(["Medium", "High"]),
                "contacts": generate_contacts(target_country, commodity_name, 2),
                "target_region_for_data": target_region_name
            }
        opportunities.append(opp)
    return opportunities

# Function to generate contacts (simplified)
def generate_contacts(country: str, commodity_name: str, num_contacts: int = 2) -> List[Dict]:
    first_names = ["Alex", "Maria", "John", "Priya", "Kenji", "Fatima"]
    last_names = ["Smith", "Patel", "Chen", "Garcia", "Müller", "Silva"]
    company_suffixes = ["Global Traders", "Food Exports", "Agri Solutions", "Commodities Inc."]
    positions = ["Procurement Director", "Sales Manager", "Supply Chain Analyst", "Regional Head"]
    contacts = []
    random.seed(sum(ord(c) for c in country + commodity_name))

    for _ in range(num_contacts):
        name = f"{random.choice(first_names)} {random.choice(last_names)}"
        company_name_base = random.choice([country, commodity_name.split(' ')[0]])
        company = f"{company_name_base} {random.choice(company_suffixes)}"
        email_name = name.lower().replace(" ", ".")
        email_domain = company_name_base.lower().replace(" ", "") + random.choice(["biz", "com", "org"])
        contacts.append({
            "name": name, "company": company, "position": random.choice(positions),
            "location": country, "email": f"{email_name}@{email_domain}",
            "phone": f"+{random.randint(1,99)} ({random.randint(100,999)}) {random.randint(100000,999999)}",
            "notes": f"Specializes in {commodity_name} trade."})
    return contacts


# Function to create image from Plotly figure
def create_chart_image(fig: go.Figure) -> str:
    try:
        img_bytes = fig.to_image(format="png", width=800, height=400, scale=2)
        return base64.b64encode(img_bytes).decode()
    except Exception as e:
        st.error(f"Failed to create chart image: {e}. Kaleido might be missing or misconfigured.")
        try:
            img = Image.new('RGB', (800, 400), color = (230, 230, 230))
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("arial.ttf", 30) # Adjusted font size
            except IOError:
                font = ImageFont.load_default()
            text = "Chart Generation Error"
            # Simple text centering logic
            bbox = draw.textbbox((0,0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (800 - text_width) / 2
            y = (400 - text_height) / 2
            draw.text((x, y), text, fill=(128,0,0), font=font)
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode()
        except Exception:
             return ""


# Function to create HTML report
def create_html_report(opportunity: Dict, commodity_name: str, main_region: str, user_type: str,
                       price_chart_b64: str, weather_chart_b64: str,
                       crop_health_chart_b64: str, trade_flow_chart_b64: str) -> str:
    report_title = f"Market Opportunity Report: {commodity_name} for {user_type} in {main_region}"
    opportunity_title = opportunity.get('title', 'N/A')
    description = opportunity.get('description', 'N/A')
    rationale = opportunity.get('rationale', 'N/A')
    potential_impact = opportunity.get('potential_impact', 'N/A')
    timeline = opportunity.get('implementation_timeline', 'N/A')
    risk = opportunity.get('risk_level', 'N/A')
    contacts_html = "<h3>Key Contacts (Illustrative)</h3>"
    if opportunity.get('contacts'):
        contacts_html += "<ul>"
        for contact in opportunity['contacts']:
            contacts_html += f"<li><b>{contact['name']}</b> ({contact['position']} at {contact['company']}) - {contact['location']}<br>"
            contacts_html += f"   Email: {contact['email']}, Phone: {contact['phone']}<br>"
            contacts_html += f"   Notes: {contact.get('notes', '')}</li>"
        contacts_html += "</ul>"
    else:
        contacts_html += "<p>No contacts generated for this illustrative opportunity.</p>"

    html_content = f"""
    <html><head><style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 900px; margin: 20px auto; padding: 20px; border: 1px solid #ddd; }}
        h1 {{ color: {PRIMARY_COLOR}; border-bottom: 2px solid {PRIMARY_COLOR}; padding-bottom: 10px; }}
        h2 {{ color: {PRIMARY_COLOR}; margin-top: 30px; border-bottom: 1px solid {SECONDARY_COLOR}; padding-bottom: 5px;}}
        h3 {{ color: {PRIMARY_COLOR}; margin-top: 20px; }}
        .highlight {{ background-color: {BG_COLOR}; padding: 15px; border-left: 5px solid {SECONDARY_COLOR}; margin-bottom:20px; border-radius: 5px; }}
        p {{ margin-bottom: 10px; }} ul {{ margin-left: 20px; }} li {{ margin-bottom: 8px; }}
        .chart-container {{ margin-top: 20px; padding:10px; border: 1px solid #eee; border-radius: 5px; text-align: center; }}
        img {{ max-width: 100%; height: auto; display: block; margin: 10px auto; border: 1px solid #ccc; }}
        .footer {{ margin-top: 30px; font-size: 0.9em; text-align: center; color: #777; }}
    </style></head><body>
        <h1>{report_title}</h1>
        <div class="highlight">
            <h2>Opportunity: {opportunity_title}</h2>
            <p><strong>Description:</strong> {description}</p><p><strong>Rationale:</strong> {rationale}</p>
            <p><strong>Potential Impact:</strong> {potential_impact}</p><p><strong>Estimated Timeline:</strong> {timeline}</p>
            <p><strong>Risk Level:</strong> {risk}</p>
        </div>
        <h2>Supporting Data Visualizations</h2>
        <p style="font-size:0.9em; color:#555;">Note: Charts below represent recent trends or relevant regional data. Price charts show the primary selected commodity. Weather and Crop Health are for the target opportunity region if specified, otherwise for the main selected region. Trade flow data is simulated.</p>
        <div class="chart-container"><h3>Price Trends for {commodity_name}</h3>
            {f'<img src="data:image/png;base64,{price_chart_b64}" alt="Price Trends for {commodity_name}">' if price_chart_b64 else "<p>Price chart not available.</p>"}</div>
        <div class="chart-container"><h3>Weather Trends (Relevant Region)</h3>
            {f'<img src="data:image/png;base64,{weather_chart_b64}" alt="Weather Impact">' if weather_chart_b64 else "<p>Weather chart not available.</p>"}</div>
        <div class="chart-container"><h3>Estimated Crop Health Indicators (Relevant Region)</h3>
            {f'<img src="data:image/png;base64,{crop_health_chart_b64}" alt="Crop Health Indicators">' if crop_health_chart_b64 else "<p>Crop health chart not available or uses simulated data.</p>"}</div>
        <div class="chart-container"><h3>Simulated Trade Flow Data</h3>
            {f'<img src="data:image/png;base64,{trade_flow_chart_b64}" alt="Trade Flow Data">' if trade_flow_chart_b64 else "<p>Trade flow chart not available.</p>"}</div>
        {contacts_html}
        <div class="footer"><p>Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Sauda Food Insights LLC - Data is for illustrative and informational purposes only.</p></div>
    </body></html>"""
    return html_content

# Function to get HTML download link
def get_html_download_link(html_content: str, filename: str, link_text: str) -> str:
    b64 = base64.b64encode(html_content.encode()).decode()
    return f'<a href="data:text/html;charset=utf-8;base64,{b64}" download="{filename}.html" class="download-button">{link_text}</a>'


# Main application
def main():
    st.sidebar.header("User Settings")
    user_type = st.sidebar.radio("I am a:", ["Buyer", "Seller"])

    st.sidebar.header("Commodity Selection")
    available_commodities = get_available_commodities()
    if not available_commodities:
        st.error("No commodities available to select. Please check configuration.")
        return

    selected_commodity_ticker = st.sidebar.selectbox(
        "Select Commodity",
        options=list(available_commodities.keys()),
        format_func=lambda x: available_commodities[x]
    )
    selected_commodity_name = available_commodities[selected_commodity_ticker]

    st.sidebar.header("Region Selection")
    region_options = ["Asia", "Africa", "South America", "North America", "Europe", "Middle East", "Oceania"]
    if user_type == "Buyer":
        region_label = "My Market Region (Destination)"
    else: # Seller
        region_label = "My Producing Region (Origin)"
    selected_region = st.sidebar.selectbox(region_label, options=region_options)

    st.title(f"{selected_commodity_name} Market Intelligence")
    st.markdown(f"**Viewing as:** `{user_type}` | **Primary Region:** `{selected_region}`")
    if user_type == "Buyer":
        st.caption(f"Insights are focused on sourcing opportunities for your market in {selected_region}.")
    else:
        st.caption(f"Insights are focused on export opportunities from your producing region, {selected_region}.")

    price_data = get_price_data(selected_commodity_ticker, selected_commodity_name, period="2y")
    weather_data = get_weather_data(selected_region, days_to_fetch_actual_weather=7)
    crop_health_data = get_crop_health_data(selected_region, selected_commodity_name)

    tab1, tab2, tab3 = st.tabs(["Market Overview", "Regional Analysis", "Market Opportunities"])

    with tab1:
        st.header(f"Price Analysis: {selected_commodity_name}")
        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        if not price_data.empty and 'Close' in price_data.columns and not price_data['Close'].empty: # Added check for non-empty 'Close' series
            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(x=price_data.index, y=price_data['Close'], mode='lines', name='Closing Price', line=dict(color=PRIMARY_COLOR)))
            fig_price.update_layout(
                title=f"{selected_commodity_name} Price Trends (Last 2 Years)",
                xaxis_title="Date", yaxis_title="Price (Currency varies by commodity)",
                height=450, showlegend=True, template="plotly_white"
            )
            st.plotly_chart(fig_price, use_container_width=True)
            
            # --- MODIFICATION FOR st.metric START ---
            latest_price_raw = price_data['Close'].iloc[-1]
            avg_price_raw = price_data['Close'].mean()

            date_label = "N/A"
            if not price_data.index.empty:
                try:
                    date_label = price_data.index[-1].strftime('%Y-%m-%d')
                except IndexError: # Should be rare if 'Close' series is not empty
                    st.caption("Could not determine latest price date.")

            latest_price_display = None
            if pd.notnull(latest_price_raw):
                if isinstance(latest_price_raw, pd.Series):
                    if not latest_price_raw.empty: # Check if Series is not empty
                        latest_price_display = float(latest_price_raw.iloc[0])
                else:
                    latest_price_display = float(latest_price_raw)
            
            avg_price_display = float(avg_price_raw) if pd.notnull(avg_price_raw) else None

            col_metric1, col_metric2 = st.columns(2) # Use columns for better layout of metrics
            with col_metric1:
                if latest_price_display is not None:
                    st.metric(label=f"Latest Price ({date_label})", value=f"{latest_price_display:.2f}")
                else:
                    st.metric(label=f"Latest Price ({date_label})", value="N/A")
            with col_metric2:
                if avg_price_display is not None:
                    st.metric(label="2-Year Average Price", value=f"{avg_price_display:.2f}")
                else:
                    st.metric(label="2-Year Average Price", value="N/A")
            # --- MODIFICATION FOR st.metric END ---
        else:
            st.warning(f"Price data for {selected_commodity_name} is currently unavailable or uses simulation. Full analysis may be limited.")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.header(f"Regional Factors in {selected_region}")
        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        st.subheader(f"Weather Impact in {selected_region}")
        if not weather_data.empty and 'Temperature' in weather_data.columns and 'Rainfall' in weather_data.columns:
            fig_weather = go.Figure()
            fig_weather.add_trace(go.Scatter(x=weather_data['Date'], y=weather_data['Temperature'], mode='lines', name='Avg Temperature (°C)', line=dict(color='red')))
            fig_weather.add_trace(go.Scatter(x=weather_data['Date'], y=weather_data['Rainfall'], mode='lines', name='Total Rainfall (mm)', yaxis='y2', line=dict(color='blue', dash='dash')))
            fig_weather.update_layout(title=f"Weather Trends in {selected_region}", xaxis_title="Date", yaxis_title="Temperature (°C)",
                                      yaxis2=dict(title="Rainfall (mm)", overlaying='y', side='right'), height=450, showlegend=True, template="plotly_white")
            st.plotly_chart(fig_weather, use_container_width=True)
        else:
            st.warning(f"Weather data for {selected_region} is currently unavailable or uses simulation.")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        st.subheader(f"Estimated Crop Health in {selected_region} for {selected_commodity_name}")
        if not crop_health_data.empty:
            fig_crop = go.Figure()
            if 'NDVI_Estimated' in crop_health_data.columns: fig_crop.add_trace(go.Scatter(x=crop_health_data['Date'], y=crop_health_data['NDVI_Estimated'], mode='lines', name='Estimated NDVI', line=dict(color='green')))
            if 'Soil_Moisture_Estimated' in crop_health_data.columns: fig_crop.add_trace(go.Scatter(x=crop_health_data['Date'], y=crop_health_data['Soil_Moisture_Estimated'], mode='lines', name='Estimated Soil Moisture', line=dict(color='brown')))
            if 'Crop_Stress_Estimated' in crop_health_data.columns: fig_crop.add_trace(go.Scatter(x=crop_health_data['Date'], y=crop_health_data['Crop_Stress_Estimated'], mode='lines', name='Estimated Crop Stress Index', yaxis='y2', line=dict(color='orange', dash='dot')))
            fig_crop.update_layout(title=f"Estimated Crop Health Indicators in {selected_region}", xaxis_title="Date", yaxis_title="Index Value (NDVI, Soil Moisture)",
                                   yaxis2=dict(title="Crop Stress Index", overlaying='y', side='right'), height=450, showlegend=True, template="plotly_white")
            if fig_crop.data: st.plotly_chart(fig_crop, use_container_width=True)
            else: st.info("No specific crop health indicators available to display from the data.")
        else:
            st.warning(f"Crop health data for {selected_region} is currently unavailable or uses simulation.")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.header("Market Opportunities & Insights")
        st.markdown("These are illustrative opportunities based on high-level data and random factors for demonstration.")
        with st.spinner("Generating potential market opportunities..."):
            opportunities = generate_market_opportunities(selected_commodity_name, selected_region, user_type)
            time.sleep(0.5)

        if not opportunities: st.info("No specific market opportunities generated at this time. Adjust selections or check back later.")
        
        for i, opp in enumerate(opportunities):
            with st.expander(f"Opportunity {i+1}: {opp['title']}", expanded=(i==0)):
                st.markdown(f"**Description:** {opp.get('description', 'N/A')}")
                st.markdown(f"**Rationale:** {opp.get('rationale', 'N/A')}")
                # ... (rest of opportunity details) ...
                opp_price_chart_b64 = ""
                if not price_data.empty and 'Close' in price_data.columns and not price_data['Close'].empty:
                    fig_price_opp = go.Figure(data=[go.Scatter(x=price_data.index, y=price_data['Close'], mode='lines', name='Price', line=dict(color=PRIMARY_COLOR))])
                    fig_price_opp.update_layout(title=f"Price Trend: {selected_commodity_name}", height=300, template="plotly_white", margin=dict(t=40, b=20, l=20, r=20))
                    opp_price_chart_b64 = create_chart_image(fig_price_opp)

                opp_weather_chart_b64 = ""
                target_data_region = opp.get("target_region_for_data", selected_region)
                opp_weather_data = get_weather_data(target_data_region, days_to_fetch_actual_weather=30)
                if not opp_weather_data.empty:
                    fig_weather_opp = go.Figure()
                    if 'Temperature' in opp_weather_data: fig_weather_opp.add_trace(go.Scatter(x=opp_weather_data['Date'], y=opp_weather_data['Temperature'],name='Temperature (°C)', line=dict(color='red')))
                    if 'Rainfall' in opp_weather_data: fig_weather_opp.add_trace(go.Scatter(x=opp_weather_data['Date'], y=opp_weather_data['Rainfall'],name='Rainfall (mm)', yaxis='y2', line=dict(color='blue')))
                    fig_weather_opp.update_layout(title=f"Weather: {target_data_region}", yaxis2=dict(overlaying='y', side='right') if 'Rainfall' in opp_weather_data else {}, height=300, template="plotly_white", margin=dict(t=40, b=20, l=20, r=20))
                    if fig_weather_opp.data : opp_weather_chart_b64 = create_chart_image(fig_weather_opp)
                
                opp_crop_health_chart_b64 = ""
                opp_crop_data = get_crop_health_data(target_data_region, selected_commodity_name)
                if not opp_crop_data.empty:
                    fig_crop_opp = go.Figure()
                    if 'NDVI_Estimated' in opp_crop_data: fig_crop_opp.add_trace(go.Scatter(x=opp_crop_data['Date'], y=opp_crop_data['NDVI_Estimated'], name='Est. NDVI'))
                    fig_crop_opp.update_layout(title=f"Est. Crop Health: {target_data_region}", height=300, template="plotly_white", margin=dict(t=40, b=20, l=20, r=20))
                    if fig_crop_opp.data: opp_crop_health_chart_b64 = create_chart_image(fig_crop_opp)

                opp_trade_flow_chart_b64 = ""
                trade_origin, trade_dest = (selected_region, target_data_region) if user_type == "Seller" else (target_data_region, selected_region)
                opp_trade_data = get_trade_flow_data(selected_commodity_name, trade_origin, trade_dest)
                if not opp_trade_data.empty and 'Volume_Simulated' in opp_trade_data:
                    fig_trade_opp = go.Figure()
                    fig_trade_opp.add_trace(go.Scatter(x=opp_trade_data['Date'], y=opp_trade_data['Volume_Simulated'], name='Sim. Volume'))
                    fig_trade_opp.update_layout(title=f"Sim. Trade: {trade_origin} to {trade_dest}", height=300, template="plotly_white", margin=dict(t=40, b=20, l=20, r=20))
                    opp_trade_flow_chart_b64 = create_chart_image(fig_trade_opp)
                
                if opp.get('contacts'):
                    st.subheader("Illustrative Contacts")
                    for contact in opp['contacts']: st.markdown(f"- **{contact['name']}** ({contact['position']})\n    - Company: {contact['company']}, Location: {contact['location']}\n    - Email: `{contact['email']}` , Phone: `{contact['phone']}`\n    - *{contact.get('notes', '')}*")
                
                try:
                    html_content = create_html_report(opp, selected_commodity_name, selected_region, user_type,
                                                      opp_price_chart_b64, opp_weather_chart_b64,
                                                      opp_crop_health_chart_b64, opp_trade_flow_chart_b64)
                    clean_title = "".join(c if c.isalnum() else "_" for c in opp['title'])
                    download_filename = f"Opportunity_{selected_commodity_name.replace(' ','_')}_{clean_title[:30]}"
                    st.markdown(get_html_download_link(html_content, download_filename, "Download Opportunity Report"), unsafe_allow_html=True)
                except Exception as e_report:
                    st.error(f"Could not generate download link for report: {e_report}")

    st.sidebar.markdown("---")
    st.sidebar.info("© Sauda Food Insights LLC. Data for illustrative purposes.")

if __name__ == "__main__":
    main()
