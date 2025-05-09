import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import requests
import base64
from PIL import Image
import io
import yfinance as yf
import random
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import time
import json
from typing import Dict, List, Optional

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
    .stSelectbox label, .stMultiselect label {{
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
        padding: 10px;
        background-color: #f9f9f9;
        border-radius: 5px;
    }}
    .download-button {{
        display: inline-block;
        padding: 10px 20px;
        background-color: {SECONDARY_COLOR};
        color: white;
        text-decoration: none;
        border-radius: 5px;
    }}
    .download-button:hover {{
        background-color: {PRIMARY_COLOR};
    }}
    @media (max-width: 768px) {{
        .stSidebar {{
            width: 100% !important;
        }}
        .chart-container {{
            height: auto !important;
        }}
    }}
</style>
""", unsafe_allow_html=True)

# Load and display logo
logo_path = "IMG_3036.png"
try:
    logo = Image.open(logo_path)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(logo, width=300)
except:
    st.title("Sauda Food Insights LLC")
    st.caption("Food Insights Platform")

# Function to get available commodities
@st.cache_data(ttl=3600)
def get_available_commodities():
    base_commodities = {
        "ZW=F": "Wheat", "ZC=F": "Corn", "ZS=F": "Soybeans", "ZM=F": "Soybean Meal",
        "ZL=F": "Soybean Oil", "ZO=F": "Oats", "ZR=F": "Rice", "KE=F": "KC Wheat",
        "JO=F": "Orange Juice", "CC=F": "Cocoa", "KC=F": "Coffee", "SB=F": "Sugar",
        "LE=F": "Live Cattle", "GF=F": "Feeder Cattle", "HE=F": "Lean Hogs",
        "CT=F": "Cotton", "LBS=F": "Lumber", "DC=F": "Class III Milk",
        "PEPPER": "Black Pepper", "CINNAMON": "Cinnamon", "ALMOND": "Almonds",
        "CASHEW": "Cashews", "SALMON": "Salmon", "SHRIMP": "Shrimp"
    }
    valid_commodities = {}
    for ticker, name in base_commodities.items():
        try:
            info = yf.Ticker(ticker).info
            if 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
                valid_commodities[ticker] = name
        except:
            valid_commodities[ticker] = name  # Keep even if no Yahoo Finance data
    
    custom_ticker = st.sidebar.text_input("Enter Custom Commodity Ticker", "")
    st.sidebar.caption("Tip: Use Yahoo Finance tickers (e.g., 'ZW=F' for Wheat).")
    if custom_ticker and custom_ticker not in valid_commodities:
        try:
            info = yf.Ticker(custom_ticker).info
            if 'regularMarketPrice' in info:
                valid_commodities[custom_ticker] = info.get('shortName', custom_ticker)
                st.sidebar.success(f"Added custom commodity: {custom_ticker}")
            else:
                st.sidebar.warning(f"Invalid ticker: {custom_ticker}. Added anyway.")
                valid_commodities[custom_ticker] = custom_ticker
        except:
            st.sidebar.warning(f"Invalid ticker: {custom_ticker}. Added anyway.")
            valid_commodities[custom_ticker] = custom_ticker
    
    return valid_commodities

# Function to get price data
@st.cache_data(ttl=3600)
def get_price_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with st.spinner(f"Fetching price data for {ticker} (Attempt {attempt + 1}/{max_retries})"):
                data = yf.download(ticker, period=period, timeout=10)
                if not data.empty:
                    return data
                raise Exception("Empty data received")
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"Error fetching data for {ticker}: {e}")
                return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
            time.sleep(2 ** attempt)
    return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])

# Function to get weather data
@st.cache_data(ttl=3600)
def get_weather_data(region: str) -> pd.DataFrame:
    try:
        api_key = st.secrets["OPENWEATHERMAP_API_KEY"]
    except KeyError:
        st.warning("OpenWeatherMap API key not found. Using simulated weather data.")
        api_key = None
    
    region_coords = {
        "Asia": (35.8617, 104.1954), "Africa": (8.7832, 34.5085),
        "South America": (-14.2350, -51.9253), "North America": (37.0902, -95.7129),
        "Europe": (54.5260, 15.2551), "Middle East": (29.3117, 47.4818),
        "Oceania": (-25.2744, 133.7751)
    }
    lat, lon = region_coords.get(region, (0, 0))
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    dates = pd.date_range(start=start_date, end=end_date, freq='M')
    
    if api_key:
        try:
            weather_data = []
            for date in dates:
                params = {
                    "lat": lat, "lon": lon, "dt": int(date.timestamp()),
                    "appid": api_key, "units": "metric", "cnt": 24
                }
                response = requests.get("https://api.openweathermap.org/data/2.5/onecall/timemachine", params=params)
                data = response.json()
                if data.get("hourly"):
                    daily_temp = np.mean([h["temp"] for h in data["hourly"]])
                    daily_rain = sum(h.get("rain", {}).get("1h", 0) for h in data["hourly"])
                    weather_data.append({"Date": date, "Temperature": daily_temp, "Rainfall": daily_rain})
            if weather_data:
                return pd.DataFrame(weather_data)
        except Exception as e:
            st.warning(f"Weather API error for {region}: {e}. Using simulated data.")
    
    # Simulated data fallback
    seed = sum(ord(c) for c in region)
    np.random.seed(seed)
    temp_base = 20 + 10 * np.sin(np.linspace(0, 4*np.pi, len(dates)))
    temp_noise = np.random.normal(0, 2, len(dates))
    rainfall = np.maximum(0, 50 + 30 * np.sin(np.linspace(0, 4*np.pi, len(dates))) + np.random.normal(0, 10, len(dates)))
    return pd.DataFrame({"Date": dates, "Temperature": temp_base + temp_noise, "Rainfall": rainfall})

# Function to get crop health data
@st.cache_data(ttl=3600)
def get_crop_health_data(region: str, commodity: str) -> pd.DataFrame:
    # TODO: Consider Planet Labs or Sentinel Hub for advanced crop health data
    api_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    region_coords = {
        "Asia": (35.8617, 104.1954), "Africa": (8.7832, 34.5085),
        "South America": (-14.2350, -51.9253), "North America": (37.0902, -95.7129),
        "Europe": (54.5260, 15.2551), "Middle East": (29.3117, 47.4818),
        "Oceania": (-25.2744, 133.7751)
    }
    lat, lon = region_coords.get(region, (0, 0))
    
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        dates = pd.date_range(start=start_date, end=end_date, freq='M')
        
        params = {
            "latitude": lat, "longitude": lon, "start": start_date.strftime("%Y%m%d"),
            "end": end_date.strftime("%Y%m%d"), "community": "ag", "parameters": "T2M,PRECTOTCOR,ALLSKY_SFC_SW_DWN",
            "format": "JSON", "user": "anonymous"
        }
        response = requests.get(api_url, params=params)
        data = response.json()["properties"]["parameter"]
        
        temp = np.array([data["T2M"][d.strftime("%Y%m%d")] for d in dates])
        precip = np.array([data["PRECTOTCOR"][d.strftime("%Y%m%d")] for d in dates])
        ndvi = 0.5 + 0.2 * np.sin(np.linspace(0, 4*np.pi, len(dates))) + (temp - np.mean(temp)) / 50 + np.random.normal(0, 0.05, len(dates))
        soil_moisture = 0.3 + 0.1 * np.sin(np.linspace(0, 4*np.pi, len(dates))) + (precip - np.mean(precip)) / 500 + np.random.normal(0, 0.03, len(dates))
        crop_stress = 30 - 15 * np.sin(np.linspace(0, 4*np.pi, len(dates))) + (np.mean(temp) - temp) / 2 + np.random.normal(0, 5, len(dates))
        
        return pd.DataFrame({
            "Date": dates, "NDVI": np.clip(ndvi, 0, 1), "Soil_Moisture": np.clip(soil_moisture, 0, 1),
            "Crop_Stress": np.clip(crop_stress, 0, 100)
        })
    except Exception as e:
        st.warning(f"Crop health API error for {region}: {e}. Using simulated data.")
        seed = sum(ord(c) for c in region) + sum(ord(c) for c in commodity)
        np.random.seed(seed)
        ndvi = 0.5 + 0.2 * np.sin(np.linspace(0, 4*np.pi, len(dates))) + np.linspace(0, 0.05, len(dates)) + np.random.normal(0, 0.05, len(dates))
        soil_moisture = 0.3 + 0.1 * np.sin(np.linspace(0, 4*np.pi, len(dates))) + np.random.normal(0, 0.03, len(dates))
        crop_stress = 30 - 15 * np.sin(np.linspace(0, 4*np.pi, len(dates))) + np.random.normal(0, 5, len(dates))
        return pd.DataFrame({
            "Date": dates, "NDVI": np.clip(ndvi, 0, 1), "Soil_Moisture": np.clip(soil_moisture, 0, 1),
            "Crop_Stress": np.clip(crop_stress, 0, 100)
        })

# Function to get trade flow data
@st.cache_data(ttl=3600)
def get_trade_flow_data(commodity: str, origin: str, destination: str) -> pd.DataFrame:
    # TODO: Integrate with FAOSTAT or UN Comtrade API for real trade data
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        dates = pd.date_range(start=start_date, end=end_date, freq='M')
        
        seed = sum(ord(c) for c in commodity) + sum(ord(c) for c in origin) + sum(ord(c) for c in destination)
        np.random.seed(seed)
        base_volume = 1000 + (sum(ord(c) for c in commodity) % 5000)
        volume = np.maximum(0, base_volume + base_volume * 0.3 * np.sin(np.linspace(0, 4*np.pi, len(dates))) + np.linspace(0, base_volume * 0.2, len(dates)) + np.random.normal(0, base_volume * 0.1, len(dates)))
        price = np.maximum(0, 100 + 20 * np.sin(np.linspace(0, 4*np.pi, len(dates))) + np.linspace(0, 30, len(dates)) + np.random.normal(0, 10, len(dates)))
        
        return pd.DataFrame({"Date": dates, "Volume": volume, "Price": price})
    except Exception as e:
        st.error(f"Trade flow data error: {e}")
        return pd.DataFrame({"Date": [], "Volume": [], "Price": []})

# Function to generate market opportunities
def generate_market_opportunities(commodity: str, region: str, user_type: str) -> List[Dict]:
    regions = {
        "Asia": ["China", "India", "Vietnam"], "Africa": ["Egypt", "South Africa", "Kenya"],
        "South America": ["Brazil", "Argentina", "Chile"], "North America": ["USA", "Canada", "Mexico"],
        "Europe": ["France", "Germany", "Italy"], "Middle East": ["UAE", "Saudi Arabia", "Turkey"],
        "Oceania": ["Australia", "New Zealand"]
    }
    other_regions = [r for r in regions.keys() if r != region]
    diversification_regions = random.sample(other_regions, min(3, len(other_regions)))
    opportunities = []
    
    if user_type == "Buyer":
        for div_region in diversification_regions:
            div_country = random.choice(regions[div_region])
            opportunities.append({
                "title": f"Explore sourcing {commodity} from {div_country} ({div_region})",
                "description": f"Recommended sourcing country: {div_country} in {div_region}",
                "rationale": random.choice([
                    f"Production in {div_country} has increased by 15% year-over-year",
                    f"New trade agreement reduces import duties to {region}",
                    f"Optimal crop health in {div_country}"
                ]),
                "potential_impact": f"Potential cost savings of {random.randint(5, 25)}%",
                "implementation_timeline": f"{random.randint(1, 3)} months",
                "risk_level": random.choice(["Low", "Medium", "High"]),
                "contacts": generate_contacts(div_country, 2)
            })
    else:  # Seller
        for div_region in diversification_regions:
            div_country = random.choice(regions[div_region])
            opportunities.append({
                "title": f"Explore exporting {commodity} to {div_country} ({div_region})",
                "description": f"Recommended market country: {div_country} in {div_region}",
                "rationale": random.choice([
                    f"{div_country} has a supply gap of 15,000 tons annually",
                    f"23% increase in {commodity} consumption in {div_country}",
                    f"New shipping routes reduce costs to {div_country}"
                ]),
                "potential_impact": f"Potential revenue increase of {random.randint(10, 30)}%",
                "implementation_timeline": f"{random.randint(2, 6)} months",
                "risk_level": random.choice(["Low", "Medium", "High"]),
                "contacts": generate_contacts(div_country, 2)
            })
    return opportunities

# Function to generate contacts
def generate_contacts(country: str, num_contacts: int = 3) -> List[Dict]:
    seed = sum(ord(c) for c in country)
    random.seed(seed)
    company_patterns = ["{country} Traders", "{country} Exports", "Distributors of {country}"]
    contact_names = {
        "China": ["Li Wei", "Zhang Min"], "India": ["Raj Sharma", "Priya Patel"],
        "USA": ["Michael Johnson", "Jennifer Smith"], "Brazil": ["Carlos Silva", "Ana Santos"]
    }
    default_names = ["John Smith", "Jane Doe"]
    names = contact_names.get(country, default_names)
    contacts = []
    commodities = ["Rice", "Wheat", "Corn", "Soybeans"]
    
    for _ in range(num_contacts):
        name = random.choice(names)
        commodity = random.choice(commodities)
        company = random.choice(company_patterns).format(country=country, commodity=commodity)
        position = random.choice(["Procurement Manager", "Supply Chain Director"])
        email = f"{name.lower().replace(' ', '.')}@{company.lower().replace(' ', '')}.com"
        phone = f"+{random.randint(1, 999)} {random.randint(100, 999)} {random.randint(1000, 9999)}"
        contacts.append({
            "name": name, "company": company, "position": position, "location": country,
            "email": email, "phone": phone, "contact": f"{name}, {position}"
        })
    return contacts

# Function to create HTML report (unchanged for brevity, uses existing logic)
def create_html_report(opportunity: Dict, commodity: str, region: str, user_type: str, price_chart: str, weather_chart: str, crop_health_chart: str, trade_flow_chart: str) -> str:
    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 800px; margin: 0 auto; padding: 20px; }}
            h1 {{ color: #2a5d4c; }}
            .highlight {{ background-color: #f8f9fa; padding: 15px; border-left: 4px solid #8bc34a; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ padding: 12px; border-bottom: 1px solid #ddd; }}
        </style>
    </head>
    <body>
        <h1>Market Opportunity Report: {commodity}</h1>
        <div class="highlight">
            <h2>Opportunity Summary</h2>
            <p><strong>{opportunity['title']}</strong></p>
            <p>{opportunity['description']}</p>
            <p><strong>Rationale:</strong> {opportunity['rationale']}</p>
        </div>
        <h2>Charts</h2>
        <img src="data:image/png;base64,{price_chart}" alt="Price Trends" style="width:100%">
        <img src="data:image/png;base64,{weather_chart}" alt="Weather Impact" style="width:100%">
    </body>
    </html>
    """
    return html_content

def create_chart_image(fig: go.Figure) -> str:
    img_bytes = fig.to_image(format="png", width=800, height=400)
    return base64.b64encode(img_bytes).decode()

def get_html_download_link(html_content: str, filename: str) -> str:
    b64 = base64.b64encode(html_content.encode()).decode()
    return f'<a href="data:text/html;base64,{b64}" download="{filename}.html" class="download-button">Download Report</a>'

# Main application
def main():
    st.sidebar.header("User Settings")
    user_type = st.sidebar.radio("Select User Type", ["Buyer", "Seller"])
    
    st.sidebar.header("Commodity Selection")
    available_commodities = get_available_commodities()
    selected_commodity = st.sidebar.selectbox(
        "Select Commodity", options=list(available_commodities.keys()),
        format_func=lambda x: available_commodities[x]
    )
    selected_commodity_name = available_commodities[selected_commodity]
    
    st.sidebar.header("Region Selection")
    if user_type == "Buyer":
        region_label = "Your Market Region (Destination)"
    else:
        region_label = "Your Producing Region (Origin)"
    selected_region = st.sidebar.selectbox(
        region_label, options=["Asia", "Africa", "South America", "North America", "Europe", "Middle East", "Oceania"]
    )
    
    st.title(f"{selected_commodity_name} Market Intelligence")
    st.subheader(f"Region: {selected_region} | View: {user_type}")
    st.caption(f"For {user_type}s, the selected region is {'your market region (destination)' if user_type == 'Buyer' else 'your producing region (origin)'}. Opportunities suggest {'sourcing from producing regions' if user_type == 'Buyer' else 'exporting to market regions'}.")
    
    tab1, tab2 = st.tabs(["Market Analysis", "Opportunities"])
    
    with tab1:
        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        st.subheader("Price Analysis")
        with st.spinner("Loading price data..."):
            price_data = get_price_data(selected_commodity)
        if not price_data.empty:
            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(x=price_data.index, y=price_data['Close'], mode='lines', name='Price', line=dict(color=PRIMARY_COLOR)))
            fig_price.update_layout(title=f"{selected_commodity_name} Price Trends", height=400, showlegend=True)
            st.plotly_chart(fig_price, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        st.subheader("Weather Impact")
        with st.spinner("Loading weather data..."):
            weather_data = get_weather_data(selected_region)
        if not weather_data.empty:
            fig_weather = go.Figure()
            fig_weather.add_trace(go.Scatter(x=weather_data['Date'], y=weather_data['Temperature'], mode='lines', name='Temperature (°C)', line=dict(color='red')))
            fig_weather.update_layout(title=f"Weather in {selected_region}", height=400, showlegend=True)
            st.plotly_chart(fig_weather, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.header("Market Opportunities")
        with st.spinner("Generating opportunities..."):
            opportunities = generate_market_opportunities(selected_commodity_name, selected_region, user_type)
        for i, opp in enumerate(opportunities):
            with st.expander(f"Opportunity {i+1}: {opp['title']}"):
                st.markdown(f"**Description:** {opp['description']}")
                st.markdown(f"**Rationale:** {opp['rationale']}")
                
                price_data = get_price_data(selected_commodity)[-24:]
                fig_price = go.Figure(data=[go.Scatter(x=price_data.index, y=price_data['Close'], mode='lines')])
                price_chart = create_chart_image(fig_price)
                
                weather_data = get_weather_data(selected_region)[-24:]
                fig_weather = go.Figure(data=[go.Scatter(x=weather_data['Date'], y=weather_data['Temperature'], mode='lines')])
                weather_chart = create_chart_image(fig_weather)
                
                html_content = create_html_report(opp, selected_commodity_name, selected_region, user_type, price_chart, weather_chart, "", "")
                st.markdown(get_html_download_link(html_content, f"report_{i}"), unsafe_allow_html=True)

if __name__ == "__main__":
    main()