import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib

from tensorflow.keras.models import load_model

# PAGE CONFIGURATION
st.set_page_config(
    page_title="StockLens",
    page_icon="📈",
    layout="wide"
)
# LOAD MODEL AND SCALER

@st.cache_resource
def load_prediction_assets():
    model = load_model(
        "final_lstm_model.keras",
        compile=False
    )

    scaler = joblib.load(
        "final_scaler.pkl"
    )

    return model, scaler


try:
    model, scaler = load_prediction_assets()

except Exception as e:
    st.error("Unable to load the forecasting model.")
    st.exception(e)
    st.stop()

# DOWNLOAD MARKET DATA
@st.cache_data(ttl=900)
def get_market_data():
    try:
        data = yf.download(
            "NVDA",
            period="6mo",
            interval="1d",
            progress=False,
            auto_adjust=False
        )

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.dropna()

        if not data.empty:
            return data, "Yahoo Finance"

    except Exception:
        pass

    # Fallback to local CSV
    data = pd.read_csv(
        "NVDA_yfinance_clean.csv",
        index_col=0,
        parse_dates=True
    )

    return data.dropna(), "Local CSV fallback"

try:
    market_data, data_source = get_market_data()

except Exception as e:
    st.error("Unable to retrieve NVIDIA market data.")
    st.exception(e)
    st.stop()


if market_data.empty:
    st.error("Market data is currently unavailable.")
    st.stop()

# HEADER
st.title("📈 StockLens")

st.write(
    "Market insights, historical trends, "
    "and data-driven price forecasting."
)

st.info(
    "Forecasts are generated from historical market data "
    "for analytical and educational purposes only. "
    "They should not be considered financial advice."
)


# MARKET OVERVIEW
st.divider()

st.header("NVIDIA Market Overview")


latest_price = float(
    market_data["Close"].iloc[-1]
)

previous_price = float(
    market_data["Close"].iloc[-2]
)

daily_change = (
    latest_price - previous_price
)

daily_change_percent = (
    daily_change / previous_price
) * 100


period_high = float(
    market_data["High"].max()
)

period_low = float(
    market_data["Low"].min()
)

latest_date = market_data.index[-1]


col1, col2, col3, col4 = st.columns(4)


col1.metric(
    "Latest Close",
    f"${latest_price:.2f}",
    f"{daily_change_percent:+.2f}%"
)


col2.metric(
    "6-Month High",
    f"${period_high:.2f}"
)


col3.metric(
    "6-Month Low",
    f"${period_low:.2f}"
)


col4.metric(
    "Last Trading Session",
    latest_date.strftime("%d %b %Y")
)
st.caption(
    f"Data source: {data_source} | "
    f"Latest record: {market_data.index[-1].strftime('%d %b %Y')}"
)


# FORECAST SECTION

st.divider()

st.header("Next-Session Forecast")

st.caption(
    "Estimate NVIDIA's next closing price using "
    "the latest 60 available trading sessions."
)


# Keep forecast after Streamlit reruns
if "forecast_result" not in st.session_state:
    st.session_state.forecast_result = None


forecast_button = st.button(
    "Generate Forecast",
    type="primary",
    key="stocklens_forecast_button"
)


if forecast_button:

    try:

        
        # LAST 60 CLOSING PRICES

        close_prices = (
            market_data["Close"]
            .astype(float)
            .to_numpy()
        )


        if len(close_prices) < 60:

            st.error(
                "At least 60 trading sessions are required "
                "to generate a forecast."
            )

        else:

            last_60_days = (
                close_prices[-60:]
                .reshape(-1, 1)
            )


            
            # SCALE DATA

            scaled_data = scaler.transform(
                last_60_days
            )


            
            # PREPARE LSTM INPUT
            # Shape = (batch, timesteps, features)
            

            X_input = np.asarray(
                scaled_data,
                dtype=np.float32
            ).reshape(
                1,
                60,
                1
            )


            
            # DIRECT MODEL INFERENCE
            # Avoid model.predict() overhead/hanging
            

            prediction_tensor = model(
                X_input,
                training=False
            )


            # Convert TensorFlow tensor to NumPy
            scaled_prediction = np.asarray(
                prediction_tensor
            ).reshape(-1, 1)


            
            # CONVERT PREDICTION BACK TO STOCK PRICE
           

            predicted_price = float(
                scaler.inverse_transform(
                    scaled_prediction
                )[0][0]
            )


            
            # CALCULATE PROJECTED MOVEMENT
            

            price_change = (
                predicted_price
                - latest_price
            )


            percentage_change = (
                price_change
                / latest_price
            ) * 100


            
            # DETERMINE DIRECTION
            

            if percentage_change > 0.5:

                direction = "UPWARD ↑"

            elif percentage_change < -0.5:

                direction = "DOWNWARD ↓"

            else:

                direction = "STABLE →"


            
            # SAVE RESULT
            

            st.session_state.forecast_result = {

                "current_price":
                    latest_price,

                "predicted_price":
                    predicted_price,

                "price_change":
                    price_change,

                "percentage_change":
                    percentage_change,

                "direction":
                    direction

            }


    except Exception as e:

        st.error(
            "The forecast could not be generated."
        )

        st.exception(e)



# DISPLAY FORECAST RESULT


if st.session_state.forecast_result is not None:

    result = st.session_state.forecast_result


    st.subheader("Forecast Summary")


    result_col1, result_col2, result_col3, result_col4 = (
        st.columns(4)
    )


    result_col1.metric(
        "Latest Close",
        f"${result['current_price']:.2f}"
    )


    result_col2.metric(
        "Forecasted Close",
        f"${result['predicted_price']:.2f}"
    )


    result_col3.metric(
        "Projected Change",
        f"${result['price_change']:+.2f}",
        f"{result['percentage_change']:+.2f}%"
    )


    result_col4.metric(
        "Projected Direction",
        result["direction"]
    )


    
    # FORECAST COMPARISON CHART
    

    st.subheader("Price Comparison")


    forecast_comparison = pd.DataFrame(

        {
            "Price": [

                result["current_price"],

                result["predicted_price"]

            ]
        },

        index=[

            "Latest Close",

            "Forecasted Close"

        ]

    )


    st.bar_chart(
        forecast_comparison,
        height=350,
        use_container_width=True
    )



# HISTORICAL PRICE PERFORMANCE


st.divider()

st.header(
    "Historical Price Performance"
)


chart_data = (

    market_data[
        ["Close"]
    ]

    .copy()

)


chart_data.columns = [
    "Closing Price"
]


st.line_chart(

    chart_data,

    height=450,

    use_container_width=True

)


st.caption(
    "NVIDIA closing-price movement "
    "over the latest six-month period."
)



# RECENT TRADING DATA


st.divider()

st.header(
    "Recent Trading Data"
)


recent_data = (

    market_data[

        [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

    ]

    .tail(10)

    .copy()

)


recent_data = recent_data.sort_index(
    ascending=False
)


recent_data.index = (

    recent_data.index.strftime(
        "%d %b %Y"
    )

)


st.dataframe(

    recent_data,

    use_container_width=True

)



# FOOTER


st.divider()

st.caption(
    "StockLens • Data-driven market analysis and forecasting"
)