import os
import pickle
import warnings
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import streamlit as st
import sklearn
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
	OneHotEncoder,
	OrdinalEncoder,
	StandardScaler,
	MinMaxScaler,
	PowerTransformer,
	FunctionTransformer
)

from feature_engine.outliers import Winsorizer
from feature_engine.datetime import DatetimeFeatures
from feature_engine.selection import SelectBySingleFeaturePerformance
from feature_engine.encoding import (
	RareLabelEncoder,
	MeanEncoder,
	CountFrequencyEncoder
)

sklearn.set_config(transform_output="pandas")

# Custom CSS for modern UI
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Poppins', sans-serif;
    }
    
    /* Main container */
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Header styles */
    .header-container {
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(135deg, #FF6B6B, #FF8E53);
        border-radius: 15px;
        color: white;
        box-shadow: 0 10px 30px rgba(255, 107, 107, 0.3);
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .subtitle {
        font-size: 1.2rem;
        margin-top: 0.5rem;
        opacity: 0.9;
        font-weight: 300;
    }
    
    /* Input section styles */
    .input-section {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 10px 30px rgba(240, 147, 251, 0.3);
    }
    
    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        text-align: center;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
    }
    
    /* Custom input styling */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .stDateInput > div > div {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .stTimeInput > div > div {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .stNumberInput > div > div {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Predict button */
    .predict-button {
        display: flex;
        justify-content: center;
        margin: 2rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        border: none;
        padding: 1rem 3rem;
        border-radius: 50px;
        font-size: 1.2rem;
        font-weight: 600;
        box-shadow: 0 10px 30px rgba(79, 172, 254, 0.4);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(79, 172, 254, 0.6);
    }
    
    /* Result styling */
    .result-container {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        margin-top: 2rem;
        box-shadow: 0 10px 30px rgba(168, 237, 234, 0.3);
    }
    
    .price-display {
        font-size: 3rem;
        font-weight: 700;
        color: #2c3e50;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .currency {
        font-size: 2rem;
        color: #7f8c8d;
    }
    
    /* Animation for result */
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animated-result {
        animation: slideIn 0.6s ease-out;
    }
    
    /* Grid layout for inputs */
    .input-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
    
    /* Feature highlights */
    .feature-box {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #fff;
    }
    
    /* Mobile responsive */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2rem;
        }
        .input-grid {
            grid-template-columns: 1fr;
        }
        .price-display {
            font-size: 2rem;
        }
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Page configuration
st.set_page_config(
    page_title="Flight Price Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# All your existing preprocessing code remains the same
air_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("grouper", RareLabelEncoder(tol=0.1, replace_with="Other", n_categories=2)),
    ("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
])

feature_to_extract = ["month", "week", "day_of_week", "day_of_year"]

doj_transformer = Pipeline(steps=[
    ("dt", DatetimeFeatures(features_to_extract=feature_to_extract, yearfirst=True, format="mixed")),
    ("scaler", MinMaxScaler())
])

location_pipe1 = Pipeline(steps=[
    ("grouper", RareLabelEncoder(tol=0.1, replace_with="Other", n_categories=2)),
    ("encoder", MeanEncoder()),
    ("scaler", PowerTransformer())
])

def is_north(X):
    columns = X.columns.to_list()
    north_cities = ["Delhi", "Kolkata", "Mumbai", "New Delhi"]
    return (
        X
        .assign(**{
            f"{col}_is_north": X.loc[:, col].isin(north_cities).astype(int)
            for col in columns
        })
        .drop(columns=columns)
    )

location_transformer = FeatureUnion(transformer_list=[
    ("part1", location_pipe1),
    ("part2", FunctionTransformer(func=is_north))
])

time_pipe1 = Pipeline(steps=[
    ("dt", DatetimeFeatures(features_to_extract=["hour", "minute"])),
    ("scaler", MinMaxScaler())
])

def part_of_day(X, morning=4, noon=12, eve=16, night=20):
    columns = X.columns.to_list()
    X_temp = X.assign(**{
        col: pd.to_datetime(X.loc[:, col]).dt.hour
        for col in columns
    })

    return (
        X_temp
        .assign(**{
            f"{col}_part_of_day": np.select(
                [X_temp.loc[:, col].between(morning, noon, inclusive="left"),
                 X_temp.loc[:, col].between(noon, eve, inclusive="left"),
                 X_temp.loc[:, col].between(eve, night, inclusive="left")],
                ["morning", "afternoon", "evening"],
                default="night"
            )
            for col in columns
        })
        .drop(columns=columns)
    )

time_pipe2 = Pipeline(steps=[
    ("part", FunctionTransformer(func=part_of_day)),
    ("encoder", CountFrequencyEncoder()),
    ("scaler", MinMaxScaler())
])

time_transformer = FeatureUnion(transformer_list=[
    ("part1", time_pipe1),
    ("part2", time_pipe2)
])

class RBFPercentileSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None, percentiles=[0.25, 0.5, 0.75], gamma=0.1):
        self.variables = variables
        self.percentiles = percentiles
        self.gamma = gamma

    def fit(self, X, y=None):
        if not self.variables:
            self.variables = X.select_dtypes(include="number").columns.to_list()

        self.reference_values_ = {
            col: (
                X
                .loc[:, col]
                .quantile(self.percentiles)
                .values
                .reshape(-1, 1)
            )
            for col in self.variables
        }

        return self

    def transform(self, X):
        objects = []
        for col in self.variables:
            columns = [f"{col}_rbf_{int(percentile * 100)}" for percentile in self.percentiles]
            obj = pd.DataFrame(
                data=rbf_kernel(X.loc[:, [col]], Y=self.reference_values_[col], gamma=self.gamma),
                columns=columns
            )
            objects.append(obj)
        return pd.concat(objects, axis=1)

def duration_category(X, short=180, med=400):
    return (
        X
        .assign(duration_cat=np.select([X.duration.lt(short),
                                        X.duration.between(short, med, inclusive="left")],
                                       ["short", "medium"],
                                       default="long"))
        .drop(columns="duration")
    )

def is_over(X, value=1000):
    return (
        X
        .assign(**{
            f"duration_over_{value}": X.duration.ge(value).astype(int)
        })
        .drop(columns="duration")
    )

duration_pipe1 = Pipeline(steps=[
    ("rbf", RBFPercentileSimilarity()),
    ("scaler", PowerTransformer())
])

duration_pipe2 = Pipeline(steps=[
    ("cat", FunctionTransformer(func=duration_category)),
    ("encoder", OrdinalEncoder(categories=[["short", "medium", "long"]]))
])

duration_union = FeatureUnion(transformer_list=[
    ("part1", duration_pipe1),
    ("part2", duration_pipe2),
    ("part3", FunctionTransformer(func=is_over)),
    ("part4", StandardScaler())
])

duration_transformer = Pipeline(steps=[
    ("outliers", Winsorizer(capping_method="iqr", fold=1.5)),
    ("imputer", SimpleImputer(strategy="median")),
    ("union", duration_union)
])

def is_direct(X):
    return X.assign(is_direct_flight=X.total_stops.eq(0).astype(int))

total_stops_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("", FunctionTransformer(func=is_direct))
])

info_pipe1 = Pipeline(steps=[
    ("group", RareLabelEncoder(tol=0.1, n_categories=2, replace_with="Other")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

def have_info(X):
    return X.assign(additional_info=X.additional_info.ne("No Info").astype(int))

info_union = FeatureUnion(transformer_list=[
    ("part1", info_pipe1),
    ("part2", FunctionTransformer(func=have_info))
])

info_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
    ("union", info_union)
])

column_transformer = ColumnTransformer(transformers=[
    ("air", air_transformer, ["airline"]),
    ("doj", doj_transformer, ["date_of_journey"]),
    ("location", location_transformer, ["source", 'destination']),
    ("time", time_transformer, ["dep_time", "arrival_time"]),
    ("dur", duration_transformer, ["duration"]),
    ("stops", total_stops_transformer, ["total_stops"]),
    ("info", info_transformer, ["additional_info"])
], remainder="passthrough")

estimator = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)

selector = SelectBySingleFeaturePerformance(
    estimator=estimator,
    scoring="r2",
    threshold=0.1
) 

preprocessor = Pipeline(steps=[
    ("ct", column_transformer),
    ("selector", selector)
])

# Load training data (cached for performance)
@st.cache_data
def load_training_data():
    train = pd.read_csv("train.csv")
    X_train = train.drop(columns="price")
    y_train = train.price.copy()
    return X_train, y_train

# Load and fit preprocessor (cached for performance)
@st.cache_resource
def load_preprocessor():
    X_train, y_train = load_training_data()
    preprocessor.fit(X_train, y_train)
    return preprocessor, X_train

# Load the preprocessor and training data
fitted_preprocessor, X_train = load_preprocessor()

# Main UI starts here
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header-container">
    <h1 class="main-title">✈️ FlightFare AI</h1>
    <p class="subtitle">Intelligent Flight Price Prediction with Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# Create columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="input-section">
        <h3 class="section-title">✈️ Flight Details</h3>
    </div>
    """, unsafe_allow_html=True)
    
    airline = st.selectbox(
        "🏢 Select Airline",
        options=X_train.airline.unique(),
        help="Choose your preferred airline"
    )
    
    doj = st.date_input(
        "📅 Date of Journey",
        help="Select your travel date"
    )
    
    col1_1, col1_2 = st.columns(2)
    with col1_1:
        source = st.selectbox(
            "🛫 From",
            options=X_train.source.unique(),
            help="Select departure city"
        )
    
    with col1_2:
        destination = st.selectbox(
            "🛬 To",
            options=X_train.destination.unique(),
            help="Select destination city"
        )

with col2:
    st.markdown("""
    <div class="input-section">
        <h3 class="section-title">⏰ Time & Duration</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col2_1, col2_2 = st.columns(2)
    with col2_1:
        dep_time = st.time_input(
            "🕐 Departure Time",
            help="Select departure time"
        )
    
    with col2_2:
        arrival_time = st.time_input(
            "🕕 Arrival Time",
            help="Select arrival time"
        )
    
    duration = st.number_input(
        "⏱️ Duration (minutes)",
        step=1,
        min_value=30,
        max_value=2000,
        value=120,
        help="Flight duration in minutes"
    )
    
    col2_3, col2_4 = st.columns(2)
    with col2_3:
        total_stops = st.number_input(
            "🔄 Total Stops",
            step=1,
            min_value=0,
            max_value=4,
            value=0,
            help="Number of stops during the flight"
        )
    
    with col2_4:
        additional_info = st.selectbox(
            "ℹ️ Additional Info",
            options=X_train.additional_info.unique(),
            help="Any additional flight information"
        )

# Create the input dataframe
x_new = pd.DataFrame(dict(
    airline=[airline],
    date_of_journey=[doj],
    source=[source],
    destination=[destination],
    dep_time=[dep_time],
    arrival_time=[arrival_time],
    duration=[duration],
    total_stops=[total_stops],
    additional_info=[additional_info]
)).astype({
    col: "str"
    for col in ["date_of_journey", "dep_time", "arrival_time"]
})

# Predict button with custom styling
st.markdown('<div class="predict-button">', unsafe_allow_html=True)
predict_clicked = st.button("🚀 Predict Flight Price", use_container_width=False)
st.markdown('</div>', unsafe_allow_html=True)

# Prediction logic
if predict_clicked:
    with st.spinner('🔮 Analyzing flight data and predicting price...'):
        try:
            # Transform the input
            x_new_pre = fitted_preprocessor.transform(x_new)
            
            # Load model and predict
            with open("xgboost-model", "rb") as f:
                model = pickle.load(f)
            x_new_xgb = xgb.DMatrix(x_new_pre)
            pred = model.predict(x_new_xgb)[0]
            
            # Display result with animation
            st.markdown(f"""
            <div class="result-container animated-result">
                <h2 style="color: #2c3e50; margin-bottom: 1rem;">🎯 Predicted Flight Price</h2>
                <div class="price-display">
                    ₹ {pred:,.0f} <span class="currency">INR</span>
                </div>
                <p style="color: #7f8c8d; font-size: 1.1rem; margin-top: 1rem;">
                    💡 This prediction is based on historical flight data and current market trends
                </p>
                <div style="margin-top: 2rem; padding: 1rem; background: rgba(52, 152, 219, 0.1); border-radius: 10px;">
                    <h4 style="color: #3498db; margin-bottom: 0.5rem;">📊 Flight Summary</h4>
                    <p style="margin: 0.25rem 0;"><strong>Route:</strong> {source} → {destination}</p>
                    <p style="margin: 0.25rem 0;"><strong>Airline:</strong> {airline}</p>
                    <p style="margin: 0.25rem 0;"><strong>Duration:</strong> {duration} minutes</p>
                    <p style="margin: 0.25rem 0;"><strong>Stops:</strong> {total_stops}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ An error occurred during prediction: {str(e)}")

# Footer
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #7f8c8d; border-top: 1px solid #ecf0f1; margin-top: 3rem;">
    <p>🤖 Powered by Machine Learning & AWS SageMaker | Built with ❤️ using Streamlit</p>
    <p style="font-size: 0.9rem; opacity: 0.7;">Disclaimer: Predictions are estimates based on historical data and may not reflect actual prices.</p>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
