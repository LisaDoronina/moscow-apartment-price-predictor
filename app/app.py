import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os, sys
from pathlib import Path

def load_model():
    try:
        model_path = Path(__file__).parent.parent / 'models' / 'moscow_forest_model.pkl'
        pipeline_path = Path(__file__).parent.parent / 'models' / 'moscow_full_pipeline.pkl'
        model = joblib.load(model_path)
        pipeline = joblib.load(pipeline_path)
        return model, pipeline
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def prepare_user_input(input_df, full_features):
    area = input_df['Area'].iloc[0]
    floor = input_df['Floor'].iloc[0]
    floors_total = input_df['Number of floors'].iloc[0]
    rooms = input_df['Number of rooms'].iloc[0]
    is_luxury = input_df['is_luxury'].iloc[0]
    minutes_to_metro = input_df['Minutes to metro'].iloc[0]

    living_area = area * 0.7
    kitchen_area = area * 0.15
    floor_ratio = floor / floors_total if floors_total > 0 else 0
    is_top_floor = 1 if floor == floors_total else 0
    is_near_metro = 1 if minutes_to_metro <= 10 else 0

    feature_dict = {
        'Minutes to metro': minutes_to_metro,
        'Number of rooms': rooms,
        'Area': area,
        'Living area': living_area,
        'Kitchen area': kitchen_area,
        'Floor': floor,
        'Number of floors': floors_total,
        'price_per_m2': 0,
        'is_luxury': int(is_luxury),
        'floor_ratio': floor_ratio,
        'is_top_floor': is_top_floor,
        'is_near_metro': is_near_metro,
        'Apartment type_Secondary': 1
    }

    user_df = pd.DataFrame([feature_dict])

    available_features = [col for col in full_features if col in user_df.columns]
    user_df = user_df[available_features]

    return user_df


def prepare_user_input(input_dict, full_features):
    # Extract values directly from the dictionary (no .iloc needed)
    area = input_dict['Area']
    floor = input_dict['Floor']
    floors_total = input_dict['Number of floors']
    rooms = input_dict['Number of rooms']
    is_luxury = input_dict['is_luxury']
    minutes_to_metro = input_dict['Minutes to metro']

    living_area = area * 0.7
    kitchen_area = area * 0.15
    floor_ratio = floor / floors_total if floors_total > 0 else 0
    is_top_floor = 1 if floor == floors_total else 0
    is_near_metro = 1 if minutes_to_metro <= 10 else 0

    feature_dict = {
        'Minutes to metro': minutes_to_metro,
        'Number of rooms': rooms,
        'Area': area,
        'Living area': living_area,
        'Kitchen area': kitchen_area,
        'Floor': floor,
        'Number of floors': floors_total,
        'price_per_m2': 0,
        'is_luxury': int(is_luxury),
        'floor_ratio': floor_ratio,
        'is_top_floor': is_top_floor,
        'is_near_metro': is_near_metro,
        'Apartment type_Secondary': 1
    }

    user_df = pd.DataFrame([feature_dict])

    available_features = [col for col in full_features if col in user_df.columns]
    user_df = user_df[available_features]

    return user_df

def user_input_form():
    """Create and return user input form"""
    col1, col2 = st.columns(2)

    with col1:
        area = st.number_input(
            "Area (square meters)",
            min_value=10.0,
            max_value=500.0,
            value=50.0,
            key="area_input"
        )
        floor = st.number_input(
            "Floor",
            min_value=1,
            max_value=100,
            value=5,
            key="floor_input"
        )
        floors_total = st.number_input(
            "Total floors in building",
            min_value=1,
            max_value=100,
            value=15,
            key="floors_total_input"
        )

    with col2:
        rooms = st.number_input(
            "Number of rooms",
            min_value=1,
            max_value=10,
            value=2,
            key="rooms_input"
        )
        time_to_metro = st.number_input(
            "Minutes to metro",
            min_value=1,
            max_value=180,
            value=10,
            key="metro_input"
        )
        is_luxury = st.checkbox(
            "Luxury property?",
            key="luxury_input"
        )

    return {
        'Area': area,
        'Floor': floor,
        'Number of floors': floors_total,
        'Number of rooms': rooms,
        'is_luxury': is_luxury,
        'Minutes to metro': time_to_metro
    }

full_features = [
    'Minutes to metro', 'Number of rooms', 'Area', 'Living area', 'Kitchen area',
    'Floor', 'Number of floors', 'price_per_m2', 'is_luxury', 'floor_ratio',
    'is_top_floor', 'is_near_metro', 'Apartment type_Secondary'
]

def main():
    st.set_page_config(
        page_title="Moscow Apartment Price Predictor",
        page_icon="🏠",
        layout="wide"
    )

    st.title("Moscow Apartment Price Predictor")

    user_inputs = user_input_form()

    if st.button("Predict Price"):
        with st.spinner("Calculating..."):
            model, pipeline = load_model()

            if model is not None and pipeline is not None:
                prepared_data = prepare_user_input(user_inputs, full_features)

                try:
                    prepared = pipeline.transform(prepared_data)
                    pred_log = model.predict(prepared)
                    prediction = np.expm1(pred_log)[0]

                    st.success(f"### Estimated Price: {prediction:,.0f} RUB")

                    price_per_m2 = prediction / user_inputs['Area']
                    st.info(f"Price per m²: {price_per_m2:,.0f} RUB")

                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")


if __name__ == "__main__":
    main()