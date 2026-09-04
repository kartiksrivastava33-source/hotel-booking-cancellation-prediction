from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd


app = FastAPI(
    title="Hotel Cancellation Prediction API",
    description="API for predicting hotel booking cancellation risk",
    version="1.0"
)


model = joblib.load("hotel_cancellation_model.pkl")


class BookingRequest(BaseModel):
    hotel: str
    arrival_date_month: str
    arrival_date_year: int
    arrival_date_week_number: int
    arrival_date_day_of_month: int

    stays_in_weekend_nights: int
    stays_in_week_nights: int

    adults: int
    children: float
    babies: int

    meal: str
    country: str
    market_segment: str
    distribution_channel: str

    reserved_room_type: str
    assigned_room_type: str
    deposit_type: str
    customer_type: str

    lead_time: int
    adr: float

    is_repeated_guest: int
    previous_cancellations: int
    previous_bookings_not_canceled: int
    booking_changes: int
    days_in_waiting_list: int

    required_car_parking_spaces: int
    total_of_special_requests: int
    has_agent: int


def cancellation_risk(probability):

    if probability >= 0.70:
        return "High Risk"

    elif probability >= 0.40:
        return "Medium Risk"

    else:
        return "Low Risk"


def booking_recommendation(probability):

    if probability >= 0.70:
        return "Require confirmation/deposit or proactive follow-up"

    elif probability >= 0.40:
        return "Send confirmation reminder"

    else:
        return "Normal booking handling"


@app.get("/")
def home():

    return {
        "message": "Hotel Cancellation Prediction API is running"
    }


@app.post("/predict")
def predict(booking: BookingRequest):

    booking_dict = booking.model_dump()

    booking_df = pd.DataFrame([booking_dict])

    # Feature engineering
    booking_df["total_nights"] = (
        booking_df["stays_in_weekend_nights"]
        + booking_df["stays_in_week_nights"]
    )

    booking_df["total_guests"] = (
        booking_df["adults"]
        + booking_df["children"]
        + booking_df["babies"]
    )

    booking_df["total_previous_bookings"] = (
        booking_df["previous_cancellations"]
        + booking_df["previous_bookings_not_canceled"]
    )

    booking_df["has_previous_booking"] = (
        booking_df["total_previous_bookings"] > 0
    ).astype(int)

    booking_df["has_special_requests"] = (
        booking_df["total_of_special_requests"] > 0
    ).astype(int)

    booking_df["is_room_changed"] = (
        booking_df["reserved_room_type"]
        != booking_df["assigned_room_type"]
    ).astype(int)

    probability = model.predict_proba(booking_df)[0, 1]

    return {
        "cancellation_probability": round(float(probability), 3),
        "risk_level": cancellation_risk(probability),
        "recommendation": booking_recommendation(probability)
    }