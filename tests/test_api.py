from fastapi.testclient import TestClient
import app


client = TestClient(app.app)


valid_booking = {
    "hotel": "Resort Hotel",
    "arrival_date_month": "July",
    "arrival_date_year": 2017,
    "arrival_date_week_number": 27,
    "arrival_date_day_of_month": 5,
    "stays_in_weekend_nights": 2,
    "stays_in_week_nights": 5,
    "adults": 2,
    "children": 0,
    "babies": 0,
    "meal": "BB",
    "country": "PRT",
    "market_segment": "Online TA",
    "distribution_channel": "TA/TO",
    "reserved_room_type": "A",
    "assigned_room_type": "A",
    "deposit_type": "No Deposit",
    "customer_type": "Transient",
    "lead_time": 180,
    "adr": 120.0,
    "is_repeated_guest": 0,
    "previous_cancellations": 0,
    "previous_bookings_not_canceled": 0,
    "booking_changes": 0,
    "days_in_waiting_list": 0,
    "required_car_parking_spaces": 0,
    "total_of_special_requests": 0,
    "has_agent": 1
}


import numpy as np


class FakeModel:
    def predict_proba(self, data):
        return np.array([[0.2, 0.8]])


app.model = FakeModel()


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert "message" in response.json()


def test_valid_prediction():
    response = client.post("/predict", json=valid_booking)

    assert response.status_code == 200

    result = response.json()

    assert "cancellation_probability" in result
    assert "risk_level" in result
    assert "recommendation" in result


def test_probability_range():
    response = client.post("/predict", json=valid_booking)

    probability = response.json()["cancellation_probability"]

    assert 0 <= probability <= 1


def test_invalid_input():
    invalid_booking = valid_booking.copy()
    invalid_booking["lead_time"] = "hello"

    response = client.post("/predict", json=invalid_booking)

    assert response.status_code == 422


def test_missing_field():
    incomplete_booking = valid_booking.copy()
    del incomplete_booking["hotel"]

    response = client.post("/predict", json=incomplete_booking)

    assert response.status_code == 422