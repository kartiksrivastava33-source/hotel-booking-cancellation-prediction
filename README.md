\# Hotel Booking Cancellation Prediction \& Decision System



A machine learning system that predicts the probability of a hotel booking being cancelled and converts the prediction into an actionable risk level and business recommendation.



\## Problem



Hotel booking cancellations can result in unused rooms, revenue loss, and inefficient inventory planning.



This project uses information available at booking time to predict whether a booking is likely to be cancelled. The predicted probability is then converted into a risk category that can support operational decisions.



\## Project Workflow



```text

Hotel Booking Data

&#x20;       ↓

Data Cleaning

&#x20;       ↓

Exploratory Data Analysis

&#x20;       ↓

Feature Engineering

&#x20;       ↓

Train/Test Split

&#x20;       ↓

Preprocessing

&#x20;       ↓

Model Training

&#x20;       ↓

Model Evaluation

&#x20;       ↓

Cancellation Probability

&#x20;       ↓

Risk Classification

&#x20;       ↓

Business Recommendation

&#x20;       ↓

REST API

&#x20;       ↓

Automated Tests

```

\## Dataset



The project uses the Hotel Booking dataset containing booking and customer-related information.



The target variable is:



is\_canceled — whether the booking was cancelled.



Data preprocessing includes handling missing values, removing duplicate records, and removing variables that would cause target leakage.



The columns reservation\_status and reservation\_status\_date were excluded because they contain information that becomes available after the booking outcome and therefore should not be used for prediction.



\## Feature Engineering



Additional features were created to provide more useful signals to the model:



total\_nights

total\_guests

total\_previous\_bookings

has\_previous\_booking

has\_special\_requests

is\_room\_changed

has\_agent

Machine Learning



Three classification models were evaluated:



Logistic Regression

Random Forest

XGBoost



The Random Forest model performed best on the test set.



Model Performance

Model	Accuracy	Precision	Recall	F1 Score	ROC-AUC

Logistic Regression	82.27%	81.06%	68.01%	73.97%	90.14%

Random Forest	89.02%	88.96%	80.34%	84.43%	95.67%

XGBoost	86.82%	84.75%	78.56%	81.54%	94.41%



Random Forest was selected as the final model based on its overall performance across the evaluated metrics.



\## Risk-Based Decision Layer



Instead of returning only a binary prediction, the system uses the model's cancellation probability to assign a risk level.



Probability	Risk Level	Recommendation

< 0.40	Low Risk	Normal booking handling

0.40–0.69	Medium Risk	Send confirmation reminder

≥ 0.70	High Risk	Require confirmation/deposit or proactive follow-up



This makes the model output more useful for operational decision-making.



\## API



The trained model is exposed through a REST API built with FastAPI.



Endpoints



GET /



Checks whether the API is running.



POST /predict



Accepts booking information and returns:



Cancellation probability

Risk level

Business recommendation



Interactive API documentation is available through Swagger UI at:



http://127.0.0.1:8000/docs

Testing



The API is tested using Pytest and FastAPI's test client.



The test suite covers:



API availability

Valid prediction requests

Probability range validation

Invalid input handling

Missing required fields



\## Current test result:



5 passed

Project Structure

hotel-booking-cancellation-prediction/

│

├── hotel\_booking\_analysis.ipynb

├── app.py

├── tests/

│   └── test\_api.py

├── .gitignore

└── README.md



The trained model file is intentionally excluded from Git because of GitHub's file-size limitations.



\## Technologies

Python

Pandas

NumPy

Scikit-learn

XGBoost

Matplotlib

Seaborn

FastAPI

Pytest

Joblib

Git \& GitHub

Future Improvements

Add model/version management

Containerize the API with Docker

Add CI/CD using GitHub Actions

Add API performance/load testing

Add model monitoring

Deploy the inference API to the cloud

\## Author



Kartik Srivastava

