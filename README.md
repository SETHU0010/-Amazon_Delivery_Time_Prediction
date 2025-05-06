# Amazon Delivery Time Prediction

## Description

This project aims to predict delivery times for e-commerce orders based on a variety of factors such as product size, distance, traffic conditions, and shipping method. [cite: 2, 3] By preprocessing the data, building a regression model, and deploying it with a user-friendly Streamlit interface, this application allows users to input relevant details and receive estimated delivery times. [cite: 3, 4]

## Skills Takeaway

* Python scripting
* Data cleaning
* Exploratory Data Analysis (EDA)
* Machine Learning
* Regression modeling
* MLflow
* Streamlit

## Domain

E-Commerce and Logistics [cite: 1]

## Problem Statement

This project aims to predict delivery times for e-commerce orders based on a variety of factors such as product size, distance, traffic conditions, and shipping method. [cite: 2] Using the provided dataset, learners will preprocess, analyze, and build regression models to accurately estimate delivery times. [cite: 3] The final application will allow users to input relevant details and receive estimated delivery times via a user-friendly interface. [cite: 4]

## Aim

The aim of this project is to develop a system that can accurately estimate delivery times for e-commerce orders. [cite: 3, 4] This involves preprocessing data, building and training regression models, and creating a user-friendly interface for inputting order details and displaying predictions. [cite: 9, 15, 16, 17]

## Scope

The scope of this project includes:

* Loading and preprocessing the provided dataset. [cite: 9]
* Performing feature engineering, such as calculating geospatial distances and extracting time-based features. [cite: 10, 13, 14]
* Training and evaluating regression models (e.g., Random Forest Regressor). [cite: 15]
* Developing a Streamlit application for user interaction. [cite: 16, 17]
* Tracking model performance and parameters using MLflow. [cite: 16]

## Advantages

* **Enhanced Delivery Logistics:** Predict delivery times to improve customer satisfaction and optimize delivery schedules. [cite: 5]
* **Dynamic Traffic and Weather Adjustments:** Adjust delivery estimates based on current traffic and weather conditions. [cite: 6]
* **Agent Performance Evaluation:** Evaluate agent efficiency and identify areas for training or improvement. [cite: 7]
* **Operational Efficiency:** Optimize resource allocation for deliveries by analyzing trends and performance metrics. [cite: 8]

## Disadvantages

*(This section requires you to think critically about potential limitations. The provided document doesn't explicitly list disadvantages, so we'll add some common ones in delivery time prediction)*

* **Data Dependency:** The accuracy of the predictions heavily relies on the quality and completeness of the input data.
* **Real-time Variability:** Unpredictable events like sudden traffic jams, accidents, or extreme weather conditions can affect actual delivery times and may not be fully captured by the model.
* **Model Complexity:** Complex models might be harder to interpret and maintain.
* **Overfitting:** There's a risk of the model overfitting the training data and not generalizing well to new, unseen data.
* **Feature Limitations:** The model's predictions are limited to the features included in the dataset. If important factors are missing, the predictions might be less accurate.

## Installation

1.  Clone the repository:

    ```bash
    git clone <your_repository_url>
    ```

2.  Navigate to the project directory:

    ```bash
    cd amazon-delivery-prediction
    ```

3.  Install the required libraries:

    ```bash
    pip install -r requirements.txt
    ```

    *(You'll need to create a `requirements.txt` file listing the dependencies like pandas, streamlit, scikit-learn, geopy, mlflow, numpy)*

## Usage

1.  Run the Streamlit application:

    ```bash
    streamlit run app.py
    ```

2.  Open your web browser to the address displayed in the terminal (usually `http://localhost:8501`).

3.  Upload a CSV file containing the delivery order data.

4.  The application will preprocess the data, train a model, and display the evaluation metrics.

## Code Structure

* `app.py`: Contains the main Streamlit application code, including data loading, preprocessing, model training, and evaluation.
* `requirements.txt`: (Create this file) Lists the Python packages required to run the project.

## MLflow Tracking

The project uses MLflow to track the model training process, log parameters (if any), and save the trained model. You can view the MLflow UI by running `mlflow ui` in your terminal.
