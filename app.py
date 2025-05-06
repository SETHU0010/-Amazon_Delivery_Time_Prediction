import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import math
from geopy.distance import geodesic

# Load the dataset
def load_data(file_path):
    """Loads the dataset from the given file path."""
    return pd.read_csv(file_path)

# Feature Engineering
def calculate_distance(df):
    """Calculates the distance between store and drop locations using the Haversine formula."""

    store_locs = list(zip(df['Store_Latitude'], df['Store_Longitude']))
    drop_locs = list(zip(df['Drop_Latitude'], df['Drop_Longitude']))
    distances = [geodesic(store, drop).km for store, drop in zip(store_locs, drop_locs)]
    df['Distance'] = distances
    return df

def extract_time_features(df):
    """Extracts hour of day and day of week from Order_Time and Order_Date."""

    df['Order_Date'] = pd.to_datetime(df['Order_Date'])

    def extract_hour(time_str):
        if isinstance(time_str, str):
            parts = time_str.split(':')
            if len(parts) > 0:
                try:
                    return int(parts[0])
                except ValueError:
                    return np.nan  # Return NaN if conversion fails
        return np.nan  # Return NaN for non-string values (including NaN)

    df['Hour_of_Day'] = df['Order_Time'].apply(extract_hour)
    df['Day_of_Week'] = df['Order_Date'].dt.day_name()
    df.drop(columns=['Order_Date'], inplace=True)  # Drop Order_Date after extraction
    return df

# Data Preprocessing
def preprocess_data(df, target_column='Delivery_Time'):
    """
    Preprocesses the data by handling missing values,
    encoding categorical features, and scaling numerical features.
    """

    # Drop unnecessary columns
    if 'Order_ID' in df.columns:
        df = df.drop(columns=['Order_ID'])

    # Separate features and target variable
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Define categorical and numerical features
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

    # Create transformers for preprocessing
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Corrected strategy here
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # Ensure dense output
    ])

    # Combine transformers using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough',  # or 'drop'
        verbose_feature_names_out=False # Prevents duplicate column names
    )

    # Create the pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    # Fit and transform the data
    X_processed = pipeline.fit_transform(X)

    # Convert the processed data back to a DataFrame (for easier handling)
    feature_names = preprocessor.get_feature_names_out(input_features=X.columns)
    X_processed_df = pd.DataFrame(X_processed, columns=feature_names)

    return X_processed_df, y

# Model Training
def train_model(X_train, y_train):
    """Trains a RandomForestRegressor model."""

    with mlflow.start_run():
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)

        # Log parameters (if any)
        # mlflow.log_param("n_estimators", model.n_estimators)

        # Log model
        mlflow.sklearn.log_model(model, "random_forest_model")

        return model

# Model Evaluation
def evaluate_model(model, X_test, y_test):
    """Evaluates the model and logs metrics to MLflow."""

    y_pred = model.predict(X_test)

    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Log metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    return rmse, mae, r2

# Streamlit App
def main():
    st.title("Amazon Delivery Time Prediction")

    # File Upload
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file is not None:
        try:
            df = load_data(uploaded_file)
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return

        # Display the first few rows of the dataframe
        st.subheader("Raw Data")
        st.dataframe(df.head())

        # Feature Engineering
        try:
            df = calculate_distance(df)
            df = extract_time_features(df)
        except Exception as e:
            st.error(f"Error in feature engineering: {e}")
            return

        # Preprocess the data
        try:
            X, y = preprocess_data(df)
        except Exception as e:
            st.error(f"Error in preprocessing: {e}")
            return

        # Split data
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        except Exception as e:
            st.error(f"Error splitting data: {e}")
            return

        # Train the model
        try:
            model = train_model(X_train, y_train)
        except Exception as e:
            st.error(f"Error training model: {e}")
            return

        # Evaluate the model
        st.subheader("Model Evaluation")
        try:
            rmse, mae, r2 = evaluate_model(model, X_test, y_test)
            st.write(f"RMSE: {rmse:.2f}")
            st.write(f"MAE: {mae:.2f}")
            st.write(f"R^2: {r2:.2f}")
        except Exception as e:
            st.error(f"Error evaluating model: {e}")
            return

        st.success("Data preprocessing, model training, and evaluation completed!")

if __name__ == '__main__':
    main()