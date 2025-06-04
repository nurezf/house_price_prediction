
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

warnings.filterwarnings('ignore')


np.random.seed(42)
n_samples = 1000
data = {
    'PlotSize_m2': np.random.randint(100, 500, n_samples),  # Plot size in square meters
    'FloorArea_m2': np.random.randint(80, 300, n_samples),  # Floor area in square meters
    'Bedrooms': np.random.randint(2, 8, n_samples),  # Number of bedrooms
    'Bathrooms': np.random.randint(1, 5, n_samples),  # Number of bathrooms
    'DistanceToRoad_km': np.random.uniform(0.1, 5, n_samples),  # Distance to main road
    'InAddisAbaba': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),  # 60% in Addis Ababa
    'Price_ETB': np.zeros(n_samples)  # Placeholder for prices
}

for i in range(n_samples):
    base_price = 1000000  # Minimum price ~1.2M ETB
    base_price += data['PlotSize_m2'][i] * 10000  # ~10,000 ETB per m²
    base_price += data['FloorArea_m2'][i] * 20000  # ~20,000 ETB per m²
    base_price += data['Bedrooms'][i] * 500000  # ~500,000 ETB per bedroom
    base_price += data['Bathrooms'][i] * 300000  # ~300,000 ETB per bathroom
    base_price -= data['DistanceToRoad_km'][i] * 200000  # Price decreases with distance
    base_price *= (1 + data['InAddisAbaba'][i] * 1.5)  # 150% higher in Addis Ababa
    data['Price_ETB'][i] = base_price + np.random.normal(0, 500000)  # Add noise

# Create DataFrame
data = pd.DataFrame(data)

# Save synthetic dataset for reproducibility
data.to_csv('ethiopian_houses.csv', index=False)

# Step 3: Data Preprocessing
# Define features (X) and target (y)
features = ['PlotSize_m2', 'FloorArea_m2', 'Bedrooms', 'Bathrooms', 'DistanceToRoad_km', 'InAddisAbaba']
X = data[features]
y = data['Price_ETB']

# Normalize/scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 5: Model Training
# Model 1: Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Model 2: Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


# Step 6: Model Evaluation
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)

    # Calculate regression metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"\n{model_name} Evaluation Metrics:")
    print(f"Root Mean Squared Error (RMSE): {rmse:,.2f} ETB")
    print(f"R² Score: {r2:.4f}")

    # Visualize Actual vs Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Price (ETB)')
    plt.ylabel('Predicted Price (ETB)')
    plt.title(f'Actual vs Predicted Prices ({model_name})')
    plt.show()

    # Feature Importance (for Random Forest)
    if model_name == "Random Forest":
        feature_importance = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        })
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        plt.figure(figsize=(8, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance)
        plt.title('Feature Importance (Random Forest)')
        plt.show()

    return rmse, r2


# Evaluate both models
lr_metrics = evaluate_model(lr_model, X_test, y_test, "Linear Regression")
rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")

# Step 7: Save the models and scaler
joblib.dump(rf_model, 'ethiopian_house_price_rf_model.pkl')
joblib.dump(scaler, 'ethiopian_scaler.pkl')


# Step 8: Interactive Prediction Function
def predict_house_price():
    print("\nEnter house features for price prediction (Ethiopian context):")
    print("Note: Use realistic values based on Ethiopian housing market.")
    print("PlotSize_m2: Plot size in square meters (e.g., 100-500)")
    print("FloorArea_m2: Floor area in square meters (e.g., 80-300)")
    print("Bedrooms: Number of bedrooms (e.g., 2-8)")
    print("Bathrooms: Number of bathrooms (e.g., 1-5)")
    print("DistanceToRoad_km: Distance to main road in kilometers (e.g., 0.1-5)")
    print("InAddisAbaba: Is the house in Addis Ababa? (1 for yes, 0 for no)")

    try:
        # Get user inputs with validation
        plot_size = float(input("PlotSize_m2 (100-500): "))
        if not 100 <= plot_size <= 500:
            raise ValueError("PlotSize_m2 must be between 100 and 500.")

        floor_area = float(input("FloorArea_m2 (80-300): "))
        if not 80 <= floor_area <= 300:
            raise ValueError("FloorArea_m2 must be between 80 and 300.")

        bedrooms = int(input("Bedrooms (2-8): "))
        if not 2 <= bedrooms <= 8:
            raise ValueError("Bedrooms must be between 2 and 8.")

        bathrooms = int(input("Bathrooms (1-5): "))
        if not 1 <= bathrooms <= 5:
            raise ValueError("Bathrooms must be between 1 and 5.")

        distance_to_road = float(input("DistanceToRoad_km (0.1-5): "))
        if not 0.1 <= distance_to_road <= 5:
            raise ValueError("DistanceToRoad_km must be between 0.1 and 5.")

        in_addis = int(input("InAddisAbaba (1 for yes, 0 for no): "))
        if in_addis not in [0, 1]:
            raise ValueError("InAddisAbaba must be 0 or 1.")

        # Create input array
        user_input = np.array([[plot_size, floor_area, bedrooms, bathrooms, distance_to_road, in_addis]])

        # Scale the input
        user_input_scaled = scaler.transform(user_input)

        # Predict using Random Forest model
        prediction = rf_model.predict(user_input_scaled)

        print(f"\nPredicted House Price: {prediction[0]:,.2f} ETB")

        # Ask if user wants to predict another
        again = input("\nWould you like to predict another house price? (yes/no): ").lower()
        if again == 'yes':
            predict_house_price()
        else:
            print("Exiting prediction mode.")

    except ValueError as e:
        print(f"Error: {e}. Please enter valid numerical values and try again.")
        predict_house_price()


# Step 9: Run the prediction function
if __name__ == "__main__":
    predict_house_price()