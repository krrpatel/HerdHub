import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# Generate Larger Dataset for Better Training
# -----------------------------
def generate_extended_data(base_data, n_samples=500):
    """Generate extended dataset based on the original small dataset"""
    np.random.seed(42)
    
    # Get unique values from original data
    animal_types = base_data['animal_type'].unique().tolist()
    breeds = base_data['breed'].unique().tolist()
    
    # Add more realistic breeds
    cow_breeds = ['Gir', 'Sahiwal', 'Red Sindhi', 'Holstein', 'Jersey', 'Tharparkar']
    buffalo_breeds = ['Murrah', 'Jaffarabadi', 'Mehsana', 'Surti']
    all_breeds = cow_breeds + buffalo_breeds
    
    extended_data = []
    
    for _ in range(n_samples):
        animal_type = np.random.choice(['Cow', 'Buffalo'])
        if animal_type == 'Cow':
            breed = np.random.choice(cow_breeds)
            base_yield = np.random.normal(8, 2)  # Base yield for cows
        else:
            breed = np.random.choice(buffalo_breeds)
            base_yield = np.random.normal(6, 1.5)  # Base yield for buffalo
        
        age = np.random.uniform(2, 10)
        lactation_num = np.random.randint(1, 6)
        days_in_milk = np.random.uniform(50, 350)
        feed_given = np.random.uniform(10, 25)
        water_intake = np.random.uniform(40, 90)
        body_condition = np.random.randint(1, 6)
        temperature = np.random.uniform(20, 35)
        humidity = np.random.uniform(40, 80)
        
        # Calculate realistic milk yield based on factors
        yield_modifier = 1.0
        yield_modifier *= (1 + 0.1 * (feed_given - 15) / 10)  # Feed effect
        yield_modifier *= (1 + 0.05 * (water_intake - 60) / 25)  # Water effect
        yield_modifier *= (1 + 0.15 * (body_condition - 3) / 2)  # Body condition
        yield_modifier *= (1 - 0.002 * days_in_milk)  # Lactation decline
        yield_modifier *= (1 + 0.03 * lactation_num if lactation_num <= 3 else 1)  # Peak lactation
        
        # Environmental stress
        if temperature > 30 or temperature < 22:
            yield_modifier *= 0.95
        if humidity > 70 or humidity < 45:
            yield_modifier *= 0.98
        
        milk_yield = max(1, base_yield * yield_modifier + np.random.normal(0, 0.5))
        
        extended_data.append({
            'animal_type': animal_type,
            'breed': breed,
            'age_years': age,
            'lactation_number': lactation_num,
            'days_in_milk': days_in_milk,
            'feed_given_kg': feed_given,
            'water_intake_liters': water_intake,
            'body_condition_score': body_condition,
            'temperature_c': temperature,
            'humidity_percent': humidity,
            'milk_yield_liters': milk_yield
        })
    
    return pd.DataFrame(extended_data)

# Original small dataset
original_data = pd.DataFrame({
    'animal_type': ['Cow','Cow','Buffalo','Cow','Buffalo'],
    'breed': ['Gir','Sahiwal','Murrah','Red Sindhi','Jaffarabadi'],
    'age_years': [4, 5, 6, 7, 3],
    'lactation_number': [1, 2, 3, 4, 1],
    'days_in_milk': [120, 200, 150, 300, 90],
    'feed_given_kg': [15, 20, 18, 12, 16],
    'water_intake_liters': [60, 80, 70, 50, 55],
    'body_condition_score': [3,4,3,3,2],
    'temperature_c': [25,28,30,27,26],
    'humidity_percent': [60,70,65,55,50],
    'milk_yield_liters': [6,8,7,5,4]
})

# Generate extended dataset
print("🔄 Generating extended training dataset...")
extended_data = generate_extended_data(original_data, n_samples=800)

# Combine original and extended data
data = pd.concat([original_data, extended_data], ignore_index=True)
print(f"✅ Total dataset size: {len(data)} samples")

# -----------------------------
# Data Preprocessing
# -----------------------------
print("🔧 Preprocessing data...")

# Encode categorical features
le_animal = LabelEncoder()
le_breed = LabelEncoder()
data['animal_type_enc'] = le_animal.fit_transform(data['animal_type'])
data['breed_enc'] = le_breed.fit_transform(data['breed'])

# Prepare features and target
X = data[['animal_type_enc','breed_enc','age_years','lactation_number',
          'days_in_milk','feed_given_kg','water_intake_liters',
          'body_condition_score','temperature_c','humidity_percent']]
y = data['milk_yield_liters']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features for algorithms that need it
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Train Multiple Models
# -----------------------------
print("🤖 Training multiple models...")

models = {
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42),
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'SVR': SVR(kernel='rbf', C=100, gamma=0.1)
}

model_results = {}
trained_models = {}

for name, model in models.items():
    print(f"Training {name}...")
    
    # Use scaled data for SVR and Ridge, original for tree-based models
    if name in ['SVR', 'Ridge Regression', 'Linear Regression']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    model_results[name] = {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'CV_R2_Mean': cv_mean,
        'CV_R2_Std': cv_std
    }
    
    trained_models[name] = model

# Display model comparison
print("\n" + "="*70)
print("📊 MODEL PERFORMANCE COMPARISON")
print("="*70)
print(f"{'Model':<20} {'RMSE':<8} {'MAE':<8} {'R2':<8} {'CV R2':<15}")
print("-"*70)

for name, metrics in model_results.items():
    print(f"{name:<20} {metrics['RMSE']:<8.3f} {metrics['MAE']:<8.3f} {metrics['R2']:<8.3f} {metrics['CV_R2_Mean']:<6.3f}±{metrics['CV_R2_Std']:<6.3f}")

# Find best model
best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['R2'])
best_model = trained_models[best_model_name]

print(f"\n🏆 Best performing model: {best_model_name}")
print(f"   R² Score: {model_results[best_model_name]['R2']:.3f}")
print(f"   RMSE: {model_results[best_model_name]['RMSE']:.3f} liters")

# Save models and encoders
print("\n💾 Saving models and encoders...")
joblib.dump(trained_models, 'milk_yield_models.pkl')
joblib.dump(le_animal, 'animal_encoder.pkl')
joblib.dump(le_breed, 'breed_encoder.pkl')
joblib.dump(scaler, 'feature_scaler.pkl')
print("✅ Models saved successfully!")

# -----------------------------
# Feature Importance (for tree-based models)
# -----------------------------
if best_model_name in ['Gradient Boosting', 'Random Forest']:
    print(f"\n🔍 Feature Importance ({best_model_name}):")
    feature_names = ['Animal Type', 'Breed', 'Age', 'Lactation#', 'Days in Milk', 
                    'Feed (kg)', 'Water (L)', 'Body Condition', 'Temperature', 'Humidity']
    importances = best_model.feature_importances_
    for feature, importance in zip(feature_names, importances):
        print(f"   {feature:<15}: {importance:.3f}")

# -----------------------------
# Interactive Prediction Function
# -----------------------------
def predict_with_all_models(animal_type, breed, age_years, lactation_number, days_in_milk,
                           feed_given_kg, water_intake_liters, body_condition_score,
                           temperature_c, humidity_percent):
    """Predict milk yield using all trained models"""
    
    # Encode categorical inputs
    try:
        animal_enc = le_animal.transform([animal_type])[0] if animal_type in le_animal.classes_ else 0
    except:
        animal_enc = 0
        print(f"⚠️ Warning: Unknown animal type '{animal_type}', using default encoding")
    
    try:
        breed_enc = le_breed.transform([breed])[0] if breed in le_breed.classes_ else 0
    except:
        breed_enc = 0
        print(f"⚠️ Warning: Unknown breed '{breed}', using default encoding")
    
    # Create input arrays
    input_features = np.array([animal_enc, breed_enc, age_years, lactation_number,
                              days_in_milk, feed_given_kg, water_intake_liters,
                              body_condition_score, temperature_c, humidity_percent]).reshape(1, -1)
    
    input_features_scaled = scaler.transform(input_features)
    
    print(f"\n🔮 Predictions for {animal_type} ({breed}):")
    print("-" * 50)
    
    predictions = {}
    for name, model in trained_models.items():
        if name in ['SVR', 'Ridge Regression', 'Linear Regression']:
            pred = model.predict(input_features_scaled)[0]
        else:
            pred = model.predict(input_features)[0]
        
        predictions[name] = max(0, pred)  # Ensure non-negative prediction
        print(f"{name:<20}: {pred:.2f} liters")
    
    # Average prediction
    avg_prediction = np.mean(list(predictions.values()))
    print(f"\n📊 Average Prediction: {avg_prediction:.2f} liters")
    print(f"🏆 Best Model ({best_model_name}): {predictions[best_model_name]:.2f} liters")
    
    return predictions

# -----------------------------
# Farmer Input Section
# -----------------------------
print("\n" + "="*70)
print("🌾 MILK YIELD PREDICTION SYSTEM")
print("="*70)

while True:
    try:
        print("\nEnter details for your animal to predict daily milk yield:")
        animal_type = input("Animal type (Cow/Buffalo): ").strip()
        breed = input("Breed: ").strip()
        age_years = float(input("Age in years: "))
        lactation_number = int(input("Lactation number: "))
        days_in_milk = float(input("Days in milk: "))
        feed_given_kg = float(input("Feed given per day (kg): "))
        water_intake_liters = float(input("Water intake per day (liters): "))
        body_condition_score = int(input("Body condition score (1-5): "))
        temperature_c = float(input("Temperature (°C): "))
        humidity_percent = float(input("Humidity (%): "))
        
        # Get predictions from all models
        predictions = predict_with_all_models(
            animal_type, breed, age_years, lactation_number, days_in_milk,
            feed_given_kg, water_intake_liters, body_condition_score,
            temperature_c, humidity_percent
        )
        
        # Ask if user wants to continue
        continue_pred = input("\n🔄 Do you want to make another prediction? (y/n): ").strip().lower()
        if continue_pred not in ['y', 'yes']:
            break
            
    except ValueError as e:
        print(f"❌ Error: Please enter valid numeric values. {e}")
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        break
    except Exception as e:
        print(f"❌ An error occurred: {e}")

print("\n✨ Thank you for using the Milk Yield Prediction System!")
print("📁 Models have been saved and can be reloaded for future use.")
