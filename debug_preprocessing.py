"""
Debug preprocessing differences between local and Azure ML
"""
import joblib
import pandas as pd
import json

# Load the deployed model files
encoders = joblib.load('ML/New folder/label_encoders_0.pkl')
scaler = joblib.load('ML/New folder/scaler_0.pkl')
metadata = joblib.load('ML/New folder/model_metadata_0.pkl')

print("=" * 80)
print("PREPROCESSING DEBUG - Sample 4 (should be FRAUD)")
print("=" * 80)

# Sample 4 data
sample_4 = {
    "DriverRating": 4,
    "Age": 41,
    "PoliceReportFiled": "No",
    "WeekOfMonthClaimed": 4,
    "PolicyType": "Utility - All Perils",
    "WeekOfMonth": 5,
    "AccidentArea": "Urban",
    "Sex": "Male",
    "Deductible": 400
}

print("\nOriginal Sample 4:")
print(json.dumps(sample_4, indent=2))

# Convert to DataFrame
df = pd.DataFrame([sample_4])

print("\n" + "=" * 80)
print("STEP 1: Categorical Encoding")
print("=" * 80)

categorical_cols = ['AccidentArea', 'Sex', 'PolicyType', 'PoliceReportFiled']
df_encoded = df.copy()

print("\nBefore encoding:")
print(df_encoded.dtypes)
print(df_encoded.values[0])

for col in categorical_cols:
    if col in encoders:
        encoder = encoders[col]
        print(f"\n{col}:")
        print(f"  Original value: {df[col].values[0]}")
        print(f"  Encoder classes: {list(encoder.classes_)}")
        
        # Encode
        df_encoded[col] = df_encoded[col].astype(str)
        known_classes = set(encoder.classes_)
        df_encoded[col] = df_encoded[col].apply(
            lambda x: x if x in known_classes else encoder.classes_[0]
        )
        df_encoded[col] = encoder.transform(df_encoded[col])
        print(f"  Encoded value: {df_encoded[col].values[0]}")

print("\nAfter encoding:")
print(df_encoded.dtypes)
print(df_encoded.values[0])

print("\n" + "=" * 80)
print("STEP 2: Feature Ordering")
print("=" * 80)

feature_order = metadata['features']
print(f"\nFeature order from metadata:")
for i, feat in enumerate(feature_order, 1):
    print(f"  {i}. {feat}")

df_ordered = df_encoded[feature_order]
print(f"\nOrdered feature values:")
print(df_ordered.values[0])

print("\n" + "=" * 80)
print("STEP 3: Scaling")
print("=" * 80)

print(f"\nScaler mean: {scaler.mean_}")
print(f"Scaler scale: {scaler.scale_}")

df_scaled = scaler.transform(df_ordered)
print(f"\nScaled values:")
print(df_scaled[0])

print("\n" + "=" * 80)
print("EXPECTED RESULTS")
print("=" * 80)
print("\nLocal prediction: 75.9% probability → FRAUD ✅")
print("Azure ML prediction: 60.2% probability → NOT FRAUD ❌")
print("\nIf Azure is using different encoding, the scaled values will be wrong!")

print("\n" + "=" * 80)
print("WHAT TO SEND TO AZURE ML")
print("=" * 80)

# Show what should be sent - with string values
send_data = {
    "DriverRating": 4,
    "Age": 41,
    "WeekOfMonthClaimed": 4,
    "WeekOfMonth": 5,
    "Deductible": 400,
    "AccidentArea": "Urban",
    "Sex": "Male",
    "PolicyType": "Utility - All Perils",
    "PoliceReportFiled": "No"
}

print("\nData to send (with categorical as strings):")
print(json.dumps(send_data, indent=2))

print("\n✅ scoring.py should encode these strings using the same encoders")
print("✅ Then order by feature_columns")
print("✅ Then scale using the same scaler")
print("✅ Result should be 75.9% probability, not 60.2%")

print("\n" + "=" * 80)
