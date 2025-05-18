# Simple linear regression model
import fastf1
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
import os

def main():
    os.makedirs("f1_cache", exist_ok=True)
    fastf1.Cache.enable_cache("f1_cache")

    # === CONFIGURABLE SECTION ===
    train_races = [
        (2022, "Emilia Romagna GP"),
        (2024, "Emilia Romagna GP"),
        (2025, "Miami GP"),
        (2025, "Saudi Arabia GP"),
        (2025, "Bahrain GP"),
        # (2025, "Emilia Romagna GP"),  # Example: use past 2025 races if available
        # Add more (year, race_name) pairs as desired
    ]
    target_race = (2025, "Emilia Romagna GP")
    session_types = ["Q", "R"]  # "R" is the target
    # ============================

    # --- Collect training data ---
    train_rows = []
    for year, race in train_races:
        driver_data = None
        for sess in session_types:
            df = get_best_lap(year, race, sess)
            if df.empty:
                continue
            if driver_data is None:
                driver_data = df
            else:
                driver_data = driver_data.merge(df, on="Driver", how="outer")
        if driver_data is not None:
            driver_data["Year"] = year
            driver_data["Race"] = race
            train_rows.append(driver_data)

    if not train_rows:
        print("No training data found for the specified configuration.")
        exit()

    train_data = pd.concat(train_rows, ignore_index=True)

    # Drop all rows with any NaNs in session columns
    session_cols = [f"{sess} (s)" for sess in session_types]
    train_data = train_data.dropna(subset=session_cols)

    if len(train_data) == 0:
        print("No valid training data after dropping NaNs.")
        exit()

    # Features and target
    feature_cols = [f"{sess} (s)" for sess in session_types if sess != "R"]
    X = train_data[feature_cols]
    y = train_data["R (s)"]

    # --- Data validation ---
    validate_training_data(X, y)
    validate_label_variation(y)
    validate_feature_target_correlation(X, y)

    # --- Train model ---
    # model = LinearRegression()
    # model = RandomForestRegressor(n_estimators=100, random_state=42)
    # model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model = ExtraTreesRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # --- Predict for target race ---
    target_year, target_race_name = target_race
    target_data = None
    for sess in session_types:
        df = get_best_lap(target_year, target_race_name, sess)
        if df.empty:
            continue
        if target_data is None:
            target_data = df
        else:
            target_data = target_data.merge(df, on="Driver", how="outer")

    if target_data is None:
        print("No data found for the target race.")
        exit()

    # Drop rows with any NaNs in feature columns
    target_data = target_data.dropna(subset=feature_cols)

    if len(target_data) == 0:
        print("No valid target data after dropping NaNs.")
        exit()

    validate_target_features(target_data, feature_cols)

    X_target = target_data[feature_cols]
    drivers = target_data["Driver"].values

    # Predict
    y_pred = model.predict(X_target)
    target_data["Predicted R (s)"] = y_pred

    # Show predictions sorted by predicted race time (fastest first)
    target_data_sorted = target_data.sort_values("Predicted R (s)")
    print(f"\nPredicted results for {target_race_name} {target_year}:")
    print(target_data_sorted[["Driver", "Predicted R (s)"]].reset_index(drop=True))

    # Print predicted winner and their predicted time
    winner_row = target_data_sorted.iloc[0]
    print(f"\n🏆 Predicted winner: {winner_row['Driver']} with a predicted race time of {winner_row['Predicted R (s)']:.3f} seconds")

    # Print actual winner and their race time if available
    if "R (s)" in target_data_sorted.columns and not target_data_sorted["R (s)"].isna().all():
        actual_sorted = target_data_sorted.sort_values("R (s)")
        actual_winner_row = actual_sorted.iloc[0]
        print(f"\n🏁 Actual winner: {actual_winner_row['Driver']} with an actual race time of {actual_winner_row['R (s)']:.3f} seconds")


def get_best_lap(year, race_name, session_type):
    try:
        session = fastf1.get_session(year, race_name, session_type)
        session.load()
        laps = session.laps
        # Use fastest lap time instead of average lap
        best_laps = laps.groupby("Driver")["LapTime"].min().reset_index()
        best_laps["LapTime (s)"] = best_laps["LapTime"].dt.total_seconds()
        best_laps = best_laps[["Driver", "LapTime (s)"]]
        best_laps = best_laps.rename(columns={"LapTime (s)": f"{session_type} (s)"})
        return best_laps
    except Exception:
        return pd.DataFrame(columns=["Driver", f"{session_type} (s)"])

def validate_training_data(X, y):
    # Check for empty data
    if X.empty or y.empty:
        print("❌ Training data is empty after preprocessing. Check your input races and session types.")
        exit()
    # Check for constant features
    for col in X.columns:
        if X[col].nunique() == 1:
            print(f"⚠️ Feature column '{col}' has the same value for all drivers. Model may not distinguish between drivers.")
    # Check for all features being identical across all rows
    if (X.nunique(axis=0) == 1).all():
        print("❌ All feature columns have the same value for every driver. Model will predict the same value for all.")
        exit()
    # Check for NaNs (shouldn't happen after dropna, but just in case)
    if X.isna().any().any() or y.isna().any():
        print("❌ NaN values detected in training data after dropna. Please check data processing.")
        exit()

def validate_label_variation(y):
    # Check for constant or near-constant target
    if y.nunique() == 1:
        print("❌ Training target (race time) is constant for all samples. Model will always predict this value.")
        exit()
    if y.max() - y.min() < 0.001:
        print("⚠️ Training target (race time) has very little variation. Model may predict nearly the same value for all.")
        
def validate_feature_target_correlation(X, y):
    # Check if features are correlated with target
    corr = pd.concat([X, y], axis=1).corr()
    target_corrs = corr[y.name][X.columns]
    if target_corrs.abs().max() < 0.01:
        print("⚠️ None of the features are correlated with the target. Model may not be able to distinguish between drivers.")
        print(target_corrs)
        
def validate_target_features(target_data, feature_cols):
    # Check if all target features are identical
    identical = all(target_data[feature_cols].nunique() == 1)
    if identical:
        print("❌ All target feature values are identical for every driver. Model will predict the same value for all.")
        print(target_data[feature_cols])
        exit()

if __name__ == "__main__":
    main()