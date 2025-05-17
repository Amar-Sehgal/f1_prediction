import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Enable FastF1 caching
fastf1.Cache.enable_cache("f1_cache")

# === CONFIGURABLE SECTION ===
# Specify which years and sessions to use for training
# Example: {"2024": ["FP1", "FP2", "Q", "R"]}
years_sessions = {
    # 2024: ["FP1", "FP2", "Q", "R"]
    2024: ["Q", "R"]
}
race_name = "Emilia Romagna GP"
# ============================

def get_fp_times(year, session_type):
    session = fastf1.get_session(year, race_name, session_type)
    session.load()
    laps = session.laps
    best_laps = laps.groupby("Driver")["LapTime"].min().reset_index()
    best_laps["LapTime (s)"] = best_laps["LapTime"].dt.total_seconds()
    return best_laps[["Driver", "LapTime (s)"]].rename(
        columns={"LapTime (s)": f"{session_type}_{year} (s)"}
    )

# Collect all requested data
all_data = []
for year, sessions in years_sessions.items():
    dfs = []
    for sess in sessions:
        session_type = "R" if sess == "R" else sess
        session = fastf1.get_session(year, race_name, session_type)
        session.load()
        laps = session.laps
        best_laps = laps.groupby("Driver")["LapTime"].min().reset_index()
        best_laps["LapTime (s)"] = best_laps["LapTime"].dt.total_seconds()
        col_name = f"Race_{year} (s)" if sess == "R" else f"{sess}_{year} (s)"
        dfs.append(
            best_laps[["Driver", "LapTime (s)"]].rename(
                columns={"LapTime (s)": col_name}
            )
        )
    # Merge all session data for this year
    df_merged = dfs[0]
    for df in dfs[1:]:
        df_merged = df_merged.merge(df, on="Driver")
    all_data.append(df_merged)

# Stack all years' data for training
if len(all_data) > 1:
    train_data = pd.concat(all_data, ignore_index=True)
else:
    train_data = all_data[0]

# Rename columns for model training
rename_dict = {}
for year, sessions in years_sessions.items():
    for sess in sessions:
        if sess == "R":
            rename_dict[f"Race_{year} (s)"] = "Race (s)"
        else:
            rename_dict[f"{sess}_{year} (s)"] = f"{sess} (s)"
train_data = train_data.rename(columns=rename_dict)

# 2025 Qualifying and Practice Data (example, fill with real values as needed)
qualifying_2025 = pd.DataFrame({
    "Driver": [
        "Oscar Piastri", "George Russell", "Lando Norris", "Max Verstappen", "Lewis Hamilton",
        "Charles Leclerc", "Isack Hadjar", "Andrea Kimi Antonelli", "Yuki Tsunoda", "Alexander Albon",
        "Esteban Ocon", "Nico Hülkenberg", "Fernando Alonso", "Lance Stroll", "Carlos Sainz Jr.",
        "Pierre Gasly", "Oliver Bearman", "Jack Doohan", "Gabriel Bortoleto", "Liam Lawson"
    ],
    # "FP1 (s)": [np.nan]*20,  # Fill with 2025 FP1 times in seconds
    # "FP2 (s)": [np.nan]*20,  # Fill with 2025 FP2 times in seconds
    "Q (s)": [90.641, 90.723, 90.793, 90.817, 90.927,
              91.021, 91.079, 91.103, 91.638, 91.706,
              91.625, 91.632, 91.688, 91.773, 91.840,
              91.992, 92.018, 92.092, 92.141, 92.174]
})

# Map full names to FastF1 3-letter codes
driver_mapping = {
    "Oscar Piastri": "PIA", "George Russell": "RUS", "Lando Norris": "NOR", "Max Verstappen": "VER",
    "Lewis Hamilton": "HAM", "Charles Leclerc": "LEC", "Isack Hadjar": "HAD", "Andrea Kimi Antonelli": "ANT",
    "Yuki Tsunoda": "TSU", "Alexander Albon": "ALB", "Esteban Ocon": "OCO", "Nico Hülkenberg": "HUL",
    "Fernando Alonso": "ALO", "Lance Stroll": "STR", "Carlos Sainz Jr.": "SAI", "Pierre Gasly": "GAS",
    "Oliver Bearman": "BEA", "Jack Doohan": "DOO", "Gabriel Bortoleto": "BOR", "Liam Lawson": "LAW"
}
qualifying_2025["DriverCode"] = qualifying_2025["Driver"].map(driver_mapping)

# Merge 2025 Data with 2024 Race Data
merged_data = qualifying_2025.merge(train_data, left_on="DriverCode", right_on="Driver")

# Use only "Q (s)" as a feature (NO CHANGE in model logic for this file)
X = merged_data[["Q (s)"]]
y = merged_data["Race (s)"]

if X.shape[0] == 0:
    raise ValueError("Dataset is empty after preprocessing. Check data sources!")

# Train Gradient Boosting Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=39)
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=39)
model.fit(X_train, y_train)

# Predict using 2025 qualifying times
predicted_lap_times = model.predict(qualifying_2025[["Q (s)"]])
qualifying_2025["PredictedRaceTime (s)"] = predicted_lap_times

# Rank drivers by predicted race time
qualifying_2025 = qualifying_2025.sort_values(by="PredictedRaceTime (s)")

# Print final predictions
print(f"\n🏁 Predicted 2025 {race_name} GP Winner with no Change in ML Model 🏁\n")
print(qualifying_2025[["Driver", "PredictedRaceTime (s)"]])

# Evaluate Model
y_pred = model.predict(X_test)
print(f"\n🔍 Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")
