import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import os

os.makedirs("f1_cache", exist_ok=True)
race_name = 'Emilia Romagna GP'
fastf1.Cache.enable_cache("f1_cache")

# === CONFIGURABLE SECTION ===
# Specify which years and sessions to use for training
# Example: {"2024": ["FP1", "FP2", "R"]} qualifying
years_sessions = {
    # 2024: ["FP1", "FP2", "Q", "R"]
    2024: ["FP1", "FP2", "R"]
}
# ============================

def get_fp_times(year, session_type):
    session = fastf1.get_session(year, race_name, session_type)
    session.load()
    laps = session.laps
    # Get best lap per driver
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
        if sess == "R":
            session = fastf1.get_session(year, race_name, "R")
            session.load()
            laps = session.laps
            best_laps = laps.groupby("Driver")["LapTime"].min().reset_index()
            best_laps["LapTime (s)"] = best_laps["LapTime"].dt.total_seconds()
            dfs.append(
                best_laps[["Driver", "LapTime (s)"]].rename(
                    columns={"LapTime (s)": f"Race_{year} (s)"}
                )
            )
        else:
            dfs.append(get_fp_times(year, sess))
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

# Rename columns for model training (assumes only one year for race, but can be extended)
rename_dict = {}
for year, sessions in years_sessions.items():
    for sess in sessions:
        if sess == "R":
            rename_dict[f"Race_{year} (s)"] = "Race (s)"
        else:
            rename_dict[f"{sess}_{year} (s)"] = f"{sess} (s)"
train_data = train_data.rename(columns=rename_dict)

# 2025 FP1 and FP2 times (manually filled)
qualifying_2025 = pd.DataFrame({
    "Driver": [
        "Oscar Piastri", "Lando Norris", "Carlos Sainz", "George Russell", "Lewis Hamilton",
        "Pierre Gasly", "Max Verstappen", "Alexander Albon", "Gabriel Bortoleto", "Nico Hulkenberg",
        "Lance Stroll", "Charles Leclerc", "Kimi Antonelli", "Fernando Alonso", "Liam Lawson",
        "Yuki Tsunoda", "Franco Colapinto", "Oliver Bearman", "Isack Hadjar", "Esteban Ocon"
    ],
    "FP1 (s)": [
        76.545, 76.577, 76.597, 76.599, 76.641,
        76.696, 76.905, 76.922, 76.925, 76.998,
        77.032, 77.077, 77.094, 77.121, 77.286,
        77.356, 77.373, 77.446, 77.641, 77.662
    ],
    "FP2 (s)": [
        75.293, 75.318, 75.934, 75.693, 75.943,
        75.569, 75.735, 75.916, 76.339, 76.419,
        76.341, 75.768, 76.406, 76.220, 76.255,
        75.827, 76.044, 76.009, 75.792, 76.420
    ],
    # "Q (s)": [np.nan]*20  # Placeholder until you have qualifying data
})

# Select features for model based on what is present in train_data and qualifying_2025
feature_cols = [col for col in train_data.columns if col.endswith("(s)") and col != "Race (s)"]
X = train_data[feature_cols]
y = train_data["Race (s)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=39)
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=39)
model.fit(X_train, y_train)

# Predict using 2025 data (must match feature_cols)
X_2025 = qualifying_2025[feature_cols]
predicted_lap_times = model.predict(X_2025)
qualifying_2025["PredictedRaceTime (s)"] = predicted_lap_times

# Rank drivers by predicted race time
qualifying_2025 = qualifying_2025.sort_values(by="PredictedRaceTime (s)")

print(f"\n🏁 Predicted 2025 {race_name} Winner 🏁\n")
print(qualifying_2025[["Driver", "PredictedRaceTime (s)"]])

# Evaluate Model
y_pred = model.predict(X_test)
print(f"\n🔍 Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")
