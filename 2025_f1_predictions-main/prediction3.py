import fastf1
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import os

os.makedirs("f1_cache", exist_ok=True)
fastf1.Cache.enable_cache("f1_cache")

# Years to use for training
train_years = [2022, 2023, 2024]
test_year = 2025

# Get all races for the test year
schedule = fastf1.get_event_schedule(test_year)
race_names = schedule["EventName"].tolist()

correct = 0
total = 0

for race_name in race_names:
    # --- Gather training data ---
    session_data = []
    race_data = []
    for year in train_years:
        # Qualifying
        try:
            q_sess = fastf1.get_session(year, race_name, "Q")
            q_sess.load()
            q_laps = q_sess.laps
            avg_q = q_laps.groupby("Driver")["LapTime"].mean().reset_index()
            avg_q["Q (s)"] = avg_q["LapTime"].dt.total_seconds()
            avg_q = avg_q[["Driver", "Q (s)"]]
        except Exception:
            continue  # Skip if session not available

        # Race
        try:
            r_sess = fastf1.get_session(year, race_name, "R")
            r_sess.load()
            r_laps = r_sess.laps
            avg_r = r_laps.groupby("Driver")["LapTime"].mean().reset_index()
            avg_r["Race (s)"] = avg_r["LapTime"].dt.total_seconds()
            avg_r = avg_r[["Driver", "Race (s)"]]
        except Exception:
            continue  # Skip if session not available

        df = avg_q.merge(avg_r, on="Driver")
        session_data.append(df)

    if not session_data:
        continue

    train_data = pd.concat(session_data)
    X = train_data[["Q (s)"]]
    y = train_data["Race (s)"]

    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]

    if len(X) == 0:
        continue

    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=39)
    model.fit(X, y)

    # --- Predict on 2024 qualifying data ---
    try:
        q_2024 = fastf1.get_session(test_year, race_name, "Q")
        q_2024.load()
        q_laps_2024 = q_2024.laps
        avg_q_2024 = q_laps_2024.groupby("Driver")["LapTime"].mean().reset_index()
        avg_q_2024["Q (s)"] = avg_q_2024["LapTime"].dt.total_seconds()
        avg_q_2024 = avg_q_2024[["Driver", "Q (s)"]].dropna()
    except Exception:
        continue

    X_2024 = avg_q_2024[["Q (s)"]]
    drivers_2024 = avg_q_2024["Driver"].values
    preds = model.predict(X_2024)
    pred_df = pd.DataFrame({"Driver": drivers_2024, "PredictedRaceTime (s)": preds})
    pred_df = pred_df.sort_values("PredictedRaceTime (s)")

    # --- Get actual 2024 race winner ---
    try:
        r_2024 = fastf1.get_session(test_year, race_name, "R")
        r_2024.load()
        r_laps_2024 = r_2024.laps
        avg_r_2024 = r_laps_2024.groupby("Driver")["LapTime"].mean().reset_index()
        avg_r_2024["Race (s)"] = avg_r_2024["LapTime"].dt.total_seconds()
        avg_r_2024 = avg_r_2024[["Driver", "Race (s)"]].dropna()
        winner_actual = avg_r_2024.sort_values("Race (s)").iloc[0]["Driver"]
    except Exception:
        continue

    winner_pred = pred_df.iloc[0]["Driver"]
    total += 1
    if winner_pred == winner_actual:
        correct += 1

    print(f"{race_name}: Predicted winner: {winner_pred}, Actual winner: {winner_actual}")

if total > 0:
    print(f"\nModel correctly picked the winner {correct}/{total} times ({100*correct/total:.1f}%)")
else:
    print("No races could be evaluated.")
