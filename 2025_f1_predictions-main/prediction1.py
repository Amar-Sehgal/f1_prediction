import fastf1
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import os

os.makedirs("f1_cache", exist_ok=True)
fastf1.Cache.enable_cache("f1_cache")

# --- User input ---
race_name = ""
train_years = [2021, 2022, 2024]
test_year = int(input("Enter test year (e.g. 2025): "))
qual_type = input("Enter qualifying session type (Q, Q1, Q2, Q3): ").strip() or "Q"
race_type = input("Enter race session type (R or S for Sprint): ").strip() or "R"

# --- Gather training data ---
session_data = []
for year in train_years:
    try:
        q_sess = fastf1.get_session(year, race_name, qual_type)
        q_sess.load()
        q_laps = q_sess.laps
        fastest_q = q_laps.groupby("Driver")["LapTime"].min().reset_index()
        fastest_q["Q (s)"] = fastest_q["LapTime"].dt.total_seconds()
        fastest_q = fastest_q[["Driver", "Q (s)"]]
    except Exception:
        continue

    try:
        r_sess = fastf1.get_session(year, race_name, race_type)
        r_sess.load()
        r_laps = r_sess.laps
        fastest_r = r_laps.groupby("Driver")["LapTime"].min().reset_index()
        fastest_r["Race (s)"] = fastest_r["LapTime"].dt.total_seconds()
        fastest_r = fastest_r[["Driver", "Race (s)"]]
    except Exception:
        continue

    df = fastest_q.merge(fastest_r, on="Driver")
    session_data.append(df)

if not session_data:
    print("No training data found.")
    exit()

train_data = pd.concat(session_data)
X = train_data[["Q (s)"]]
y = train_data["Race (s)"]

mask = ~(X.isna().any(axis=1) | y.isna())
X = X[mask]
y = y[mask]

if len(X) == 0:
    print("No valid training data after filtering.")
    exit()

model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=39)
model.fit(X, y)

# --- Predict on test year qualifying data ---
try:
    q_test = fastf1.get_session(test_year, race_name, qual_type)
    q_test.load()
    q_laps_test = q_test.laps
    fastest_q_test = q_laps_test.groupby("Driver")["LapTime"].min().reset_index()
    fastest_q_test["Q (s)"] = fastest_q_test["LapTime"].dt.total_seconds()
    fastest_q_test = fastest_q_test[["Driver", "Q (s)"]].dropna()
except Exception:
    print("Could not load qualifying data for test year.")
    exit()

X_test = fastest_q_test[["Q (s)"]]
drivers_test = fastest_q_test["Driver"].values
preds = model.predict(X_test)
pred_df = pd.DataFrame({"Driver": drivers_test, "PredictedRaceTime (s)": preds})
pred_df = pred_df.sort_values("PredictedRaceTime (s)")

print("\nPredicted finishing order:")
print(pred_df)

# --- Get actual winner if available ---
try:
    r_test = fastf1.get_session(test_year, race_name, race_type)
    r_test.load()
    r_laps_test = r_test.laps
    fastest_r_test = r_laps_test.groupby("Driver")["LapTime"].min().reset_index()
    fastest_r_test["Race (s)"] = fastest_r_test["LapTime"].dt.total_seconds()
    fastest_r_test = fastest_r_test[["Driver", "Race (s)"]].dropna()
    winner_actual = fastest_r_test.sort_values("Race (s)").iloc[0]["Driver"]
    print(f"\nActual winner: {winner_actual}")
except Exception:
    print("\nActual race data not available for test year.")
