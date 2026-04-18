"""
Smart EV Charging Station Availability Prediction System
Upgraded Flask Backend — app.py v2.0
New: city search, smart suggestions, wait time, directions endpoint, all-India stations
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib, json, math

app = Flask(__name__)

rf = joblib.load("rf.pkl")
dt = joblib.load("dt.pkl")

df_train    = pd.read_csv("ev_large_data.csv")
stations_df = pd.read_csv("india_ev_stations.csv")

# Sorted city list for dropdown
CITIES = sorted(stations_df["city"].unique().tolist())

# ── Haversine (km) ──────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

# ── Smart suggestions based on hour + congestion ────────────────
def generate_suggestions(hour, congestion_label, availability_pct, rain):
    tips = []
    # Best hour suggestion
    if congestion_label == "High":
        if hour < 10:
            tips.append("👉 Try charging after 11 AM — congestion usually drops mid-morning.")
        elif hour < 14:
            tips.append("👉 Peak hours! Try charging before 9 AM or after 8 PM for less wait.")
        elif hour < 19:
            tips.append("👉 Evening rush ahead. Charge now or wait until after 9 PM.")
        else:
            tips.append("👉 Late-night charging (10 PM–6 AM) has the lowest congestion. ✅")
    elif congestion_label == "Medium":
        tips.append("👉 Moderate traffic. Off-peak hours (6–9 AM or 9–11 PM) are better.")
    else:
        tips.append("👉 Great time to charge! Low congestion right now. ✅")

    # Availability advice
    if availability_pct < 30:
        tips.append("⚠️ Low availability predicted. Consider a nearby backup station.")
    elif availability_pct > 75:
        tips.append("✅ High availability — you're likely to find a free slot immediately.")

    # Rain tip
    if rain == 1:
        tips.append("🌧️ Rainy conditions: Charging is safe but allow extra travel time.")

    # General best-time tip
    tips.append("💡 Statistically, 6–9 AM and 10 PM–12 AM are the least busy charging windows.")

    return tips

# ── Estimated wait time (minutes) ──────────────────────────────
def estimate_wait(station, congestion_idx):
    avail = station.get("available_slots", 1)
    stored_wait = station.get("wait_time_min", 0)
    if avail > 0:
        return 0
    # Factor in ML congestion
    base = stored_wait if stored_wait > 0 else 10
    multiplier = [1.0, 1.3, 1.8][congestion_idx]
    return round(base * multiplier)

# ── Chart data ──────────────────────────────────────────────────
def build_chart_data(hour=None, temp=None, rain=None):
    hourly_avg = df_train.groupby("hour")["availability_percentage"].mean().reset_index()
    if hour is not None:
        variation = (int(hour) + float(temp) + int(rain)) * 0.4
        hourly_avg["availability_percentage"] += np.sin(hourly_avg["hour"]*np.pi/12)*variation
        hourly_avg["availability_percentage"] = hourly_avg["availability_percentage"].clip(5,100)
    return {"hours": hourly_avg["hour"].tolist(),
            "values": hourly_avg["availability_percentage"].round(2).tolist()}

# ── Routes ───────────────────────────────────────────────────────
@app.route("/")
def home():
    chart_data = build_chart_data()
    return render_template("index.html",
                           chart_data=json.dumps(chart_data),
                           cities=CITIES)

@app.route("/stations_all")
def stations_all():
    """Return ALL stations for initial map display."""
    records = stations_df.to_dict(orient="records")
    return jsonify({"stations": records})

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    try:
        hour = int(payload["hour"])
        day  = int(payload["day"])
        temp = float(payload["temp"])
        rain = int(payload["rain"])
        mode = payload.get("mode", "gps")          # "gps" or "city"
        city = payload.get("city", "").strip()
        lat  = payload.get("latitude")
        lon  = payload.get("longitude")
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    # ── ML Predictions ──
    features = pd.DataFrame([[hour, day, temp, rain]],
                            columns=["hour","day","temperature","rain"])
    availability_pct = round(float(rf.predict(features)[0]), 2)
    congestion_idx   = int(dt.predict(features)[0])
    congestion_label = ["Low","Medium","High"][congestion_idx]

    # ── Station Filtering ──
    sdf = stations_df.copy()
    busy_map = {"Low":1,"Medium":2,"High":3}
    sdf["busy_score"] = sdf["busy_level"].map(busy_map)

    if mode == "gps" and lat is not None and lon is not None:
        ref_lat, ref_lon = float(lat), float(lon)
        sdf["distance_km"] = sdf.apply(
            lambda r: round(haversine(ref_lat, ref_lon, r["latitude"], r["longitude"]), 2), axis=1)
        max_d = sdf["distance_km"].max() or 1
        sdf["norm_distance"] = (sdf["distance_km"]/max_d)*3
        sdf["final_score"] = sdf["norm_distance"] + sdf["busy_score"]
        sdf = sdf.sort_values("final_score").head(10).reset_index(drop=True)
    elif mode == "city" and city:
        filtered = sdf[sdf["city"].str.strip().str.lower() == city.lower()].copy()
        if filtered.empty:
            return jsonify({"error": f"No stations found in '{city}'"}), 404
        # City center as reference
        ref_lat = filtered["latitude"].mean()
        ref_lon = filtered["longitude"].mean()
        filtered["distance_km"] = filtered.apply(
            lambda r: round(haversine(ref_lat, ref_lon, r["latitude"], r["longitude"]), 2), axis=1)
        max_d = filtered["distance_km"].max() or 1
        filtered["norm_distance"] = (filtered["distance_km"]/max_d)*3
        filtered["final_score"] = filtered["norm_distance"] + filtered["busy_score"]
        sdf = filtered.sort_values("final_score").reset_index(drop=True)
        lat, lon = ref_lat, ref_lon   # map center = city center
    else:
        return jsonify({"error": "Provide GPS coordinates or select a city."}), 400

    # ── Enrich stations ──
    stations = []
    for i, row in sdf.iterrows():
        s = row.to_dict()
        s["wait_estimated"] = estimate_wait(s, congestion_idx)
        s["is_best"] = (i == 0)
        stations.append(s)

    suggestions = generate_suggestions(hour, congestion_label, availability_pct, rain)
    chart_data  = build_chart_data(hour, temp, rain)

    return jsonify({
        "availability":   availability_pct,
        "congestion":     congestion_label,
        "congestion_idx": congestion_idx,
        "stations":       stations,
        "suggestions":    suggestions,
        "chart_data":     chart_data,
        "user_lat":       lat,
        "user_lon":       lon,
        "mode":           mode,
        "city":           city
    })

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
