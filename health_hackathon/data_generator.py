import pandas as pd
import numpy as np
import os

def generate_multi_source_data(days=90):
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=days)
    
    ### INTERNAL BIOLOGY SIMULATION ###
    caffeine = np.random.choice([0, 50, 100, 200, 400], days)
    screen_time = np.random.uniform(2, 7, days)
    sleep_hrs = np.clip(8 - (screen_time * 0.3) + np.random.normal(0, 0.5, days), 3, 10) # sleep impacted by screen time

    ### HEART RATE SOURCE (Wearable) ###
    hr_df = pd.DataFrame({
        "date": dates,
        "heart_rate_bpm": np.random.normal(70, 5, days)
    })
    
    ### ACTIVITY SOURCE (Watch) ###
    workout_intensity = np.clip((sleep_hrs * 10) + np.random.normal(0, 15, days), 10, 100) # intensity influenced by sleep
    active_min = np.clip((workout_intensity * 0.8) + np.random.normal(10, 5, days), 0, None)
    
    activity_df = pd.DataFrame({
        "date": dates,
        "active_min": active_min,
        "workout_intensity": workout_intensity,
        "steps": (active_min * 110) + np.random.normal(0, 500, days),
        "sleep_hrs": sleep_hrs,
    })
    
    ### SOURCE (Phone) ###
    sleep_df = pd.DataFrame({
        "date": dates,
        "screen_hrs": screen_time,
    })
    
    ### NUTRITION SOURCE (Logging App) ###
    nutrition_df = pd.DataFrame({
        "date": dates,
        "caffeine_mg": caffeine,
        "calories": 1800 + (workout_intensity * 8),
        "protein_g": np.random.normal(100, 15, days)
    })

    # INJECT ANOMALIES (Day 45: high caffeine spike)
    hr_df.loc[45, 'heart_rate_bpm'] = 118 
    nutrition_df.loc[45, 'caffeine_mg'] = 650

    # Save individual files to mimic different API exports
    hr_df.to_csv("heart_rate.csv", index=False)
    activity_df.to_csv("activity.csv", index=False)
    sleep_df.to_csv("sleep.csv", index=False)
    nutrition_df.to_csv("nutrition.csv", index=False)
    
    print("✅ Generated 4 source CSVs with cross-device correlations.")

if __name__ == "__main__":
    generate_multi_source_data()