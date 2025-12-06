# app/ai/preprocessing.py
import pandas as pd
import numpy as np
from geopy.distance import geodesic

def docs_to_dataframe(docs): 
    rows = []
    for d in docs:
        gps = d.get('gps', {})
        battery = d.get('battery', {})
        lat = gps.get('lat', np.nan)
        lon = gps.get('lon', np.nan)
        percent = battery.get('percent', np.nan)
        voltage = battery.get('voltage', np.nan)
        ts = battery.get('ts_ms', None)
        rows.append({
            'device_id': d.get('device_id'),
            'lat': float(lat) if lat is not None else np.nan,
            'lon': float(lon) if lon is not None else np.nan,
            'battery_percent': float(percent) if percent is not None else np.nan,
            'battery_voltage': float(voltage) if voltage is not None else np.nan,
            'ts_ms': int(ts) if ts is not None else None
        })
    df = pd.DataFrame(rows)
    df = df.dropna(subset=['lat', 'lon', 'battery_percent', 'battery_voltage'])
    df = df.sort_values(['device_id', 'ts_ms']).reset_index(drop=True)
    return df


def add_derived_features(df):
    df['speed_m_s'] = np.nan
    df['battery_drop_per_s'] = 0.0
    for dev, group in df.groupby('device_id'):
        group = group.sort_values('ts_ms')
        speeds, drops = [], []
        prev = None
        for _, row in group.iterrows():
            if prev is None:
                speeds.append(0.0)
                drops.append(0.0)
            else:
                prev_coord = (prev['lat'], prev['lon'])
                cur_coord = (row['lat'], row['lon'])
                try:
                    dist_m = geodesic(prev_coord, cur_coord).meters
                except:
                    dist_m = 0.0
                dt = max(1, (row['ts_ms'] - prev['ts_ms']) / 1000.0)
                speed = dist_m / dt
                battery_drop = max(0.0, prev['battery_percent'] - row['battery_percent']) / dt
                speeds.append(speed)
                drops.append(battery_drop)
            prev = row
        df.loc[group.index, 'speed_m_s'] = speeds
        df.loc[group.index, 'battery_drop_per_s'] = drops
    df['speed_m_s'] = df['speed_m_s'].replace([np.inf, -np.inf], 0.0)
    df['battery_drop_per_s'] = df['battery_drop_per_s'].replace([np.inf, -np.inf], 0.0)
    return df

def prepare_features(df, feature_cols=None):
    if feature_cols is None:
        feature_cols = ['lat', 'lon', 'battery_percent', 'battery_voltage', 'speed_m_s', 'battery_drop_per_s']
    X = df[feature_cols].values
    return X, feature_cols
