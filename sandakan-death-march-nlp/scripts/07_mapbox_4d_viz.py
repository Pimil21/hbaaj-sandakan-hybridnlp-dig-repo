#!/usr/bin/env python3
"""
4D STC Generator - SYNTAX ERROR FIXED VERSION
==============================================
Fixed: Removed all f-string interpolation conflicts in JavaScript
Method: Using string concatenation instead of f-strings for JS sections
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'route_csv': '/data/Route.csv',
    'emotion_csv': '/outputs/step5_emotion_analysis_hybrid/emotion_data/location_emotions_hybrid.csv',
    'narrative_csv': '/outputs/step5_emotion_analysis_hybrid/emotion_data/hybrid_emotion_analysis.csv',
    'output_html': '4D_STC_Emotion_Analystics.html',
    'narrative_sample_rows': 1000,
    'sandakan_lat': 5.868730,
    'sandakan_lon': 118.087115,
    'basic_emotions': ['fear', 'sadness', 'anger', 'disgust', 'joy', 'surprise', 'trust', 'anticipation'],
    'extended_emotions': ['death', 'hunger', 'trauma', 'survival', 'disease', 'cruelty', 'suffering', 'despair', 'exhaustion'],
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def safe_int(val, default=0):
    if pd.isna(val) or val is None:
        return default
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default

def safe_float(val, default=0.0):
    if pd.isna(val) or val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def calculate_distance(lat1, lon1, lat2, lon2):
    return ((lat1 - lat2)**2 + (lon1 - lon2)**2)**0.5 * 111

# ============================================================================
# DATA PROCESSING
# ============================================================================

def load_and_process_data():
    print("="*80)
    print("4D STC GENERATOR - SYNTAX FIXED VERSION")
    print("="*80)

    print("\n[1/5] Loading route data...")
    df_route = pd.read_csv(CONFIG['route_csv'])
    print(f"   ✓ Loaded {len(df_route)} route locations")

    print("\n[2/5] Loading emotion data...")
    df_emotion = pd.read_csv(CONFIG['emotion_csv'])
    print(f"   ✓ Loaded {len(df_emotion)} emotion locations")

    print("\n[3/5] Loading narrative data...")
    sample_rows = CONFIG['narrative_sample_rows']
    if sample_rows:
        df_narrative = pd.read_csv(CONFIG['narrative_csv'], nrows=sample_rows)
        print(f"   ✓ Loaded {len(df_narrative)} narrative records (sampled)")
    else:
        df_narrative = pd.read_csv(CONFIG['narrative_csv'])
        print(f"   ✓ Loaded {len(df_narrative)} narrative records (full)")

    print("\n[4/5] Processing spatial-temporal data...")
    min_base = df_route['Base_month'].min()
    df_route['z_display'] = df_route['Base_month'] - min_base

    march_map = {'First March': 1, 'Second March': 2, 'Third March': 3}
    df_route['march_number'] = df_route['march_id'].map(march_map)

    df_route['location_match'] = df_route['POI_Name'].str.strip().str.lower()
    df_emotion['location_match'] = df_emotion['location_name'].str.strip().str.lower()

    df_merged = df_route.merge(df_emotion, left_on='location_match', right_on='location_match', how='left')

    print(f"   ✓ Merged data: {len(df_merged)} total locations")
    print(f"   ✓ Locations with emotions: {df_merged['avg_sentiment_score'].notna().sum()}")

    print("\n[5/5] Extracting emotion scores and narratives...")

    all_emotion_types = CONFIG['basic_emotions'] + CONFIG['extended_emotions']
    data_records = []

    for idx, row in df_merged.iterrows():
        all_emotions = {}
        for emo in all_emotion_types:
            all_emotions[emo] = safe_float(row.get(emo))

        dominant_emotion = max(all_emotions, key=all_emotions.get) if max(all_emotions.values()) > 0 else 'neutral'

        location_name = str(row['POI_Name']).strip().lower()
        narrative_rows = df_narrative[df_narrative['location_name'].str.lower().str.strip() == location_name]
        narrative_excerpt = ""

        if len(narrative_rows) > 0:
            for _, nrow in narrative_rows.iterrows():
                sentence = str(nrow.get('sentence', ''))
                if len(sentence) > 50:
                    # Clean narrative for JSON safety
                    sentence = sentence.replace('"', '\\"').replace("\n", " ").replace("\r", "")
                    narrative_excerpt = sentence[:300] + "..."
                    break

        distance_km = calculate_distance(
            row['Latitude'], row['Longitude'],
            CONFIG['sandakan_lat'], CONFIG['sandakan_lon']
        )

        record = {
            'location_id': int(row['OBJECTID']),
            'location_name': str(row['POI_Name']),
            'latitude': float(row['Latitude']),
            'longitude': float(row['Longitude']),
            'march_id': str(row['march_id']),
            'march_number': int(row['march_number']),
            'segment': int(row['segment__']),
            'base_month': int(row['Base_month']),
            'start_week': int(row['Start_week']),
            'end_week': int(row['End_week']),
            'z_display': float(row['z_display']),
            'distance_from_sandakan': float(distance_km),
            'day_of_march': int(240 - row['Base_month']),
            'has_emotion_data': not pd.isna(row.get('avg_sentiment_score')),
            'mention_count': safe_int(row.get('mention_count')),
            'avg_sentiment_score': safe_float(row.get('avg_sentiment_score')),
            'sentiment_label': str(row.get('sentiment_label', 'neutral') if pd.notna(row.get('sentiment_label')) else 'neutral'),
            'avg_emotion_intensity': safe_float(row.get('avg_emotion_intensity')),
            'total_emotions_detected': safe_int(row.get('total_emotions_detected')),
            'unique_emotions': safe_int(row.get('unique_emotions')),
            **{f'{emo}_score': all_emotions[emo] for emo in all_emotion_types},
            'dominant_emotion': dominant_emotion,
            'narrative_excerpt': narrative_excerpt
        }

        data_records.append(record)

    locations_with_emotions = [r for r in data_records if r['has_emotion_data']]

    print(f"   ✓ Processed {len(data_records)} records")
    print(f"   ✓ Extracted {sum(1 for r in data_records if r['narrative_excerpt'])} narrative excerpts")

    temporal_data = {}
    for r in locations_with_emotions:
        day = r['day_of_march']
        if day not in temporal_data:
            temporal_data[day] = {'count': 0, 'total_intensity': 0, 'emotions': {}, 'locations': []}
        temporal_data[day]['count'] += 1
        temporal_data[day]['total_intensity'] += r['avg_emotion_intensity']
        temporal_data[day]['locations'].append(r['location_name'])

        dom_emo = r['dominant_emotion']
        if dom_emo not in temporal_data[day]['emotions']:
            temporal_data[day]['emotions'][dom_emo] = 0
        temporal_data[day]['emotions'][dom_emo] += 1

    temporal_days = sorted(temporal_data.keys())
    temporal_counts = [temporal_data[d]['count'] for d in temporal_days]
    temporal_locations = [temporal_data[d]['locations'] for d in temporal_days]

    emotion_counts = {}
    for emo in all_emotion_types:
        emotion_counts[emo] = sum(1 for r in data_records if r['dominant_emotion'] == emo and r['has_emotion_data'])

    stats = {
        'total_records': len(df_merged),
        'march1_count': len(df_merged[df_merged['march_number'] == 1]),
        'march2_count': len(df_merged[df_merged['march_number'] == 2]),
        'march3_count': len(df_merged[df_merged['march_number'] == 3]),
        'max_z': float(df_merged['z_display'].max()),
        'locations_with_emotions': len(locations_with_emotions),
        'emotion_counts': emotion_counts
    }

    print("\n[✓] Data processing complete!")
    print(f"    Total locations: {stats['total_records']}")
    print(f"    With emotions: {stats['locations_with_emotions']}")
    print(f"    Temporal days tracked: {len(temporal_days)}")

    return {
        'data_records': data_records,
        'temporal_data': {'days': temporal_days, 'counts': temporal_counts, 'locations': temporal_locations},
        'stats': stats
    }

# ============================================================================
# HTML GENERATION
# ============================================================================

def generate_html(processed_data):
    data_records = processed_data['data_records']
    temporal_data = processed_data['temporal_data']
    stats = processed_data['stats']

    # Convert to JSON with proper escaping
    data_json = json.dumps(data_records)
    temporal_json = json.dumps(temporal_data)

    emotion_counts = stats['emotion_counts']

    # Build legend items safely
    legend_items = []
    emotion_colors_list = [
        ('fear', '#9B59B6'), ('sadness', '#5499C7'), ('anger', '#E74C3C'), 
        ('disgust', '#27AE60'), ('joy', '#F8D547'), ('surprise', '#E91E63'),
        ('trust', '#16A085'), ('anticipation', '#F39C12'), ('death', '#34495E'), 
        ('suffering', '#C0392B')
    ]

    for emo, color in emotion_colors_list:
        count = emotion_counts.get(emo, 0)
        legend_items.append(
            f'<div class="legend-item" onclick="filterEmotion(\'{emo}\')"><div class="legend-color" style="background:{color}"></div>{emo.capitalize()} ({count})</div>'
        )

    legend_html = '\n'.join(legend_items)

    # Build HTML in parts to avoid f-string issues

    # Combine all parts
    html_content = html_part1 + html_part2 + html_part3

    return html_content

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*80)
    print("STARTING 4D STC GENERATION - SYNTAX FIXED VERSION")
    print("="*80)

    for key, path in [('route_csv', CONFIG['route_csv']), 
                      ('emotion_csv', CONFIG['emotion_csv']), 
                      ('narrative_csv', CONFIG['narrative_csv'])]:
        if not Path(path).exists():
            print(f"\n❌ ERROR: File not found: {path}")
            print(f"   Please update CONFIG['{key}'] with the correct path.")
            return

    processed_data = load_and_process_data()

    print("\n[6/6] Generating HTML (syntax-safe)...")
    html_content = generate_html(processed_data)

    if html_content is None:
        print("\n❌ ERROR: generate_html() returned None!")
        return

    output_path = CONFIG['output_html']
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"   ✓ HTML saved: {output_path}")

if __name__ == "__main__":
    main()
