#!/usr/bin/env python3
"""
SOLUTION A: RO2 STATISTICAL DASHBOARD - UPDATED WITH STEP 5 HYBRID DATA
Runs locally for analysis - uses enhanced emotion detection results

Author: Your Name
Date: January 2026
"""

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    BASE_DIR = Path(r"your/base/directory/path")  # Update this path accordingly
    
    # Step 5 hybrid emotion data (NEW - enhanced detection)
    STEP5_DIR = BASE_DIR / "outputs" / "step5_emotion_analysis_hybrid"
    HYBRID_EMOTIONS = STEP5_DIR / "emotion_data" / "hybrid_emotion_analysis.csv"
    LOCATION_EMOTIONS = STEP5_DIR / "emotion_data" / "location_emotions_hybrid.csv"
    STATISTICS = STEP5_DIR / "reports" / "hybrid_emotion_statistics.json"

# ============================================================================
# INSIDE OUT EMOTION COLOR PALETTE
# ============================================================================

INSIDE_OUT_COLORS = {
    'joy': '#F8D547',          # Yellow
    'sadness': '#5499C7',      # Blue
    'anger': '#E74C3C',        # Red
    'fear': '#9B59B6',         # Purple
    'disgust': '#27AE60',      # Green
    'trust': '#3498DB',        # Light Blue
    'anticipation': '#F39C12', # Orange
    'surprise': '#E91E63',     # Pink
    'suffering': '#8B4513',    # Brown
    'death': '#2C3E50',        # Dark gray
    'exhaustion': '#7F8C8D',   # Gray
    'courage': '#16A085',      # Teal
    'despair': '#34495E',      # Slate
    'no_data': '#95A5A6'       # Gray
}

# ============================================================================
# LOAD AND PREPARE DATA FROM STEP 5
# ============================================================================

def load_step5_data():
    """Load enhanced emotion data from Step 5 hybrid analysis"""
    
    print("📂 Loading Step 5 hybrid emotion data...")
    
    # Load hybrid emotions
    hybrid_emotions = pd.read_csv(Config.HYBRID_EMOTIONS)
    
    # Parse emotion lists
    for col in ['emotions', 'bert_top_emotions', 'combined_emotions']:
        if col in hybrid_emotions.columns:
            hybrid_emotions[col] = hybrid_emotions[col].apply(
                lambda x: eval(x) if pd.notna(x) and isinstance(x, str) and x.startswith('[') else []
            )
    
    # Load location-level aggregated data
    location_emotions = pd.read_csv(Config.LOCATION_EMOTIONS)
    
    print(f"✓ Loaded {len(hybrid_emotions)} emotion analyses")
    print(f"✓ Loaded {len(location_emotions)} unique locations")
    
    # Create RO2-compatible dataframe
    ro2_data = create_ro2_compatible_data(hybrid_emotions, location_emotions)
    
    return ro2_data

def create_ro2_compatible_data(hybrid_emotions, location_emotions):
    """Transform Step 5 data into RO2 statistical format"""
    
    records = []
    
    for idx, row in hybrid_emotions.iterrows():
        # Get location details
        location_name = row['entity_text']
        
        # Get coordinates
        loc_info = location_emotions[location_emotions['location_name'] == location_name]
        if len(loc_info) > 0:
            lat = loc_info.iloc[0]['latitude']
            lon = loc_info.iloc[0]['longitude']
        else:
            lat, lon = None, None
        
        # Get dominant emotion from combined emotions
        combined_emotions = row.get('combined_emotions', [])
        if combined_emotions and len(combined_emotions) > 0:
            from collections import Counter
            emotion_counts = Counter(combined_emotions)
            dominant_emotion = emotion_counts.most_common(1)[0][0]
            emotion_intensity = len(combined_emotions) / 10  # Normalize
        else:
            dominant_emotion = 'unknown'
            emotion_intensity = 0
        
        # Sentiment
        sentiment_polarity = row.get('sentiment_score', 0)
        sentiment_label = row.get('sentiment_label', 'neutral')
        
        # Emotion scores (mock from combined emotions for compatibility)
        emotion_scores = {}
        for emotion in combined_emotions:
            emotion_scores[f'{emotion}_score'] = 1
        
        # March number (assign based on location if available)
        march_number = assign_march_number(location_name)
        
        # Distance (placeholder - calculate if needed)
        distance_from_start_km = 0
        
        record = {
            'location_name': location_name,
            'latitude': lat,
            'longitude': lon,
            'dominant_emotion': dominant_emotion,
            'emotion_intensity': emotion_intensity,
            'sentiment_polarity': sentiment_polarity,
            'sentiment_label': sentiment_label,
            'march_number': march_number,
            'distance_from_start_km': distance_from_start_km,
            'date': None,
            **emotion_scores
        }
        
        records.append(record)
    
    df = pd.DataFrame(records)
    
    # Fill NaN distances
    df['distance_from_start_km'] = df['distance_from_start_km'].fillna(0)
    
    print(f"✓ Created RO2 dataset with {len(df)} records")
    
    return df

def assign_march_number(location_name):
    """Assign march number based on location name"""
    loc_lower = str(location_name).lower()
    
    # First march locations
    if any(x in loc_lower for x in ['sandakan', 'mile 8', 'beluran']):
        return 1
    # Second march
    elif any(x in loc_lower for x in ['paginatan', 'telupid']):
        return 2
    # Third march
    elif any(x in loc_lower for x in ['boto']):
        return 3
    # Common endpoint
    elif 'ranau' in loc_lower:
        return 1  # Default to first march
    else:
        return 1  # Default

# ============================================================================
# STATISTICAL ANALYSIS FUNCTIONS
# ============================================================================

def calculate_moran_i(df):
    """Calculate Moran's I for spatial autocorrelation"""
    
    emotions = df.groupby('location_name')['emotion_intensity'].mean()
    n = len(emotions)
    
    if n < 2:
        return 0.0
    
    mean_emotion = emotions.mean()
    locations = df[['location_name', 'latitude', 'longitude']].drop_duplicates()
    locations = locations[locations['latitude'].notna() & locations['longitude'].notna()]
    
    if len(locations) < 2:
        return 0.0
    
    numerator = 0
    denominator = 0
    weights_sum = 0
    
    for i, row1 in locations.iterrows():
        for j, row2 in locations.iterrows():
            if i != j:
                dist = np.sqrt((row1['latitude'] - row2['latitude'])**2 + 
                             (row1['longitude'] - row2['longitude'])**2)
                weight = 1 / dist if dist > 0 else 0
                weights_sum += weight
                
                val1 = emotions.get(row1['location_name'], 0)
                val2 = emotions.get(row2['location_name'], 0)
                numerator += weight * (val1 - mean_emotion) * (val2 - mean_emotion)
    
    for val in emotions:
        denominator += (val - mean_emotion)**2
    
    if denominator > 0 and weights_sum > 0:
        moran_i = (n / weights_sum) * (numerator / denominator)
    else:
        moran_i = 0
    
    return moran_i

def get_statistical_summary(df):
    """Generate statistical summary metrics"""
    
    # Calculate correlation
    valid_data = df[['emotion_intensity', 'sentiment_polarity']].dropna()
    if len(valid_data) > 1:
        corr, p_value = pearsonr(valid_data['emotion_intensity'], 
                                 valid_data['sentiment_polarity'])
    else:
        corr, p_value = 0, 1.0
    
    # Moran's I
    moran_i = calculate_moran_i(df)
    interpretation = "Dispersed" if moran_i < 0 else "Clustered" if moran_i > 0.3 else "Random"
    
    # Dominant emotion
    dominant = df['dominant_emotion'].mode()[0] if len(df['dominant_emotion'].mode()) > 0 else 'N/A'
    
    summary = {
        'Total Emotion Records': len(df),
        'Unique POI Locations': df['location_name'].nunique(),
        "Moran's I": f"{moran_i:.4f}",
        'Interpretation': interpretation,
        'Avg Emotion Score': f"{df['emotion_intensity'].mean():.3f}",
        'Avg Sentiment Polarity': f"{df['sentiment_polarity'].mean():.3f}",
        'Emotion-Sentiment Corr': f"{corr:.3f}",
        'Correlation P-value': f"{p_value:.4f}",
        'Dominant Emotion': dominant.capitalize()
    }
    
    return summary

# ============================================================================
# VISUALIZATION FUNCTIONS (continuing...)
# ============================================================================

def create_emotion_distribution_chart(df):
    """Emotion Distribution by March Phase"""
    
    df_filtered = df[df['march_number'].isin([1, 2, 3])].copy()
    
    if len(df_filtered) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data for marches 1-3", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=300, title='Emotion Distribution by March Phase')
        return fig
    
    march_labels = {1: 'First March', 2: 'Second March', 3: 'Third March'}
    df_filtered['march_phase'] = df_filtered['march_number'].map(march_labels)
    
    phase_emotion = df_filtered.groupby(['march_phase', 'dominant_emotion'], 
                                        observed=False).size().reset_index(name='count')
    
    fig = px.bar(phase_emotion, x='march_phase', y='count',
                 color='dominant_emotion',
                 title='Emotion Distribution by March Phase (Hybrid Detection)',
                 labels={'march_phase': 'March Phase', 'count': 'Count'},
                 color_discrete_map=INSIDE_OUT_COLORS)
    
    fig.update_layout(height=300, margin=dict(l=40, r=20, t=40, b=60),
                     xaxis_tickangle=-45,
                     legend=dict(orientation="v", yanchor="top", y=1, xanchor="right", x=1.15))
    
    return fig

def create_emotion_score_boxplot(df):
    """Emotion Score Distribution"""
    
    # Get all emotion columns
    emotion_cols = [col for col in df.columns if col.endswith('_score')]
    
    plot_data = []
    for col in emotion_cols:
        emotion_name = col.replace('_score', '').capitalize()
        values = df[col].dropna().values
        plot_data.extend([{'Emotion': emotion_name, 'Emotion_Score': val} 
                         for val in values if val > 0])
    
    if len(plot_data) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No emotion score data available", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=300, title='Emotion Score Distribution')
        return fig
    
    plot_df = pd.DataFrame(plot_data)
    
    color_map = {emotion.capitalize(): INSIDE_OUT_COLORS.get(emotion, '#95A5A6') 
                 for emotion in INSIDE_OUT_COLORS.keys()}
    
    fig = px.box(plot_df, x='Emotion', y='Emotion_Score',
                 title='Emotion Score Distribution (BERT + Lexicon)',
                 color='Emotion', color_discrete_map=color_map)
    
    fig.update_layout(height=300, margin=dict(l=40, r=20, t=40, b=60), showlegend=False)
    
    return fig

def create_emotion_sentiment_scatter(df):
    """Emotion Score vs Sentiment Polarity"""
    
    fig = px.scatter(df, x='emotion_intensity', y='sentiment_polarity',
                    color='dominant_emotion',
                    title='Emotion Score vs Sentiment Polarity',
                    labels={'emotion_intensity': 'Emotion Score', 
                           'sentiment_polarity': 'Sentiment Polarity'},
                    hover_data=['location_name'],
                    color_discrete_map=INSIDE_OUT_COLORS)
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(height=300, margin=dict(l=40, r=20, t=40, b=40),
                     legend=dict(orientation="v", yanchor="top", y=1, xanchor="right", x=1.15))
    
    return fig

def create_distance_decay_chart(df):
    """Distance Decay Analysis"""
    
    fear_data = df[(df['dominant_emotion'] == 'fear') & 
                   (df['distance_from_start_km'].notna()) &
                   (df['distance_from_start_km'] > 0)]
    
    if len(fear_data) < 3:
        fear_data = df[(df['distance_from_start_km'].notna()) &
                      (df['distance_from_start_km'] > 0)]
        title = 'Distance Decay: All Emotions vs Distance'
        color_col = 'dominant_emotion'
        color_map = INSIDE_OUT_COLORS
    else:
        title = 'Distance Decay: Fear Emotion'
        color_col = None
        color_map = None
    
    if len(fear_data) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No valid distance data", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=300, title=title)
        return fig
    
    if color_col:
        fig = px.scatter(fear_data, x='distance_from_start_km', y='emotion_intensity',
                        color=color_col, color_discrete_map=color_map,
                        hover_data=['location_name'])
    else:
        fig = px.scatter(fear_data, x='distance_from_start_km', y='emotion_intensity',
                        hover_data=['location_name'],
                        color_discrete_sequence=[INSIDE_OUT_COLORS['fear']])
    
    fig.update_traces(marker=dict(size=8, opacity=0.6))
    fig.update_layout(height=300, margin=dict(l=40, r=20, t=40, b=40),
                     title=title,
                     xaxis_title='Distance from Sandakan (km)',
                     yaxis_title='Emotion Score')
    
    return fig

def create_avg_emotion_by_phase(df):
    """Average Emotion Intensity by March Phase"""
    
    df_filtered = df[df['march_number'].isin([1, 2, 3])].copy()
    
    if len(df_filtered) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=300, title='Avg Emotion Intensity')
        return fig
    
    march_labels = {1: 'First March', 2: 'Second March', 3: 'Third March'}
    df_filtered['march_phase'] = df_filtered['march_number'].map(march_labels)
    
    phase_avg = df_filtered.groupby('march_phase')['emotion_intensity'].mean().reset_index()
    
    phase_order = ['First March', 'Second March', 'Third March']
    phase_avg['march_phase'] = pd.Categorical(phase_avg['march_phase'], 
                                              categories=phase_order, ordered=True)
    phase_avg = phase_avg.sort_values('march_phase')
    
    fig = px.bar(phase_avg, x='march_phase', y='emotion_intensity',
                title='Avg Emotion Intensity by March Phase',
                labels={'march_phase': 'March Phase', 
                       'emotion_intensity': 'Avg Emotion Score'},
                color_discrete_sequence=['#FA8072'])
    
    fig.update_layout(height=300, margin=dict(l=40, r=20, t=40, b=60),
                     xaxis_tickangle=-45)
    
    return fig

def create_statistical_summary_table(summary):
    """Statistical Summary Table"""
    
    table_data = [[key, str(value)] for key, value in summary.items()]
    
    fig = go.Figure(data=[go.Table(
        header=dict(values=['<b>Metric</b>', '<b>Value</b>'],
                   fill_color='#4682B4', align='left',
                   font=dict(color='white', size=12)),
        cells=dict(values=list(zip(*table_data)),
                  fill_color='lavender', align='left',
                  font=dict(size=11), height=25)
    )])
    
    fig.update_layout(title='Statistical Summary (Hybrid Analysis)',
                     height=300, margin=dict(l=20, r=20, t=40, b=20))
    
    return fig

# ============================================================================
# DASH APP
# ============================================================================

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Load data
print("\n" + "="*80)
print("LOADING STEP 5 HYBRID EMOTION DATA FOR RO2 ANALYSIS")
print("="*80)
df = load_step5_data()
summary_stats = get_statistical_summary(df)

app.layout = dbc.Container([
    
    dbc.Row([
        dbc.Col([
            html.H2("Statistical Spatial Analysis: Sandakan-Ranau Death Marches",
                   className='text-center mb-2 mt-3', style={'fontWeight': 'bold'}),
            html.H5("Enhanced with Hybrid Emotion Detection (BERT + Lexicon)",
                   className='text-center mb-4', style={'color': '#666'}),
            html.P("Research Objective 2 (RO2): Analyze spatial-temporal sentiment patterns",
                   className='text-center text-muted', style={'fontSize': '0.9rem'})
        ])
    ]),
    
    # Top Row
    dbc.Row([
        dbc.Col([dcc.Graph(id='emotion-distribution', 
                          figure=create_emotion_distribution_chart(df))], width=4),
        dbc.Col([dcc.Graph(id='emotion-boxplot', 
                          figure=create_emotion_score_boxplot(df))], width=4),
        dbc.Col([dcc.Graph(id='emotion-sentiment-scatter', 
                          figure=create_emotion_sentiment_scatter(df))], width=4)
    ], className='mb-3'),
    
    # Bottom Row
    dbc.Row([
        dbc.Col([dcc.Graph(id='distance-decay', 
                          figure=create_distance_decay_chart(df))], width=4),
        dbc.Col([dcc.Graph(id='avg-emotion-phase', 
                          figure=create_avg_emotion_by_phase(df))], width=4),
        dbc.Col([dcc.Graph(id='summary-table', 
                          figure=create_statistical_summary_table(summary_stats))], width=4)
    ])
    
], fluid=True, style={'backgroundColor': '#f8f9fa'})

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == '__main__':

