"""
Emotion Intensity Heatmap Dashboard
Sandakan-Ranau Death Marches Spatial Emotion Analysis
Using Inside Out Color Scheme (Ekman & Keltner, 2015)
"""

# ==================== CONFIGURATION ====================
class Config:
    HYBRID_EMOTIONS = "/outputs/step5_emotion_analysis_hybrid/emotion_data/hybrid_emotion_analysis.csv"

    # Inside Out / Ekman color scheme
    EMOTION_COLORS = {
        'fear': '#9D27CD',      # Purple (Fear)
        'sadness': '#4A90E2',   # Blue (Sadness)
        'disgust': '#7CB342',   # Green (Disgust)
        'anger': '#E53935',     # Red (Anger)
        'joy': '#FFD700',       # Yellow (Joy)
        'surprise': '#FF9800',  # Orange (Surprise)
        'neutral': '#9E9E9E',   # Gray (Neutral)
        'death': '#8B0000',     # Dark Red (Death)
        'despair': '#424242',   # Dark Gray (Despair)
        'cruelty': '#6A1B9A'    # Dark Purple (Cruelty)
    }

# ==================== DATA LOADING ====================
def load_emotion_data():
    """Load and prepare emotion data"""
    try:
        df = pd.read_csv(Config.HYBRID_EMOTIONS)
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

# ==================== EMOTION PARSING ====================
def parse_emotion_column(emotion_text):
    """Parse emotion data from text-based dictionary format"""
    if pd.isna(emotion_text) or not emotion_text:
        return {}

    try:
        parsed = ast.literal_eval(str(emotion_text))

        if isinstance(parsed, dict):
            result = {}
            for key, value in parsed.items():
                if isinstance(value, (int, float)):
                    result[key] = float(value)
                elif isinstance(value, list) and len(value) > 0:
                    if isinstance(value[0], (int, float)):
                        result[key] = float(value[0])
                    else:
                        result[key] = 1.0
            return result

        elif isinstance(parsed, list):
            return {emotion: 1.0 for emotion in parsed if isinstance(emotion, str)}

    except:
        try:
            emotion_words = ['anger', 'fear', 'sadness', 'joy', 'surprise', 
                           'disgust', 'neutral', 'hunger', 'despair', 'cruelty', 'death']
            result = {}
            text_lower = str(emotion_text).lower()

            for emotion in emotion_words:
                if emotion in text_lower:
                    pattern = rf"'{emotion}':\s*([\d.]+)"
                    match = re.search(pattern, text_lower)
                    if match:
                        result[emotion] = float(match.group(1))
                    else:
                        result[emotion] = 1.0

            return result
        except:
            return {}

    return {}

def hex_to_rgba(hex_color, alpha):
    """Convert hex color to rgba format"""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'

def extract_emotion_spatial_data(df, lat_col=None, lon_col=None, emotion_cols=None):
    """Extract emotions with spatial coordinates - SWAPPED VERSION FOR YOUR DATA"""
    emotion_data = []

    # Auto-detect coordinate columns if not provided
    if lat_col is None or lon_col is None:
        for col in df.columns:
            col_lower = col.lower()
            if lat_col is None and any(pattern in col_lower for pattern in ['lat', 'latitude']):
                lat_col = col
            if lon_col is None and any(pattern in col_lower for pattern in ['lon', 'long', 'longitude']):
                lon_col = col

    if lat_col is None or lon_col is None:
        print(f"❌ ERROR: Could not find coordinate columns!")
        return pd.DataFrame()

    print(f"✅ Using coordinate columns: '{lat_col}' and '{lon_col}'")

    # Auto-detect emotion columns
    if emotion_cols is None:
        emotion_cols = []
        dict_emotion_columns = ['emotions', 'bert_emotions', 'combined_emotions', 
                               'emotion_dict', 'emotion_data', 'bert_top_emotions']

        for col in dict_emotion_columns:
            if col in df.columns:
                emotion_cols.append(col)

    if not emotion_cols:
        print("❌ ERROR: No emotion columns found!")
        return pd.DataFrame()

    print(f"✅ Using emotion columns: {emotion_cols}")

    # Process each row
    records_with_coords = 0

    for idx, row in df.iterrows():
        # FIX: Your CSV has swapped columns
        # Column 'latitude' has longitude values, 'longitude' has latitude values
        lon_value = row.get(lat_col, None)  # From 'latitude' column
        lat_value = row.get(lon_col, None)  # From 'longitude' column

        # Skip if coordinates are missing
        if pd.isna(lat_value) or pd.isna(lon_value):
            continue

        records_with_coords += 1

        # Get location name
        location_name = row.get('location_name', row.get('entity_text', row.get('Location', 'Unknown')))

        # Parse emotions
        for col in emotion_cols:
            if col in row and not pd.isna(row[col]):
                parsed = parse_emotion_column(row[col])
                for emotion, score in parsed.items():
                    emotion_data.append({
                        'emotion': emotion,
                        'latitude': float(lat_value),
                        'longitude': float(lon_value),
                        'intensity': score,
                        'location': str(location_name)
                    })

    result_df = pd.DataFrame(emotion_data)

    print(f"\n📊 EXTRACTION RESULTS:")
    print(f"   Rows with coordinates: {records_with_coords} / {len(df)}")
    print(f"   Total emotion records: {len(result_df)}")

    if not result_df.empty:
        print(f"   Unique emotions: {result_df['emotion'].nunique()}")
        print(f"   Unique locations: {result_df['location'].nunique()}")
        print(f"\n📍 COORDINATE VERIFICATION:")
        print(f"   Latitude range: {result_df['latitude'].min():.2f}° to {result_df['latitude'].max():.2f}°")
        print(f"   Longitude range: {result_df['longitude'].min():.2f}° to {result_df['longitude'].max():.2f}°")

    return result_df

def get_top_emotions(emotion_df, n=6):
    """Get top N emotions by occurrence count"""
    if emotion_df.empty:
        return []

    emotion_counts = emotion_df['emotion'].value_counts().head(n)
    return emotion_counts.index.tolist()

def create_emotion_heatmap(emotion_df, emotion_name):
    """Create KDE heatmap with FIXED Plotly syntax"""
    emotion_subset = emotion_df[emotion_df['emotion'] == emotion_name].copy()

    print(f"\n🎨 Generating heatmap for {emotion_name}:")
    print(f"   Total occurrences: {len(emotion_subset)}")

    if len(emotion_subset) < 3:
        print(f"   ⚠️ Insufficient data (need 3+)")
        return create_insufficient_data_figure(emotion_name, len(emotion_subset))

    # Extract data
    lons = emotion_subset['longitude'].values
    lats = emotion_subset['latitude'].values
    intensities = emotion_subset['intensity'].values
    locations = emotion_subset['location'].values

    # Check for coordinate variation
    lat_unique = len(np.unique(lats))
    lon_unique = len(np.unique(lons))

    print(f"   Unique latitudes: {lat_unique}")
    print(f"   Unique longitudes: {lon_unique}")

    if lat_unique < 2 or lon_unique < 2:
        print(f"   ⚠️ Insufficient spatial variation for KDE")
        return create_scatter_map(emotion_subset, emotion_name)

    # Create KDE with error handling
    try:
        print(f"   Attempting KDE calculation...")

        # Add small jitter to identical points
        lons_jittered = lons + np.random.normal(0, 0.001, len(lons))
        lats_jittered = lats + np.random.normal(0, 0.001, len(lats))

        positions = np.vstack([lons_jittered, lats_jittered])

        # Use bw_method='scott' for automatic bandwidth
        kernel = gaussian_kde(positions, bw_method='scott', weights=intensities)

        # Create grid
        lon_min, lon_max = lons.min(), lons.max()
        lat_min, lat_max = lats.min(), lats.max()

        # Ensure minimum range
        lon_range = lon_max - lon_min
        lat_range = lat_max - lat_min

        if lon_range < 0.01:
            lon_pad = 0.05
        else:
            lon_pad = lon_range * 0.2

        if lat_range < 0.01:
            lat_pad = 0.05
        else:
            lat_pad = lat_range * 0.2

        lon_grid = np.linspace(lon_min - lon_pad, lon_max + lon_pad, 80)
        lat_grid = np.linspace(lat_min - lat_pad, lat_max + lat_pad, 80)

        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        grid_coords = np.vstack([lon_mesh.ravel(), lat_mesh.ravel()])

        density = kernel(grid_coords).reshape(lon_mesh.shape)

        print(f"   ✅ KDE successful!")

        # Get emotion color
        color_hex = Config.EMOTION_COLORS.get(emotion_name, '#808080')

        # Create colorscale with rgba format (FIXED)
        colorscale = [
            [0, hex_to_rgba(color_hex, 0)],
            [0.2, hex_to_rgba(color_hex, 0.2)],
            [0.5, hex_to_rgba(color_hex, 0.5)],
            [0.8, hex_to_rgba(color_hex, 0.8)],
            [1, hex_to_rgba(color_hex, 1)]
        ]

        # Create figure
        fig = go.Figure()

        # Add contour heatmap with FIXED colorbar syntax
        fig.add_trace(go.Contour(
            x=lon_grid,
            y=lat_grid,
            z=density,
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(
                title=dict(text="Density", side="right"),  # FIXED: nested title dict
                len=0.6,
                thickness=12
            ),
            contours=dict(
                coloring='heatmap',
                showlabels=False
            ),
            hovertemplate='Lon: %{x:.3f}<br>Lat: %{y:.3f}<br>Density: %{z:.4f}<extra></extra>'
        ))

        # Add scatter points
        fig.add_trace(go.Scatter(
            x=lons,
            y=lats,
            mode='markers',
            marker=dict(
                size=8,
                color='black',
                symbol='circle',
                line=dict(width=1.5, color='white')
            ),
            text=[f"{loc}<br>Intensity: {i:.2f}" for loc, i in zip(locations, intensities)],
            hovertemplate='<b>%{text}</b><br>Lon: %{x:.4f}<br>Lat: %{y:.4f}<extra></extra>',
            name='Locations',
            showlegend=False
        ))

        # Update layout
        fig.update_layout(
            title=dict(
                text=f"{emotion_name.capitalize()} Intensity Heatmap<br><sub>({len(emotion_subset)} occurrences, KDE method)</sub>",
                font=dict(size=15, color=color_hex, family='Arial Black'),
                x=0.5,
                xanchor='center'
            ),
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            height=450,
            plot_bgcolor='rgba(245,245,245,1)',
            paper_bgcolor='white',
            font=dict(size=10),
            margin=dict(l=50, r=50, t=70, b=50)
        )

        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='lightgray', scaleanchor="x", scaleratio=1)

        return fig

    except Exception as e:
        print(f"   ❌ KDE failed: {str(e)}")
        print(f"   Falling back to scatter map...")
        return create_scatter_map(emotion_subset, emotion_name)

def create_scatter_map(emotion_df, emotion_name):
    """Create scatter map with sized markers as fallback"""

    # Aggregate by location
    location_agg = emotion_df.groupby(['latitude', 'longitude', 'location']).agg({
        'intensity': ['sum', 'count', 'mean']
    }).reset_index()

    location_agg.columns = ['latitude', 'longitude', 'location', 'total_intensity', 'count', 'avg_intensity']

    color_hex = Config.EMOTION_COLORS.get(emotion_name, '#808080')

    # Create colorscale with rgba
    colorscale = [[0, hex_to_rgba(color_hex, 0.3)], [1, hex_to_rgba(color_hex, 1)]]

    fig = go.Figure()

    fig.add_trace(go.Scattermapbox(
        lon=location_agg['longitude'],
        lat=location_agg['latitude'],
        mode='markers',
        marker=dict(
            size=location_agg['count'] * 5,
            color=location_agg['avg_intensity'],
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title=dict(text="Avg Intensity")),
            sizemode='diameter'
        ),
        text=location_agg['location'],
        hovertemplate='<b>%{text}</b><br>Count: ' + location_agg['count'].astype(str) + 
                     '<br>Avg Intensity: %{marker.color:.2f}<extra></extra>'
    ))

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(
            center=dict(lat=location_agg['latitude'].mean(), lon=location_agg['longitude'].mean()),
            zoom=8
        ),
        title=f"{emotion_name.capitalize()} Intensity Map<br><sub>({len(emotion_df)} occurrences)</sub>",
        height=450,
        margin=dict(l=0, r=0, t=50, b=0)
    )

    return fig

def create_insufficient_data_figure(emotion_name, count):
    """Create placeholder figure for insufficient data"""
    color = Config.EMOTION_COLORS.get(emotion_name, '#808080')

    fig = go.Figure()

    fig.add_annotation(
        text=f"Insufficient data for {emotion_name.capitalize()}<br><br>{count} record{'s' if count != 1 else ''}<br><br>Minimum 3 occurrences required",
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14, color='gray'),
        align='center'
    )

    fig.update_layout(
        title=f"{emotion_name.capitalize()} Intensity Heatmap",
        height=450,
        plot_bgcolor='rgba(250,250,250,1)',
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False)
    )

    return fig

# ==================== LOAD DATA ====================
print("\n🔍 Loading and processing data...\n")

emotion_df = load_emotion_data()

if emotion_df is not None:
    spatial_emotion_df = extract_emotion_spatial_data(emotion_df)
else:
    spatial_emotion_df = pd.DataFrame()

top_emotions = get_top_emotions(spatial_emotion_df, n=6)

print("\n" + "="*60)
print("📊 DASHBOARD READY")
print("="*60)
print(f"Emotion records: {len(spatial_emotion_df)}")
print(f"Top 6 emotions: {top_emotions}")
print("="*60 + "\n")

# ==================== DASH APP ====================
app = dash.Dash(__name__)
app.title = "Emotion Intensity Heatmap Dashboard"

# ==================== LAYOUT ====================
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("SANDAKAN-RANAU DEATH MARCHES", style={
            'textAlign': 'center', 'color': '#FFD700', 'fontFamily': 'Courier New', 
            'textShadow': '2px 2px 4px rgba(0,0,0,0.5)', 'margin': '0', 'fontSize': '2.5em'
        }),
        html.H3("Emotion Intensity Analysis - Inside Out Colour Scheme (Ekman & Keltner, 2015)", style={
            'textAlign': 'center', 'color': '#D7CCC8', 'fontFamily': 'Arial', 
            'fontWeight': '400', 'margin': '10px 0', 'fontSize': '1.3em'
        }),
        html.P("Spatial-Emotional Pattern Analysis Using Kernel Density Estimation", style={
            'textAlign': 'center', 'color': '#C3B091', 'fontStyle': 'italic', 'margin': '5px 0'
        })
    ], style={
        'background': 'linear-gradient(135deg, #3E2723 0%, #5D4E37 100%)',
        'padding': '30px', 'borderRadius': '10px', 'border': '3px solid #8B7355',
        'marginBottom': '30px', 'boxShadow': '0 4px 12px rgba(0,0,0,0.3)'
    }),

    # Statistics
    html.Div([
        html.H4("📊 Dataset Statistics", style={'color': '#3E2723', 'marginBottom': '15px'}),
        html.Div([
            html.Div([
                html.Div("Emotion Records", style={'fontSize': '0.9em', 'color': '#666'}),
                html.Div(f"{len(spatial_emotion_df)}", style={'fontSize': '2em', 'fontWeight': 'bold', 'color': '#8B0000'})
            ], style={'flex': '1', 'textAlign': 'center', 'padding': '15px', 'background': '#f5f5f5', 'borderRadius': '8px', 'margin': '5px'}),

            html.Div([
                html.Div("Unique Locations", style={'fontSize': '0.9em', 'color': '#666'}),
                html.Div(f"{spatial_emotion_df['location'].nunique()}" if not spatial_emotion_df.empty else "0", 
                        style={'fontSize': '2em', 'fontWeight': 'bold', 'color': '#4A5D3F'})
            ], style={'flex': '1', 'textAlign': 'center', 'padding': '15px', 'background': '#f5f5f5', 'borderRadius': '8px', 'margin': '5px'}),

            html.Div([
                html.Div("Emotion Types", style={'fontSize': '0.9em', 'color': '#666'}),
                html.Div(f"{spatial_emotion_df['emotion'].nunique()}" if not spatial_emotion_df.empty else "0", 
                        style={'fontSize': '2em', 'fontWeight': 'bold', 'color': '#9D27CD'})
            ], style={'flex': '1', 'textAlign': 'center', 'padding': '15px', 'background': '#f5f5f5', 'borderRadius': '8px', 'margin': '5px'}),
        ], style={'display': 'flex', 'justifyContent': 'space-around'})
    ], style={'background': 'white', 'padding': '20px', 'borderRadius': '8px', 'boxShadow': '0 2px 8px rgba(0,0,0,0.1)', 'marginBottom': '30px'}),

    # Dropdown
    html.Div([
        html.Label("🎯 Select Emotion:", style={'fontSize': '1.1em', 'fontWeight': 'bold', 'marginBottom': '10px'}),
        dcc.Dropdown(
            id='emotion-selector',
            options=[{'label': f"{e.capitalize()} ({spatial_emotion_df[spatial_emotion_df['emotion']==e].shape[0]} occurrences)", 
                     'value': e} for e in top_emotions] if top_emotions else [],
            value=top_emotions[0] if top_emotions else None,
            style={'marginBottom': '20px'}
        )
    ], style={'marginBottom': '30px'}),

    # Single heatmap
    html.Div([dcc.Graph(id='emotion-heatmap', config={'displayModeBar': True, 'displaylogo': False})]),

    # Top 6 grid (2 rows x 3 columns)
    html.Div([
        html.H3("Top 6 Emotion Intensity Heatmaps", style={'color': '#3E2723', 'marginTop': '40px', 'marginBottom': '20px'}),
        html.Div([
            html.Div([dcc.Graph(figure=create_emotion_heatmap(spatial_emotion_df, emotion), config={'displayModeBar': True, 'displaylogo': False})], 
                    style={'width': '31%', 'display': 'inline-block', 'margin': '1%', 'verticalAlign': 'top'})
            for emotion in top_emotions
        ])
    ]) if top_emotions else html.Div("No data", style={'textAlign': 'center', 'color': 'red', 'padding': '50px'}),

    # Methodology
    html.Div([
        html.H4("ℹ️ Methodology", style={'color': '#3E2723', 'marginBottom': '15px'}),
        html.P("Gaussian Kernel Density Estimation (KDE) with automatic bandwidth selection (Scott's rule). Fallback to scatter maps for insufficient spatial variation."),
        html.P([html.Strong("Color Scheme: "), "Inside Out emotion colors based on Ekman's basic emotions model."]),
        html.P([html.Strong("Reference: "), "Ekman, P., & Keltner, D. (2015). Universal facial expressions of emotion. ",
                html.Em("California Mental Health Research Digest, 8(4), 151-158.")])
    ], style={'background': '#FFF8DC', 'padding': '20px', 'borderRadius': '8px', 'border': '2px solid #DEB887', 'marginTop': '40px'}),

], style={'maxWidth': '1400px', 'margin': '0 auto', 'padding': '30px', 'fontFamily': 'Arial', 'backgroundColor': '#fafafa'})

# ==================== CALLBACKS ====================
@callback(Output('emotion-heatmap', 'figure'), Input('emotion-selector', 'value'))
def update_heatmap(selected_emotion):
    if selected_emotion is None or spatial_emotion_df.empty:
        return go.Figure()
    return create_emotion_heatmap(spatial_emotion_df, selected_emotion)

# ==================== RUN ====================
if __name__ == '__main__':
    import webbrowser
    from threading import Timer
    Timer(1.5, lambda: webbrowser.open_new('http://localhost:8050/')).start()
    app.run(debug=True, port=8050, use_reloader=False)
