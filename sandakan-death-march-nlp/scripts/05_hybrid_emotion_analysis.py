#!/usr/bin/env python3
"""
STEP 5: HYBRID EMOTION ANALYSIS - BERT + LEXICON
Sandakan-Ranau Death Marches - Advanced Emotion Detection

ENHANCEMENTS:
- BERT transformer model for contextual emotion understanding
- Traditional lexicon-based analysis for transparency
- Hybrid fusion strategy (Lexicon + Transformer)
- FIX: Calculates 'Dominant Emotion' for direct GIS mapping
- FIX: Exports directly to Shapefile (.shp) for ArcGIS Pro compatibility

Author: Your Name
Date: February 2026
"""

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for emotion analysis"""
    
    # Base paths - UPDATE IF NEEDED
    BASE_DIR = Path(r"your_project_directory")  # <-- Change this to your project directory
    
    # Input from Step 4 (Ensure this points to your Step 4 output)
    STEP4_OUTPUT = BASE_DIR / "outputs" / "step4_ner" 
    GEO_ENTITIES_FILE = STEP4_OUTPUT / "geographic_entities.csv"
    
    # Output directories
    OUTPUT_DIR = BASE_DIR / "outputs" / "step5_emotion_analysis_hybrid_v2"
    EMOTION_DATA_DIR = OUTPUT_DIR / "emotion_data"
    VIZ_DIR = OUTPUT_DIR / "visualizations"
    GIS_DIR = OUTPUT_DIR / "gis_exports"
    REPORTS_DIR = OUTPUT_DIR / "reports"
    
    # Lexicon paths
    LEXICON_DIR = BASE_DIR / "data" / "lexicons"
    NRC_LEXICON = LEXICON_DIR / "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
    
    # Emotion categories
    EMOTION_CATEGORIES = [
        'fear', 'anger', 'sadness', 'disgust', 
        'surprise', 'anticipation', 'trust', 'joy'
    ]
    
    SENTIMENT_CATEGORIES = ['positive', 'negative', 'neutral']
    
    # BERT model for emotion detection
    # Using DistilRoBERTa for speed/accuracy balance on historical text
    BERT_MODEL = "j-hartmann/emotion-english-distilroberta-base" 
    BERT_BATCH_SIZE = 16 


# ============================================================================
# EMOTION LEXICONS (Traditional Approach)
# ============================================================================

class EmotionLexicon:
    """Load and manage emotion lexicons"""
    
    def __init__(self):
        self.nrc_lexicon = {}
        self.historical_emotion_words = {}
        self.load_lexicons()
    
    def load_lexicons(self):
        """Load emotion lexicons"""
        
        if Config.NRC_LEXICON.exists():
            self.load_nrc_lexicon()
        else:
            print(f"⚠ NRC lexicon not found at: {Config.NRC_LEXICON}")
            print("  Proceeding with historical lexicon only.")
        
        self.load_historical_emotions()
    
    def load_nrc_lexicon(self):
        """Load NRC Emotion Lexicon"""
        try:
            with open(Config.NRC_LEXICON, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 3:
                        word, emotion, association = parts
                        if int(association) == 1:
                            if word not in self.nrc_lexicon:
                                self.nrc_lexicon[word] = []
                            self.nrc_lexicon[word].append(emotion)
            print(f"✓ Loaded NRC lexicon: {len(self.nrc_lexicon)} words")
        except Exception as e:
            print(f"⚠ Error loading NRC lexicon: {e}")
    
    def load_historical_emotions(self):
        """Load custom historical war-related emotion words (Domain Specific)"""
        
        self.historical_emotion_words = {
            'suffering': ['pain', 'suffering', 'agony', 'torture', 'torment', 'misery', 
                          'hardship', 'ordeal', 'affliction', 'anguish'],
            'death': ['death', 'dying', 'died', 'dead', 'killed', 'murder', 'execution',
                      'massacre', 'perished', 'casualties', 'fatalities', 'burial', 'bayoneted'],
            'fear': ['fear', 'afraid', 'terror', 'dread', 'panic', 'frightened', 
                     'scared', 'horrified', 'terrified', 'anxiety'],
            'despair': ['despair', 'hopeless', 'desperate', 'helpless', 'desperation',
                        'forlorn', 'despondent', 'dejected'],
            'exhaustion': ['exhausted', 'tired', 'weary', 'fatigue', 'worn', 'drained',
                           'depleted', 'spent', 'weakness'],
            'hunger': ['hunger', 'hungry', 'starvation', 'starving', 'famished', 
                       'malnourished', 'food', 'rations'],
            'disgust': ['disease', 'illness', 'sick', 'sickness', 'malaria', 'dysentery',
                        'infection', 'epidemic', 'fever', 'filth', 'foul', 'rotting'],
            'cruelty': ['cruel', 'cruelty', 'brutal', 'brutality', 'savage', 'barbaric',
                        'inhumane', 'atrocity', 'violence'],
            'joy': ['courage', 'brave', 'bravery', 'heroic', 'valor', 'gallant',
                        'fortitude', 'resilience', 'endurance', 'relief', 'liberation', 'rescued'],
            'sadness': ['grief', 'sorrow', 'mourning', 'bereavement', 'lament', 'weep',
                       'crying', 'tears', 'tragic'],
            'trauma': ['trauma', 'traumatic', 'shock', 'horror', 'nightmare', 'haunted',
                       'memories']
        }
        
        print(f"✓ Loaded historical emotion words: {len(self.historical_emotion_words)} categories")
    
    def get_emotions(self, word):
        """Get emotions associated with a word"""
        emotions = []
        word_lower = word.lower()
        
        if word_lower in self.nrc_lexicon:
            emotions.extend(self.nrc_lexicon[word_lower])
        
        for emotion, words in self.historical_emotion_words.items():
            if word_lower in words:
                emotions.append(emotion)
        
        return list(set(emotions))


# ============================================================================
# BERT EMOTION ANALYZER (Transformer Approach)
# ============================================================================

class BERTEmotionAnalyzer:
    """BERT-based emotion classification using DistilRoBERTa"""
    
    def __init__(self, model_name: str = Config.BERT_MODEL):
        """Initialize BERT emotion classifier"""
        
        print(f"\n🤖 Initializing BERT emotion classifier...")
        print(f"   Model: {model_name}")
        
        try:
            # Check for GPU
            device = 0 if torch.cuda.is_available() else -1
            device_name = "GPU" if device == 0 else "CPU"
            print(f"   Device: {device_name}")
            
            # Load pre-trained emotion classification model
            self.classifier = pipeline(
                "text-classification",
                model=model_name,
                top_k=None,  # Return all emotion scores
                device=device
            )
            
            print(f"✓ BERT emotion classifier loaded successfully")
            
        except Exception as e:
            print(f"⚠ Error loading BERT model: {e}")
            print("Falling back to lexicon-only analysis")
            self.classifier = None
    
    def analyze_emotion(self, text: str):
        """Analyze emotions in text using BERT"""
        
        if not self.classifier or not text or len(text.strip()) < 5:
            return {}
        
        try:
            # Truncate if too long (BERT limit is usually 512 tokens)
            if len(text) > 500:
                text = text[:500]
            
            # Get emotion predictions
            results = self.classifier(text)[0]
            
            # Convert to dictionary
            emotions = {}
            for result in results:
                emotion = result['label'].lower()
                score = result['score']
                emotions[emotion] = score
            
            return emotions
            
        except Exception as e:
            return {}


# ============================================================================
# HYBRID EMOTION ANALYZER
# ============================================================================

class HybridEmotionAnalyzer:
    """Combines BERT and lexicon-based approaches"""
    
    def __init__(self):
        self.lexicon = EmotionLexicon()
        self.bert = BERTEmotionAnalyzer()
        self.stop_words = set(stopwords.words('english'))
    
    def analyze_lexicon(self, context_text):
        """Lexicon-based emotion analysis"""
        
        if pd.isna(context_text) or not context_text:
            return {
                'emotions': [],
                'emotion_words': [],
                'sentiment_score': 0.0,
                'sentiment_label': 'neutral',
                'emotion_intensity': 0.0
            }
        
        # Tokenize
        try:
            words = word_tokenize(str(context_text).lower())
        except:
            words = str(context_text).lower().split()
        
        words = [w for w in words if w.isalpha() and w not in self.stop_words]
        
        # Detect emotions
        emotions = []
        emotion_words = []
        
        for word in words:
            word_emotions = self.lexicon.get_emotions(word)
            if word_emotions:
                emotions.extend(word_emotions)
                emotion_words.append((word, word_emotions))
        
        # Sentiment analysis using TextBlob
        try:
            blob = TextBlob(str(context_text))
            sentiment_score = blob.sentiment.polarity
        except:
            sentiment_score = 0.0
        
        if sentiment_score > 0.1:
            sentiment_label = 'positive'
        elif sentiment_score < -0.1:
            sentiment_label = 'negative'
        else:
            sentiment_label = 'neutral'
        
        emotion_intensity = len(emotions) / max(len(words), 1)
        
        return {
            'emotions': list(set(emotions)),
            'emotion_words': emotion_words,
            'sentiment_score': sentiment_score,
            'sentiment_label': sentiment_label,
            'emotion_intensity': emotion_intensity
        }
    
    def analyze_entity(self, entity_row):
        """Analyze emotions for a geographic entity using HYBRID approach"""
        
        # Combine all context
        contexts = []
        for col in ['before_context', 'after_context', 'sentence']:
            if col in entity_row and pd.notna(entity_row[col]):
                contexts.append(str(entity_row[col]))
        
        full_context = ' '.join(contexts)
        
        # 1. Lexicon-based analysis
        lexicon_result = self.analyze_lexicon(full_context)
        
        # 2. BERT-based analysis
        bert_emotions = self.bert.analyze_emotion(full_context)
        
        # 3. Combine results (Hybrid Fusion Strategy)
        # Filter BERT emotions with high confidence
        bert_top_emotions = []
        if bert_emotions:
            # Threshold: 0.3 (adjusted for historical subtlety)
            bert_top_emotions = [
                emotion for emotion, score in bert_emotions.items() 
                if score > 0.3
            ]
        
        # Combine: union of both approaches
        combined_emotions = list(set(lexicon_result['emotions'] + bert_top_emotions))
        
        # Determine primary method used
        analysis_method = "hybrid"
        if not bert_emotions:
            analysis_method = "lexicon_only"
        elif not lexicon_result['emotions']:
            analysis_method = "bert_only"
        
        return {
            **lexicon_result,
            'bert_emotions': bert_emotions,
            'bert_top_emotions': bert_top_emotions,
            'combined_emotions': combined_emotions,
            'analysis_method': analysis_method
        }


# ============================================================================
# LOCATION AGGREGATION (ENHANCED FOR GIS)
# ============================================================================

def aggregate_location_emotions(emotion_df):
    """Aggregate emotions by location and calculate Dominant Emotion"""
    
    location_groups = emotion_df.groupby('entity_text')
    
    aggregated = []
    
    for location, group in location_groups:
        
        # Collect combined emotions (hybrid approach)
        all_emotions = []
        for emotions in group['combined_emotions']:
            if isinstance(emotions, list):
                all_emotions.extend(emotions)
        
        emotion_counts = Counter(all_emotions)
        
        # --- DOMINANT EMOTION LOGIC (CRITICAL FOR MAPPING) ---
        dominant_emotion = "neutral"
        max_count = 0
        
        if emotion_counts:
            # Get the most common emotion
            most_common = emotion_counts.most_common(1)
            if most_common:
                dominant_emotion = most_common[0][0]
                max_count = most_common[0][1]
        # -----------------------------------------------------

        # Average sentiment
        avg_sentiment = group['sentiment_score'].mean()
        
        # Most common sentiment label
        sentiment_mode = group['sentiment_label'].mode()
        sentiment_label = sentiment_mode[0] if len(sentiment_mode) > 0 else 'neutral'
        
        # Average emotion intensity
        avg_intensity = group['emotion_intensity'].mean()
        
        # Coordinates (Check for valid float conversion)
        lat = group['latitude'].iloc[0] if 'latitude' in group.columns else None
        lon = group['longitude'].iloc[0] if 'longitude' in group.columns else None
        
        # Build record
        record = {
            'location_name': location,
            'mention_count': len(group),
            'avg_sentiment_score': round(avg_sentiment, 4),
            'sentiment_label': sentiment_label,
            'dominant_emotion': dominant_emotion,  # Used for Unique Value Symbology
            'dominant_emotion_count': max_count,
            'avg_emotion_intensity': round(avg_intensity, 4),
            'latitude': lat,
            'longitude': lon
        }
        
        # Add individual emotion counts (for Heatmap generation)
        std_emotions = ['fear', 'anger', 'sadness', 'disgust', 'surprise', 
                       'anticipation', 'trust', 'joy']
                       
        for emotion in std_emotions:
            record[f'emotion_{emotion}'] = emotion_counts.get(emotion, 0)
        
        aggregated.append(record)
    
    return pd.DataFrame(aggregated)


# ============================================================================
# VISUALIZATION & EXPORT
# ============================================================================

class EmotionVisualizer:
    """Create visualizations for emotion analysis"""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def plot_emotion_comparison(self, emotion_df):
        """Compare lexicon vs BERT vs hybrid emotions"""
        
        # Count emotions by method
        lexicon_emotions = []
        bert_emotions = []
        hybrid_emotions = []
        
        for _, row in emotion_df.iterrows():
            if isinstance(row.get('emotions'), list):
                lexicon_emotions.extend(row['emotions'])
            
            if isinstance(row.get('bert_top_emotions'), list):
                bert_emotions.extend(row['bert_top_emotions'])
            
            if isinstance(row.get('combined_emotions'), list):
                hybrid_emotions.extend(row['combined_emotions'])
        
        lexicon_counts = Counter(lexicon_emotions).most_common(10)
        bert_counts = Counter(bert_emotions).most_common(10)
        hybrid_counts = Counter(hybrid_emotions).most_common(10)
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for ax, counts, title, color in zip(
            axes,
            [lexicon_counts, bert_counts, hybrid_counts],
            ['Lexicon-Based', 'BERT Transformer', 'Hybrid Approach'],
            ['steelblue', 'coral', 'green']
        ):
            if counts:
                emotions, values = zip(*counts)
                ax.barh(emotions, values, color=color, alpha=0.7)
                ax.set_xlabel('Frequency')
                ax.set_title(f'{title}\n({sum(values)} total)', fontweight='bold')
                ax.invert_yaxis()
        
        plt.tight_layout()
        output_file = self.output_dir / "emotion_method_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Emotion comparison chart saved: {output_file}")


def export_to_shapefile(df, output_path):
    """Export DataFrame to ESRI Shapefile (GIS Ready)"""
    
    # Force coordinates to numeric
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    
    # Filter valid coordinates
    valid_df = df.dropna(subset=['latitude', 'longitude'])
    valid_df = valid_df[valid_df['longitude'] != 0] # Remove null island
    
    if len(valid_df) == 0:
        print("⚠ No valid coordinates for Shapefile export.")
        return

    # Create Geometry (Longitude = X, Latitude = Y)
    geometry = [Point(xy) for xy in zip(valid_df['longitude'], valid_df['latitude'])]
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(valid_df, geometry=geometry, crs="EPSG:4326")
    
    # Convert object columns to strings (Shapefile limitation)
    for col in gdf.columns:
        if gdf[col].dtype == 'object':
            gdf[col] = gdf[col].astype(str)
            
    # Save
    try:
        gdf.to_file(str(output_path), driver='ESRI Shapefile')
        print(f"✓ Shapefile exported successfully: {output_path}")
    except Exception as e:
        print(f"❌ Shapefile export failed: {e}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main execution function"""
    
    print("\n" + "=" * 80)
    print("STEP 5: HYBRID EMOTION ANALYSIS (BERT + LEXICON)")
    print("Sandakan-Ranau Death Marches - Advanced Emotion Detection")
    print("=" * 80 + "\n")
    
    # Create output directories
    for directory in [Config.OUTPUT_DIR, Config.EMOTION_DATA_DIR, Config.VIZ_DIR,
                      Config.GIS_DIR, Config.REPORTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzers
    print("🔬 Initializing hybrid emotion analyzer...")
    analyzer = HybridEmotionAnalyzer()
    
    # Load geographic entities from Step 4
    print("\n📂 Loading geographic entities from Step 4...")
    if not Config.GEO_ENTITIES_FILE.exists():
        print(f"❌ Error: Geographic entities file not found: {Config.GEO_ENTITIES_FILE}")
        print("Please run Step 4 (NER) first!")
        return
    
    geo_entities = pd.read_csv(Config.GEO_ENTITIES_FILE)
    print(f"✓ Loaded {len(geo_entities)} geographic entities\n")
    
    # Analyze emotions
    print("🔍 Analyzing emotions with HYBRID approach (BERT + Lexicon)...")
    print("NOTE: This may take several minutes for BERT processing...")
    
    results = []
    # Using tqdm for progress bar
    for idx, row in tqdm(geo_entities.iterrows(), total=len(geo_entities), desc="Analyzing entities"):
        analysis = analyzer.analyze_entity(row)
        result_row = row.to_dict()
        result_row.update(analysis)
        results.append(result_row)
    
    emotion_df = pd.DataFrame(results)
    
    # Save detailed raw results
    output_file = Config.EMOTION_DATA_DIR / "hybrid_emotion_analysis_raw.csv"
    emotion_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n✓ Detailed emotion analysis saved: {output_file}")
    
    # Aggregate by location (Calculation of Dominant Emotion happens here)
    print("📊 Aggregating emotions by location...")
    location_emotions = aggregate_location_emotions(emotion_df)
    
    # Save aggregated CSV
    csv_output = Config.EMOTION_DATA_DIR / "location_emotions_hybrid.csv"
    location_emotions.to_csv(csv_output, index=False, encoding='utf-8')
    print(f"✓ Location emotions (CSV) saved: {csv_output}")
    
    # EXPORT TO GIS (SHAPEFILE)
    print("🗺️ Exporting to GIS Shapefile...")
    shp_output = Config.GIS_DIR / "emotion_hotspots.shp"
    export_to_shapefile(location_emotions, shp_output)
    
    # Create visualizations
    print("🎨 Creating visualizations...")
    visualizer = EmotionVisualizer(Config.VIZ_DIR)
    try:
        visualizer.plot_emotion_comparison(emotion_df)
    except Exception as e:
        print(f"⚠ Warning: Could not create visualization: {e}")
    
    # Generate summary statistics
    print("\n📊 Generating summary statistics...")
    stats = {
        'total_entities': len(emotion_df),
        'unique_locations': emotion_df['entity_text'].nunique(),
        'analysis_methods': emotion_df['analysis_method'].value_counts().to_dict(),
        'hybrid_coverage_improvement': "Calculated in chart"
    }
    
    stats_file = Config.REPORTS_DIR / "hybrid_emotion_statistics.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    

if __name__ == "__main__":
    main()