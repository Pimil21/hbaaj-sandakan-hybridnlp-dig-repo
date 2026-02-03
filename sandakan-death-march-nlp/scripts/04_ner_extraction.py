#!/usr/bin/env python3
"""
STEP 4: ENHANCED NAMED ENTITY RECOGNITION WITH TRANSFORMERS
Optimized for Sandakan-Ranau Death Marches Historical Documents

ENHANCEMENTS:
- Using spaCy Transformer model (RoBERTa-based) for better context understanding
- Automatic fallback to standard model if build tools unavailable
- Custom historical entity patterns
- POI database integration for validation
- Comprehensive validation metrics
- Export to GIS formats (GeoJSON + Shapefile)
- FIX: Forces numeric coordinates to prevent ArcGIS display errors

Author: Your Name
Date: January 2026
"""

# ============================================================================
# CONFIGURATION
# ============================================================================

class NERConfig:
    """Configuration for NER processing"""
    
    # Paths - UPDATE THESE FOR YOUR SYSTEM
    BASE_DIR = Path(r"your_project_directory_here")  # <-- UPDATE THIS PATH
    PREPROCESSED_DATA_DIR = BASE_DIR / r"outputs" / "step3_preprocessed"
    POI_DATABASE_PATH = Path(r"\data\poi_locations.csv")
    OUTPUT_DIR = BASE_DIR / "outputs" / "step4_ner"
    
    # spaCy model - UPGRADED TO TRANSFORMER (with fallback)
    SPACY_MODEL = "en_core_web_trf"  # Transformer model (RoBERTa-based) for better accuracy
    FALLBACK_MODEL = "en_core_web_md"  # Fallback if transformer fails
    
    # Processing parameters
    BATCH_SIZE = 50  # Reduced for transformer (more memory intensive)
    MIN_SENTENCE_LENGTH = 10
    
    # Custom entity thresholds
    MIN_ENTITY_CONFIDENCE = 0.5
    MIN_LOCATION_OCCURRENCE = 2
    
    # Output formats
    OUTPUT_FORMATS = ['csv', 'json', 'geojson', 'shapefile']
    
    @classmethod
    def setup_paths(cls):
        """Create output directories"""
        Path(cls.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        (Path(cls.OUTPUT_DIR) / "visualizations").mkdir(parents=True, exist_ok=True)
        (Path(cls.OUTPUT_DIR) / "validation").mkdir(parents=True, exist_ok=True)
        (Path(cls.OUTPUT_DIR) / "gis_exports").mkdir(parents=True, exist_ok=True)
        print(f"✓ Output directory created: {cls.OUTPUT_DIR}")

# Initialize paths
NERConfig.setup_paths()


# ============================================================================
# POI DATABASE HANDLER
# ============================================================================

class POIDatabase:
    """Load and manage historical POI database for validation"""
    
    def __init__(self, poi_path: str):
        self.poi_df = pd.read_csv(poi_path)
        
        # --- CRITICAL FIX FOR ARCGIS ---
        # Force latitude and longitude to be numeric (floats).
        # 'coerce' turns invalid strings (like empty spaces) into NaN so we can drop them.
        self.poi_df['latitude'] = pd.to_numeric(self.poi_df['latitude'], errors='coerce')
        self.poi_df['longitude'] = pd.to_numeric(self.poi_df['longitude'], errors='coerce')
        
        # Drop rows that don't have valid coordinates
        initial_len = len(self.poi_df)
        self.poi_df = self.poi_df.dropna(subset=['latitude', 'longitude'])
        cleaned_len = len(self.poi_df)
        
        if initial_len != cleaned_len:
            print(f"⚠ Dropped {initial_len - cleaned_len} POIs due to invalid coordinates")
        
        self.location_mapping = {}
        self.alternate_names = {}
        print(f"✓ Loaded {len(self.poi_df)} valid historical POIs")
        self.build_location_mapping()
    
    def build_location_mapping(self):
        """Build comprehensive location mapping with alternate names"""
        
        for _, row in self.poi_df.iterrows():
            poi_name = str(row['poi_name']).strip()
            
            # Add primary name
            self.location_mapping[poi_name.lower()] = {
                'original': poi_name,
                'poi_id': row['poi_id'],
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'march_id': row['march_id'],
                'march_name': row['march_name'],
                'poi_detail': row['poi_detail']
            }
            
            # Extract alternate names from poi_detail
            if pd.notna(row['poi_detail']):
                detail = str(row['poi_detail'])
                
                # Find alternate names in parentheses
                alt_names = re.findall(r'\((.*?)\)', detail)
                for alt in alt_names:
                    if ',' in alt:
                        # Multiple names separated by commas
                        for name in alt.split(','):
                            clean_name = name.strip().lower()
                            if clean_name and len(clean_name) > 2:
                                self.alternate_names[clean_name] = poi_name.lower()
                    else:
                        clean_name = alt.strip().lower()
                        if clean_name and len(clean_name) > 2:
                            self.alternate_names[clean_name] = poi_name.lower()
        
        # Add known alternate names
        known_alternates = {
            'Jesselton': 'Kota Kinabalu',
            'British North Borneo': 'Sabah',
            'Sandakan No. 1 Camp': 'Sandakan Camp 1',
            'Sandakan No. 2 Camp': 'Sandakan Camp 2',
            'Ranau No. 1 Camp': 'Ranau Camp 1',
            'Ranau No. 2 Camp': 'Ranau Camp 2'
        }
        
        for alt, primary in known_alternates.items():
            if alt.lower() not in self.location_mapping:
                self.alternate_names[alt.lower()] = primary.lower()
        
        print(f"  - Primary locations: {len(self.location_mapping)}")
        print(f"  - Alternate names: {len(self.alternate_names)}")
    
    def is_historical_location(self, location_name: str) -> Tuple[bool, Dict]:
        """Check if location exists in historical database"""
        
        location_lower = location_name.lower().strip()
        
        # Check primary names
        if location_lower in self.location_mapping:
            return True, self.location_mapping[location_lower]
        
        # Check alternate names
        if location_lower in self.alternate_names:
            primary_name = self.alternate_names[location_lower]
            return True, self.location_mapping.get(primary_name, {})
        
        # Check partial matches (for locations like "Mile 8 POW Camp")
        for primary_name, data in self.location_mapping.items():
            if location_lower in primary_name or primary_name in location_lower:
                return True, data
        
        return False, {}
    
    def get_all_locations(self) -> List[str]:
        """Get all location names (primary and alternate)"""
        all_locations = list(self.location_mapping.keys())
        all_locations.extend(list(self.alternate_names.keys()))
        return all_locations
    
    def get_march_locations(self, march_id: int) -> List[Dict]:
        """Get locations for a specific march"""
        march_locations = []
        for loc_data in self.location_mapping.values():
            if loc_data.get('march_id') == march_id:
                march_locations.append(loc_data)
        return march_locations


# ============================================================================
# TRANSFORMER-BASED NER EXTRACTOR
# ============================================================================

class TransformerNERExtractor:
    """Advanced NER extractor using Transformer model (RoBERTa)
    
    Optimized for historical Death Marches documents
    Integrates with POI database for validation
    """
    
    def __init__(self, model_name: str = "en_core_web_trf", poi_database: Optional[POIDatabase] = None):
        """Initialize NER extractor with spaCy Transformer model"""
        
        print(f"\n🤖 Initializing Transformer NER extractor with model: {model_name}")
        
        # Try to load Transformer model with fallback
        model_loaded = False
        actual_model = model_name
        
        try:
            self.nlp = spacy.load(model_name)
            print(f"✓ Loaded spaCy Transformer model: {model_name}")
            model_loaded = True
        except OSError:
            print(f"⚠ Model '{model_name}' not found. Attempting to download...")
            print("NOTE: Transformer models require 'spacy-transformers' and Microsoft C++ Build Tools")
            
            import subprocess
            try:
                result = subprocess.run(['python', '-m', 'spacy', 'download', model_name], 
                                      capture_output=True, text=True, check=True)
                self.nlp = spacy.load(model_name)
                print(f"✓ Model downloaded and loaded: {model_name}")
                model_loaded = True
            except Exception as e:
                print(f"⚠ Could not download transformer model")
                print(f"  Reason: Build tools required for Windows")
                print(f"  Falling back to standard model: {NERConfig.FALLBACK_MODEL}")
                
                try:
                    self.nlp = spacy.load(NERConfig.FALLBACK_MODEL)
                    actual_model = NERConfig.FALLBACK_MODEL
                    model_loaded = True
                except OSError:
                    subprocess.run(['python', '-m', 'spacy', 'download', NERConfig.FALLBACK_MODEL])
                    self.nlp = spacy.load(NERConfig.FALLBACK_MODEL)
                    actual_model = NERConfig.FALLBACK_MODEL
                    model_loaded = True
        
        if not model_loaded:
            raise RuntimeError("Failed to load any spaCy model")
        
        # Initialize matchers
        self.matcher = Matcher(self.nlp.vocab)
        self.phrase_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        
        # Store POI database
        self.poi_db = poi_database
        
        # Add custom patterns
        self.add_custom_patterns()
        
        # Statistics
        self.entity_stats = defaultdict(Counter)
        self.validation_stats = {
            'historical_matches': 0,
            'non_historical': 0
        }
        
        print(f"✓ NER extractor initialized successfully")
        print(f"  Model: {actual_model}")
        print(f"  Model type: {type(self.nlp.pipeline[0][1]).__name__}")
    
    def add_custom_patterns(self):
        """Add custom entity patterns for historical Death Marches"""
        
        # Pattern 1: POW Camps
        camp_patterns = [
            [{"LOWER": {"IN": ["sandakan", "ranau", "telupid"]}}, 
             {"LOWER": {"IN": ["camp", "camps"]}}, 
             {"OP": "?"},
             {"IS_DIGIT": True, "OP": "?"}],
            
            [{"LOWER": "camp"}, {"IS_DIGIT": True}],
            
            [{"LOWER": {"IN": ["pow", "p.o.w.", "prisoner"]}}, 
             {"LOWER": "camp"}]
        ]
        
        for idx, pattern in enumerate(camp_patterns):
            self.matcher.add(f"CAMP_{idx}", [pattern])
        
        # Pattern 2: Military Units
        military_patterns = [
            [{"TEXT": {"REGEX": r"\d+"}},
             {"LOWER": {"IN": ["st", "nd", "rd", "th"]}, "OP": "?"},
             {"LOWER": {"IN": ["battalion", "regiment", "division", "brigade"]}}],
            
            [{"LOWER": {"IN": ["australian", "british", "japanese", "allied"]}},
             {"LOWER": {"IN": ["imperial", "army", "force", "troops"]}, "OP": "+"}]
        ]
        
        for idx, pattern in enumerate(military_patterns):
            self.matcher.add(f"MILITARY_{idx}", [pattern])
        
        # Pattern 3: Distance markers
        distance_patterns = [
            [{"LOWER": "mile"}, {"IS_DIGIT": True}],
            [{"IS_DIGIT": True}, {"LOWER": {"IN": ["mile", "miles", "kilometer", "kilometers", "km"]}}]
        ]
        
        for idx, pattern in enumerate(distance_patterns):
            self.matcher.add(f"DISTANCE_{idx}", [pattern])
        
        # Add POI locations as phrase patterns
        if self.poi_db:
            print(f"📍 Adding {len(self.poi_db.location_mapping)} POIs to phrase matcher...")
            
            poi_patterns = []
            for location_name in self.poi_db.location_mapping.keys():
                if len(location_name) > 2:
                    doc = self.nlp.make_doc(location_name)
                    poi_patterns.append(doc)
            
            if poi_patterns:
                self.phrase_matcher.add("HISTORICAL_POIS", poi_patterns)
                print(f"✓ Added {len(poi_patterns)} POI patterns")
    
    def extract_entities(self, text: str, context_window: int = 50) -> List[Dict]:
        """Extract entities from text with context"""
        
        if not text or len(text) < NERConfig.MIN_SENTENCE_LENGTH:
            return []
        
        # Process with model
        doc = self.nlp(text)
        
        entities = []
        
        # Extract spaCy NER entities
        for ent in doc.ents:
            
            # Get context
            start_char = max(0, ent.start_char - context_window)
            end_char = min(len(text), ent.end_char + context_window)
            before_context = text[start_char:ent.start_char].strip()
            after_context = text[ent.end_char:end_char].strip()
            
            # Validate against POI database
            is_historical = False
            poi_data = {}
            
            if self.poi_db:
                is_historical, poi_data = self.poi_db.is_historical_location(ent.text)
                
                if is_historical:
                    self.validation_stats['historical_matches'] += 1
                else:
                    self.validation_stats['non_historical'] += 1
            
            entity_info = {
                'entity_text': ent.text,
                'entity_type': ent.label_,
                'start_pos': ent.start_char,
                'end_pos': ent.end_char,
                'sentence': text,
                'before_context': before_context,
                'after_context': after_context,
                'is_historical_location': is_historical,
                'confidence_score': 1.0,
                'latitude': poi_data.get('latitude') if is_historical else None,
                'longitude': poi_data.get('longitude') if is_historical else None,
                'march_id': poi_data.get('march_id') if is_historical else None,
                'march_name': poi_data.get('march_name') if is_historical else None
            }
            
            entities.append(entity_info)
            self.entity_stats[ent.label_][ent.text] += 1
        
        # Extract custom pattern matches
        matches = self.matcher(doc)
        for match_id, start, end in matches:
            span = doc[start:end]
            
            if any(e['entity_text'] == span.text for e in entities):
                continue
            
            pattern_name = self.nlp.vocab.strings[match_id]
            
            start_char = span.start_char
            end_char = span.end_char
            before_start = max(0, start_char - context_window)
            after_end = min(len(text), end_char + context_window)
            
            entity_info = {
                'entity_text': span.text,
                'entity_type': f'CUSTOM_{pattern_name.split("_")[0]}',
                'start_pos': start_char,
                'end_pos': end_char,
                'sentence': text,
                'before_context': text[before_start:start_char].strip(),
                'after_context': text[end_char:after_end].strip(),
                'is_historical_location': False,
                'confidence_score': 0.8,
                'latitude': None,
                'longitude': None,
                'march_id': None,
                'march_name': None
            }
            
            entities.append(entity_info)
            self.entity_stats[f'CUSTOM_{pattern_name.split("_")[0]}'][span.text] += 1
        
        # Extract phrase matches (POI locations)
        phrase_matches = self.phrase_matcher(doc)
        for match_id, start, end in phrase_matches:
            span = doc[start:end]
            
            if any(e['entity_text'] == span.text for e in entities):
                continue
            
            is_historical, poi_data = self.poi_db.is_historical_location(span.text)
            
            start_char = span.start_char
            end_char = span.end_char
            before_start = max(0, start_char - context_window)
            after_end = min(len(text), end_char + context_window)
            
            entity_info = {
                'entity_text': span.text,
                'entity_type': 'GPE',
                'start_pos': start_char,
                'end_pos': end_char,
                'sentence': text,
                'before_context': text[before_start:start_char].strip(),
                'after_context': text[end_char:after_end].strip(),
                'is_historical_location': is_historical,
                'confidence_score': 0.95,
                'latitude': poi_data.get('latitude'),
                'longitude': poi_data.get('longitude'),
                'march_id': poi_data.get('march_id'),
                'march_name': poi_data.get('march_name')
            }
            
            entities.append(entity_info)
            self.entity_stats['GPE'][span.text] += 1
            self.validation_stats['historical_matches'] += 1
        
        return entities
    
    def batch_extract_entities(self, sentences: List[str], sentence_metadata: List[Dict] = None, 
                               show_progress: bool = True) -> pd.DataFrame:
        """Extract entities from multiple sentences with batching"""
        
        from tqdm import tqdm
        
        all_entities = []
        
        iterator = tqdm(sentences, desc="Extracting entities") if show_progress else sentences
        
        for idx, sentence in enumerate(iterator):
            entities = self.extract_entities(sentence)
            
            for entity in entities:
                if sentence_metadata and idx < len(sentence_metadata):
                    entity.update(sentence_metadata[idx])
                
                entity['sentence_id'] = idx
                all_entities.append(entity)
        
        df = pd.DataFrame(all_entities)
        
        print(f"\n✓ Extracted {len(df)} entities from {len(sentences)} sentences")
        
        return df
    
    def extract_geographic_entities(self, entity_df: pd.DataFrame) -> pd.DataFrame:
        """Extract and filter geographic entities"""
        
        geographic_types = ['GPE', 'LOC', 'FAC', 'CUSTOM_CAMP', 'CUSTOM_DISTANCE']
        
        geo_df = entity_df[entity_df['entity_type'].isin(geographic_types)].copy()
        
        geo_df['location_name'] = geo_df['entity_text'].str.strip()
        
        location_counts = geo_df.groupby('location_name').agg({
            'entity_text': 'count',
            'is_historical_location': 'first',
            'latitude': 'first',
            'longitude': 'first',
            'march_id': 'first',
            'march_name': 'first'
        }).rename(columns={'entity_text': 'mention_count'})
        
        frequent_locations = location_counts[
            location_counts['mention_count'] >= NERConfig.MIN_LOCATION_OCCURRENCE
        ]
        
        geo_df = geo_df.merge(
            frequent_locations[['mention_count']], 
            left_on='location_name', 
            right_index=True, 
            how='inner'
        )
        
        print(f"✓ Extracted {len(geo_df)} geographic entity mentions")
        print(f"  - Unique locations: {geo_df['location_name'].nunique()}")
        print(f"  - Historical locations: {geo_df['is_historical_location'].sum()}")
        
        return geo_df
    
    def get_statistics(self) -> Dict:
        """Get extraction statistics"""
        
        stats = {}
        
        stats['entity_type_counts'] = {
            etype: sum(counts.values()) for etype, counts in self.entity_stats.items()
        }
        
        stats['unique_entities'] = {
            etype: len(counts) for etype, counts in self.entity_stats.items()
        }
        
        stats['top_entities'] = {
            etype: counts.most_common(10) for etype, counts in self.entity_stats.items()
        }
        
        stats['validation_stats'] = {
            'historical_matches': self.validation_stats['historical_matches'],
            'non_historical': self.validation_stats['non_historical'],
            'validation_rate': self.validation_stats['historical_matches'] / 
                             max(1, self.validation_stats['historical_matches'] + 
                                  self.validation_stats['non_historical'])
        }
        
        return stats


# ============================================================================
# VISUALIZATION
# ============================================================================

class NERVisualizer:
    """Create visualizations for NER results"""
    
    @staticmethod
    def create_entity_distribution_chart(entity_stats: Dict, output_path: str):
        """Create bar chart of entity distribution"""
        
        sorted_data = sorted(zip(entity_stats.keys(), entity_stats.values()), 
                           key=lambda x: x[1], reverse=True)
        entity_types = [x[0] for x in sorted_data]
        counts = [x[1] for x in sorted_data]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(entity_types, counts, 
                      color=plt.cm.Set3(np.arange(len(entity_types))/len(entity_types)))
        
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom', fontsize=9)
        
        plt.title('Named Entity Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Entity Type', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Entity distribution chart saved: {output_path}")
    
    @staticmethod
    def create_validation_metrics_chart(validation_stats: Dict, output_path: str):
        """Create chart showing validation metrics"""
        
        labels = ['Historical Matches', 'Non-Historical']
        values = [validation_stats.get('historical_matches', 0), 
                 validation_stats.get('non_historical', 0)]
        
        if sum(values) == 0:
            return
        
        colors = ['#2ecc71', '#e74c3c']
        
        plt.figure(figsize=(8, 6))
        plt.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('NER Validation Against Historical POI Database', 
                 fontsize=14, fontweight='bold')
        plt.axis('equal')
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Validation metrics chart saved: {output_path}")
    
    @staticmethod
    def create_location_wordcloud(geo_df: pd.DataFrame, output_path: str):
        """Create word cloud of location mentions"""
        
        if len(geo_df) == 0:
            print("⚠ No geographic entities for word cloud")
            return
        
        location_freq = geo_df['location_name'].value_counts().to_dict()
        
        wordcloud = WordCloud(
            width=1200,
            height=600,
            background_color='white',
            colormap='Reds',
            max_words=50
        ).generate_from_frequencies(location_freq)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Location Mention Word Cloud', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Location word cloud saved: {output_path}")


# ============================================================================
# GIS EXPORT
# ============================================================================

def export_to_gis_formats(geo_df: pd.DataFrame, output_dir: Path):
    """Export geographic entities to GIS formats (GeoJSON and Shapefile)
    
    CRITICAL UPDATES FOR ARCGIS:
    1. Forces numeric coordinates.
    2. Filters out (0,0) or null coordinates.
    3. Converts object columns to strings (Shapefiles cannot store lists).
    """
    
    # 1. Ensure coordinates are numeric floats
    geo_df['latitude'] = pd.to_numeric(geo_df['latitude'], errors='coerce')
    geo_df['longitude'] = pd.to_numeric(geo_df['longitude'], errors='coerce')

    # 2. Filter valid coordinates (and remove 0,0 if it exists)
    geo_with_coords = geo_df[
        (geo_df['latitude'].notna()) & 
        (geo_df['longitude'].notna()) &
        (geo_df['longitude'] != 0)
    ].copy()
    
    if len(geo_with_coords) == 0:
        print("⚠ No entities with valid coordinates to export")
        return
    
    print(f"📍 Exporting {len(geo_with_coords)} entities with coordinates...")
    
    # 3. Create Geometry
    # Point takes (x, y) which is (Longitude, Latitude)
    geometry = [Point(xy) for xy in zip(geo_with_coords['longitude'], 
                                        geo_with_coords['latitude'])]
    
    # 4. Create GeoDataFrame
    gdf = gpd.GeoDataFrame(geo_with_coords, geometry=geometry, crs="EPSG:4326")
    
    # 5. Clean columns for Shapefile compatibility
    # Shapefiles cannot store complex types like Lists or Dicts, so we convert them to strings.
    for col in gdf.columns:
        if gdf[col].dtype == 'object':
            gdf[col] = gdf[col].astype(str)
            
    # Export GeoJSON
    geojson_path = output_dir / "geographic_entities.geojson"
    try:
        gdf.to_file(str(geojson_path), driver='GeoJSON')
        print(f"✓ GeoJSON exported: {geojson_path}")
    except Exception as e:
        print(f"❌ GeoJSON export failed: {e}")

    # Export Shapefile (Preferred for ArcGIS Pro)
    shp_path = output_dir / "geographic_entities.shp"
    try:
        gdf.to_file(str(shp_path), driver='ESRI Shapefile')
        print(f"✓ Shapefile exported: {shp_path}")
    except Exception as e:
        print(f"❌ Shapefile export failed: {e}")

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_ner_pipeline():
    """Main NER processing pipeline"""
    
    print("\n" + "=" * 80)
    print("STARTING NER PIPELINE")
    print("=" * 80 + "\n")
    
    # STEP 1: Load POI database
    print("STEP 1: Loading POI database...")
    
    if not NERConfig.POI_DATABASE_PATH.exists():
        print(f"⚠ POI database not found: {NERConfig.POI_DATABASE_PATH}")
        print("Proceeding without POI validation")
        poi_db = None
    else:
        poi_db = POIDatabase(NERConfig.POI_DATABASE_PATH)
    
    print()
    
    # STEP 2: Find preprocessed files
    print("STEP 2: Loading preprocessed data...")
    
    preprocessed_files = list(Path(NERConfig.PREPROCESSED_DATA_DIR).glob("*preprocessed*.csv"))
    
    if not preprocessed_files:
        print(f"❌ No preprocessed files found in: {NERConfig.PREPROCESSED_DATA_DIR}")
        print("Please run Step 3 first.")
        return
    
    print(f"✓ Found {len(preprocessed_files)} preprocessed files")
    print()
    
    # STEP 3: Initialize NER extractor
    print("STEP 3: Initializing NER extractor...")
    ner_extractor = TransformerNERExtractor(
        model_name=NERConfig.SPACY_MODEL,
        poi_database=poi_db
    )
    print()
    
    # STEP 4: Load all sentences
    print("STEP 4: Loading sentences...")
    
    all_sentences = []
    all_metadata = []
    
    for filepath in preprocessed_files:
        df = pd.read_csv(filepath)
        
        sentences = df['sentence'].tolist()
        
        metadata_columns = [col for col in df.columns if col != 'sentence']
        metadata = df[metadata_columns].to_dict('records')
        
        all_sentences.extend(sentences)
        all_metadata.extend(metadata)
        
        print(f"  - {filepath.name}: {len(sentences)} sentences")
    
    print(f"✓ Total sentences loaded: {len(all_sentences)}\n")
    
    # STEP 5: Extract entities
    print("STEP 5: Extracting entities with NER model...")
    print("NOTE: This may take a few minutes depending on corpus size")
    
    entity_df = ner_extractor.batch_extract_entities(
        sentences=all_sentences,
        sentence_metadata=all_metadata,
        show_progress=True
    )
    
    entity_output = NERConfig.OUTPUT_DIR / "all_entities.csv"
    entity_df.to_csv(entity_output, index=False, encoding='utf-8')
    print(f"✓ All entities saved: {entity_output}\n")
    
    # STEP 6: Extract geographic entities
    print("STEP 6: Extracting geographic entities for GIS...")
    
    geo_df = ner_extractor.extract_geographic_entities(entity_df)
    
    geo_output = NERConfig.OUTPUT_DIR / "geographic_entities.csv"
    geo_df.to_csv(geo_output, index=False, encoding='utf-8')
    print(f"✓ Geographic entities saved: {geo_output}\n")
    
    # STEP 7: Get statistics
    print("STEP 7: Generating statistics...")
    
    stats = ner_extractor.get_statistics()
    
    stats_output = NERConfig.OUTPUT_DIR / "ner_statistics.json"
    with open(stats_output, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"✓ Statistics saved: {stats_output}\n")
    
    # STEP 8: Create visualizations
    print("STEP 8: Creating visualizations...")
    
    viz = NERVisualizer()
    
    entity_chart_path = Path(NERConfig.OUTPUT_DIR) / "visualizations" / "entity_distribution.png"
    viz.create_entity_distribution_chart(stats['entity_type_counts'], entity_chart_path)
    
    if poi_db and 'validation_stats' in stats:
        validation_path = Path(NERConfig.OUTPUT_DIR) / "visualizations" / "validation_metrics.png"
        viz.create_validation_metrics_chart(stats['validation_stats'], validation_path)
    
    wordcloud_path = Path(NERConfig.OUTPUT_DIR) / "visualizations" / "location_wordcloud.png"
    viz.create_location_wordcloud(geo_df, wordcloud_path)
    
    print()
    
    # STEP 9: Export to GIS formats
    print("STEP 9: Exporting to GIS formats...")
    
    gis_output_dir = Path(NERConfig.OUTPUT_DIR) / "gis_exports"
    export_to_gis_formats(geo_df, gis_output_dir)
    
    print()
    
    # STEP 10: Generate summary report
    print("STEP 10: Creating validation report...")
    
    validation_report = {
        'extraction_summary': {
            'total_sentences': len(all_sentences),
            'sentences_with_entities': len(entity_df[entity_df['entity_text'].notna()]),
            'total_entities': len(entity_df),
            'geographic_mentions': len(geo_df),
            'unique_locations': geo_df['location_name'].nunique() if len(geo_df) > 0 else 0
        },
        'entity_distribution': stats['entity_type_counts'],
        'validation_metrics': stats.get('validation_stats', {}),
        'top_locations': geo_df['location_name'].value_counts().head(20).to_dict() if len(geo_df) > 0 else {},
        'processing_details': {
            'spacy_model': NERConfig.SPACY_MODEL,
            'processing_date': pd.Timestamp.now().isoformat(),
            'poi_database_used': poi_db is not None,
            'poi_database_size': len(poi_db.poi_df) if poi_db else 0
        }
    }
    
    report_output = NERConfig.OUTPUT_DIR / "validation_report.json"
    with open(report_output, 'w', encoding='utf-8') as f:
        json.dump(validation_report, f, indent=2, ensure_ascii=False)
    

if __name__ == "__main__":
    run_ner_pipeline()