
# ============================================
# STEP 3: TEXT PREPROCESSING AND CLEANING (ENHANCED)
# Optimized for Step 2 Sentence CSV Output
# ============================================
"""
This module preprocesses sentences extracted from Step 2
while preserving critical metadata (locations, dates, emotions)
for downstream NER and emotion analysis.

IMPORTANT PRESERVATION:
- Proper nouns (locations, person names)
- Temporal expressions (dates, times)
- Emotion-bearing words
- Sentence metadata from Step 2
"""

# Download required NLTK data
for resource in ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'punkt_tab']:
    try:
        nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else 
                      f'corpora/{resource}' if resource in ['stopwords', 'wordnet'] else 
                      f'taggers/{resource}')
    except LookupError:
        print(f"📥 Downloading {resource}...")
        nltk.download(resource)


# ============================================
# ENHANCED TEXT CLEANER WITH METADATA PRESERVATION
# ============================================

class MetadataPreservingCleaner:
    """
    Text cleaning that preserves critical information for historical narrative analysis
    Works with sentence-level data from Step 2
    """

    # Words to preserve (emotion, violence, suffering related)
    PRESERVE_KEYWORDS = {
        # Emotions
        'fear', 'afraid', 'terrified', 'scared', 'terror', 'dread',
        'sad', 'sadness', 'grief', 'sorrow', 'mourning', 'despair',
        'anger', 'angry', 'rage', 'fury', 'outrage', 'hatred',
        'disgust', 'disgusted', 'revolted', 'repulsed',
        'joy', 'happy', 'relief', 'hope', 'hopeful',
        'surprise', 'shocked', 'astonished', 'amazed',

        # Suffering and hardship
        'died', 'death', 'killed', 'murdered', 'executed', 'perished',
        'starved', 'starvation', 'hunger', 'hungry', 'thirst', 'thirsty',
        'exhausted', 'exhaustion', 'fatigue', 'weak', 'weakness',
        'sick', 'disease', 'illness', 'malaria', 'dysentery',
        'beaten', 'torture', 'tortured', 'abuse', 'abused',
        'suffered', 'suffering', 'pain', 'painful', 'agony',

        # Movement and journey
        'marched', 'march', 'walked', 'walk', 'trudged', 'stumbled',
        'journey', 'travelled', 'moved', 'proceeded',

        # Violence
        'shot', 'bayoneted', 'attacked', 'assault', 'violence',
        'brutal', 'brutality', 'cruel', 'cruelty', 'savage',

        # Survival
        'survived', 'survivor', 'escape', 'escaped', 'rescue', 'rescued',
        'witness', 'witnessed', 'saw', 'observed',

        # Places (proper nouns)
        'sandakan', 'ranau', 'telupid', 'paginatan', 'tangkul',
        'mile', 'camp', 'jungle', 'forest', 'river', 'mountain',

        # Military/Historical
        'prisoner', 'pow', 'japanese', 'soldier', 'guard', 'officer',
        'war', 'wwii', 'allied', 'australian', 'british'
    }

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        # Remove critical words from stopwords
        keep_words = {'not', 'no', 'never', 'none', 'nothing', 
                     'nor', 'neither', 'before', 'after', 'until', 'when', 'where'}
        self.stop_words = self.stop_words - keep_words

    def normalize_text(self, text: str) -> str:
        """
        Basic text normalization without aggressive cleaning
        """
        # Normalize quotes
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r"['']", "'", text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        # Fix common contractions
        contractions = {
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am"
        }
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)

        return text.strip()

    def fix_encoding_issues(self, text: str) -> str:
        """Fix common encoding problems in historical documents"""

        fixes = {
            'â€™': "'",
            'â€œ': '"',
            'â€�': '"',
            'â€"': '-',
            'Ã©': 'e',
            'Ã¡': 'a',
            'Ã³': 'o'
        }

        for bad, good in fixes.items():
            text = text.replace(bad, good)

        return text

    def preserve_entities(self, text: str) -> Tuple[str, Dict]:
        """
        Temporarily replace entities with placeholders to prevent modification

        Returns:
            (modified_text, entity_map)
        """
        entity_map = {}
        counter = [0]

        def replace_with_placeholder(pattern, prefix):
            def replacer(match):
                entity = match.group(0)
                placeholder = f"__{prefix}{counter[0]}__"
                entity_map[placeholder] = entity
                counter[0] += 1
                return placeholder
            return replacer

        # Preserve dates
        date_patterns = [
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b',
            r'\b\d{4}\b'  # Years
        ]
        for pattern in date_patterns:
            text = re.sub(pattern, replace_with_placeholder(pattern, 'DATE'), text)

        # Preserve locations (capitalized phrases)
        text = re.sub(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', 
                     replace_with_placeholder(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', 'LOC'), text)

        # Preserve numbers with units (e.g., "8 miles", "100 men")
        text = re.sub(r'\b\d+(?:\.\d+)?\s+(?:miles?|kilometers?|km|men|prisoners?|days?|hours?)\b',
                     replace_with_placeholder(r'\b\d+(?:\.\d+)?\s+(?:miles?|kilometers?|km|men|prisoners?|days?|hours?)\b', 'NUM'), text)

        return text, entity_map

    def restore_entities(self, text: str, entity_map: Dict) -> str:
        """Restore preserved entities"""
        for placeholder, entity in entity_map.items():
            text = text.replace(placeholder, entity)
        return text

    def clean_sentence(self, sentence: str, preserve_structure: bool = True) -> Dict:
        """
        Clean a single sentence with metadata preservation

        Args:
            sentence: Input sentence
            preserve_structure: Keep original structure for context

        Returns:
            Dictionary with original, cleaned, and tokenized versions
        """
        # Fix encoding issues
        sentence = self.fix_encoding_issues(sentence)

        # Normalize
        normalized = self.normalize_text(sentence)

        # Preserve entities
        if preserve_structure:
            protected, entity_map = self.preserve_entities(normalized)
        else:
            protected = normalized
            entity_map = {}

        # Remove excessive punctuation but keep sentence structure
        cleaned = re.sub(r'([.!?,;:]){2,}', r'\1', protected)
        cleaned = re.sub(r'\s+([.!?,;:])', r'\1', cleaned)

        # Restore entities
        if entity_map:
            cleaned = self.restore_entities(cleaned, entity_map)

        # Tokenize
        tokens = word_tokenize(cleaned.lower())

        # POS tagging
        pos_tags = pos_tag(tokens)

        # Filter tokens (but preserve keywords)
        filtered_tokens = []
        for token, pos in pos_tags:
            # Always keep preserved keywords
            if token in self.PRESERVE_KEYWORDS:
                filtered_tokens.append(token)
            # Keep proper nouns
            elif pos in ['NNP', 'NNPS']:
                filtered_tokens.append(token)
            # Keep other content words
            elif pos.startswith(('NN', 'VB', 'JJ', 'RB')):
                if token not in self.stop_words and len(token) > 2:
                    filtered_tokens.append(token)
            # Keep numbers
            elif pos == 'CD':
                filtered_tokens.append(token)

        # Lemmatize (but preserve emotion words in original form)
        lemmatized = []
        for token in filtered_tokens:
            if token in self.PRESERVE_KEYWORDS:
                lemmatized.append(token)  # Keep as-is
            else:
                lemmatized.append(self.lemmatizer.lemmatize(token))

        return {
            'original': sentence,
            'normalized': normalized,
            'cleaned': cleaned,
            'tokens': filtered_tokens,
            'lemmatized': lemmatized,
            'token_count': len(filtered_tokens),
            'preserved_keywords': [t for t in filtered_tokens if t in self.PRESERVE_KEYWORDS]
        }


# ============================================
# SENTENCE DATAFRAME PROCESSOR
# ============================================

class SentencePreprocessor:
    """
    Process sentence CSV files from Step 2 with metadata preservation
    """

    def __init__(self, csv_path: str):
        """
        Load sentence CSV from Step 2

        Args:
            csv_path: Path to {document}_sentences.csv from Step 2
        """
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.cleaner = MetadataPreservingCleaner()
        self.processed_df = None

        print(f"✓ Loaded {len(self.df)} sentences from {Path(csv_path).name}")

    def preprocess_all_sentences(self, preserve_structure: bool = True) -> pd.DataFrame:
        """
        Preprocess all sentences while preserving Step 2 metadata

        Args:
            preserve_structure: Preserve entities and structure

        Returns:
            Enhanced DataFrame with cleaned text and tokens
        """
        print(f"\n🔍 Preprocessing {len(self.df)} sentences...")

        processed_sentences = []

        for idx, row in self.df.iterrows():
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(self.df)} sentences...")

            sentence = row['sentence']

            # Clean sentence
            cleaned_data = self.cleaner.clean_sentence(sentence, preserve_structure)

            # Combine original metadata with cleaned data
            processed_row = {
                **row.to_dict(),  # All original Step 2 metadata
                'text_normalized': cleaned_data['normalized'],
                'text_cleaned': cleaned_data['cleaned'],
                'tokens': ' '.join(cleaned_data['tokens']),
                'lemmatized': ' '.join(cleaned_data['lemmatized']),
                'token_count': cleaned_data['token_count'],
                'preserved_keywords': ', '.join(cleaned_data['preserved_keywords']),
                'keyword_count': len(cleaned_data['preserved_keywords']),
                'has_emotion_words': len(cleaned_data['preserved_keywords']) > 0,
                'is_informative': cleaned_data['token_count'] >= 5  # Minimum meaningful content
            }

            processed_sentences.append(processed_row)

        self.processed_df = pd.DataFrame(processed_sentences)

        print(f"\n✓ Preprocessing complete!")
        print(f"  - Sentences with emotion words: {self.processed_df['has_emotion_words'].sum()}")
        print(f"  - Informative sentences: {self.processed_df['is_informative'].sum()}")
        print(f"  - Avg tokens per sentence: {self.processed_df['token_count'].mean():.1f}")

        return self.processed_df

    def get_preprocessing_statistics(self) -> Dict:
        """Generate preprocessing statistics"""

        if self.processed_df is None:
            self.preprocess_all_sentences()

        stats = {
            'total_sentences': len(self.processed_df),
            'informative_sentences': int(self.processed_df['is_informative'].sum()),
            'sentences_with_emotions': int(self.processed_df['has_emotion_words'].sum()),
            'sentences_with_locations': int(self.processed_df['location_count'].gt(0).sum()) if 'location_count' in self.processed_df else 0,
            'sentences_with_dates': int(self.processed_df['date_count'].gt(0).sum()) if 'date_count' in self.processed_df else 0,
            'avg_tokens': float(self.processed_df['token_count'].mean()),
            'avg_keywords': float(self.processed_df['keyword_count'].mean()),
            'sentence_type_distribution': self.processed_df['sentence_type'].value_counts().to_dict() if 'sentence_type' in self.processed_df else {},
            'top_preserved_keywords': self._get_top_keywords(10)
        }

        return stats

    def _get_top_keywords(self, n: int = 10) -> List[Tuple[str, int]]:
        """Get most frequent preserved keywords"""

        all_keywords = []
        for keywords_str in self.processed_df['preserved_keywords'].dropna():
            if keywords_str:
                all_keywords.extend(keywords_str.split(', '))

        keyword_counts = Counter(all_keywords)
        return keyword_counts.most_common(n)

    def filter_quality_sentences(self, 
                                 min_tokens: int = 5,
                                 require_narrative_value: bool = True) -> pd.DataFrame:
        """
        Filter for high-quality sentences suitable for NER and emotion analysis

        Args:
            min_tokens: Minimum token count
            require_narrative_value: Only keep sentences marked as having narrative value

        Returns:
            Filtered DataFrame
        """
        filtered = self.processed_df.copy()

        # Apply filters
        filtered = filtered[filtered['token_count'] >= min_tokens]

        if require_narrative_value and 'has_narrative_value' in filtered.columns:
            filtered = filtered[filtered['has_narrative_value'] == True]

        print(f"\n📊 Quality filtering:")
        print(f"  Original sentences: {len(self.processed_df)}")
        print(f"  After filtering: {len(filtered)}")
        print(f"  Retention rate: {len(filtered)/len(self.processed_df)*100:.1f}%")

        return filtered

    def save_preprocessed(self, output_path: str, filtered: bool = False):
        """
        Save preprocessed sentences to CSV

        Args:
            output_path: Output CSV path
            filtered: Whether to save only quality-filtered sentences
        """
        df_to_save = self.filter_quality_sentences() if filtered else self.processed_df

        df_to_save.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\n✓ Saved {len(df_to_save)} preprocessed sentences to: {output_path}")


# ============================================
# BATCH PROCESSING FOR MULTIPLE DOCUMENTS
# ============================================

def batch_preprocess_sentences(input_directory: str, 
                               output_directory: str,
                               preserve_structure: bool = True,
                               save_filtered: bool = True) -> Dict:
    """
    Batch preprocess all sentence CSV files from Step 2

    Args:
        input_directory: Directory with Step 2 sentence CSV files
        output_directory: Where to save preprocessed CSVs
        preserve_structure: Preserve entities during cleaning
        save_filtered: Also save quality-filtered versions

    Returns:
        Dictionary with statistics for all documents
    """

    print("="*80)
    print("STEP 3: BATCH SENTENCE PREPROCESSING")
    print("="*80)
    print(f"Input: {input_directory}")
    print(f"Output: {output_directory}")
    print("="*80)

    input_dir = Path(input_directory)
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all sentence CSV files from Step 2
    sentence_files = list(input_dir.glob("*_sentences.csv"))

    if not sentence_files:
        print(f"\n❌ No sentence CSV files found in {input_directory}")
        print("   Expected files: {document}_sentences.csv from Step 2")
        return {}

    print(f"\nFound {len(sentence_files)} sentence CSV files\n")

    all_stats = {}

    for idx, csv_file in enumerate(sentence_files, 1):
        print(f"\n{'='*80}")
        print(f"[{idx}/{len(sentence_files)}] Processing: {csv_file.name}")
        print('='*80)

        try:
            # Initialize preprocessor
            processor = SentencePreprocessor(str(csv_file))

            # Preprocess all sentences
            processed_df = processor.preprocess_all_sentences(preserve_structure)

            # Get statistics
            stats = processor.get_preprocessing_statistics()

            # Save preprocessed (all sentences)
            doc_name = csv_file.stem.replace('_sentences', '')
            output_file = output_dir / f"{doc_name}_preprocessed.csv"
            processor.save_preprocessed(output_file, filtered=False)

            # Save filtered version
            if save_filtered:
                filtered_file = output_dir / f"{doc_name}_preprocessed_filtered.csv"
                processor.save_preprocessed(filtered_file, filtered=True)

            # Save statistics
            stats_file = output_dir / f"{doc_name}_preprocessing_stats.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2)

            all_stats[doc_name] = stats

            print(f"\n✓ {doc_name} complete!")

        except Exception as e:
            print(f"\n❌ Error processing {csv_file.name}: {e}")
            continue

    # Overall summary
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE - OVERALL SUMMARY")
    print("="*80)

    total_sentences = sum(s['total_sentences'] for s in all_stats.values())
    total_informative = sum(s['informative_sentences'] for s in all_stats.values())
    total_with_emotions = sum(s['sentences_with_emotions'] for s in all_stats.values())

    print(f"Documents processed: {len(all_stats)}")
    print(f"Total sentences: {total_sentences:,}")
    print(f"Informative sentences: {total_informative:,} ({total_informative/total_sentences*100:.1f}%)")
    print(f"Sentences with emotions: {total_with_emotions:,} ({total_with_emotions/total_sentences*100:.1f}%)")

    # Save overall summary
    summary_file = output_dir / "batch_preprocessing_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, indent=2)
    print(f"\n✓ Summary saved: {summary_file}")

    return all_stats


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":

    # ==========================================
    # CONFIGURATION
    # ==========================================

    # Input: Step 2 output directory (contains *_sentences.csv files)
    INPUT_DIRECTORY = "/outputs/step2_extraction"

    # Output: Preprocessed sentences directory
    OUTPUT_DIRECTORY = "/outputs/step3_preprocessed"

    # Preserve structure (entities, dates, locations)
    PRESERVE_STRUCTURE = True

    # Save filtered versions (only high-quality sentences)
    SAVE_FILTERED = True

    # ==========================================
    # RUN PREPROCESSING
    # ==========================================

    results = batch_preprocess_sentences(
        input_directory=INPUT_DIRECTORY,
        output_directory=OUTPUT_DIRECTORY,
        preserve_structure=PRESERVE_STRUCTURE,
        save_filtered=SAVE_FILTERED
    )

    print("\n" + "="*80)
    print("✅ STEP 3 COMPLETE!")
    print("="*80)
    print(f"\n📁 Output files saved to: {OUTPUT_DIRECTORY}")
    print("\n📋 Files per document:")
    print("   1. {document}_preprocessed.csv - All preprocessed sentences")
    print("   2. {document}_preprocessed_filtered.csv - Quality-filtered sentences")
    print("   3. {document}_preprocessing_stats.json - Statistics")
    print("\n🔄 Next: Use *_preprocessed_filtered.csv for Step 4 (NER)")
    print("="*80)
