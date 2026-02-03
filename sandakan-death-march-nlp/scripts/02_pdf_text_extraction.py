
# ============================================
# STEP 2: ENHANCED PDF TEXT EXTRACTION
# Optimized for Historical Narrative Documents
# Sandakan-Ranau Death Marches NLP Processing
# ============================================

"""
INSTALLATION:
pip install pymupdf4llm PyMuPDF nltk pandas numpy
python -m nltk.downloader punkt punkt_tab
"""

# NLTK for sentence tokenization
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("📥 Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')
    nltk.download('punkt_tab')
    from nltk.tokenize import sent_tokenize


# ============================================
# QUALITY CONTROL & VALIDATION
# ============================================

class TextQualityChecker:
    """
    Quality control for extracted text to ensure accuracy
    """

    @staticmethod
    def assess_text_quality(text: str) -> Dict:
        """
        Assess quality of extracted text
        Returns quality metrics and flags potential issues
        """
        if not text or not text.strip():
            return {
                'quality_score': 0.0,
                'issues': ['Empty text'],
                'is_valid': False
            }

        issues = []
        quality_indicators = {
            'char_count': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(sent_tokenize(text)),
            'avg_word_length': 0,
            'special_char_ratio': 0,
            'uppercase_ratio': 0,
            'digit_ratio': 0
        }

        words = text.split()
        if words:
            quality_indicators['avg_word_length'] = np.mean([len(w) for w in words])

        # Calculate character ratios
        total_chars = len(text)
        quality_indicators['special_char_ratio'] = len(re.findall(r'[^a-zA-Z0-9\s]', text)) / total_chars
        quality_indicators['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / total_chars
        quality_indicators['digit_ratio'] = sum(1 for c in text if c.isdigit()) / total_chars

        # Quality checks
        if quality_indicators['char_count'] < 100:
            issues.append('Text too short')

        if quality_indicators['avg_word_length'] < 2 or quality_indicators['avg_word_length'] > 15:
            issues.append('Unusual word length - possible OCR errors')

        if quality_indicators['special_char_ratio'] > 0.3:
            issues.append('High special character ratio - possible corruption')

        if quality_indicators['uppercase_ratio'] > 0.5:
            issues.append('Excessive uppercase - possible OCR issues')

        # Calculate quality score (0-100)
        score = 100.0
        if issues:
            score -= len(issues) * 15

        # Bonus for good text characteristics
        if 3 < quality_indicators['avg_word_length'] < 8:
            score += 10
        if quality_indicators['sentence_count'] > 5:
            score += 10

        score = max(0, min(100, score))

        return {
            'quality_score': round(score, 2),
            'indicators': quality_indicators,
            'issues': issues if issues else ['No issues detected'],
            'is_valid': score >= 50
        }

    @staticmethod
    def clean_ocr_artifacts(text: str) -> str:
        """
        Clean common OCR artifacts and errors
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Fix common OCR mistakes
        ocr_corrections = {
            r'\bl\b': 'I',  # lowercase L mistaken for I
            r'\b0\b': 'O',  # zero mistaken for O in words
            r'[\|\[\]\{\}]': '',  # Remove stray brackets
            r'~': '-',  # Replace tildes with hyphens
            r'`': "'",  # Backticks to apostrophes
        }

        for pattern, replacement in ocr_corrections.items():
            text = re.sub(pattern, replacement, text)

        # Remove page numbers and footers
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'Page\s+\d+', '', text, flags=re.IGNORECASE)

        return text.strip()

    @staticmethod
    def detect_narrative_sections(text: str) -> List[Dict]:
        """
        Detect different narrative sections (testimonies, diary entries, reports)
        """
        sections = []

        # Patterns for different narrative types
        patterns = {
            'diary_entry': r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}',
            'testimony': r'(?:testified|stated|recalled|remembered|witnessed)',
            'report': r'(?:Report|Document|Record|Statement)',
            'quote': r'"[^"]{50,}"',
            'location_mention': r'(?:Sandakan|Ranau|Telupid|Paginatan|Mile\s+\d+)'
        }

        for section_type, pattern in patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                sections.append({
                    'type': section_type,
                    'position': match.start(),
                    'text': match.group(),
                    'context_start': max(0, match.start() - 100),
                    'context_end': min(len(text), match.end() + 100)
                })

        return sections


# ============================================
# ENHANCED PDF EXTRACTOR CLASS
# ============================================

class EnhancedPyMuPDF4LLMExtractor:
    """
    Enhanced PDF extractor with quality control and narrative intelligence
    Optimized for historical Death Marches documents
    """

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.markdown_text = ""
        self.pages_data = []
        self.metadata = {}
        self.quality_checker = TextQualityChecker()
        self.extraction_quality = {}

    def extract_with_quality_control(self, 
                                     preserve_images: bool = False,
                                     min_quality_score: float = 50.0) -> Tuple[str, Dict]:
        """
        Extract text with quality validation

        Args:
            preserve_images: Extract and save images
            min_quality_score: Minimum quality threshold (0-100)

        Returns:
            Tuple of (extracted_text, quality_report)
        """
        try:
            print(f"\n📄 Extracting from: {Path(self.pdf_path).name}")

            # Extract markdown
            self.markdown_text = pymupdf4llm.to_markdown(
                doc=self.pdf_path,
                write_images=preserve_images,
                image_path="extracted_images" if preserve_images else None,
                dpi=300  # High quality for historical documents
            )

            # Clean OCR artifacts
            cleaned_text = self.quality_checker.clean_ocr_artifacts(self.markdown_text)

            # Assess quality
            quality_report = self.quality_checker.assess_text_quality(cleaned_text)

            print(f"  ✓ Extracted {quality_report['indicators']['char_count']:,} characters")
            print(f"  ✓ Quality Score: {quality_report['quality_score']}/100")

            if quality_report['quality_score'] < min_quality_score:
                print(f"  ⚠️ WARNING: Quality below threshold ({min_quality_score})")
                print(f"     Issues: {', '.join(quality_report['issues'])}")

            # Detect narrative sections
            sections = self.quality_checker.detect_narrative_sections(cleaned_text)
            quality_report['narrative_sections'] = len(sections)
            quality_report['section_types'] = Counter([s['type'] for s in sections])

            print(f"  ✓ Detected {len(sections)} narrative sections")

            self.markdown_text = cleaned_text
            self.extraction_quality = quality_report

            return cleaned_text, quality_report

        except Exception as e:
            print(f"  ❌ Extraction failed: {e}")
            return None, {'quality_score': 0, 'error': str(e)}

    def extract_pages_with_context(self, 
                                    start_page: Optional[int] = None,
                                    end_page: Optional[int] = None) -> List[Dict]:
        """
        Extract pages with enhanced metadata and context

        Args:
            start_page: Starting page (0-indexed)
            end_page: Ending page (0-indexed)

        Returns:
            List of page dictionaries with metadata
        """
        try:
            import pymupdf

            # Determine page range
            doc = pymupdf.open(self.pdf_path)
            total_pages = len(doc)
            doc.close()

            pages = None
            if start_page is not None or end_page is not None:
                start = start_page if start_page is not None else 0
                end = end_page if end_page is not None else total_pages - 1
                pages = list(range(start, end + 1))

            # Extract with page chunks
            raw_pages = pymupdf4llm.to_markdown(
                doc=self.pdf_path,
                pages=pages,
                page_chunks=True
            )

            # Enhance each page with quality metrics and context
            enhanced_pages = []

            for idx, page_data in enumerate(raw_pages):
                page_text = page_data.get('text', '')
                page_num = page_data.get('page', idx)

                # Clean text
                cleaned_text = self.quality_checker.clean_ocr_artifacts(page_text)

                # Quality assessment
                quality = self.quality_checker.assess_text_quality(cleaned_text)

                # Detect sections
                sections = self.quality_checker.detect_narrative_sections(cleaned_text)

                enhanced_page = {
                    'page': page_num,
                    'text': cleaned_text,
                    'original_text': page_text,
                    'char_count': len(cleaned_text),
                    'word_count': len(cleaned_text.split()),
                    'quality_score': quality['quality_score'],
                    'quality_issues': quality['issues'],
                    'narrative_sections': sections,
                    'section_count': len(sections),
                    'has_locations': any(s['type'] == 'location_mention' for s in sections),
                    'has_dates': any(s['type'] == 'diary_entry' for s in sections),
                    'has_testimony': any(s['type'] == 'testimony' for s in sections)
                }

                enhanced_pages.append(enhanced_page)

                # Progress indicator
                if (idx + 1) % 10 == 0:
                    print(f"  Processed {idx + 1}/{len(raw_pages)} pages...")

            self.pages_data = enhanced_pages

            print(f"\n  ✓ Enhanced {len(enhanced_pages)} pages with metadata")
            print(f"  ✓ Pages with locations: {sum(1 for p in enhanced_pages if p['has_locations'])}")
            print(f"  ✓ Pages with dates: {sum(1 for p in enhanced_pages if p['has_dates'])}")
            print(f"  ✓ Pages with testimonies: {sum(1 for p in enhanced_pages if p['has_testimony'])}")

            return enhanced_pages

        except Exception as e:
            print(f"  ❌ Page extraction failed: {e}")
            return []

    def extract_sentences_with_metadata(self, min_sentence_length: int = 15) -> pd.DataFrame:
        """
        Extract sentences with comprehensive metadata
        Bridges Step 2 (extraction) with Step 3 (preprocessing)

        Args:
            min_sentence_length: Minimum character length for valid sentences

        Returns:
            DataFrame with sentence-level data
        """
        if not self.pages_data:
            print("⚠️ No page data available. Running page extraction first...")
            self.extract_pages_with_context()

        print(f"\n🔍 Extracting sentences from {len(self.pages_data)} pages...")

        sentences = []
        sentence_id = 0

        for page_info in self.pages_data:
            page_num = page_info['page']
            page_text = page_info['text']
            page_quality = page_info['quality_score']

            # Tokenize into sentences
            page_sentences = sent_tokenize(page_text)

            for sent_idx, sent in enumerate(page_sentences):
                sent = sent.strip()

                # Filter very short or invalid sentences
                if len(sent) < min_sentence_length:
                    continue

                # Check if sentence contains narrative value
                has_narrative_value = any([
                    re.search(r'\b(?:marched|walked|died|killed|suffered|exhausted)\b', sent, re.IGNORECASE),
                    re.search(r'\b(?:fear|sad|angry|terrible|awful|horrible)\b', sent, re.IGNORECASE),
                    re.search(r'\b(?:Sandakan|Ranau|Telupid|Mile\s+\d+)\b', sent, re.IGNORECASE),
                    re.search(r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}', sent, re.IGNORECASE)
                ])

                # Extract potential location mentions
                location_pattern = r'\b(?:Sandakan|Ranau|Telupid|Paginatan|Tangkul|Mile\s+\d+|Beluran|Poring|Keningau)\b'
                locations = re.findall(location_pattern, sent, re.IGNORECASE)

                # Extract potential dates
                date_pattern = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}'
                dates = re.findall(date_pattern, sent, re.IGNORECASE)

                # Classify sentence type
                sent_lower = sent.lower()
                sentence_type = 'general'
                if any(word in sent_lower for word in ['testified', 'stated', 'recalled', 'said', 'told']):
                    sentence_type = 'testimony'
                elif any(word in sent_lower for word in ['marched', 'walked', 'moved', 'travelled']):
                    sentence_type = 'movement'
                elif any(word in sent_lower for word in ['died', 'killed', 'death', 'casualty']):
                    sentence_type = 'casualty'
                elif locations:
                    sentence_type = 'location_reference'

                sentence_id += 1

                sentences.append({
                    'sentence_id': sentence_id,
                    'page': page_num,
                    'page_quality': page_quality,
                    'sentence_index': sent_idx,
                    'sentence': sent,
                    'char_count': len(sent),
                    'word_count': len(sent.split()),
                    'has_narrative_value': has_narrative_value,
                    'sentence_type': sentence_type,
                    'potential_locations': ', '.join(locations) if locations else None,
                    'location_count': len(locations),
                    'potential_dates': ', '.join(dates) if dates else None,
                    'date_count': len(dates),
                    'has_emotion_keywords': bool(re.search(r'\b(?:fear|terror|sad|grief|anger|rage|disgust|joy|surprise)\b', sent, re.IGNORECASE))
                })

        df = pd.DataFrame(sentences)

        print(f"\n  ✓ Extracted {len(df)} sentences")
        print(f"  ✓ Sentences with narrative value: {df['has_narrative_value'].sum()}")
        print(f"  ✓ Sentences with locations: {df['location_count'].gt(0).sum()}")
        print(f"  ✓ Sentences with dates: {df['date_count'].gt(0).sum()}")
        print(f"  ✓ Sentences with emotion keywords: {df['has_emotion_keywords'].sum()}")

        print(f"\n  📊 Sentence type distribution:")
        for stype, count in df['sentence_type'].value_counts().items():
            print(f"     - {stype}: {count}")

        return df

    def save_enhanced_outputs(self, output_directory: str, document_name: str):
        """
        Save all extracted data with comprehensive metadata
        """
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save cleaned Markdown
        if self.markdown_text:
            md_file = output_dir / f"{document_name}.md"
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(self.markdown_text)
            print(f"  ✓ Markdown saved: {md_file}")

        # 2. Save enhanced page data as JSON
        if self.pages_data:
            json_file = output_dir / f"{document_name}_pages_enhanced.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(self.pages_data, f, indent=2, ensure_ascii=False)
            print(f"  ✓ Enhanced pages saved: {json_file}")

        # 3. Save sentence-level data as CSV
        df_sentences = self.extract_sentences_with_metadata()
        if not df_sentences.empty:
            csv_file = output_dir / f"{document_name}_sentences.csv"
            df_sentences.to_csv(csv_file, index=False, encoding='utf-8')
            print(f"  ✓ Sentences saved: {csv_file}")

        # 4. Save quality report
        if self.extraction_quality:
            quality_file = output_dir / f"{document_name}_quality_report.json"
            with open(quality_file, 'w', encoding='utf-8') as f:
                json.dump(self.extraction_quality, f, indent=2)
            print(f"  ✓ Quality report saved: {quality_file}")

        # 5. Save extraction summary
        summary = {
            'document_name': document_name,
            'extraction_date': datetime.now().isoformat(),
            'total_pages': len(self.pages_data),
            'total_characters': len(self.markdown_text),
            'total_words': len(self.markdown_text.split()),
            'total_sentences': len(df_sentences),
            'quality_score': self.extraction_quality.get('quality_score', 0),
            'quality_issues': self.extraction_quality.get('issues', []),
            'narrative_sections': self.extraction_quality.get('narrative_sections', 0),
            'high_quality_pages': sum(1 for p in self.pages_data if p['quality_score'] >= 70),
            'pages_with_locations': sum(1 for p in self.pages_data if p['has_locations']),
            'pages_with_dates': sum(1 for p in self.pages_data if p['has_dates']),
            'sentences_with_narrative_value': df_sentences['has_narrative_value'].sum()
        }

        summary_file = output_dir / f"{document_name}_extraction_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print(f"  ✓ Extraction summary saved: {summary_file}")


# ============================================
# BATCH PROCESSING WITH QUALITY CONTROL
# ============================================

def batch_extract_with_quality_control(pdf_directory: str, 
                                       output_directory: str,
                                       min_quality_score: float = 50.0) -> Dict:
    """
    Batch extract PDFs with comprehensive quality control

    Args:
        pdf_directory: Folder containing PDF files
        output_directory: Where to save extracted data
        min_quality_score: Minimum acceptable quality score (0-100)

    Returns:
        Dictionary with extraction results and statistics
    """

    print("="*80)
    print("ENHANCED PDF EXTRACTION - SANDAKAN-RANAU DEATH MARCHES")
    print("Historical Narrative Intelligence Pipeline")
    print("="*80)
    print(f"Source: {pdf_directory}")
    print(f"Output: {output_directory}")
    print(f"Quality Threshold: {min_quality_score}/100")
    print("="*80)

    # Verify directory exists
    if not os.path.exists(pdf_directory):
        print(f"❌ ERROR: PDF directory not found: {pdf_directory}")
        return {}

    # Create output directory
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    # Find all PDFs
    pdf_files = list(Path(pdf_directory).glob("*.pdf"))

    if not pdf_files:
        print(f"❌ WARNING: No PDF files found in {pdf_directory}")
        return {}

    print(f"\nFound {len(pdf_files)} PDF files to process...\n")

    results = {
        'successful': [],
        'failed': [],
        'low_quality': [],
        'statistics': {}
    }

    for idx, pdf_file in enumerate(pdf_files, 1):
        print(f"\n{'='*80}")
        print(f"[{idx}/{len(pdf_files)}] Processing: {pdf_file.name}")
        print(f"{'='*80}")

        try:
            extractor = EnhancedPyMuPDF4LLMExtractor(str(pdf_file))

            # Extract with quality control
            text, quality_report = extractor.extract_with_quality_control(
                min_quality_score=min_quality_score
            )

            if not text:
                results['failed'].append({
                    'document': pdf_file.name,
                    'error': quality_report.get('error', 'Unknown error')
                })
                continue

            # Extract pages with context
            extractor.extract_pages_with_context()

            # Save all outputs
            doc_name = pdf_file.stem
            extractor.save_enhanced_outputs(output_directory, doc_name)

            # Track results
            result_entry = {
                'document': pdf_file.name,
                'quality_score': quality_report['quality_score'],
                'pages': len(extractor.pages_data),
                'characters': len(text),
                'words': len(text.split()),
                'narrative_sections': quality_report.get('narrative_sections', 0)
            }

            if quality_report['quality_score'] >= min_quality_score:
                results['successful'].append(result_entry)
            else:
                results['low_quality'].append(result_entry)

            print(f"\n✓ {pdf_file.name} processing complete!")

        except Exception as e:
            print(f"\n❌ Error processing {pdf_file.name}: {e}")
            results['failed'].append({
                'document': pdf_file.name,
                'error': str(e)
            })

    # Calculate statistics
    results['statistics'] = {
        'total_documents': len(pdf_files),
        'successful': len(results['successful']),
        'low_quality': len(results['low_quality']),
        'failed': len(results['failed']),
        'success_rate': round(len(results['successful']) / len(pdf_files) * 100, 2) if pdf_files else 0,
        'total_pages': sum(r['pages'] for r in results['successful']),
        'total_characters': sum(r['characters'] for r in results['successful']),
        'total_words': sum(r['words'] for r in results['successful']),
        'avg_quality_score': round(np.mean([r['quality_score'] for r in results['successful']]), 2) if results['successful'] else 0
    }

    # Print final summary
    print("\n" + "="*80)
    print("EXTRACTION COMPLETE - FINAL SUMMARY")
    print("="*80)
    print(f"Total Documents: {results['statistics']['total_documents']}")
    print(f"✓ Successful: {results['statistics']['successful']}")
    print(f"⚠️ Low Quality: {results['statistics']['low_quality']}")
    print(f"❌ Failed: {results['statistics']['failed']}")
    print(f"Success Rate: {results['statistics']['success_rate']}%")
    print(f"\nTotal Pages Extracted: {results['statistics']['total_pages']}")
    print(f"Total Characters: {results['statistics']['total_characters']:,}")
    print(f"Total Words: {results['statistics']['total_words']:,}")
    print(f"Average Quality Score: {results['statistics']['avg_quality_score']}/100")

    # Save batch summary
    summary_file = Path(output_directory) / "batch_extraction_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Batch summary saved: {summary_file}")

    return results


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":

    # ==========================================
    # CONFIGURATION - EDIT YOUR PATHS HERE
    # ==========================================

    PDF_DIRECTORY = "/PDF"
    OUTPUT_DIRECTORY = "/outputs/step2_extraction"

    # Quality threshold (0-100)
    # 70+ = High quality, 50-70 = Acceptable, <50 = Low quality
    MIN_QUALITY_SCORE = 50.0

    # ==========================================
    # RUN ENHANCED EXTRACTION
    # ==========================================

    results = batch_extract_with_quality_control(
        pdf_directory=PDF_DIRECTORY,
        output_directory=OUTPUT_DIRECTORY,
        min_quality_score=MIN_QUALITY_SCORE
    )

    print("\n" + "="*80)
    print("✅ STEP 2 COMPLETE - PDF EXTRACTION WITH QUALITY CONTROL")
    print("="*80)
    print(f"\n📁 All outputs saved to: {OUTPUT_DIRECTORY}")
    print("\n📋 Generated files per document:")
    print("   1. {document}.md - Cleaned Markdown text")
    print("   2. {document}_pages_enhanced.json - Page-level data with metadata")
    print("   3. {document}_sentences.csv - Sentence-level data (ready for Step 3)")
    print("   4. {document}_quality_report.json - Quality assessment")
    print("   5. {document}_extraction_summary.json - Extraction statistics")
    print("\n🔄 Next Step: Use {document}_sentences.csv for Step 3 (Text Preprocessing)")
