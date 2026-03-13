"""
Text Preprocessing Service

Section 7: Text Preprocessing
Before sending text to TTS or classification:
- Clean OCR artifacts
- Normalize punctuation
- Fix common OCR errors
- Handle special characters
- Preserve reading order

This service runs BEFORE text_box_classifier to ensure clean input.
"""

from email.mime import text
import re
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Handles text normalization and cleaning for OCR pipeline.
    
    Section 7: Text Preprocessing
    - Remove stray symbols
    - Normalize punctuation
    - Fix common OCR errors (optional dictionary-based)
    
    Used BEFORE classification to improve accuracy.
    """
    
    def __init__(self):
        """Initialize text preprocessor."""
        logger.info("TextPreprocessor initialized")
        
        # Common OCR errors and their corrections
        self.ocr_corrections = {
            # Common letter substitutions
            'l': 'I',   # lowercase L often confused with I in context
            '0': 'O',   # Zero to letter O in words
            '1': 'I',   # One to letter I in words
            '5': 'S',   # Five to letter S in certain contexts
        }
        
        # Characters to completely remove (noise)
        self.noise_chars = r'[~`{}\[\]|\\]'
        
        # Symbol replacements (for TTS clarity)
        self.symbol_replacements = {
            '&': ' and ',
            '%': ' percent ',
            '@': ' at ',
            '#': ' number ',
            '$': ' dollar ',
            '+': ' plus ',
            '=': ' equals ',
        }
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text for better TTS results and classification.
        
        Section 8.2: Text Normalization
        - Handle special characters and punctuation
        - Remove ellipsis and non-standard punctuation
        - Clean up whitespace
        - Preserve word boundaries when removing punctuation
        
        Args:
            text: Raw OCR text
            
        Returns:
            Normalized text string
            
        Examples:
            >>> preprocessor = TextPreprocessor()
            >>> preprocessor.normalize_text("HELLO....THERE")
            "HELLO THERE"
            >>> preprocessor.normalize_text("hello...jeff")
            "hello jeff"
            >>> preprocessor.normalize_text("WHAT?!?!")
            "WHAT?!"
        """
        if not text:
            return ""
        
        original_text = text
        
        # 1. Handle excessive ellipsis (3+ dots) → SPACE (not removal)
        # "WAIT...." → "WAIT " (preserves word boundary)
        # "hello....jeff" → "hello jeff"
        text = re.sub(r'\.{3,}', ' ', text)
        
        # 2. Replace ellipsis character (…) with space
        # "WAIT…" → "WAIT "
        text = re.sub(r'…', ' ', text)
        
        # 3. Replace remaining single/double dots between words with space
        # This catches cases like "hello..jeff" or "word.word"
        # But preserves sentence-ending periods: "End. Start" → "End. Start"
        text = re.sub(r'(?<=[a-zA-Z])\.{1,2}(?=[a-zA-Z])', ' ', text)
        
        # 4. Normalize multiple punctuation of the same type
        # "WHAT?!?!" → "WHAT?!"
        # "NO!!!" → "NO!"
        text = re.sub(r'([!?])\1{2,}', r'\1', text)
        
        # 5. Remove noise characters (tildes, backticks, brackets, etc.)
        # But replace them with spaces if they're between words
        # "HELLO~THERE" → "HELLO THERE" (not "HELLOTHERE")
        text = re.sub(r'(?<=[a-zA-Z])[~`{}\[\]|\\]+(?=[a-zA-Z])', ' ', text)
        # Then remove remaining noise chars at start/end
        text = re.sub(self.noise_chars, '', text)
        
        # 6. Replace symbols with words (for TTS clarity)
        # Add spaces around replacements to ensure word boundaries
        for symbol, replacement in self.symbol_replacements.items():
            # "5&5" → "5 and 5" (not "5and5")
            text = text.replace(symbol, f' {replacement.strip()} ')
        
        # 7. Handle dashes/hyphens between words
        # "HELLO-THERE" could be intentional (hyphenated word) or separator
        # "HELLO - THERE" is clearly a separator → "HELLO THERE"
        # Keep hyphens that are part of compound words
        text = re.sub(r'\s+-\s+', ' ', text)  # "word - word" → "word word"
        text = re.sub(r'(?<=[a-zA-Z])--+(?=[a-zA-Z])', ' ', text)  # "word--word" → "word word"
        
        # 8. Normalize whitespace (multiple spaces → single space)
        # This catches all the spaces we added in previous steps
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 9. Remove spaces before punctuation
        # "HELLO !" → "HELLO!"
        # "HELLO . THERE" → "HELLO. THERE"
        text = re.sub(r'\s+([!?.,;:])', r'\1', text)
        
        # 10. Ensure space after sentence-ending punctuation if followed by letter
        # "End.Start" → "End. Start"
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)

        #11 get rid of * or common OCR misread
        text = re.sub(r'\*+', ' ', text)  # Replace multiple asterisks with space
        text = re.sub(r'(?<=[a-zA-Z])\*+(?=[a-zA-Z])', ' ', text)  # Replace asterisks between words
        
        if text != original_text:
            logger.debug(f"Normalized: '{original_text}' → '{text}'")

        return text

    

    def clean_ocr_artifacts(self, text: str) -> str:
        """
        Remove common OCR artifacts and noise.
        
        Section 7: Clean OCR artifacts
        - Remove stray symbols (NOT punctuation)
        - Fix spacing issues
        - Remove isolated single characters (likely noise)
        - PRESERVE sentence punctuation (!?.,;:)
        
        Args:
            text: Raw OCR text
            
        Returns:
            Cleaned text
            
        Examples:
            >>> preprocessor = TextPreprocessor()
            >>> preprocessor.clean_ocr_artifacts("HELLO | THERE")
            "HELLO THERE"
            >>> preprocessor.clean_ocr_artifacts("HELLO !")
            "HELLO !"
            >>> preprocessor.clean_ocr_artifacts("A HELLO B THERE C")
            "HELLO THERE"
        """
        if not text:
            return ""
        
        original_text = text
        
        # 1. Remove single stray characters surrounded by spaces
        # "HELLO | THERE" → "HELLO THERE"
        # But PRESERVE:
        # - Single-letter words like "I" and "A"
        # - Punctuation marks (!?.,;:)
        # Only remove noise symbols like |, ~, `, etc.
        # Match: space + (NOT word char, NOT space, NOT I/A, NOT punctuation) + space
        text = re.sub(r'\s+[^\w\sIAia!?.,;:—-]\s+', ' ', text)
        
        
        # 3. Fix multiple spaces
        text = re.sub(r'\s{2,}', ' ', text).strip()
        
        # 4. Remove standalone single dashes/symbols between words (but not punctuation)
        # "HELLO - THERE" → "HELLO THERE" (if dash is alone)
        # But preserve "HELLO! THERE" (punctuation)
        text = re.sub(r'\s+[–—]\s+', ' ', text)  # Remove em/en dashes
        
        if text != original_text:
            logger.debug(f"Cleaned artifacts: '{original_text}' → '{text}'")
        
        return text
    
    def fix_common_ocr_errors(self, text: str) -> str:
        """
        Fix common OCR character substitution errors.
        
        Section 7: Fix common OCR errors
        - Cautiously fix obvious mistakes
        - Only apply when high confidence
        
        Args:
            text: Text with potential OCR errors
            
        Returns:
            Text with corrections applied
            
        Examples:
            >>> preprocessor = TextPreprocessor()
            >>> preprocessor.fix_common_ocr_errors("HEll0 THERE")
            "HELLO THERE"
        """
        if not text:
            return ""
        
        # Fix only in specific contexts to avoid false positives
        # This is very conservative to avoid breaking valid text
        
        # Fix '0' (zero) to 'O' when surrounded by letters
        # "HEll0" → "HELLO"
        text = re.sub(r'(?<=[A-Z])0(?=[A-Z])', 'O', text)
        text = re.sub(r'(?<=[a-z])0(?=[a-z])', 'o', text)
        
        # Fix '1' (one) to 'I' when surrounded by letters
        # "HE1LO" → "HELLO" (rare case)
        text = re.sub(r'(?<=[A-Z])1(?=[A-Z])', 'I', text)
        text = re.sub(r'(?<=[a-z])1(?=[a-z])', 'l', text)

        text = re.sub(r'\b\d+(?=[a-zA-Z])', '', text)
        
        return text
    
    def remove_repeated_characters(self, text: str, max_repeats: int = 2) -> str:
        """
        Remove excessive character repetition (likely OCR errors).
        
        Handles cases like "HELLLLLO" → "HELLO"
        Preserves intentional doubles like "HELLO" and "GOOD"
        
        Args:
            text: Input text
            max_repeats: Maximum allowed character repetition (default: 2)
            
        Returns:
            Text with excessive repetition removed
            
        Examples:
            >>> preprocessor = TextPreprocessor()
            >>> preprocessor.remove_repeated_characters("HELLLLLO")
            "HELLO"
            >>> preprocessor.remove_repeated_characters("GOOOD")
            "GOOD"
        """
        if not text:
            return ""
        
        # Replace 3+ repeated characters with max_repeats
        # "HELLLLLO" → "HELLO" (if max_repeats=2)
        # "AAAA" → "AA"
        pattern = r'(.)\1{' + str(max_repeats) + r',}'
        text = re.sub(pattern, r'\1' * max_repeats, text)
        
        return text
    

    def fix_ocr_corruptions(self, text: str) -> str:
        """
        Fix OCR corruptions like merged numbers and symbols.
        
        Args:
            text: Text to fix
            
        Returns:
            Fixed text with corruptions removed
        """
        # 1. Number prefix merged into word: "06suchan" → "suchan"
        text = re.sub(r'\b\d+(?=[a-zA-Z])', '', text)
    
        # 2. Number suffix merged into word: "suchan06" → "suchan"
        text = re.sub(r'(?<=[a-zA-Z])\d+\b', '', text)
    
        # 3. Number sandwiched in word: "suc06han" → "suchan"
        text = re.sub(r'(?<=[a-zA-Z])\d+(?=[a-zA-Z])', '', text)
    
        # 4. Special chars merged into word: "#suchan" → "suchan"
        text = re.sub(r'^[^a-zA-Z]+(?=[a-zA-Z])', '', text)
    
        # 5. Clean up any double spaces created
        text = re.sub(r'\s+', ' ', text).strip()
    
        return text
    
    def preprocess_for_classification(self, text: str) -> str:
        """
        Preprocess text BEFORE classification.
        
        Section 7: Text Preprocessing (before classification)
        This ensures the classifier receives clean, normalized input.
        
        Pipeline:
        1. Clean OCR artifacts (remove noise)
        2. Fix common OCR errors (character substitutions)
        3. Normalize text (punctuation, whitespace)
        
        Args:
            text: Raw OCR text
            
        Returns:
            Preprocessed text ready for classification
            
        Examples:
            >>> preprocessor = TextPreprocessor()
            >>> preprocessor.preprocess_for_classification("HEll0.... | THERE!!!")
            "HELLO THERE!"
        """
        if not text or not text.strip():
            return ""
        
        original_text = text

        text = text.lower()
        
        # Step 1: Clean OCR artifacts (remove noise symbols)
        text = self.clean_ocr_artifacts(text)
        
        # Step 2: Fix common OCR character errors
        text = self.fix_common_ocr_errors(text)
        
        # Step 3: Normalize text (punctuation, whitespace, symbols)
        text = self.normalize_text(text)
        
        # Step 4: Remove excessive repetition (likely errors)
        text = self.remove_repeated_characters(text, max_repeats=2)

        #step 5: Fix OCR corruptions like merged numbers/symbols
        text = self.fix_ocr_corruptions(text)
        
        if text != original_text:
            logger.info(f"Preprocessed for classification: '{original_text}' → '{text}'")
        
        return text
    
    def preprocess_for_tts(self, text: str) -> str:
        """
        Preprocess text AFTER classification, BEFORE TTS.
        
        Section 8.2: Text Normalization for TTS
        Additional TTS-specific preprocessing.
        
        Pipeline:
        1. Normalize text (already done in classification)
        2. Handle numbers (convert to words if needed)
        3. Expand abbreviations
        4. Final cleanup
        
        Args:
            text: Classified dialogue text
            
        Returns:
            Text optimized for TTS
            
        Examples:
            >>> preprocessor = TextPreprocessor()
            >>> preprocessor.preprocess_for_tts("MEET ME @ 3PM")
            "MEET ME at 3PM"
        """
        if not text or not text.strip():
            return ""
        
        # Already normalized during classification, but apply again for safety
        text = self.normalize_text(text)
        
        # Additional TTS-specific processing can go here
        # For MVP, normalization is sufficient
        # Future: number-to-word conversion, abbreviation expansion
        
        logger.debug(f"Preprocessed for TTS: '{text}'")
        return text
    
    def preprocess_batch(
        self,
        texts: List[str],
        for_classification: bool = True
    ) -> List[str]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of raw OCR texts
            for_classification: If True, use classification preprocessing;
                              if False, use TTS preprocessing
            
        Returns:
            List of preprocessed texts (same order)
            
        Examples:
            >>> preprocessor = TextPreprocessor()
            >>> preprocessor.preprocess_batch(["HELLO....", "WAIT!!!"], for_classification=True)
            ["HELLO", "WAIT!"]
        """
        if for_classification:
            return [self.preprocess_for_classification(text) for text in texts]
        else:
            return [self.preprocess_for_tts(text) for text in texts]


# ============================================================================
# STANDALONE FUNCTIONS (for backwards compatibility)
# ============================================================================

def normalize_text(text: str) -> str:
    """
    Standalone normalize_text function for backwards compatibility.
    
    Creates a TextPreprocessor instance and calls normalize_text.
    
    Args:
        text: Raw OCR text
        
    Returns:
        Normalized text
    """
    preprocessor = TextPreprocessor()
    return preprocessor.normalize_text(text)


def preprocess_for_classification(text: str) -> str:
    """
    Standalone preprocessing function for classification.
    
    Args:
        text: Raw OCR text
        
    Returns:
        Preprocessed text ready for classification
    """
    preprocessor = TextPreprocessor()
    return preprocessor.preprocess_for_classification(text)


def preprocess_for_tts(text: str) -> str:
    """
    Standalone preprocessing function for TTS.
    
    Args:
        text: Classified dialogue text
        
    Returns:
        Preprocessed text ready for TTS
    """
    preprocessor = TextPreprocessor()
    return preprocessor.preprocess_for_tts(text)
