"""
Language-Based Feature Extraction for OCR Dialogue Classification

This module provides linguistic heuristics to distinguish valid dialogue text
from OCR noise, background objects, and UI elements in webtoon/comic images.

Features:
- Dictionary word ratio
- Alphabetic character ratio
- Word frequency scoring
- Character trigram language modeling
- OCR noise detection
- Complete feature extraction and classification
"""

import logging
import re
import math
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from collections import Counter

logger = logging.getLogger(__name__)


# ============================================================================
# 1. DICTIONARY AND WORD LISTS
# ============================================================================

# Common English words (top ~1000 for lightweight implementation)
# These are the most frequently used words in English
COMMON_ENGLISH_WORDS = {
    # Articles & Determiners
    'a', 'an', 'the', 'this', 'that', 'these', 'those', 'my', 'your', 'his',
    'her', 'its', 'our', 'their', 'some', 'any', 'all', 'each', 'every', 'no',
    
    # Pronouns
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
    'who', 'what', 'which', 'where', 'when', 'why', 'how',
    
    # Common verbs
    'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
    'will', 'would', 'shall', 'should', 'can', 'could', 'may', 'might', 'must',
    'go', 'get', 'make', 'know', 'think', 'take', 'see', 'come', 'want',
    'look', 'use', 'find', 'give', 'tell', 'work', 'call', 'try', 'need',
    'feel', 'become', 'leave', 'put', 'mean', 'keep', 'let', 'begin',
    'seem', 'help', 'talk', 'turn', 'start', 'show', 'hear', 'play',
    'run', 'move', 'like', 'live', 'believe', 'hold', 'bring', 'happen',
    'write', 'sit', 'stand', 'lose', 'pay', 'meet', 'include', 'continue',
    'set', 'learn', 'change', 'lead', 'understand', 'watch', 'follow', 'stop',
    'create', 'speak', 'read', 'allow', 'add', 'spend', 'grow', 'open', 'walk',
    'win', 'offer', 'remember', 'love', 'consider', 'appear', 'buy', 'wait',
    'serve', 'die', 'send', 'expect', 'build', 'stay', 'fall', 'cut', 'reach',
    'kill', 'remain', 'suggest', 'raise', 'pass', 'sell', 'require', 'report',
    
    # Common adjectives
    'good', 'new', 'first', 'last', 'long', 'great', 'little', 'own', 'other',
    'old', 'right', 'big', 'high', 'different', 'small', 'large', 'next',
    'early', 'young', 'important', 'few', 'public', 'bad', 'same', 'able',
    
    # Common adverbs
    'not', 'so', 'up', 'out', 'just', 'now', 'how', 'then', 'more', 'also',
    'here', 'well', 'only', 'very', 'even', 'back', 'there', 'down', 'still',
    'in', 'as', 'too', 'where', 'why', 'when', 'much', 'before', 'again',
    'away', 'off', 'over', 'always', 'never', 'really', 'maybe', 'perhaps',
    
    # Common nouns
    'time', 'person', 'year', 'way', 'day', 'thing', 'man', 'world', 'life',
    'hand', 'part', 'child', 'eye', 'woman', 'place', 'work', 'week', 'case',
    'point', 'government', 'company', 'number', 'group', 'problem', 'fact',
    
    # Prepositions & Conjunctions
    'of', 'to', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'about', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'between', 'under',
    'and', 'or', 'but', 'if', 'because', 'as', 'until', 'while', 'than', 'so',
    
    # Common interjections & short words (critical for webtoon dialogue!)
    'oh', 'ah', 'hey', 'yes', 'no', 'ok', 'okay', 'yeah', 'nah', 'uh', 'um',
    'wow', 'whoa', 'huh', 'hmm', 'ooh', 'ow', 'ugh', 'yay', 'yep', 'nope',
    'sure', 'fine', 'wait', 'stop', 'help', 'please', 'thanks', 'sorry',
    
    # Common contractions (store without apostrophe for OCR tolerance)
    "dont", "doesnt", "didnt", "wont", "wouldnt", "shouldnt", "couldnt",
    "cant", "cannot", "im", "youre", "hes", "shes", "its", "were", "theyre",
    "ive", "youve", "weve", "theyve", "ill", "youll", "hell", "shell",
    "well", "theyll", "isnt", "arent", "wasnt", "werent", "hasnt", "havent",
    "hadnt", "whats", "wheres", "whens", "whos", "hows", "thats", "theres"
}

# Extended dictionary for validation (include common webtoon-specific terms)
EXTENDED_DICTIONARY = COMMON_ENGLISH_WORDS | {
    # Webtoon-specific dialogue
    'gonna', 'wanna', 'gotta', 'kinda', 'sorta', 'dunno', 'lemme',
    'cmon', 'yall', 'cause', 'cuz', 'cos',
    
    # Onomatopoeia (common in comics)
    'bang', 'boom', 'crash', 'thud', 'whoosh', 'zoom', 'pow', 'bam',
    'splash', 'thump', 'slam', 'crack', 'snap', 'click', 'beep',
    
    # Emotions & reactions
    'laugh', 'cry', 'sigh', 'gasp', 'scream', 'yell', 'shout', 'whisper',
    'smile', 'frown', 'grin', 'smirk', 'giggle', 'chuckle', 'sob', 'weep'
}


# Word frequency scores (log probabilities for common words)
# Higher scores = more common words
# Based on frequency analysis of English text
WORD_FREQUENCIES = {
    # Top tier (very common)
    'the': 5.0, 'be': 4.8, 'to': 4.7, 'of': 4.6, 'and': 4.5, 'a': 4.5, 'in': 4.4,
    'that': 4.3, 'have': 4.2, 'i': 4.2, 'it': 4.1, 'for': 4.0, 'not': 4.0,
    'on': 3.9, 'with': 3.9, 'he': 3.8, 'as': 3.8, 'you': 3.8, 'do': 3.7,
    'at': 3.7, 'this': 3.6, 'but': 3.6, 'his': 3.5, 'by': 3.5, 'from': 3.4,
    
    # High frequency
    'they': 3.4, 'we': 3.3, 'say': 3.3, 'her': 3.2, 'she': 3.2, 'or': 3.2,
    'an': 3.1, 'will': 3.1, 'my': 3.1, 'one': 3.0, 'all': 3.0, 'would': 3.0,
    'there': 2.9, 'their': 2.9, 'what': 2.9, 'so': 2.8, 'up': 2.8, 'out': 2.8,
    'if': 2.7, 'about': 2.7, 'who': 2.7, 'get': 2.7, 'which': 2.6, 'go': 2.6,
    'me': 2.6, 'when': 2.5, 'make': 2.5, 'can': 2.5, 'like': 2.5, 'time': 2.5,
    'no': 2.4, 'just': 2.4, 'him': 2.4, 'know': 2.4, 'take': 2.4, 'people': 2.4,
    
    # Medium frequency
    'into': 2.3, 'year': 2.3, 'your': 2.3, 'good': 2.3, 'some': 2.3, 'could': 2.3,
    'them': 2.2, 'see': 2.2, 'other': 2.2, 'than': 2.2, 'then': 2.2, 'now': 2.2,
    'look': 2.1, 'only': 2.1, 'come': 2.1, 'its': 2.1, 'over': 2.1, 'think': 2.1,
    'also': 2.0, 'back': 2.0, 'after': 2.0, 'use': 2.0, 'two': 2.0, 'how': 2.0,
    'our': 1.9, 'work': 1.9, 'first': 1.9, 'well': 1.9, 'way': 1.9, 'even': 1.9,
    
    # Dialogue-specific (interjections & common short phrases)
    'oh': 2.5, 'hey': 2.3, 'yes': 2.2, 'yeah': 2.1, 'okay': 2.0, 'ok': 2.0,
    'wait': 2.1, 'stop': 2.0, 'help': 1.9, 'please': 2.0, 'sorry': 1.9,
    'thanks': 1.8, 'sure': 1.9, 'wow': 1.7, 'ah': 1.8, 'uh': 1.6, 'um': 1.5,
    'hmm': 1.4, 'huh': 1.5, 'nah': 1.4, 'yep': 1.4, 'nope': 1.4,
}

# Default score for unknown words
DEFAULT_WORD_FREQUENCY = 0.5


# Character trigram frequencies (sample - would be larger in production)
# These are common character sequences in English
# Format: trigram -> log probability
CHARACTER_TRIGRAMS = {
    # Very common trigrams
    'the': 4.5, 'ing': 4.3, 'and': 4.2, 'ion': 4.0, 'tio': 3.9, 'ent': 3.8,
    'for': 3.8, 'her': 3.7, 'ter': 3.6, 'hat': 3.6, 'tha': 3.6, 'ere': 3.5,
    'ate': 3.4, 'his': 3.4, 'con': 3.3, 'res': 3.3, 'ver': 3.2, 'all': 3.2,
    'ons': 3.1, 'nce': 3.1, 'men': 3.0, 'ith': 3.0, 'ted': 3.0, 'ers': 3.0,
    'pro': 2.9, 'thi': 2.9, 'wit': 2.9, 'are': 2.9, 'ess': 2.8, 'not': 2.8,
    'ive': 2.8, 'was': 2.7, 'ect': 2.7, 'rea': 2.7, 'com': 2.7, 'eve': 2.6,
    
    # Common in dialogue
    'you': 3.5, 'out': 3.2, 'wha': 2.9, 'ght': 2.7, 'oul': 2.6, 'can': 2.8,
    'don': 2.5, 'ont': 2.4, 'ldn': 2.3, 'now': 2.7, 'kno': 2.5, 'abo': 2.6,
    'ain': 2.5, 'hin': 2.6, 'ght': 2.7, 'ell': 2.6, 'ill': 2.7, 'one': 2.8,
    
    # Common word endings
    'ght': 2.7, 'nce': 3.1, 'nde': 2.5, 'our': 2.6, 'ous': 2.7, 'use': 2.5,
}

# Default score for unseen trigrams (with smoothing)
DEFAULT_TRIGRAM_SCORE = 0.3


# ============================================================================
# 2. HELPER FUNCTIONS
# ============================================================================

def normalize_text(text: str) -> str:
    """
    Normalize text for analysis.
    
    - Convert to lowercase
    - Remove excessive whitespace
    - Keep punctuation for analysis
    
    Args:
        text: Input text string
        
    Returns:
        Normalized text
    """
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    return text


def tokenize_words(text: str) -> List[str]:
    """
    Tokenize text into words.
    
    - Splits on whitespace and punctuation
    - Handles contractions (don't -> dont)
    - Converts to lowercase
    
    Args:
        text: Input text string
        
    Returns:
        List of word tokens
    """
    # Remove common punctuation but keep apostrophes for contractions
    text = text.lower()
    
    # Handle contractions: remove apostrophes (don't -> dont)
    text = text.replace("'", "")
    
    # Remove other punctuation except hyphens (for compound words)
    text = re.sub(r'[^\w\s-]', ' ', text)
    
    # Split on whitespace
    words = text.split()
    
    # Remove empty strings and single characters (except 'a' and 'i')
    words = [w for w in words if len(w) > 0]
    
    return words


# ============================================================================
# 3. FEATURE EXTRACTION FUNCTIONS
# ============================================================================

def compute_dictionary_ratio(text: str) -> float:
    """
    Compute the ratio of valid dictionary words to total words.
    
    Higher ratio = more likely to be real language
    Lower ratio = more likely to be OCR noise or gibberish
    
    Args:
        text: Input OCR text
        
    Returns:
        Dictionary ratio [0.0, 1.0]
        Returns 0.0 if no words found
    
    Examples:
        >>> compute_dictionary_ratio("HELLO THERE")
        1.0
        >>> compute_dictionary_ratio("XKJF QWER")
        0.0
        >>> compute_dictionary_ratio("HELLO XKJF")
        0.5
    """
    words = tokenize_words(text)
    
    if not words:
        return 0.0
    
    valid_count = sum(1 for word in words if word in EXTENDED_DICTIONARY)
    ratio = valid_count / len(words)
    
    return ratio


def compute_alphabet_ratio(text: str) -> float:
    """
    Compute the ratio of alphabetic characters to total characters.
    
    Excludes spaces from calculation.
    Higher ratio = more likely to be real text
    Lower ratio = more likely to be symbols, numbers, or noise
    
    Args:
        text: Input OCR text
        
    Returns:
        Alphabet ratio [0.0, 1.0]
        Returns 0.0 if no non-space characters
    
    Examples:
        >>> compute_alphabet_ratio("HELLO")
        1.0
        >>> compute_alphabet_ratio("HI!!!")
        0.4
        >>> compute_alphabet_ratio("12345")
        0.0
    """
    # Remove spaces for calculation
    text_no_space = text.replace(' ', '')
    
    if not text_no_space:
        return 0.0
    
    alpha_count = sum(1 for c in text_no_space if c.isalpha())
    ratio = alpha_count / len(text_no_space)
    
    return ratio


def compute_word_frequency_score(text: str) -> float:
    """
    Compute average word frequency score.
    
    Uses log-frequency of words in common English text.
    Higher score = words are more common in English
    Lower score = rare/unknown words or gibberish
    
    Args:
        text: Input OCR text
        
    Returns:
        Average log-frequency score [0.0, 5.0]
        Returns DEFAULT_WORD_FREQUENCY if no words found
    
    Examples:
        >>> compute_word_frequency_score("the cat")  # Common words
        4.0+
        >>> compute_word_frequency_score("xylophone quixotic")  # Rare words
        ~0.5
    """
    words = tokenize_words(text)
    
    if not words:
        return DEFAULT_WORD_FREQUENCY
    
    scores = [WORD_FREQUENCIES.get(word, DEFAULT_WORD_FREQUENCY) for word in words]
    avg_score = sum(scores) / len(scores)
    
    return avg_score


def compute_trigram_language_score(text: str) -> float:
    """
    Compute character trigram language model score.
    
    Measures how "English-like" the character sequences are.
    Uses character-level trigrams with smoothing.
    
    Higher score = more English-like character patterns
    Lower score = random or non-English patterns
    
    Args:
        text: Input OCR text (will be normalized to lowercase)
        
    Returns:
        Average trigram log-probability score [0.0, 5.0]
        Returns DEFAULT_TRIGRAM_SCORE if insufficient characters
    
    Examples:
        >>> compute_trigram_language_score("the quick")  # English patterns
        3.0+
        >>> compute_trigram_language_score("xqz zzz")  # Unusual patterns
        ~0.5
    """
    text = normalize_text(text)
    
    # Remove spaces and punctuation for trigram analysis
    text = re.sub(r'[^a-z]', '', text)
    
    if len(text) < 3:
        return DEFAULT_TRIGRAM_SCORE
    
    # Extract all trigrams
    trigrams = [text[i:i+3] for i in range(len(text) - 2)]
    
    if not trigrams:
        return DEFAULT_TRIGRAM_SCORE
    
    # Score each trigram
    scores = [CHARACTER_TRIGRAMS.get(tri, DEFAULT_TRIGRAM_SCORE) for tri in trigrams]
    avg_score = sum(scores) / len(scores)
    
    return avg_score


def compute_ocr_noise_score(text: str) -> float:
    """
    Detect OCR noise and garbage patterns.
    
    Checks for:
    - Repeated characters (|||, OOO, ---)
    - Symbol-dominated strings
    - Very low alphabet ratio
    - Single-character repetition
    
    Higher score = more likely to be noise
    Lower score = more likely to be real text
    
    Args:
        text: Input OCR text
        
    Returns:
        Noise score [0.0, 1.0]
        0.0 = clean text, 1.0 = definite noise
    
    Examples:
        >>> compute_ocr_noise_score("HELLO THERE")
        0.0
        >>> compute_ocr_noise_score("|||||||")
        1.0
        >>> compute_ocr_noise_score("####***###")
        0.9+
    """
    if not text or len(text.strip()) == 0:
        return 1.0
    
    noise_score = 0.0
    
    # 1. Check for repeated characters
    # Pattern: 3+ of the same character in a row
    repeated_pattern = re.compile(r'(.)\1{2,}')
    if repeated_pattern.search(text):
        # Count how much of the text is repetition
        matches = repeated_pattern.findall(text)
        repetition_chars = sum(len(m) + 2 for m in matches)  # +2 for minimum length
        repetition_ratio = min(1.0, repetition_chars / len(text))
        noise_score += repetition_ratio * 0.4
    
    # 2. Symbol domination
    # If more than 40% of non-space characters are symbols
    text_no_space = text.replace(' ', '')
    if text_no_space:
        symbol_count = sum(1 for c in text_no_space if not c.isalnum())
        symbol_ratio = symbol_count / len(text_no_space)
        
        if symbol_ratio > 0.4:
            noise_score += (symbol_ratio - 0.4) * 0.5
    
    # 3. Very low alphabet ratio
    alpha_ratio = compute_alphabet_ratio(text)
    if alpha_ratio < 0.3:
        noise_score += (0.3 - alpha_ratio) * 0.3
    
    # 4. Single repeated character (like "AAA" or "---")
    unique_chars = len(set(text.replace(' ', '')))
    if unique_chars <= 2 and len(text.strip()) > 2:
        noise_score += 0.5
    
    # 5. Only punctuation/symbols
    if text_no_space and not any(c.isalnum() for c in text_no_space):
        noise_score += 0.6
    
    # Cap at 1.0
    return min(1.0, noise_score)


# ============================================================================
# 4. COMPLETE FEATURE EXTRACTION
# ============================================================================

@dataclass
class BBoxMetadata:
    """Bounding box metadata for spatial features."""
    width: int
    height: int
    x: int = 0
    y: int = 0
    image_width: int = 1000
    image_height: int = 1000


def extract_language_features(
    text: str,
    bbox_metadata: BBoxMetadata = None
) -> Dict[str, float]:
    """
    Extract complete feature vector including language-based features.
    
    Combines spatial (bounding box) features with linguistic features
    for comprehensive dialogue classification.
    
    Args:
        text: Input OCR text
        bbox_metadata: Optional bounding box information
        
    Returns:
        Dictionary of feature names to values
        
    Example:
        >>> bbox = BBoxMetadata(width=200, height=80, image_width=1000, image_height=800)
        >>> features = extract_language_features("HELLO THERE!", bbox)
        >>> features['dictionary_ratio']
        1.0
    """
    features = {}
    
    # ========================================
    # SPATIAL FEATURES (if bbox provided)
    # ========================================
    if bbox_metadata:
        bbox_area = bbox_metadata.width * bbox_metadata.height
        image_area = bbox_metadata.image_width * bbox_metadata.image_height
        
        # Bounding box area ratio
        features['bbox_area'] = bbox_area / image_area if image_area > 0 else 0.0
        
        # Aspect ratio
        features['aspect_ratio'] = (
            bbox_metadata.width / bbox_metadata.height 
            if bbox_metadata.height > 0 else 1.0
        )
        
        # Text density (characters per pixel)
        char_count = len(text)
        features['text_density'] = char_count / bbox_area if bbox_area > 0 else 0.0
    else:
        # Default spatial features
        features['bbox_area'] = 0.5
        features['aspect_ratio'] = 3.0
        features['text_density'] = 0.001
    
    # ========================================
    # BASIC TEXT FEATURES
    # ========================================
    words = tokenize_words(text)
    features['word_count'] = len(words)
    
    # Punctuation count
    punctuation_chars = set('.,!?;:-—…')
    punct_count = sum(1 for c in text if c in punctuation_chars)
    features['punctuation_count'] = punct_count
    
    # ========================================
    # LANGUAGE-BASED FEATURES
    # ========================================
    
    # 1. Dictionary word ratio
    features['dictionary_ratio'] = compute_dictionary_ratio(text)
    
    # 2. Alphabetic character ratio
    features['alphabet_ratio'] = compute_alphabet_ratio(text)
    
    # 3. Word frequency score
    features['word_frequency_score'] = compute_word_frequency_score(text)
    
    # 4. Character trigram language score
    features['trigram_language_score'] = compute_trigram_language_score(text)
    
    # 5. OCR noise score
    features['ocr_noise_score'] = compute_ocr_noise_score(text)
    
    return features


# ============================================================================
# 5. DIALOGUE CLASSIFICATION
# ============================================================================

class DialogueClassifier:
    """
    Weighted linear classifier for dialogue vs background text.
    
    Uses feature weights to compute a final score, then applies a threshold
    to classify text as dialogue or background.
    
    Designed to be easily replaceable with an ML model in the future.
    """
    
    def __init__(
        self,
        threshold: float = 0.55,
        weights: Dict[str, float] = None
    ):
        """
        Initialize dialogue classifier.
        
        Args:
            threshold: Classification threshold [0, 1]
            weights: Feature weights dict (if None, uses defaults)
        """
        self.threshold = threshold
        
        # Default weights (tuned for webtoon dialogue)
        if weights is None:
            self.weights = {
                # Spatial features (lower weight - less reliable alone)
                'bbox_area': 0.08,
                'aspect_ratio': 0.07,
                'text_density': 0.08,
                'word_count': 0.10,
                'punctuation_count': 0.07,
                
                # Language features (higher weight - more discriminative)
                'dictionary_ratio': 0.20,  # Very important
                'alphabet_ratio': 0.15,    # Good for filtering symbols
                'word_frequency_score': 0.12,  # Common words = dialogue
                'trigram_language_score': 0.08,  # English-like patterns
                'ocr_noise_score': 0.05,   # Low noise = real text
            }
        else:
            self.weights = weights
        
        # Validate weights sum to ~1.0
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Weights sum to {total:.3f}, not 1.0. Normalizing...")
            norm_factor = 1.0 / total
            self.weights = {k: v * norm_factor for k, v in self.weights.items()}
    
    def _normalize_feature(self, feature_name: str, value: float) -> float:
        """
        Normalize feature value to [0, 1] range.
        
        Different features have different scales and need normalization.
        
        Args:
            feature_name: Name of the feature
            value: Raw feature value
            
        Returns:
            Normalized value [0, 1]
        """
        if feature_name == 'bbox_area':
            # Optimal dialogue box: 1-5% of image
            # Score based on how close to optimal range
            if 0.01 <= value <= 0.05:
                return 1.0
            elif 0.005 <= value <= 0.10:
                return 0.7
            elif value < 0.005:
                return 0.3
            else:
                return 0.2
        
        elif feature_name == 'aspect_ratio':
            # Dialogue boxes typically 2:1 to 5:1
            if 2.0 <= value <= 5.0:
                return 1.0
            elif 1.0 <= value <= 7.0:
                return 0.7
            else:
                return 0.3
        
        elif feature_name == 'text_density':
            # Lower density = dialogue (well-spaced text in bubble)
            # Optimal: 0.0005 - 0.002 chars/pixel
            if 0.0005 <= value <= 0.002:
                return 1.0
            elif 0.0002 <= value <= 0.005:
                return 0.7
            elif value > 0.01:
                return 0.2
            else:
                return 0.5
        
        elif feature_name == 'word_count':
            # Short dialogue is common ("NO!", "WAIT!") but so is UI text ("Follow", "Search")
            # Rely on other features, but give some credit to multi-word text
            if value == 0:
                return 0.0
            elif value == 1:
                return 0.4  # Neutral - could be dialogue or UI
            elif value == 2:
                return 0.6
            elif value >= 3:
                return min(1.0, 0.7 + (value - 3) * 0.1)
            else:
                return 0.5
        
        elif feature_name == 'punctuation_count':
            # Dialogue often has punctuation
            if value >= 1:
                return 1.0
            else:
                return 0.3
        
        elif feature_name == 'dictionary_ratio':
            # Already [0, 1], but boost high values
            if value >= 0.8:
                return 1.0
            elif value >= 0.5:
                return 0.7
            elif value >= 0.3:
                return 0.4
            else:
                return value * 0.3
        
        elif feature_name == 'alphabet_ratio':
            # Already [0, 1], prefer high values
            if value >= 0.7:
                return 1.0
            elif value >= 0.5:
                return 0.8
            elif value >= 0.3:
                return 0.5
            else:
                return value
        
        elif feature_name == 'word_frequency_score':
            # Log-frequency scale [0, 5], normalize to [0, 1]
            # Higher = more common words = likely dialogue
            normalized = value / 5.0
            if normalized >= 0.6:
                return 1.0
            elif normalized >= 0.4:
                return 0.8
            else:
                return normalized
        
        elif feature_name == 'trigram_language_score':
            # Log-probability scale [0, 5], normalize
            normalized = value / 5.0
            return normalized
        
        elif feature_name == 'ocr_noise_score':
            # Invert: low noise = good
            return 1.0 - value
        
        else:
            # Unknown feature - return as-is
            return value
    
    def classify_dialogue(self, features: Dict[str, float]) -> Tuple[str, float]:
        """
        Classify text as dialogue or background.
        
        Args:
            features: Feature dictionary from extract_language_features()
            
        Returns:
            Tuple of (label, score) where:
                label: "dialogue" or "background"
                score: Classification confidence [0, 1]
        
        Examples:
            >>> classifier = DialogueClassifier()
            >>> features = extract_language_features("HELLO THERE!")
            >>> label, score = classifier.classify_dialogue(features)
            >>> label
            "dialogue"
        """
        # Compute weighted score
        score = 0.0
        
        for feature_name, weight in self.weights.items():
            if feature_name in features:
                raw_value = features[feature_name]
                normalized_value = self._normalize_feature(feature_name, raw_value)
                score += normalized_value * weight
            else:
                # Missing feature - use neutral score
                logger.warning(f"Missing feature: {feature_name}")
                score += 0.5 * weight
        
        # Classify based on threshold
        label = "dialogue" if score >= self.threshold else "background"
        
        return label, score
    
    def classify_text(
        self,
        text: str,
        bbox_metadata: BBoxMetadata = None
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        End-to-end classification from text to label.
        
        Convenience method that extracts features and classifies in one call.
        
        Args:
            text: Input OCR text
            bbox_metadata: Optional bounding box information
            
        Returns:
            Tuple of (label, score, features)
        
        Examples:
            >>> classifier = DialogueClassifier()
            >>> label, score, features = classifier.classify_text("WHAT?!")
            >>> label
            "dialogue"
        """
        features = extract_language_features(text, bbox_metadata)
        label, score = self.classify_dialogue(features)
        
        return label, score, features


# ============================================================================
# 6. EDGE CASE HANDLING
# ============================================================================

def handle_short_dialogue(text: str, features: Dict[str, float]) -> Dict[str, float]:
    """
    Apply special handling for short dialogue edge cases.
    
    Webtoons often have very short dialogue like:
    - "NO"
    - "HEY"
    - "WHAT?!"
    - "..."
    - "STOP"
    
    These should not be filtered out despite low word count.
    
    Args:
        text: Input text
        features: Feature dictionary
        
    Returns:
        Modified feature dictionary with boosted scores for valid short dialogue
    """
    text_normalized = normalize_text(text)
    words = tokenize_words(text)
    
    # Boost features for valid short exclamations
    if len(words) <= 2:
        # Check if it's a known exclamation/interjection
        short_dialogue_patterns = {
            'no', 'yes', 'yeah', 'nah', 'yep', 'nope',
            'oh', 'ah', 'hey', 'wow', 'stop', 'wait', 'help',
            'what', 'why', 'how', 'when', 'where', 'who',
            'okay', 'ok', 'sure', 'fine', 'please', 'thanks', 'sorry'
        }
        
        # Check if entire text (without punctuation) is in pattern set
        text_alpha = re.sub(r'[^a-z]', '', text_normalized)
        
        if text_alpha in short_dialogue_patterns:
            # Boost word count score
            features['word_count'] = max(features.get('word_count', 0), 3.0)
            # Boost dictionary ratio
            features['dictionary_ratio'] = 1.0
            # Boost word frequency (check both possible key names)
            freq_key = 'word_frequency' if 'word_frequency' in features else 'word_frequency_score'
            features[freq_key] = max(features.get(freq_key, 0), 2.5)
        
        # Check for punctuation (exclamation/question marks boost confidence)
        if any(p in text for p in ['!', '?', '...']):
            punct_key = 'punctuation' if 'punctuation' in features else 'punctuation_count'
            features[punct_key] = max(features.get(punct_key, 0), 2.0)
    


    return features
