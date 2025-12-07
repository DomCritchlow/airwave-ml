#!/usr/bin/env python3
"""
Training Text Generator

Generates diverse text for training CTC-based audio decoders.
Designed for PATTERN coverage, not language modeling.

Key principles:
1. All characters should appear with roughly equal frequency
2. All character PAIRS should be represented (bigram coverage)
3. Include realistic ham radio content (callsigns, Q-codes, etc.)
4. Include adversarial/random sequences
5. Mode-agnostic: works for CW, PSK31, RTTY, etc.

Usage:
    from generate_training_text import TextGenerator
    
    gen = TextGenerator()
    texts = gen.generate_batch(1000)
"""

import random
import string
from typing import List, Set, Tuple, Optional
from dataclasses import dataclass
from collections import Counter
import itertools


@dataclass
class TextGeneratorConfig:
    """Configuration for text generation."""
    # Character set
    include_letters: bool = True
    include_numbers: bool = True
    include_punctuation: bool = False  # Basic punctuation
    include_lowercase: bool = False    # For PSK31 conversational
    include_extended_punct: bool = False  # For RTTY Baudot figures
    
    # Content mix (should sum to 1.0)
    # Balanced for general-purpose decoding with radio capability
    callsign_ratio: float = 0.15      # W1ABC, K3XYZ patterns
    qcode_ratio: float = 0.05         # QTH, QSL, QRZ
    abbreviation_ratio: float = 0.05  # CQ, DE, 73, etc.
    number_ratio: float = 0.10        # RST, frequencies, times
    word_ratio: float = 0.35          # Real words from dictionary (largest!)
    sentence_ratio: float = 0.15      # Full sentences/phrases
    random_ratio: float = 0.15        # Random character sequences
    punctuation_ratio: float = 0.00   # Punctuation-heavy samples
    
    # Length settings
    min_length: int = 10
    max_length: int = 50
    
    # Special patterns
    include_prosigns: bool = True     # AR, SK, BT
    
    # Mode-specific character sets
    mode: str = 'GENERAL'  # CW, PSK31, RTTY, FT8, GENERAL


# Mode-specific character sets
MODE_CHARSETS = {
    'CW': {
        'letters': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
        'numbers': '0123456789',
        'punctuation': '.,?/-=',
        'special': ' ',
    },
    'PSK31': {
        # PSK31 Varicode supports full ASCII
        'letters': 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
        'numbers': '0123456789',
        'punctuation': '.,?!-/:;()\'\"@#$%&*+=[]{}|\\<>^_`~',
        'special': ' ',
    },
    'RTTY': {
        # RTTY Baudot has limited character set
        'letters': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
        'numbers': '0123456789',
        'punctuation': '-?:$!&#()\'.+,;/',  # Baudot FIGURES
        'special': ' ',
    },
    'FT8': {
        # FT8 has very limited character set
        'letters': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
        'numbers': '0123456789',
        'punctuation': '/+-.',
        'special': ' ',
    },
    'GENERAL': {
        'letters': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
        'numbers': '0123456789',
        'punctuation': '.,?!-',
        'special': ' ',
    },
}


# Ham radio callsign prefixes by country
CALLSIGN_PREFIXES = [
    # USA
    'W', 'K', 'N', 'AA', 'AB', 'AC', 'AD', 'AE', 'AF', 'AG', 'AI', 'AJ', 'AK', 'AL',
    'KA', 'KB', 'KC', 'KD', 'KE', 'KF', 'KG', 'KI', 'KJ', 'KK', 'KM', 'KN', 'KO',
    'NA', 'NB', 'NC', 'ND', 'NE', 'NF', 'NG', 'NI', 'NJ', 'NK', 'NM', 'NN', 'NO',
    'WA', 'WB', 'WC', 'WD', 'WE', 'WF', 'WG', 'WI', 'WJ', 'WK', 'WM', 'WN', 'WO',
    # Canada
    'VE', 'VA', 'VY',
    # UK
    'G', 'M', '2E', '2M',
    # Germany
    'DL', 'DA', 'DB', 'DC', 'DD', 'DF', 'DG', 'DH', 'DI', 'DJ', 'DK', 'DM', 'DO',
    # Japan
    'JA', 'JE', 'JF', 'JG', 'JH', 'JI', 'JJ', 'JK', 'JL', 'JM', 'JN', 'JO', 'JP', 'JQ', 'JR', 'JS',
    # Australia
    'VK',
    # Russia
    'UA', 'RA', 'R',
    # Others
    'EA', 'F', 'I', 'LA', 'OH', 'OZ', 'PA', 'SM', 'SP', 'YL', 'LY', 'ES', 'UR',
]

# Q-codes used in amateur radio
Q_CODES = [
    'QTH', 'QSL', 'QRZ', 'QSO', 'QRM', 'QRN', 'QSB', 'QRT', 'QRX', 'QRV',
    'QSY', 'QTR', 'QRG', 'QRL', 'QRP', 'QRO', 'QRS', 'QRQ', 'QSK', 'QTC',
]

# Common ham radio abbreviations
ABBREVIATIONS = [
    'CQ', 'DE', 'DX', 'ES', 'FB', 'GM', 'GA', 'GE', 'GN', 'HI', 'HR', 'HW',
    'NR', 'OM', 'OP', 'PSE', 'PWR', 'RIG', 'RST', 'RX', 'TX', 'TU', 'UR',
    'WX', 'XYL', 'YL', '73', '88', 'ANT', 'BK', 'CL', 'CFM', 'CU', 'CUL',
    'AGN', 'BT', 'SK', 'AR', 'KN', 'AS', 'SN', 'VA', 'RPT', 'RPTR', 'QSP',
]

# Prosigns (sent as single units in Morse)
PROSIGNS = ['AR', 'SK', 'BT', 'KN', 'AS', 'SN', 'VA', 'VVV']

# RST reports
RST_REPORTS = [f'{r}{s}{t}' for r in '345' for s in '56789' for t in '9']

# Common words (expanded list for better coverage)
COMMON_WORDS = [
    # Articles, prepositions
    'THE', 'A', 'AN', 'IN', 'ON', 'AT', 'TO', 'FOR', 'OF', 'BY', 'WITH', 'FROM',
    # Common verbs
    'IS', 'ARE', 'WAS', 'WERE', 'BE', 'BEEN', 'HAVE', 'HAS', 'HAD', 'DO', 'DOES',
    'WILL', 'CAN', 'MAY', 'COULD', 'WOULD', 'SHOULD', 'GET', 'GOT', 'MAKE', 'MADE',
    # Nouns
    'TIME', 'YEAR', 'PEOPLE', 'WAY', 'DAY', 'MAN', 'THING', 'WOMAN', 'LIFE', 'CHILD',
    'WORLD', 'SCHOOL', 'STATE', 'FAMILY', 'STUDENT', 'GROUP', 'COUNTRY', 'PROBLEM',
    'HAND', 'PART', 'PLACE', 'CASE', 'WEEK', 'COMPANY', 'SYSTEM', 'PROGRAM', 'QUESTION',
    # Radio-specific
    'RADIO', 'ANTENNA', 'SIGNAL', 'BAND', 'FREQUENCY', 'METER', 'POWER', 'WATTS',
    'NOISE', 'COPY', 'SOLID', 'WEAK', 'STRONG', 'CLEAR', 'FADE', 'STATIC',
    # Adjectives
    'GOOD', 'NEW', 'FIRST', 'LAST', 'LONG', 'GREAT', 'LITTLE', 'OWN', 'OTHER',
    'OLD', 'RIGHT', 'BIG', 'HIGH', 'DIFFERENT', 'SMALL', 'LARGE', 'NEXT', 'EARLY',
    # Adverbs
    'UP', 'OUT', 'JUST', 'NOW', 'HOW', 'THEN', 'MORE', 'ALSO', 'HERE', 'WELL',
    'ONLY', 'VERY', 'EVEN', 'BACK', 'THERE', 'DOWN', 'STILL', 'THROUGH', 'MUCH',
    # Numbers as words
    'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE', 'TEN',
    # Connectors
    'AND', 'OR', 'BUT', 'IF', 'WHEN', 'THAN', 'BECAUSE', 'WHILE', 'AFTER', 'BEFORE',
    # Question words
    'WHAT', 'WHO', 'WHICH', 'WHERE', 'WHY', 'WHEN', 'HOW',
    # Weather
    'SUNNY', 'CLOUDY', 'RAIN', 'SNOW', 'WIND', 'STORM', 'COLD', 'HOT', 'WARM', 'COOL',
    # Locations
    'NORTH', 'SOUTH', 'EAST', 'WEST', 'CITY', 'TOWN', 'HOME', 'WORK', 'OFFICE',
    # More variety
    'HELLO', 'THANKS', 'PLEASE', 'SORRY', 'YES', 'NO', 'MAYBE', 'SURE', 'RIGHT', 'WRONG',
    'TODAY', 'TOMORROW', 'YESTERDAY', 'MORNING', 'NIGHT', 'EVENING', 'AFTERNOON',
    'WATER', 'FIRE', 'EARTH', 'AIR', 'LIGHT', 'DARK', 'BLACK', 'WHITE', 'RED', 'BLUE',
    'GREEN', 'YELLOW', 'FAST', 'SLOW', 'START', 'STOP', 'OPEN', 'CLOSE', 'SEND', 'RECEIVE',
]

# Grid squares (Maidenhead locator system)
GRID_LETTERS_1 = 'ABCDEFGHIJKLMNOPQR'
GRID_LETTERS_2 = 'ABCDEFGHIJKLMNOPQRSTUVWX'

# General sentences/phrases for language variety
GENERAL_SENTENCES = [
    # News headlines
    "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG",
    "BREAKING NEWS FROM AROUND THE WORLD",
    "WEATHER FORECAST CALLS FOR SUNNY SKIES",
    "STOCK MARKET REACHES NEW HIGH TODAY",
    "SCIENTISTS DISCOVER NEW SPECIES",
    "ELECTION RESULTS EXPECTED TONIGHT",
    "TECHNOLOGY COMPANY ANNOUNCES MERGER",
    "SPORTS TEAM WINS CHAMPIONSHIP GAME",
    "CONCERT TICKETS ON SALE NOW",
    "TRAFFIC DELAYS EXPECTED ON HIGHWAY",
    # Conversational
    "HELLO HOW ARE YOU TODAY",
    "THANK YOU FOR YOUR MESSAGE",
    "PLEASE SEND MORE INFORMATION",
    "LOOKING FORWARD TO HEARING FROM YOU",
    "THE MEETING IS SCHEDULED FOR TOMORROW",
    "I WILL BE THERE AT NOON",
    "SEE YOU LATER",
    "HAVE A GREAT DAY",
    "BEST WISHES TO YOU AND YOUR FAMILY",
    "CONGRATULATIONS ON YOUR SUCCESS",
    # Descriptive
    "THE SUN IS SHINING BRIGHTLY",
    "IT IS A BEAUTIFUL DAY OUTSIDE",
    "THE MOUNTAINS ARE COVERED WITH SNOW",
    "THE OCEAN WAVES CRASH ON THE SHORE",
    "BIRDS ARE SINGING IN THE TREES",
    "THE CITY LIGHTS SPARKLE AT NIGHT",
    # Instructions
    "PLEASE TURN LEFT AT THE CORNER",
    "OPEN THE DOOR AND COME INSIDE",
    "PRESS THE BUTTON TO START",
    "WAIT FOR THE SIGNAL BEFORE CROSSING",
    "FOLLOW THE PATH TO THE END",
    # Questions
    "WHAT TIME DOES THE TRAIN LEAVE",
    "WHERE IS THE NEAREST STATION",
    "HOW MUCH DOES IT COST",
    "CAN YOU HELP ME WITH THIS",
    "DO YOU KNOW THE WAY",
    # Technical
    "SYSTEM STATUS IS NORMAL",
    "DATA TRANSMISSION COMPLETE",
    "SIGNAL STRENGTH IS GOOD",
    "FREQUENCY ADJUSTMENT NEEDED",
    "EQUIPMENT CHECK REQUIRED",
    # Numbers in context
    "THE TEMPERATURE IS TWENTY DEGREES",
    "THERE ARE FIFTY PEOPLE HERE",
    "THE BUILDING HAS TEN FLOORS",
    "WE NEED THREE MORE UNITS",
    "DELIVERY EXPECTED IN FIVE DAYS",
]

# Additional common words for variety
EXTRA_COMMON_WORDS = [
    # Time words
    "ALWAYS", "NEVER", "SOMETIMES", "OFTEN", "USUALLY", "RARELY",
    "SOON", "LATER", "EARLY", "LATE", "ALREADY", "YET",
    # Size/amount
    "SMALL", "LARGE", "TINY", "HUGE", "FEW", "MANY", "SEVERAL", "SOME",
    # Feelings
    "HAPPY", "SAD", "ANGRY", "CALM", "EXCITED", "WORRIED", "TIRED",
    # Actions
    "RUN", "WALK", "JUMP", "CLIMB", "SWIM", "FLY", "DRIVE", "RIDE",
    "READ", "WRITE", "SPEAK", "LISTEN", "WATCH", "LOOK", "SEE", "HEAR",
    "EAT", "DRINK", "SLEEP", "WAKE", "WORK", "PLAY", "REST", "MOVE",
    # Objects
    "BOOK", "PHONE", "CAR", "HOUSE", "DOOR", "WINDOW", "TABLE", "CHAIR",
    "TREE", "FLOWER", "RIVER", "MOUNTAIN", "ROAD", "BRIDGE", "BUILDING",
    # People
    "FRIEND", "FAMILY", "MOTHER", "FATHER", "BROTHER", "SISTER", "CHILD",
    "TEACHER", "DOCTOR", "WORKER", "DRIVER", "STUDENT", "MANAGER",
    # Abstract
    "IDEA", "PLAN", "GOAL", "DREAM", "HOPE", "FEAR", "LOVE", "HATE",
    "TRUTH", "PEACE", "FREEDOM", "POWER", "MONEY", "HEALTH", "SUCCESS",
]


class TextGenerator:
    """
    Generates diverse text for training audio-to-text models.
    
    Optimized for CTC training where we need pattern coverage,
    not language modeling.
    """
    
    def __init__(self, config: Optional[TextGeneratorConfig] = None):
        self.config = config or TextGeneratorConfig()
        self._build_charset()
    
    def _build_charset(self):
        """Build the character set based on config and mode."""
        self.charset = set()
        
        # Get mode-specific character set
        mode_chars = MODE_CHARSETS.get(self.config.mode, MODE_CHARSETS['GENERAL'])
        
        if self.config.include_letters:
            self.charset.update(mode_chars['letters'])
        
        if self.config.include_numbers:
            self.charset.update(mode_chars['numbers'])
        
        if self.config.include_punctuation or self.config.include_extended_punct:
            self.charset.update(mode_chars['punctuation'])
        
        if self.config.include_lowercase and 'abcdefghijklmnopqrstuvwxyz' in mode_chars['letters']:
            self.charset.update(string.ascii_lowercase)
        
        self.charset.update(mode_chars['special'])
        self.charset = sorted(self.charset)
        
        # Store mode-specific chars for coverage
        self.mode_punctuation = mode_chars['punctuation']
    
    def generate_callsign(self) -> str:
        """
        Generate a realistic ham radio callsign.
        
        Formats:
        - W1ABC (1 letter prefix + 1 digit + 2-3 letters)
        - KA1ABC (2 letter prefix + 1 digit + 2-3 letters)
        - W10ABC (for special events)
        """
        prefix = random.choice(CALLSIGN_PREFIXES)
        
        # Number (usually 0-9, occasionally 10)
        if random.random() < 0.02:
            num = '10'
        else:
            num = str(random.randint(0, 9))
        
        # Suffix (1-3 letters)
        suffix_len = random.choice([1, 2, 2, 2, 3, 3, 3])  # 2-3 most common
        suffix = ''.join(random.choices(string.ascii_uppercase, k=suffix_len))
        
        return f"{prefix}{num}{suffix}"
    
    def generate_callsign_exchange(self) -> str:
        """
        Generate a typical callsign exchange.
        
        Examples:
        - "CQ CQ CQ DE W1ABC W1ABC K"
        - "W1ABC DE K3XYZ K3XYZ"
        - "TU 73 DE W1ABC SK"
        """
        patterns = [
            lambda: f"CQ CQ CQ DE {self.generate_callsign()} {self.generate_callsign()} K",
            lambda: f"CQ DX DE {self.generate_callsign()} K",
            lambda: f"{self.generate_callsign()} DE {self.generate_callsign()} {self.generate_callsign()} K",
            lambda: f"TU {random.choice(['73', '88'])} DE {self.generate_callsign()} SK",
            lambda: f"{self.generate_callsign()} {self.generate_callsign()} DE {self.generate_callsign()} K",
            lambda: f"QRZ DE {self.generate_callsign()} K",
        ]
        
        return random.choice(patterns)()
    
    def generate_qcode_text(self) -> str:
        """Generate text using Q-codes."""
        qcode = random.choice(Q_CODES)
        
        # Sometimes with ? (asking) or without (telling)
        if random.random() < 0.3:
            return f"{qcode}?"
        
        # Sometimes with additional info
        if qcode == 'QTH':
            locations = ['NEW YORK', 'LOS ANGELES', 'LONDON', 'TOKYO', 'PARIS', 
                        'BERLIN', 'SYDNEY', 'TORONTO', 'MOSCOW', 'BEIJING']
            return f"QTH {random.choice(locations)}"
        elif qcode == 'RST':
            return f"RST {random.choice(RST_REPORTS)}"
        elif qcode == 'QRG':
            freq = random.randint(3500, 28500)
            return f"QRG {freq} KHZ"
        
        return qcode
    
    def generate_abbreviation_text(self) -> str:
        """Generate text using ham radio abbreviations."""
        num_abbrevs = random.randint(1, 4)
        abbrevs = random.sample(ABBREVIATIONS, min(num_abbrevs, len(ABBREVIATIONS)))
        return ' '.join(abbrevs)
    
    def generate_number_text(self) -> str:
        """
        Generate number-heavy text.
        
        Examples: RST reports, frequencies, times, dates
        """
        patterns = [
            # RST report
            lambda: f"UR RST {random.choice(RST_REPORTS)}",
            # Frequency
            lambda: f"{random.randint(1800, 28500)} KHZ",
            # Power
            lambda: f"{random.choice([5, 10, 25, 50, 100, 500, 1000, 1500])} WATTS",
            # Time
            lambda: f"{random.randint(0, 23):02d}{random.randint(0, 59):02d} UTC",
            # Date
            lambda: f"{random.randint(1, 31):02d} {random.choice(['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'])}",
            # Grid square
            lambda: f"{random.choice(GRID_LETTERS_1)}{random.choice(GRID_LETTERS_1)}{random.randint(0, 9)}{random.randint(0, 9)}{random.choice(GRID_LETTERS_2.lower()).upper()}{random.choice(GRID_LETTERS_2.lower()).upper()}",
            # Contest exchange
            lambda: f"NR {random.randint(1, 999)} {random.choice(['CA', 'NY', 'TX', 'FL', 'OH', 'PA', 'IL', 'MI', 'GA', 'NC'])}",
            # Temperature
            lambda: f"{random.randint(-20, 40)} DEGREES",
            # Random numbers
            lambda: ' '.join(str(random.randint(0, 9)) for _ in range(random.randint(3, 8))),
        ]
        
        return random.choice(patterns)()
    
    def generate_word_text(self) -> str:
        """Generate text from word list."""
        num_words = random.randint(2, 8)
        
        # Mix words from both lists
        all_words = COMMON_WORDS + EXTRA_COMMON_WORDS
        words = random.sample(all_words, min(num_words, len(all_words)))
        return ' '.join(words)
    
    def generate_sentence_text(self) -> str:
        """
        Generate full sentences or sentence fragments.
        Good for teaching natural language patterns.
        """
        # Sometimes use full sentence
        if random.random() < 0.6:
            return random.choice(GENERAL_SENTENCES)
        
        # Sometimes use sentence fragment + extra words
        sentence = random.choice(GENERAL_SENTENCES)
        words = sentence.split()
        
        # Take a chunk
        start = random.randint(0, max(0, len(words) - 3))
        length = random.randint(3, min(6, len(words) - start))
        fragment = ' '.join(words[start:start + length])
        
        # Maybe add extra words
        if random.random() < 0.3:
            extra = random.sample(COMMON_WORDS, random.randint(1, 2))
            fragment = fragment + ' ' + ' '.join(extra)
        
        return fragment
    
    def generate_random_text(self, length: int = None) -> str:
        """
        Generate random character sequence.
        
        This is important for:
        - Ensuring all character pairs are seen
        - Adversarial training
        - Preventing overfitting to common patterns
        """
        if length is None:
            length = random.randint(
                self.config.min_length,
                self.config.max_length
            )
        
        # Mix of different random patterns
        pattern_type = random.choice(['uniform', 'words', 'mixed'])
        
        if pattern_type == 'uniform':
            # Pure random characters
            chars = random.choices(
                [c for c in self.charset if c != ' '],
                k=length
            )
            # Add some spaces
            for i in range(0, len(chars), random.randint(3, 6)):
                if i < len(chars):
                    chars[i] = ' '
            return ''.join(chars).strip()
        
        elif pattern_type == 'words':
            # Random "words" of 2-5 chars
            words = []
            total = 0
            while total < length:
                word_len = random.randint(2, 5)
                word = ''.join(random.choices(
                    [c for c in self.charset if c != ' '],
                    k=word_len
                ))
                words.append(word)
                total += word_len + 1
            return ' '.join(words)[:length]
        
        else:  # mixed
            # Mix of random and real words
            parts = []
            for _ in range(random.randint(2, 5)):
                if random.random() < 0.5:
                    parts.append(random.choice(COMMON_WORDS))
                else:
                    parts.append(''.join(random.choices(
                        [c for c in self.charset if c != ' '],
                        k=random.randint(2, 5)
                    )))
            return ' '.join(parts)
    
    def generate_punctuation_text(self) -> str:
        """
        Generate text with heavy punctuation usage.
        Only uses punctuation valid for the current mode.
        """
        punct = self.mode_punctuation
        patterns = []
        
        # Build patterns based on available punctuation
        if '?' in punct:
            patterns.append(lambda: "WHAT? HOW? WHY?")
            patterns.append(lambda: "QRZ? QTH? QSL?")
        
        if '!' in punct:
            patterns.append(lambda: "HELLO! YES! OK!")
            patterns.append(lambda: "GREAT! THANKS!")
        
        if '(' in punct and ')' in punct:
            patterns.append(lambda: "SIGNAL (GOOD) HERE")
            patterns.append(lambda: "FREQ (7040) KHZ")
        
        if ':' in punct:
            patterns.append(lambda: "TIME: 1430 UTC")
            patterns.append(lambda: "RST: 599")
        
        if ',' in punct:
            patterns.append(lambda: "A, B, C, D")
            patterns.append(lambda: "YES, NO, MAYBE")
        
        if '.' in punct:
            patterns.append(lambda: "HELLO. GOODBYE.")
            patterns.append(lambda: "1. FIRST 2. SECOND")
        
        if '+' in punct:
            patterns.append(lambda: "5+3 10+4")
        
        if '-' in punct:
            patterns.append(lambda: "10-4 OVER-OUT")
        
        if '/' in punct:
            patterns.append(lambda: "W1ABC/P PORTABLE")
            patterns.append(lambda: "YES/NO AND/OR")
        
        if "'" in punct:
            patterns.append(lambda: "IT'S JOHN'S")
            patterns.append(lambda: "DON'T CAN'T")
        
        if '$' in punct:
            patterns.append(lambda: "PRICE $50 $100")
        
        if '#' in punct:
            patterns.append(lambda: "#1 #2 #3")
        
        if '&' in punct:
            patterns.append(lambda: "A&B C&D")
        
        if ';' in punct:
            patterns.append(lambda: "HELLO; GOODBYE")
        
        # Always add random punctuation sequence
        if punct:
            patterns.append(lambda: ' '.join(random.choices(list(punct), k=random.randint(3, 6))))
        
        if not patterns:
            return "NO PUNCTUATION"
        
        return random.choice(patterns)()
    
    def generate_lowercase_text(self) -> str:
        """
        Generate mixed-case conversational text.
        For modes like PSK31 that support lowercase.
        """
        patterns = [
            # Natural conversational
            "Hello, how are you today?",
            "Thanks for the contact, 73!",
            "The weather here is nice and sunny.",
            "I'm using a dipole antenna.",
            "What's your QTH?",
            "Nice to meet you!",
            "Can you hear me okay?",
            "Let's try again later.",
            "Good luck in the contest!",
            "See you on the bands.",
            # Technical with mixed case
            "Running 100w into a Yagi.",
            "Signal is 5/9 here!",
            "Using FLdigi for PSK31.",
            "My rig is an Icom IC-7300.",
            "Antenna is at 40 feet.",
            # Casual chat
            "Hi there, first time on PSK31.",
            "Great signal, very clean!",
            "Sorry, QRM here.",
            "Will QSL via LoTW.",
            "Take care, bye for now.",
        ]
        
        text = random.choice(patterns)
        
        # Sometimes add random case variations
        if random.random() < 0.3:
            text = ''.join(
                c.lower() if random.random() < 0.5 else c.upper()
                for c in text
            )
        
        return text
    
    def generate_one(self) -> str:
        """
        Generate a single text sample based on configured ratios.
        """
        r = random.random()
        cumulative = 0
        
        # Words (largest ratio - general language)
        cumulative += self.config.word_ratio
        if r < cumulative:
            text = self.generate_word_text()
            # For PSK31, sometimes convert to lowercase
            if self.config.include_lowercase and random.random() < 0.3:
                text = text.lower()
            return text
        
        # Sentences (natural language)
        cumulative += self.config.sentence_ratio
        if r < cumulative:
            text = self.generate_sentence_text()
            # For PSK31, sometimes use mixed case
            if self.config.include_lowercase and random.random() < 0.5:
                return self.generate_lowercase_text()
            return text
        
        # Callsign (radio-specific)
        cumulative += self.config.callsign_ratio
        if r < cumulative:
            return self.generate_callsign_exchange()
        
        # Random (adversarial pattern coverage)
        cumulative += self.config.random_ratio
        if r < cumulative:
            return self.generate_random_text()
        
        # Punctuation (mode-specific)
        cumulative += self.config.punctuation_ratio
        if r < cumulative:
            return self.generate_punctuation_text()
        
        # Numbers
        cumulative += self.config.number_ratio
        if r < cumulative:
            return self.generate_number_text()
        
        # Q-code
        cumulative += self.config.qcode_ratio
        if r < cumulative:
            return self.generate_qcode_text()
        
        # Abbreviation
        cumulative += self.config.abbreviation_ratio
        if r < cumulative:
            return self.generate_abbreviation_text()
        
        # Fallback to words
        return self.generate_word_text()
    
    def generate_batch(self, count: int, ensure_coverage: bool = True) -> List[str]:
        """
        Generate a batch of text samples.
        
        Args:
            count: Number of samples to generate
            ensure_coverage: If True, ensure all characters and common bigrams appear
        """
        texts = []
        
        if ensure_coverage:
            # First, generate samples that ensure character coverage
            coverage_texts = self._generate_coverage_samples()
            texts.extend(coverage_texts)
            count -= len(coverage_texts)
        
        # Generate remaining samples
        for _ in range(max(0, count)):
            texts.append(self.generate_one())
        
        # Shuffle
        random.shuffle(texts)
        
        return texts
    
    def _generate_coverage_samples(self) -> List[str]:
        """
        Generate samples that ensure all characters appear.
        Also ensures common bigrams are represented.
        Mode-specific to cover all supported characters.
        """
        coverage_samples = []
        
        # Single character samples (each char appears solo)
        for char in self.charset:
            if char != ' ':
                # Char repeated with spaces
                coverage_samples.append(f"{char} {char} {char}")
        
        # Bigram coverage (important pairs)
        letters = [c for c in self.charset if c.isalpha()]
        important_pairs = []
        for c1 in letters[:10]:  # First 10 letters
            for c2 in letters[:10]:
                important_pairs.append(f"{c1}{c2}")
        
        # Add some of these
        for pair in random.sample(important_pairs, min(50, len(important_pairs))):
            coverage_samples.append(f"{pair} {pair} {pair}")
        
        # All digits
        coverage_samples.append('0 1 2 3 4 5 6 7 8 9')
        coverage_samples.append('0123456789')
        
        # Mode-specific punctuation coverage
        if self.mode_punctuation:
            # Each punctuation mark in context
            for p in self.mode_punctuation:
                coverage_samples.append(f"A{p}B {p} C{p}D")
            
            # Only add context samples with characters valid for this mode
            if '?' in self.mode_punctuation and '!' in self.mode_punctuation:
                coverage_samples.append(f"HELLO? YES! OK")
            if ':' in self.mode_punctuation and '$' in self.mode_punctuation:
                coverage_samples.append(f"PRICE: $50")
            if '(' in self.mode_punctuation and ')' in self.mode_punctuation:
                coverage_samples.append(f"SIGNAL (GOOD)")
            if '/' in self.mode_punctuation:
                coverage_samples.append(f"A/B W1ABC/P")
            if '+' in self.mode_punctuation:
                coverage_samples.append(f"C+D 5+3")
            if '-' in self.mode_punctuation:
                coverage_samples.append(f"E-F 10-4")
            if "'" in self.mode_punctuation:
                coverage_samples.append(f"IT'S JOHN'S")
            if ':' in self.mode_punctuation:
                coverage_samples.append(f"TIME: 14:30")
            if '#' in self.mode_punctuation:
                coverage_samples.append(f"#1 #2 #3")
            if '&' in self.mode_punctuation:
                coverage_samples.append(f"A&B C&D")
        
        # Lowercase coverage for PSK31
        if self.config.include_lowercase:
            for char in string.ascii_lowercase:
                coverage_samples.append(f"{char} {char} {char}")
            
            # Mixed case patterns
            coverage_samples.append("Hello World")
            coverage_samples.append("The Quick Brown Fox")
            coverage_samples.append("abcdefghijklm nopqrstuvwxyz")
            coverage_samples.append("AaBbCcDdEeFf")
        
        # Prosigns if enabled
        if self.config.include_prosigns:
            for prosign in PROSIGNS:
                coverage_samples.append(prosign)
        
        return coverage_samples
    
    def analyze_coverage(self, texts: List[str]) -> dict:
        """
        Analyze character coverage in generated texts.
        """
        all_text = ' '.join(texts)
        char_counts = Counter(all_text)
        
        # Bigram counts
        bigrams = [all_text[i:i+2] for i in range(len(all_text) - 1)]
        bigram_counts = Counter(bigrams)
        
        # Missing characters
        missing_chars = [c for c in self.charset if c not in char_counts]
        
        # Character frequency distribution
        total_chars = sum(char_counts.values())
        char_freq = {c: count / total_chars for c, count in char_counts.items()}
        
        return {
            'total_samples': len(texts),
            'total_chars': total_chars,
            'unique_chars': len(char_counts),
            'missing_chars': missing_chars,
            'unique_bigrams': len(bigram_counts),
            'char_counts': dict(char_counts.most_common()),
            'min_char_freq': min(char_freq.values()) if char_freq else 0,
            'max_char_freq': max(char_freq.values()) if char_freq else 0,
        }


def generate_texts_for_mode(
    mode: str,
    count: int = 1000
) -> List[str]:
    """
    Generate texts optimized for a specific mode.
    
    All modes include general language capability with mode-specific additions.
    Each mode uses its full supported character set.
    
    Args:
        mode: 'CW', 'PSK31', 'RTTY', 'FT8', 'GENERAL'
        count: Number of samples
        
    Returns:
        List of text strings
    """
    configs = {
        # General-purpose: balanced for any audio source
        'GENERAL': TextGeneratorConfig(
            mode='GENERAL',
            word_ratio=0.40,           # Natural language
            sentence_ratio=0.20,       # Full sentences
            callsign_ratio=0.10,       # Some radio
            number_ratio=0.10,         # Numbers
            random_ratio=0.15,         # Adversarial
            punctuation_ratio=0.00,
            qcode_ratio=0.03,
            abbreviation_ratio=0.02,
            include_prosigns=False,
            include_punctuation=False,
            include_lowercase=False,
        ),
        # CW: Mix of radio and general, limited punctuation
        'CW': TextGeneratorConfig(
            mode='CW',
            word_ratio=0.30,           # General language
            sentence_ratio=0.15,       # Full sentences
            callsign_ratio=0.20,       # Radio callsigns
            number_ratio=0.10,         # Numbers
            random_ratio=0.15,         # Adversarial
            punctuation_ratio=0.00,
            qcode_ratio=0.05,
            abbreviation_ratio=0.05,
            include_prosigns=True,
            include_punctuation=True,  # CW has .,?/-=
            include_lowercase=False,
        ),
        # PSK31: Conversational with full ASCII (lowercase + punctuation)
        'PSK31': TextGeneratorConfig(
            mode='PSK31',
            word_ratio=0.25,           # General language
            sentence_ratio=0.20,       # Full sentences
            callsign_ratio=0.15,       # Radio callsigns
            number_ratio=0.08,         # Numbers
            random_ratio=0.12,         # Adversarial
            punctuation_ratio=0.10,    # Punctuation-heavy samples
            qcode_ratio=0.05,
            abbreviation_ratio=0.05,
            include_prosigns=False,
            include_punctuation=True,  # Full ASCII punctuation
            include_lowercase=True,    # PSK31 is conversational
            include_extended_punct=True,
        ),
        # RTTY: News/teletype style with Baudot figures
        'RTTY': TextGeneratorConfig(
            mode='RTTY',
            word_ratio=0.20,           # Natural language
            sentence_ratio=0.20,       # Headlines
            callsign_ratio=0.15,       # Radio callsigns
            number_ratio=0.15,         # Numbers with figures
            random_ratio=0.10,         # Adversarial
            punctuation_ratio=0.10,    # Baudot figures (!#$&'()+,-./:;?)
            qcode_ratio=0.05,
            abbreviation_ratio=0.05,
            include_prosigns=False,
            include_punctuation=True,  # Baudot FIGURES
            include_lowercase=False,   # RTTY is uppercase only
            include_extended_punct=True,
        ),
        # FT8: Highly structured (mostly callsigns), minimal punctuation
        'FT8': TextGeneratorConfig(
            mode='FT8',
            callsign_ratio=0.45,       # FT8 is callsign-heavy
            number_ratio=0.25,         # Grid squares, signal reports
            word_ratio=0.10,
            sentence_ratio=0.05,
            random_ratio=0.10,
            punctuation_ratio=0.03,    # Just /+-.
            qcode_ratio=0.02,
            abbreviation_ratio=0.00,
            include_prosigns=False,
            include_punctuation=True,
            include_lowercase=False,
        ),
        # VOICE: For speech-to-text, mostly natural language
        'VOICE': TextGeneratorConfig(
            mode='GENERAL',
            word_ratio=0.35,
            sentence_ratio=0.40,       # Natural speech
            number_ratio=0.10,
            random_ratio=0.10,
            punctuation_ratio=0.00,
            callsign_ratio=0.03,
            qcode_ratio=0.01,
            abbreviation_ratio=0.01,
            include_prosigns=False,
            include_punctuation=False,
            include_lowercase=False,
        ),
    }
    
    config = configs.get(mode.upper(), configs['GENERAL'])
    generator = TextGenerator(config)
    
    return generator.generate_batch(count, ensure_coverage=True)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate training text')
    parser.add_argument('--mode', type=str, default='CW',
                       choices=['CW', 'PSK31', 'RTTY', 'FT8'],
                       help='Mode to generate text for')
    parser.add_argument('--count', type=int, default=100,
                       help='Number of samples')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze coverage')
    
    args = parser.parse_args()
    
    texts = generate_texts_for_mode(args.mode, args.count)
    
    print(f"Generated {len(texts)} samples for {args.mode}")
    print("\nSamples:")
    for i, text in enumerate(texts[:20]):
        print(f"  {i+1:3d}: {text}")
    
    if args.analyze:
        gen = TextGenerator()
        analysis = gen.analyze_coverage(texts)
        print(f"\nCoverage Analysis:")
        print(f"  Total chars: {analysis['total_chars']}")
        print(f"  Unique chars: {analysis['unique_chars']}")
        print(f"  Missing chars: {analysis['missing_chars']}")
        print(f"  Unique bigrams: {analysis['unique_bigrams']}")
        print(f"  Min char freq: {analysis['min_char_freq']:.4f}")
        print(f"  Max char freq: {analysis['max_char_freq']:.4f}")

