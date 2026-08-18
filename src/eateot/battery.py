"""IQ battery scoring logic.

The question spec itself lives in versioned YAML under
``config/questionnaires/`` (see ``eateot.questionnaires``) — this module only
holds the pure ``evaluate_response`` scoring function.
"""

import re

from .questionnaires import DEFAULT_BATTERY, load_battery

# The default battery loaded from YAML. Swap it at runtime by calling
# ``load_battery("other_name")`` and passing the result to ``run_iq_test``.
IQ_TEST_BATTERY = load_battery(DEFAULT_BATTERY)


# A compact lexicon for category-fluency scoring. Large enough for realistic
# 60-second animal-fluency responses; add categories as needed. Matching is
# case/plural-insensitive via naive singularization (see _extract_fluency_tokens).
ANIMAL_LEXICON = frozenset({
    "dog", "cat", "bird", "fish", "horse", "cow", "pig", "sheep", "goat",
    "chicken", "duck", "turkey", "rooster", "rabbit", "mouse", "rat",
    "hamster", "guinea pig", "gerbil", "ferret", "bear", "lion", "tiger",
    "leopard", "cheetah", "wolf", "fox", "coyote", "deer", "elk", "moose",
    "bison", "buffalo", "elephant", "giraffe", "zebra", "rhinoceros",
    "hippopotamus", "kangaroo", "koala", "panda", "monkey", "ape", "gorilla",
    "chimpanzee", "orangutan", "baboon", "lemur", "sloth", "anteater",
    "armadillo", "hedgehog", "porcupine", "squirrel", "chipmunk", "beaver",
    "otter", "raccoon", "skunk", "badger", "weasel", "mink", "mole", "bat",
    "whale", "dolphin", "porpoise", "seal", "sea lion", "walrus", "shark",
    "octopus", "squid", "crab", "lobster", "shrimp", "clam", "oyster",
    "starfish", "jellyfish", "crocodile", "alligator", "lizard", "snake",
    "turtle", "tortoise", "frog", "toad", "salamander", "newt", "eagle",
    "hawk", "owl", "falcon", "vulture", "raven", "crow", "magpie",
    "sparrow", "robin", "bluebird", "cardinal", "woodpecker", "parrot",
    "penguin", "ostrich", "emu", "flamingo", "swan", "goose", "peacock",
    "pigeon", "seagull", "pelican", "stork", "heron", "hummingbird",
    "butterfly", "bee", "wasp", "ant", "spider", "scorpion", "snail",
    "worm", "ladybug", "grasshopper", "dragonfly", "mosquito", "fly",
    "cricket", "caterpillar", "tadpole", "platypus", "wombat", "possum",
    "opossum", "camel", "llama", "alpaca", "donkey", "mule", "panther",
    "jaguar", "cougar", "puma", "hyena", "jackal", "dingo", "aardvark",
})


# Semantic-category lexicons for fluency scoring. Matching is case/plural-
# insensitive via naive singularization (see _extract_fluency_tokens).
FRUIT_LEXICON = frozenset({
    "apple", "banana", "orange", "grape", "strawberry", "blueberry",
    "raspberry", "blackberry", "cherry", "peach", "plum", "apricot",
    "nectarine", "pear", "mango", "papaya", "pineapple", "watermelon",
    "cantaloupe", "honeydew", "melon", "kiwi", "lemon", "lime",
    "grapefruit", "tangerine", "mandarin", "pomegranate", "fig", "date",
    "coconut", "avocado", "guava", "lychee", "passion fruit", "cranberry",
    "gooseberry", "currant", "persimmon", "quince", "durian", "mulberry",
    "boysenberry", "elderberry", "raisin", "prune", "olive", "tomato",
})
VEHICLE_LEXICON = frozenset({
    "car", "truck", "bus", "motorcycle", "bicycle", "bike", "scooter",
    "van", "taxi", "limousine", "sedan", "coupe", "convertible",
    "pickup", "suv", "jeep", "tractor", "forklift", "crane",
    "bulldozer", "excavator", "train", "subway", "tram", "trolley",
    "airplane", "plane", "jet", "helicopter", "glider", "hot air balloon",
    "blimp", "ship", "boat", "ferry", "yacht", "sailboat", "canoe",
    "kayak", "submarine", "hovercraft", "skateboard", "rollerskate",
    "sled", "wagon", "carriage", "gondola", "rickshaw", "moped",
})
FLUENCY_LEXICONS = {
    "animals": ANIMAL_LEXICON,
    "fruits": FRUIT_LEXICON,
    "vehicles": VEHICLE_LEXICON,
}

# Function words are not credited in letter-fluency tests (FAS convention),
# though articles followed by whitespace are already stripped during extraction.
FUNCTION_WORDS = frozenset({
    "a", "an", "and", "at", "as", "of", "or", "for", "from", "in", "on",
    "to", "with", "is", "was", "be", "by", "it", "its", "the", "that",
    "this", "so", "if", "but", "not", "no", "yes", "when", "what", "how",
    "why", "who", "which", "are", "were", "will", "would", "can", "could",
    "should", "shall", "may", "might", "must", "do", "does", "did", "have",
    "has", "had", "you", "your", "we", "they", "them", "he", "she", "his",
    "her", "my", "our", "their", "me", "us", "all", "any", "some", "each",
    "more", "most", "other", "such", "only", "own", "same", "too", "very",
})


def _in_lexicon(lexicon: frozenset, token: str) -> bool:
    """Lexicon membership that tolerates naive over-singularization.

    The naive stemmer turns 'bus' -> 'bu' and 'walrus' -> 'walru'; try the
    token plus a trailing 's' as a fallback so root-s words still match.
    """
    return token in lexicon or (token + "s") in lexicon


def evaluate_fluency(raw_output: str, category: str, target: int, max_points: int,
                     letter: str | None = None):
    """Score a category- or letter-fluency response.

    Semantic categories (``category`` in ``FLUENCY_LEXICONS``) count distinct
    items from the lexicon; a ``letter`` (e.g. "f" for the FAS test) counts
    distinct words starting with that letter; any other category falls back
    to counting all distinct words. Awards proportional points up to
    ``target`` distinct items and penalizes each repetition by one point (a
    perseveration marker). Returns ``(points, status, accuracy_pct, metrics)``
    where ``metrics`` carries total/distinct/repeats/type-token ratio.
    """
    empty_metrics = {"total": 0, "distinct": 0, "repeats": 0, "type_token_ratio": 0.0}
    if not raw_output or raw_output.strip() in ("[NO OUTPUT GENERATED]", "[INFERENCE ERROR]"):
        return 0, "FAILED (Null / Empty Generation)", 0.0, empty_metrics

    tokens = _extract_fluency_tokens(raw_output)
    if letter:
        # Phonemic fluency credits words, not bare letters or function words
        # (FAS convention).
        items = [
            t for t in tokens
            if len(t) > 1 and t not in FUNCTION_WORDS and t.startswith(letter.lower())
        ]
    else:
        lexicon = FLUENCY_LEXICONS.get(category)
        items = [t for t in tokens if _in_lexicon(lexicon, t)] if lexicon else tokens

    if not items:
        return 0, "FAILED (No Category Items)", 0.0, empty_metrics

    total = len(items)
    distinct = len(set(items))
    repeats = total - distinct

    ratio = min(distinct / target, 1.0)
    points = max(0, round(max_points * ratio) - repeats)
    accuracy_pct = round(ratio * 100, 1)

    if points >= max_points:
        status = "PASSED (Full Fluency)"
    elif distinct >= target:
        status = f"PARTIAL (Full Count, {repeats} Repeated)"
    elif distinct > 0:
        status = f"PARTIAL ({distinct}/{target} distinct)"
    else:
        status = "FAILED (No Category Items)"

    metrics = {
        "total": total,
        "distinct": distinct,
        "repeats": repeats,
        "type_token_ratio": round(distinct / total, 3),
    }
    return points, status, accuracy_pct, metrics


def _extract_fluency_tokens(raw_output: str) -> list[str]:
    """Split a fluency response into normalized candidate item tokens."""
    text = raw_output.lower()
    parts = re.split(r"[,;\n.!?()]|\band\b", text)
    tokens = []
    for part in parts:
        tok = re.sub(r"^[\W\d]+", "", part.strip())
        tok = re.sub(r"[\W\d]+$", "", tok)
        # Drop leading articles: 'a dog' / 'the cat' count as their noun.
        tok = re.sub(r"^(a|an|the)\s+", "", tok)
        if not tok:
            continue
        # Naive singularization: foxes -> fox, dogs -> dog (keeps 'bus' intact).
        if tok.endswith("es") and len(tok) > 3:
            tok = tok[:-2]
        elif tok.endswith("s") and len(tok) > 2:
            tok = tok[:-1]
        if tok:
            tokens.append(tok)
    return tokens


def evaluate_question(raw_output: str, item: dict):
    """Dispatch a battery question to its evaluator by ``item['type']``.

    Returns a 4-tuple ``(points, status, accuracy_pct, metrics)``; ``metrics``
    is ``None`` for anchored questions and a diagnostics dict for fluency ones.
    """
    q_type = item.get("type", "anchored")
    if q_type == "fluency":
        cfg = item.get("fluency", {})
        if not isinstance(cfg, dict) or not cfg:
            raise ValueError(
                f"Fluency question '{item.get('domain', '?')}' has no 'fluency' "
                "config (need a category or letter + target)."
            )
        return evaluate_fluency(
            raw_output,
            cfg.get("category", ""),
            int(cfg.get("target", 1)),
            int(item["max_points"]),
            letter=cfg.get("letter"),
        )
    points, status, pct = evaluate_response(
        raw_output, item.get("ground_truth_anchors", []), item["max_points"]
    )
    return points, status, pct, None


def evaluate_response(raw_output: str, ground_truth_anchors: list[list[str]], max_points: int):
    """
    Evaluates response against multiple anchor groups (synonyms).
    Awards proportional partial credit based on how many distinct anchors were matched.
    """
    if not raw_output or raw_output.strip() in ["[NO OUTPUT GENERATED]", "[INFERENCE ERROR]"]:
        return 0, "FAILED (Null / Empty Generation)", 0.0

    text_lower = raw_output.lower()

    # 1. Genuine Perseveration Check (Consecutive phrase loops)
    consecutive_loop_pattern = r'(\b\w+(?:\s+\w+){1,4}\b)(?:\s*\1){3,}'
    if re.search(consecutive_loop_pattern, text_lower):
        return 0, "FAILED (Perseveration / Attractor Loop)", 0.0

    # 2. Normalize text: Unwrap LaTeX \boxed{}, remove markups
    normalized = re.sub(r'\\boxed\{([^}]+)\}', r'\1', raw_output)
    normalized = re.sub(r'[\\()\[\]*_`#]', ' ', normalized).lower()

    # 3. Check each anchor group
    total_anchors = len(ground_truth_anchors)
    matched_anchors = 0

    for anchor_group in ground_truth_anchors:
        group_matched = False
        for synonym in anchor_group:
            syn_clean = synonym.lower().strip()
            pattern = r'\b' + re.escape(syn_clean) + r'\b'
            if re.search(pattern, normalized):
                group_matched = True
                break

        if group_matched:
            matched_anchors += 1

    # 4. Proportional Partial Point Scoring
    match_ratio = matched_anchors / total_anchors
    earned_points = int(round(max_points * match_ratio))

    if matched_anchors == total_anchors:
        status = "PASSED (Full Logical Match)"
    elif matched_anchors > 0:
        status = f"PARTIAL ({matched_anchors}/{total_anchors} anchors matched)"
    else:
        status = "FAILED (Logical Inaccuracy / Confabulation)"

    return earned_points, status, round(match_ratio * 100, 1)
