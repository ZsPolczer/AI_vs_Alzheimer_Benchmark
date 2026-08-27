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
        raw_output, item.get("ground_truth_anchors", []), item["max_points"],
        question=item.get("question"),
    )
    return points, status, pct, None


# Echolalia rejection + fractional-credit scoring thresholds.
ECHO_BIGRAM_COVERAGE = 0.85  # fraction of the question's content-word bigrams the response must reproduce
ECHO_MIN_LENGTH_RATIO = 0.8   # response must be at least this long (vs the question) to count as an echo
PARTIAL_CREDIT_FLOOR = 0.5    # minimum synonym-token coverage to earn fractional group credit


def evaluate_response(raw_output: str, ground_truth_anchors: list[list[str]], max_points: int,
                      question: str | None = None):
    """
    Evaluates response against multiple anchor groups (synonyms).

    Stricter than a plain keyword count:

    * **Echolalia rejection** — when ``question`` is supplied and the response
      is a near-verbatim reproduction of the prompt (a classic degradation
      failure), the response is deliberately failed with 0 points. This
      prevents prompt echoes from scoring full marks just because the question
      text itself contains anchor words (e.g. 'State Yes or No').
    * **Fractional group credit** — a group earns full credit when any of its
      synonyms appears verbatim; otherwise it earns partial credit scaled by
      how many of the synonym's content words are present (when coverage is at
      least ``PARTIAL_CREDIT_FLOOR``). Accuracy therefore spans a finer range
      (e.g. 0/25/33/50/67/75/100) instead of jumping in large proportional
      gaps like 0/50/100.
    """
    if not raw_output or raw_output.strip() in ["[NO OUTPUT GENERATED]", "[INFERENCE ERROR]"]:
        return 0, "FAILED (Null / Empty Generation)", 0.0

    # 1. Echolalia rejection: reproducing the prompt must never score.
    if _is_echo_response(raw_output, question):
        return 0, "FAILED (Echolalia / Prompt Echo)", 0.0

    text_lower = raw_output.lower()

    # 2. Genuine Perseveration Check (Consecutive phrase loops)
    consecutive_loop_pattern = r'(\b\w+(?:\s+\w+){1,4}\b)(?:\s*\1){3,}'
    if re.search(consecutive_loop_pattern, text_lower):
        return 0, "FAILED (Perseveration / Attractor Loop)", 0.0

    # 3. Normalize text: Unwrap LaTeX \boxed{}, remove markups
    normalized = re.sub(r'\\boxed\{([^}]+)\}', r'\1', raw_output)
    normalized = re.sub(r'[\\()\[\]*_`#]', ' ', normalized).lower()

    # 4. Score each anchor group: full credit on a verbatim synonym, otherwise
    #    fractional credit from content-word coverage (see docstring).
    total_anchors = len(ground_truth_anchors)
    resp_tokens = set(_content_words(normalized))
    group_credits = []

    for anchor_group in ground_truth_anchors:
        best = 0.0
        for synonym in anchor_group:
            syn_clean = synonym.lower().strip()
            if re.search(r'\b' + re.escape(syn_clean) + r'\b', normalized):
                best = 1.0
                break
            syn_tokens = [t for t in re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", syn_clean)
                          if t not in FUNCTION_WORDS]
            if syn_tokens:
                covered = sum(1 for t in syn_tokens if t in resp_tokens) / len(syn_tokens)
                if covered >= PARTIAL_CREDIT_FLOOR:
                    best = max(best, covered)
        group_credits.append(best)

    # 5. Proportional point scoring over the (possibly fractional) credits.
    match_ratio = sum(group_credits) / total_anchors
    earned_points = int(round(max_points * match_ratio))
    pct = round(match_ratio * 100, 1)

    if match_ratio >= 1.0:
        status = "PASSED (Full Logical Match)"
    elif match_ratio > 0.0:
        status = f"PARTIAL ({pct}% accuracy)"
    else:
        status = "FAILED (Logical Inaccuracy / Confabulation)"

    return earned_points, status, pct


def _content_words(text: str) -> list[str]:
    """Lowercased content words (function words stripped) for lexical metrics."""
    tokens = re.findall(r"\b[a-z]+(?:'[a-z]+)?\b", text.lower())
    return [t for t in tokens if t not in FUNCTION_WORDS]


def _is_echo_response(raw_output: str, question: str | None) -> bool:
    """True when the response is a near-verbatim reproduction of the question.

    Echolalia — the model parroting the prompt instead of answering — is a
    classic degradation failure and must never earn credit. The response
    counts as an echo when it reproduces at least ``ECHO_BIGRAM_COVERAGE`` of
    the question's content-word bigrams AND is at least ``ECHO_MIN_LENGTH_RATIO``
    as long as the question (a terse "Yes, because …" answer is far shorter
    and quotes only part of the premises, so it is not flagged).
    """
    q_tokens = _content_words(question or "")
    r_tokens = _content_words(raw_output or "")
    if len(q_tokens) < 8 or len(r_tokens) < 8:
        return False  # too short to distinguish an echo from a terse answer
    q_bigrams = set(zip(q_tokens, q_tokens[1:]))
    if not q_bigrams:
        return False
    r_bigrams = set(zip(r_tokens, r_tokens[1:]))
    coverage = sum(1 for b in q_bigrams if b in r_bigrams) / len(q_bigrams)
    length_ratio = len(r_tokens) / len(q_tokens)
    return coverage >= ECHO_BIGRAM_COVERAGE and length_ratio >= ECHO_MIN_LENGTH_RATIO


def grade_deterioration(raw_output: str, item: dict | None = None,
                        clean_response: str | None = None) -> float:
    """Compute a 0–100 deterioration grade for a generated answer (higher = worse).

    Self-contained (no clean baseline required). Three normalized signals are
    combined — correctness, lexical repetition, and perseveration:

    * ``correctness``  — anchored questions: fraction of ground-truth anchor
      groups matched (reuses ``evaluate_question``); fluency: distinct/target
      ratio. Defaults to 0.5 (neutral) when no ``item`` is supplied.
    * ``repetition``   — redundancy from the content-word type-token ratio
      (0 when TTR ≥ 0.5, ramps linearly to 1 as TTR → 0).
    * ``perseveration``— 1.0 when the consecutive-loop detector fires.

    Echolalia (the response reproducing the question instead of answering) is
    scored 100.0 outright — it is a severe sign of degradation and must never
    be graded pristine just because the prompt text contained anchor words.

    Grade = 100 · (0.5·(1−correctness) + 0.25·repetition + 0.25·perseveration).

    When ``clean_response`` is given, the content-word bigram overlap (Jaccard)
    with the undegraded answer is folded in as an extra 20% weight, so the
    grade then measures deterioration relative to a clean baseline. Null /
    empty generations score 100 (fully deteriorated).
    """
    if not raw_output or raw_output.strip() in ("[NO OUTPUT GENERATED]",
                                                "[INFERENCE ERROR]", "[NO OUTPUT GENERATED]".lower()):
        return 100.0

    # Echolalia (reproducing the prompt) is a severe clinical sign: fully
    # deteriorated, regardless of any anchor words the question text contains.
    if item is not None and _is_echo_response(raw_output, item.get("question", "")):
        return 100.0

    if item is not None:
        _pts, _status, pct, _metrics = evaluate_question(raw_output, item)
        correctness = pct / 100.0 if pct is not None else 0.5
    else:
        correctness = 0.5
    correctness = min(1.0, max(0.0, correctness))

    words = _content_words(raw_output)
    if len(words) >= 2:
        ttr = len(set(words)) / len(words)
        repetition = min(1.0, max(0.0, (0.5 - ttr) * 2.0))
    else:
        repetition = 1.0  # degenerate / near-empty content

    text_lower = raw_output.lower()
    consecutive_loop_pattern = r'(\b\w+(?:\s+\w+){1,4}\b)(?:\s*\1){3,}'
    perseveration = 1.0 if re.search(consecutive_loop_pattern, text_lower) else 0.0

    grade = 100.0 * (0.5 * (1.0 - correctness) + 0.25 * repetition
                     + 0.25 * perseveration)

    if clean_response and clean_response.strip():
        clean_words = _content_words(clean_response)
        if words and clean_words:
            own = set(zip(words, words[1:]))
            ref = set(zip(clean_words, clean_words[1:]))
            overlap = len(own & ref) / len(own | ref) if (own | ref) else 0.0
            grade = 0.8 * grade + 0.2 * 100.0 * (1.0 - overlap)

    return round(min(100.0, max(0.0, grade)), 1)
