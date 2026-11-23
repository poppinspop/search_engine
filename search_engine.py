"""
Basic search engine for Text and Web Intelligence.
Run:  python search_engine.py
Only the standard library is required (NLTK's stemmer is used if installed).
"""
import math
import re
from collections import defaultdict

# ---------------------------------------------------------------
# Sample corpus (replace with your own text if you want)
# ---------------------------------------------------------------
DOCS = {
    1: "In July 1969 the Apollo 11 mission landed the first humans on the Moon. Neil Armstrong and Buzz Aldrin walked on the lunar surface while Michael Collins stayed in orbit around the Moon. The crew returned to Earth with rock samples and a splashdown in the Pacific Ocean.",
    2: "The International Space Station orbits Earth about every ninety minutes at an altitude of roughly four hundred kilometres. Astronauts from many countries live and work there for months, running experiments in microgravity. The station has been continuously occupied since the year 2000.",
    3: "Mars rovers such as Curiosity and Perseverance explore the red planet. They drive across rocky terrain, drill into rocks and search for signs of ancient life. Perseverance landed in Jezero crater in 2021 and is collecting rock samples for a future return to Earth.",
    4: "A rocket works by throwing hot gas out of its engine, which pushes the vehicle upward. Reusable rockets have boosters that fly back and land after launch, which lowers the cost of reaching orbit. Engineers study aerodynamics and propulsion to make launches safer and cheaper.",
    5: "Astronaut training takes years. Candidates practise spacewalks in huge water pools, ride centrifuges to feel high gravity, and learn robotics, medicine and survival skills. They also train to work as a team during long missions in a small spacecraft.",
    6: "Space telescopes see the universe without interference from the atmosphere. The Hubble telescope has photographed distant galaxies for decades, and the James Webb telescope uses infrared light to study the first stars and galaxies. These observations help scientists understand black holes and how the universe began.",
    7: "Satellites orbit Earth for communication, navigation, weather forecasting and observation. GPS satellites help phones and aircraft find their position, while weather satellites track storms and cyclones. Thousands of satellites are now in orbit, which creates a growing problem with space debris.",
    8: "India's space agency ISRO built the Chandrayaan missions to explore the Moon. Chandrayaan-3 landed near the lunar south pole in 2023, making India the fourth country to land on the Moon. ISRO also launched the Mangalyaan orbiter to Mars and is preparing the Gaganyaan mission to send Indian astronauts into space.",
}
# Link graph for PageRank (page -> pages it links to). Treat it like hyperlinks between articles.
LINKS = {1: [2, 5, 8], 2: [5, 7, 1], 3: [4, 6], 4: [7, 2, 8], 5: [2, 1], 6: [3, 7], 7: [2, 6], 8: [1, 3, 4]}


# ---------------------------------------------------------------
# UNIT I: Preprocessing (tokenize, stop words, normalize, stem)
# ---------------------------------------------------------------
STOPWORDS = {"a", "an", "the", "of", "to", "and", "or", "not", "is", "for",
             "on", "in", "each", "it", "like", "using", "between", "them"}
try:
    from nltk.stem import PorterStemmer
    _stem = PorterStemmer().stem
except ImportError:  # crude fallback stemmer
    def _stem(w):
        for suf in ("ing", "ed", "s"):
            if w.endswith(suf) and len(w) > len(suf) + 2:
                return w[: -len(suf)]
        return w


def preprocess(text, remove_stop=True):
    tokens = re.findall(r"[a-z0-9]+", text.lower())      # tokenize + lowercase
    if remove_stop:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return [_stem(t) for t in tokens]                     # stemming


# ---------------------------------------------------------------
# UNIT I: Inverted index (dictionary -> postings list)
# ---------------------------------------------------------------
index = defaultdict(set)      # term -> set of doc ids
tf = {}                       # doc id -> {term: count}
for doc_id, text in DOCS.items():
    terms = preprocess(text)
    tf[doc_id] = {t: terms.count(t) for t in set(terms)}
    for t in terms:
        index[t].add(doc_id)
postings = {t: sorted(d) for t, d in index.items()}
ALL_DOCS = set(DOCS)


# ---------------------------------------------------------------
# UNIT I: Boolean retrieval  e.g. "index AND NOT stemming"
# (left-to-right evaluation, no parentheses, to keep it simple)
# ---------------------------------------------------------------
def boolean_search(query):
    tokens = query.split()
    result, op, negate = None, "AND", False
    for tok in tokens:
        if tok in ("AND", "OR"):
            op = tok
        elif tok == "NOT":
            negate = True
        else:
            words = preprocess(tok, False)
            if not words:
                continue
            docs = set(postings.get(words[0], []))
            if negate:
                docs, negate = ALL_DOCS - docs, False
            if result is None:
                result = docs
            else:
                result = result & docs if op == "AND" else result | docs
    return sorted(result or [])


# ---------------------------------------------------------------
# UNIT I: Wildcard query  e.g. "retriev*"
# ---------------------------------------------------------------
def wildcard_search(pattern):
    regex = re.compile("^" + pattern.lower().replace("*", ".*") + "$")
    docs = set()
    for text_id, text in DOCS.items():
        if any(regex.match(w) for w in re.findall(r"[a-z0-9]+", text.lower())):
            docs.add(text_id)
    return sorted(docs)


# ---------------------------------------------------------------
# UNIT II: tf-idf + cosine similarity ranking
# ---------------------------------------------------------------
N = len(DOCS)
idf = {t: math.log10(N / len(d)) for t, d in index.items()}


def vec(counts):
    return {t: (1 + math.log10(c)) * idf.get(t, 0) for t, c in counts.items()}


doc_vecs = {d: vec(c) for d, c in tf.items()}


def cosine(a, b):
    dot = sum(a[t] * b.get(t, 0) for t in a)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def ranked_search(query, k=3):
    terms = preprocess(query)
    q = vec({t: terms.count(t) for t in set(terms)})
    scores = [(cosine(q, v), d) for d, v in doc_vecs.items()]
    return [(d, round(s, 3)) for s, d in sorted(scores, reverse=True)[:k] if s > 0]


# ---------------------------------------------------------------
# UNIT IV: PageRank (power iteration)
# ---------------------------------------------------------------
def pagerank(links, d=0.85, iters=50):
    pages = list(links)
    pr = {p: 1 / len(pages) for p in pages}
    for _ in range(iters):
        new = {}
        for p in pages:
            inbound = sum(pr[q] / len(links[q]) if links[q] else pr[q] / len(pages)
                          for q in pages if p in links[q] or not links[q])
            new[p] = (1 - d) / len(pages) + d * inbound
        pr = new
    return pr


PR = pagerank(LINKS)


def search(query, k=3):
    """Final engine: tf-idf relevance combined with PageRank importance."""
    hits = []
    for doc_id, cos in ranked_search(query, k=N):
        hits.append((doc_id, 0.8 * cos + 0.2 * PR[doc_id] * N / 2))
    hits.sort(key=lambda x: -x[1])
    return hits[:k]


# ---------------------------------------------------------------
# Demo / simple command-line interface
# ---------------------------------------------------------------
if __name__ == "__main__":
    print("Inverted index (first 5 terms):")
    for t in sorted(postings)[:5]:
        print(f"  {t}: {postings[t]}")
    print("\nBoolean 'moon OR mars AND NOT rover' ->", boolean_search("moon OR mars AND NOT rover"))
    print("Wildcard 'orbit*'                ->", wildcard_search("orbit*"))
    print("PageRank:", {p: round(v, 3) for p, v in PR.items()})
    print("\nType a query (blank to quit). Use AND/OR/NOT for Boolean, * for wildcard, or plain words for ranked search.")
    while True:
        q = input("search> ").strip()
        if not q:
            break
        words = q.split()
        if any(w in ("AND", "OR", "NOT") for w in words):      # Boolean query
            for doc_id in boolean_search(q):
                print(f"  doc {doc_id}: {DOCS[doc_id][:100]}...")
        elif "*" in q:                                         # wildcard query
            for doc_id in wildcard_search(q):
                print(f"  doc {doc_id}: {DOCS[doc_id][:100]}...")
        else:                                                  # ranked query
            for doc_id, score in search(q):
                print(f"  [{score:.3f}] doc {doc_id}: {DOCS[doc_id][:100]}...")