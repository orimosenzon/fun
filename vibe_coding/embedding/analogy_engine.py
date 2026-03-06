import gensim.downloader as api
import random

CATEGORIES = {
    "Animal Sounds": [
        ("dog", "bark"), ("cat", "meow"), ("cow", "moo"),
        ("lion", "roar"), ("bird", "tweet"), ("duck", "quack"),
        ("snake", "hiss"), ("bee", "buzz"), ("horse", "neigh"),
        ("frog", "croak"), ("wolf", "howl"),
    ],
    "Countries & Capitals": [
        ("france", "paris"), ("germany", "berlin"), ("japan", "tokyo"),
        ("italy", "rome"), ("spain", "madrid"), ("russia", "moscow"),
        ("china", "beijing"), ("egypt", "cairo"), ("greece", "athens"),
        ("india", "delhi"), ("brazil", "brasilia"),
    ],
    "Gender Pairs": [
        ("king", "queen"), ("man", "woman"), ("boy", "girl"),
        ("actor", "actress"), ("father", "mother"), ("brother", "sister"),
        ("husband", "wife"), ("son", "daughter"), ("prince", "princess"),
        ("uncle", "aunt"), ("waiter", "waitress"),
    ],
    "Object & Material": [
        ("window", "glass"), ("table", "wood"), ("ring", "gold"),
        ("tire", "rubber"), ("book", "paper"), ("coin", "metal"),
        ("shoe", "leather"), ("shirt", "cotton"), ("sword", "steel"),
    ],
    "Profession & Tool": [
        ("painter", "brush"), ("writer", "pen"), ("chef", "knife"),
        ("carpenter", "hammer"), ("surgeon", "scalpel"), ("teacher", "chalk"),
        ("farmer", "plow"), ("archer", "bow"), ("soldier", "rifle"),
    ],
    "Big & Small": [
        ("ocean", "lake"), ("mountain", "hill"), ("city", "village"),
        ("river", "stream"), ("mansion", "cottage"), ("highway", "path"),
        ("elephant", "mouse"), ("whale", "fish"),
    ],
}


class AnalogyEngine:
    def __init__(self, model_name="glove-wiki-gigaword-100"):
        print(f"Loading model: {model_name} ...")
        self.model = api.load(model_name)
        print("Model loaded.")
        self.categories = self._verify_pairs()
        self._word_pool = self._build_pool()

    def _verify_pairs(self):
        verified = {}
        for category, pairs in CATEGORIES.items():
            valid = [(a, b) for a, b in pairs if a in self.model and b in self.model]
            if len(valid) >= 3:
                verified[category] = valid
        return verified

    def _build_pool(self):
        words = set()
        for pairs in self.categories.values():
            for a, b in pairs:
                words.add(a)
                words.add(b)
        return list(words)

    def get_word_pool(self, n=24):
        return random.sample(self._word_pool, min(n, len(self._word_pool)))

    def generate_question(self):
        category = random.choice(list(self.categories.keys()))
        pairs = self.categories[category]
        pair1, pair2 = random.sample(pairs, 2)
        a, b = pair1
        c, answer = pair2

        distractors = self._generate_distractors(a, b, c, answer)
        choices = [answer] + distractors[:3]
        random.shuffle(choices)

        return {
            "category": category,
            "a": a, "b": b, "c": c,
            "answer": answer,
            "choices": choices,
        }

    def _generate_distractors(self, a, b, c, answer):
        exclude = {a, b, c, answer}
        try:
            similar = self.model.most_similar(positive=[b, c], negative=[a], topn=20)
        except Exception:
            similar = self.model.most_similar(answer, topn=20)
        return [w for w, _ in similar if w not in exclude and w.isalpha()]

    def get_2d_projection(self, n: int = 20, method: str = "pca", category: str = None):
        import numpy as np
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE

        if category and category in self.categories:
            pool = list({w for pair in self.categories[category] for w in pair})
        else:
            pool = self._word_pool

        n = min(n, len(pool))
        words = random.sample(pool, n)
        vectors = np.array([self.model[w] for w in words])

        if method == "tsne":
            perplexity = min(5, n - 1)
            reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        else:
            reducer = PCA(n_components=2)

        coords = reducer.fit_transform(vectors)
        return [{"word": w, "x": float(coords[i, 0]), "y": float(coords[i, 1])}
                for i, w in enumerate(words)]

    def nearest_words(self, word: str, topn: int = 7):
        word = word.lower().strip()
        if word not in self.model:
            return None, f"Word not in vocabulary: '{word}'"
        try:
            results = self.model.most_similar(word, topn=topn)
            return [(w, round(s, 3)) for w, s in results], None
        except Exception as e:
            return None, str(e)

    def answer_analogy(self, a: str, b: str, c: str, topn: int = 5):
        a, b, c = a.lower().strip(), b.lower().strip(), c.lower().strip()
        missing = [w for w in [a, b, c] if w not in self.model]
        if missing:
            return None, f"Words not in vocabulary: {', '.join(missing)}"
        try:
            results = self.model.most_similar(positive=[b, c], negative=[a], topn=topn + 5)
            filtered = [(w, s) for w, s in results if w not in {a, b, c}][:topn]
            return filtered, None
        except Exception as e:
            return None, str(e)
