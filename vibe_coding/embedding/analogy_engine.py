import gensim.downloader as api


class AnalogyEngine:
    def __init__(self, model_name="glove-wiki-gigaword-100", on_progress=None):
        def progress(pct, msg):
            print(f"[{pct}%] {msg}")
            if on_progress:
                on_progress(pct, msg)

        progress(5, "Checking model cache...")
        import os
        cached = os.path.exists(os.path.join(api.BASE_DIR, model_name))

        if cached:
            progress(10, "Loading GloVe model into memory...")
        else:
            progress(10, "Downloading GloVe model (~170 MB, first run only)...")

        self.model = api.load(model_name)
        progress(100, "Ready!")

    def nearest_words(self, word: str, topn: int = 7):
        word = word.lower().strip()
        if word not in self.model:
            return None, f"Word not in vocabulary: '{word}'"
        try:
            results = self.model.most_similar(word, topn=topn)
            return [(w, round(s, 3)) for w, s in results], None
        except Exception as e:
            return None, str(e)

    def explore_analogy(self, a: str, b: str, c: str = None):
        import numpy as np
        from sklearn.decomposition import PCA

        a, b = a.lower().strip(), b.lower().strip()
        to_check = [a, b]
        if c:
            c = c.lower().strip()
            to_check.append(c)
        missing = [w for w in to_check if w not in self.model]
        if missing:
            return {"error": f"Words not in vocabulary: {', '.join(missing)}"}

        va = self.model[a]
        vb = self.model[b]
        diff = vb - va

        words_list = [a, b]
        vectors = [va, vb]
        result_word = None

        if c:
            vc = self.model[c]
            vd = vc + diff
            try:
                similar = self.model.similar_by_vector(vd, topn=15)
                for w, _ in similar:
                    if w not in {a, b, c}:
                        result_word = w
                        break
            except Exception:
                pass
            words_list = [a, b, c, result_word or "?"]
            vectors = [va, vb, vc, vd]

        vecs_np = np.array(vectors, dtype=float)
        n_components = min(2, len(vectors))
        pca = PCA(n_components=n_components)
        coords = pca.fit_transform(vecs_np)
        if coords.shape[1] < 2:
            coords = np.hstack([coords, np.zeros((len(coords), 1))])

        points = [
            {"word": w, "x": float(coords[i, 0]), "y": float(coords[i, 1])}
            for i, w in enumerate(words_list)
        ]
        out = {"points": points}
        if c is not None:
            out["result_word"] = result_word
        return out
