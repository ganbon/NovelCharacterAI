from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import numpy as np

def encode_line(
        tone_model: SentenceTransformer, utterance_list: list[str]
    ) -> np.ndarray:
        pca = PCA(n_components=100)
        for i, sentence in enumerate(utterance_list):
            if i == 0:
                vector_data = (
                    tone_model.encode(sentence[1:-1], convert_to_tensor=True)
                    .reshape(-1, 768)
                    .to("cpu")
                    .detach()
                    .numpy()
                    .copy()
                )
            else:
                vector_data = np.concatenate(
                    [
                        vector_data,
                        tone_model.encode(sentence[1:-1], convert_to_tensor=True)
                        .reshape(-1, 768)
                        .to("cpu")
                        .detach()
                        .numpy()
                        .copy(),
                    ]
                )
        pca.fit(np.array(vector_data))
        pca_vector = pca.transform(np.array(vector_data))
        return [pca for pca in pca_vector]
        # return pca_vector