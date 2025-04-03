import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class VoiceRegistry:
    def __init__(self, threshold=0.7):
        """
        Initializes the voice registry.
        :param threshold: The similarity threshold to consider a match (default: 0.7)
        """
        self.registry = {}  # Stores embeddings with associated identities
        self.threshold = threshold

    def register_voice(self, person_id, embedding):
        """
        Registers a new voice embedding.
        :param person_id: Unique identifier for the person
        :param embedding: Numpy array representing the voice embedding
        """
        self.registry[person_id] = embedding

    def find_closest_match(self, embedding):
        """
        Finds the closest matching voice in the registry.
        :param embedding: Numpy array of the input voice embedding
        :return: The matched person_id if above threshold, otherwise None
        """
        if not self.registry:
            print("Can't find closest match, registry is empty")
            return None
        
        best_match = None
        best_score = 0
        
        for person_id, stored_embedding in self.registry.items():
            similarity = np.inner(embedding, stored_embedding)
            print(similarity)
            if similarity > best_score:
                best_score = similarity
                best_match = person_id
        
        return best_match if best_score >= self.threshold else None

    def process_voice(self, embedding):
        """
        Processes an incoming voice embedding.
        If a match is found, returns the existing person_id.
        Otherwise, registers a new voice.
        :param embedding: Numpy array of the input voice embedding
        :return: Matched or newly assigned person_id
        """
        match = self.find_closest_match(embedding)
        if match:
            return match
        
        new_id = f"Person_{len(self.registry) + 1}"
        self.register_voice(new_id, embedding)
        return new_id
