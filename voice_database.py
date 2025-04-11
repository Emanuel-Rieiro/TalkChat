import numpy as np
import pickle
import os

class VoiceRegistry:
    def __init__(self, threshold=0.8, filepath="voice_registry.pkl"):
        """
        Initializes the voice registry.
        
        Args:
            threshold (float): The similarity threshold to consider a match (default: 0.8)
        """
        self.registry = {}  # Stores embeddings with associated identities
        self.registry_inputs_count = {}
        self.threshold = threshold

        self.load_registry(filepath)  # Load saved data if exists

    def register_voice(self, person_id, embedding):
        """
        Registers a new voice embedding.

        Args:
            person_id (str): Unique identifier for the person
            embedding (np.ndarray): Numpy array representing the voice embedding
        """
        self.registry[person_id] = embedding
        self.registry_inputs_count[person_id] = 1

    def find_closest_match(self, embedding):
        """
        Finds the closest matching voice in the registry.

        Args:
            embedding (np.ndarray): Numpy array of the input voice embedding

        Returns: 
            str: The matched person_id if above threshold, otherwise None
        """
        if not self.registry:
            print("Can't find closest match, registry is empty")
            return None
        
        best_match = None
        best_score = 0
        
        for person_id, stored_embedding in self.registry.items():
            similarity = np.inner(embedding, stored_embedding)
            print(person_id, similarity)
            if similarity > best_score:
                best_score = similarity
                best_match = person_id
        
        return best_match if best_score >= self.threshold else None

    def process_voice(self, embedding, update = False):
        """
        Processes an incoming voice embedding. If a match is found, returns the existing person_id. Otherwise, registers a new voice.
        
        Args:
            embedding (np.ndarray): Numpy array of the input voice embedding
        
        Returns:
            str: Matched or newly assigned person_id
        """
        match = self.find_closest_match(embedding)
        if match:
            if update:
                self.update_embedding(match, embedding)
            return match
        
        new_id = f"Person_{len(self.registry) + 1}"
        self.register_voice(new_id, embedding)
        return new_id

    def update_embedding(self, person_id, embedding):
        """
        Updates the stored embedding for a given person using a running average.

        Args:
            person_id (str): Identifier for the person whose embedding is being updated.
            embedding (np.ndarray): The new embedding to incorporate into the registry.

        Returns:
            str: A success message after the update.

        Note:
            The update is performed using the formula for a running average:
            new_average = old_average + (new_value - old_average) / count
            This assumes that self.registry[person_id] holds the current average
            and self.registry_inputs_count[person_id] holds the number of embeddings seen so far.
        """
        self.registry[person_id] = self.registry[person_id] + (embedding - self.registry[person_id]) / self.registry_inputs_count[person_id]
    
        return 'Successfull update'

    def save_registry(self, filepath="voice_registry.pkl"):
        """
        Saves the registry and registry_inputs_count to a file.

        Args:
            filepath (str): Path to the file where data will be saved.
        """
        with open(filepath, "wb") as f:
            pickle.dump({
                'registry': self.registry,
                'registry_inputs_count': self.registry_inputs_count
            }, f)
        print(f"Registry saved to {filepath}")

    def load_registry(self, filepath="voice_registry.pkl"):
        """
        Loads the registry and registry_inputs_count from a file.

        Args:
            filepath (str): Path to the file where data is stored.
        """
        if not os.path.exists(filepath):
            print(f"No file found at {filepath}. Starting with an empty registry.")
            return

        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.registry = data.get('registry', {})
            self.registry_inputs_count = data.get('registry_inputs_count', {})
        print(f"Registry loaded from {filepath}")
    
    def delete_person(self, person_id):
        """
        Deletes the voice data for a given person_id from the registry.
        
        Args:
            person_id (str): Unique identifier for the person to delete
        
        Returns:
            (bool) True if the person was removed, False if not found
        """
        found = False
        if person_id in self.registry:
            del self.registry[person_id]
            found = True
        if person_id in self.registry_inputs_count:
            del self.registry_inputs_count[person_id]
            found = True

        print(f'{person_id} deleted')
        return found

    def reset_registry(self):
        """
        Resets the registry for the current class instance.
        """
        self.registry = {}
        self.registry_inputs_count = {}
        print(f"Registry reseted")