import os
from datasets import load_dataset, load_from_disk, concatenate_datasets

class DataLoader:
    def __init__(self, dataset_name, cache_dir='./data/eval'):
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.export_file_dir = os.path.join(cache_dir, dataset_name)
    
    def load_dataset(self):
        if os.path.exists(self.export_file_dir):
            print(f"Dataset already exists at {self.export_file_dir}. Loading from disk...")
            return load_from_disk(self.export_file_dir)
        return self.preprocess_dataset()

    def preprocess_dataset(self):
        raise NotImplementedError("Subclasses should implement dataset-specific preprocessing logic.")

    def concatenate_datasets(self, datasets):
        if len(datasets) > 1:
            return concatenate_datasets(datasets)
        return datasets[0]
