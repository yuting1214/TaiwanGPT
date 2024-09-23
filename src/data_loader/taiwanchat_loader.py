import uuid
from src.data_loader.base_loader import DataLoader
from datasets import DatasetDict, load_dataset, get_dataset_config_names


class TaiwanChatDataLoader(DataLoader):
    def preprocess_dataset(self):
        dataset = load_dataset(self.dataset_name, cache_dir=self.cache_dir)
        transformed_dataset = dataset['train'].map(self.transform_example, remove_columns=['conversations', 'id'])

        # Save the entire DatasetDict to disk
        transformed_dataset.save_to_disk(self.export_file_dir)
        print(f"Dataset saved to {self.export_file_dir}.")
        
        return transformed_dataset

    def transform_example(self, example):
        return {
            'dataset_name': example['id'],  # Rename 'id' to 'dataset_name'
            'messages': example['messages'],  # Keep 'messages'
            'instance_id': str(uuid.uuid4())  # Generate a new UUID for 'instance_id'
        }

    def fetch_subset(self, specific_ds):
        # Load the transformed dataset from disk
        transformed_dataset = load_dataset(self.export_file_dir)

        # Filter the dataset based on the specific dataset name
        filtered_dataset = transformed_dataset.filter(lambda example: example['dataset_name'] == specific_ds)
        
        return filtered_dataset