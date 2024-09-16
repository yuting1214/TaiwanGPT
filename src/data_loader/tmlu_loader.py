from src.data_loader.base_loader import DataLoader
from datasets import DatasetDict, load_dataset, get_dataset_config_names

class TMLUDataLoader(DataLoader):
    def preprocess_dataset(self):
        """
        Preprocess the 'miulab/tmlu' dataset by downloading all subsets,
        adding config names as columns, and concatenating them into a DatasetDict.
        
        Returns:
        - Concatenated DatasetDict with 'test' and 'dev' splits.
        """
        configs = self.get_configs()
        datasets = {
            'test': [],
            'dev': []
        }

        for config in configs:
            print(f"Downloading and processing subset: {config}")
            dataset = load_dataset(self.dataset_name, config, cache_dir=self.cache_dir)
            
            # Add the config name as a new column to distinguish subsets
            for split in dataset:
                dataset[split] = dataset[split].map(lambda example: {"subject": config})
                datasets[split].append(dataset[split])

        # Concatenate the 'test' and 'dev' datasets separately
        concatenated_datasets = {}
        for split in datasets:
            if len(datasets[split]) > 1:
                concatenated_datasets[split] = self.concatenate_datasets(datasets[split])
            else:
                concatenated_datasets[split] = datasets[split][0]
        
        # Create a DatasetDict containing both 'test' and 'dev'
        final_dataset_dict = DatasetDict(concatenated_datasets)

        # Save the entire DatasetDict to disk
        final_dataset_dict.save_to_disk(self.export_file_dir)
        print(f"Concatenated DatasetDict saved to {self.export_file_dir}.")
        
        return final_dataset_dict

    def get_configs(self):
        """
        Retrieve the available subsets (configurations) for the TMLU dataset.
        
        Returns:
        - List of dataset configurations (subsets).
        """
        return get_dataset_config_names(self.dataset_name)
