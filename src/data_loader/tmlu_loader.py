from src.data_loader.base_loader import DataLoader
from datasets import DatasetDict, load_dataset, get_dataset_config_names

class TMLUDataLoader(DataLoader):
    def preprocess_dataset(self):
        """
        Preprocess the 'miulab/tmlu' dataset by downloading all subsets,
        adding config names as columns, creating prompts, and concatenating them into a DatasetDict.
        
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
                # Add config name
                dataset[split] = dataset[split].map(lambda example: {"subject": config})
                # Create user prompts
                dataset[split] = dataset[split].map(self.create_user_prompt)
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
    
    def create_user_prompt(self, example):
        # Template for the question
        prompt_template_part_one = '''
        以下選擇題為出自臺灣的考題，答案為其中一個選項。

        問題:
        {question}

        '''
        prompt_template_part_two = '''
        正確答案：(
        '''
        # List of possible options (A to F)
        options = ["A", "B", "C", "D", "E", "F"]
        
        # Check if each option exists in the example and construct answer string
        answer_template = ' '.join([f"({option}) {example[option]}" for option in options if example.get(option)])
        
        # Fill the question in the prompt template
        prompt = prompt_template_part_one.format(question=example["question"]) + answer_template + prompt_template_part_two
        
        # Add the final prompt to 'user_content'
        example['user_content'] = prompt
        
        return example

