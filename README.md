# TaiwanGPT: Fine-Tuning GPT-4 Mini with Traditional Chinese Corpus for TMLU Benchmarking

## Overview

TaiwanGPT is a project focused on fine-tuning the GPT-4 mini model using a Traditional Chinese corpus, with an emphasis on instruction tuning. The fine-tuned model is benchmarked against the [TMLU](https://huggingface.co/datasets/miulab/tmlu) (Taiwanese Mandarin Language Understanding) evaluation suite, which assesses advanced knowledge and reasoning capabilities across 37 subjects, ranging from middle school to professional levels in Taiwanese Mandarin.

## Project Structure

- **data/**: Contains the Traditional Chinese corpus used for fine-tuning, and the TMLU benchmark dataset.
- **scripts/**: Includes scripts for data preprocessing, model fine-tuning, and evaluation.
- **models/**: Directory where trained models and checkpoints are saved.
- **notebooks/**: Jupyter notebooks for exploratory data analysis, fine-tuning, and evaluation.
- **results/**: Contains results, including performance metrics and analysis on the TMLU benchmark.

```
TaiwanGPT/
│
├── data/
│   ├── fine_tuning/
│   │   ├── raw_corpus/             # Raw Traditional Chinese corpus files
│   │   ├── processed/              # Preprocessed data ready for fine-tuning
│   │   └── prompts/                # Instruction prompts for fine-tuning
│   │
│   ├── eval/
│   │   ├── TMLU_raw/               # Raw TMLU benchmark data
│   │   └── TMLU_processed/         # Processed TMLU data formatted for evaluation
│   │
│   └── README.md                   # Documentation for the data directory
│
├── models/
│   ├── checkpoints/                # Model checkpoints saved during training
│   └── taiwan_gpt_final/           # Final fine-tuned model
│
├── notebooks/
│   ├── data_exploration.ipynb      # Exploratory data analysis and preprocessing
│   ├── fine_tuning.ipynb           # Fine-tuning experiments and results
│   └── evaluation.ipynb            # Evaluation of the fine-tuned model on TMLU
│
├── results/
│   ├── fine_tuning/                # Logs, metrics, and visualizations from fine-tuning
│   ├── evaluation/                 # Performance results from TMLU benchmark
│   └── analysis/                   # Detailed analysis and reports on model performance
│
├── scripts/
│   ├── preprocess_data.py          # Script for preprocessing the Traditional Chinese corpus
│   ├── generate_prompts.py         # Script for generating instructional prompts
│   ├── train.py                    # Script for fine-tuning the GPT-4 mini model
│   ├── evaluate.py                 # Script for running evaluations against the TMLU benchmark
│   ├── api_client.py               
│   ├── calc_accuracy.py            
│   ├── tracking_db.py              # New: Script to interact with SQLite database
│   └── data_loader/                # New directory for data loading-related scripts
│       ├── __init__.py             # For initializing the data_loader module
│       ├── base_loader.py          # Contains the base DataLoader class
│       ├── tmlu_loader.py          # TMLU-specific DataLoader subclass
│       └── custom_loader.py        # (Optional) Another dataset-specific loader
│
├── configs/
│   ├── fine_tuning.yaml            # Configuration file for fine-tuning parameters
│   └── evaluation.yaml             # Configuration file for evaluation settings
|
├── tracking/
│   ├── fine_tuning.db              # SQLite database for tracking fine-tuning results
│   └── schema.sql                  # SQL schema to create necessary tables
│
├── README.md                       # Project overview and instructions
├── requirements.txt                # Python dependencies
├── .env                            # Environment variables (API keys, paths, etc.)
└── LICENSE                         # License for the project

```

## Installation

To get started with TaiwanGPT, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/TaiwanGPT.git
   cd TaiwanGPT
   ```
2. **Set up a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
3. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up the environment variables:** Create a .env file in the project root and add any necessary API keys or environment configurations.

## Data Preparation

1. Traditional Chinese Corpus Collection
The corpus is collected from a variety of sources relevant to the TMLU's subjects, including social sciences, STEM, humanities, and Taiwan-specific content. The data is preprocessed to ensure consistency and quality, focusing on tokenization, noise removal, and handling Traditional Chinese-specific characters.

2. Instruction Tuning Dataset
A set of instructional prompts relevant to the TMLU subjects is generated and formatted for fine-tuning. These prompts are designed to enhance the model's ability to understand and respond accurately in a wide range of contexts.

## Model Fine-Tuning

The fine-tuning process uses the GPT-4 mini model as a base, with the Traditional Chinese corpus and instructional prompts as input. Key hyperparameters such as learning rate, batch size, and epochs are carefully selected to optimize performance.

Run the fine-tuning process using the following command:

```
python scripts/train.py --config configs/fine_tuning.yaml
```

## Benchmarking with TMLU

After fine-tuning, the model is benchmarked against the TMLU evaluation suite, which includes 37 subjects across various domains. The evaluation results are used to assess the model's knowledge and reasoning capabilities in Taiwanese Mandarin.

Run the benchmarking process:

```
python scripts/evaluate.py --model_path models/taiwan_gpt_final
```

## Results and Analysis
The results of the benchmarking are stored in the results/ directory. The analysis includes accuracy metrics, advanced reasoning assessments, and subject-wise performance breakdowns.

Key Findings
* Strengths: Detailed analysis of the model's performance across different subjects, highlighting strengths in specific areas.
* Weaknesses: Identification of areas where the model underperforms, with suggestions for future improvements.

## Further Development
TaiwanGPT is an ongoing project, and contributions are welcome! Future work includes iterating on the model based on the benchmark results, integrating more diverse data sources, and potentially deploying the model for broader use.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgements
* The TMLU benchmark creators for providing a comprehensive evaluation suite.
* OpenAI for the GPT-4 mini model.
* Contributors who helped with data collection and preprocessing.

## Contact
For any inquiries or contributions, please reach out to [your_email@example.com].
