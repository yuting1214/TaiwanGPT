from src.backend.dependencies.database import init_db, get_db
from src.backend.crud import add_evaluation_experiment, add_evaluation_metric


if __name__ == "__main__":
    init_db()

    # Get a database session
    db = next(get_db())

    # Example usage of the utility functions
    experiment = add_evaluation_experiment(db, "Model A", "Prompt 1", "Data Set 1")
    print(f"Added Evaluation Experiment: {experiment}")

    metric = add_evaluation_metric(db, experiment.experiment_id, "accuracy", 0.85)
    print(f"Added Evaluation Metric: {metric}")

