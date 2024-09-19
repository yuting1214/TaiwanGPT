from sqlalchemy.orm import Session
from src.backend.models import (
    EvaluationExperiment,
    EvaluationMetric,
    FineTuningExperiment,
    FineTuningMetric,
    LLMResponse
)

def add_evaluation_experiment(db: Session, model_name: str, prompt_name: str, data_name: str):
    experiment = EvaluationExperiment(
        model_name=model_name,
        prompt_name=prompt_name,
        data_name=data_name,
    )
    db.add(experiment)
    db.commit()
    db.refresh(experiment)
    return experiment

def add_evaluation_metric(db: Session, experiment_id: str, metric_name: str, metric_value: float):
    metric = EvaluationMetric(
        experiment_id=experiment_id,
        metric_name=metric_name,
        metric_value=metric_value,
    )
    db.add(metric)
    db.commit()
    db.refresh(metric)
    return metric

def add_fine_tuning_experiment(db: Session, model_name: str, dataset_name: str, num_epochs: int, learning_rate: float, tokens_used_million: float):
    experiment = FineTuningExperiment(
        model_name=model_name,
        dataset_name=dataset_name,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        tokens_used_million=tokens_used_million,
    )
    db.add(experiment)
    db.commit()
    db.refresh(experiment)
    return experiment

def add_fine_tuning_metric(db: Session, experiment_id: str, metric_name: str, metric_value: float):
    metric = FineTuningMetric(
        experiment_id=experiment_id,
        metric_name=metric_name,
        metric_value=metric_value,
    )
    db.add(metric)
    db.commit()
    db.refresh(metric)
    return metric

def add_llm_response(db: Session, experiment_id: str, input_id: str, model_response: str):
    response = LLMResponse(
        experiment_id=experiment_id,
        input_id=input_id,
        model_response=model_response,
    )
    db.add(response)
    db.commit()
    db.refresh(response)
    return response
