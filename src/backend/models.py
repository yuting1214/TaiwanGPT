import uuid
from sqlalchemy import Column, String, Integer, Text, Float, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base

# Base class for the database models
Base = declarative_base()

# Evaluation Experiment Table
class EvaluationExperiment(Base):
    __tablename__ = "evaluation_experiments"
    experiment_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name = Column(String, nullable=False)
    prompt_name = Column(String, nullable=False)
    data_name = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False, default=func.now())
    # Relationships
    responses = relationship("LLMResponse", back_populates="evaluation_experiment")
    metrics = relationship("EvaluationMetric", back_populates="experiment")

# Evaluation Metric Table
class EvaluationMetric(Base):
    __tablename__ = "evaluation_metrics"
    metric_id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(UUID(as_uuid=True), ForeignKey("evaluation_experiments.experiment_id"))
    metric_name = Column(String, nullable=False)  # Name of the metric, e.g., "accuracy"
    metric_value = Column(Float, nullable=False)  # Value of the metric, e.g., 0.85
    created_at = Column(DateTime, nullable=False, default=func.now())
    # Relationships
    experiment = relationship("EvaluationExperiment", back_populates="metrics")

# Fine-Tuning Experiment Table
class FineTuningExperiment(Base):
    __tablename__ = "fine_tuning_experiments"
    experiment_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name = Column(String, nullable=False)
    dataset_name = Column(String, nullable=False)  # Dataset used for fine-tuning
    num_epochs = Column(Integer, nullable=False)  # Number of training epochs
    learning_rate = Column(Float, nullable=False)  # Learning rate for fine-tuning
    tokens_used_million = Column(Float, nullable=False)
    created_at = Column(DateTime, nullable=False, default=func.now())
    # Relationships
    metrics = relationship("FineTuningMetric", back_populates="experiment")

    def __repr__(self):
        return (f"<FineTuningExperiment(experiment_id='{self.experiment_id}', "
                f"model_name='{self.model_name}', dataset_name='{self.dataset_name}', "
                f"num_epochs={self.num_epochs}, learning_rate={self.learning_rate}, "
                f"tokens_used_million={self.tokens_used_million}, created_at={self.created_at})>")

# Fine-Tuning Metric Table
class FineTuningMetric(Base):
    __tablename__ = "fine_tuning_metrics"
    metric_id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(UUID(as_uuid=True), ForeignKey("fine_tuning_experiments.experiment_id"))
    metric_name = Column(String, nullable=False)  # Name of the metric, e.g., "training_loss"
    metric_value = Column(Float, nullable=False)  # Value of the metric, e.g., 0.02 for loss
    created_at = Column(DateTime, nullable=False, default=func.now())
    # Relationships
    experiment = relationship("FineTuningExperiment", back_populates="metrics")

class LLMResponse(Base):
    __tablename__ = "llm_responses"
    response_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_id = Column(UUID(as_uuid=True), ForeignKey("evaluation_experiments.experiment_id"))
    input_id = Column(String, nullable=False)
    model_response = Column(Text, nullable=False)
    # Relationships
    evaluation_experiment = relationship("EvaluationExperiment", back_populates="responses")

