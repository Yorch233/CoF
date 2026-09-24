"""Lightning entry point for explicitly selected posttraining methods."""

from cof.pipeline.base import BaseTrainingPipeline


class PosttrainPipeline(BaseTrainingPipeline):
    """Execute any compatible posttraining method with an explicit stage identity."""

    stage_name = "post_training"
