"""Lightning entry point for explicitly selected pretraining methods."""

from cof.pipeline.base import BaseTrainingPipeline


class PretrainPipeline(BaseTrainingPipeline):
    """Execute a pretraining method under the pretrain stage contract."""

    stage_name = "pretrain"
