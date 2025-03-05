import os
import sys
from ..utils import read_json
from fairseq.tasks.language_modeling import LanguageModelingTask, LanguageModelingConfig
from fairseq.models import register_model, register_model_architecture
from fairseq.tasks import register_task
from fairseq.tasks.language_modeling import LanguageModelingTask, LanguageModelingConfig

def load_fairseq_checkpoint(
        ckpt_path = "/home/longshen/work/StyleTrans/modules/musecoco/checkpoints/large.pt", 
        dict_fp = "/home/longshen/work/StyleTrans/modules/musecoco/checkpoints/dicts/large/dict.txt", 
        args_fp = "/home/longshen/work/StyleTrans/modules/musecoco/checkpoints/large_args.json",
    ):
    """
    Load the fairseq checkpoint
    """
    if not os.path.exists(args_fp):
        raise Exception("Args path {} not exist.".format(args_fp))

    from fairseq.checkpoint_utils import load_checkpoint_to_cpu, load_model_ensemble
    from argparse import Namespace
    from transformers.models.musecoco.linear_transformer.transformer_lm import (
        LinearTransformerLanguageModel,
    )

    fairseq_args = read_json(args_fp)
    fairseq_args["data"] = os.path.dirname(dict_fp)  # Overwrite the dictionary path
    fairseq_args["path"] = ckpt_path
    fairseq_args = Namespace(**fairseq_args)
    fairseq_task = LanguageModelingTaskWithControl.setup_task(fairseq_args)
    models, _model_args = load_model_ensemble(
        fairseq_args.path.split(os.pathsep),
        # arg_overrides=eval(fairseq_args.model_overrides),
        task=fairseq_task,
        suffix=getattr(fairseq_args, "checkpoint_suffix", ""),
        strict=(fairseq_args.checkpoint_shard_count == 1),
        num_shards=fairseq_args.checkpoint_shard_count,
    )
    assert len(models) == 1
    model = models[0]
    dict = fairseq_task.dictionary

    return model, dict

def load_fairseq_checkpoint_from_arg_dict(ckpt_path, dict_fp, fairseq_args):
    """
    Load the fairseq checkpoint
    """
    from fairseq.checkpoint_utils import load_checkpoint_to_cpu, load_model_ensemble
    from argparse import Namespace
    from transformers.models.musecoco.linear_transformer.transformer_lm import (
        LinearTransformerLanguageModel,
    )

    fairseq_args["data"] = os.path.dirname(dict_fp)  # Overwrite the dictionary path
    fairseq_args["path"] = ckpt_path
    fairseq_args = Namespace(**fairseq_args)
    fairseq_task = LanguageModelingTaskWithControl.setup_task(fairseq_args)
    models, _model_args = load_model_ensemble(
        fairseq_args.path.split(os.pathsep),
        arg_overrides=fairseq_args.model_overrides,
        task=fairseq_task,
        suffix=getattr(fairseq_args, "checkpoint_suffix", ""),
        strict=(fairseq_args.checkpoint_shard_count == 1),
        num_shards=fairseq_args.checkpoint_shard_count,
    )
    assert len(models) == 1
    model = models[0]
    dict = fairseq_task.dictionary

    return model, dict

@register_task("language_modeling_control", dataclass=LanguageModelingConfig)
class LanguageModelingTaskWithControl(LanguageModelingTask):
    """
    A util class for Fairseq to load the model checkpoint
    """

    @classmethod
    def add_args(cls, parser):
        """
        Some original fairseq model's args.
        """
        super().add_args(parser)  # This line is unnecessary
        parser.add_argument("--truncated_length", type=int, default=5868)
        parser.add_argument("--padding_to_max_length", type=int, default=0)
        parser.add_argument("--command_path", type=str)
        parser.add_argument("--command_embed_dim", type=int)
        parser.add_argument("--command_mask_prob", type=float, default=0.4)
        parser.add_argument("--is_inference", type=bool, default=False)
