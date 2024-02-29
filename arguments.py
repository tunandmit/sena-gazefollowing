from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/vidpr"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained vidpr downloaded from s3"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    stage: Optional[str] = field(
        default='pretrain',
        metadata={"help" : "support pretrain and finetune"}
    )
    gaze_cone: Optional[int] = field(
        default=60,
        metadata={"help" : "gaze cone angle"}
    )
    gaze_heatmap_size: Optional[int] = field(
        default=64,
        metadata={"help" : "heatmap size"}
    )
    num_gaze_queries: Optional[int] = field(
        default=20,
        metadata={"help" : "number of queries sens condetr"}
    )
    num_head_decoder: Optional[int] = field(
        default=8,
        metadata={"help" : "number of queries sens condetr"}
    )
    num_gaze_decoder_layers: Optional[int] = field(
        default=6,
        metadata={"help" : "number of queries sens condetr"}
    )
    gaze_dropout: Optional[float] = field(
        default=0.1,
        metadata={"help" : "number of queries sens condetr"}
    )
    

@dataclass
class DataArguments:
    dataset_name_train: str = field(
        default=None, metadata={"help": "huggingface dataset name"}
    )
    dataset_name_val: str = field(
        default=None, metadata={"help": "huggingface dataset name"}
    )
    dataset_name_test: str = field(
        default=None, metadata={"help": "huggingface dataset name"}
    )
    train_dir: str = field(default=None, metadata={"help": "Path to train directory"})
    max_train_samples: int = field(
        default=None, metadata={"help": "Max number of training samples"}
    )
    valid_dir: str = field(
        default=None, metadata={"help": "Path to validation directory"}
    )
    max_valid_samples: int = field(
        default=None, metadata={"help": "Max number of validation samples"}
    )
    test_dir: str = field(
        default=None, metadata={"help": "Path to validation directory"}
    )
    max_test_samples: int = field(
        default=None, metadata={"help": "Max number of validation samples"}
    )
    passage_field_separator: str = field(default=" ")
    num_workers: int = field(
        default=4, metadata={"help": "number of process used in dataset preprocessing"}
    )
    data_cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the data downloaded from huggingface"
        },
    )
    streaming: bool = field(
        default=False,
        metadata={"help": "whether to use streaming dataset for training"},
    ) 
    object_path: str = field(
        default=None,
        metadata={"help": "objects name classes in detection"},
    )
    topknn : Optional[int] = field(
        default=10,
        metadata={"help": "topKNN of target object"},
    )