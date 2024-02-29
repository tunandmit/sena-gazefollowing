import os
os.environ['WANDB_PROJECT'] = 'SenaGaze'

import sys
import logging

from transformers import (
    AutoImageProcessor,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    AutoModelForObjectDetection
)
from transformers.trainer_utils import get_last_checkpoint
from .arguments import (
    ModelArguments,
    DataArguments,
)
from .data.dataloader import DataProcessor, SenDataProcessor, DataCollator
from .models import SenAGaze
from .trainer import TrainerWithLogs


logger = logging.getLogger(__name__)



def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # check output_dir path
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}; "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info("MODEL parameters %s", model_args)

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    set_seed(training_args.seed)

    # Prepare processor
    image_processor = AutoImageProcessor.from_pretrained(model_args.model_name_or_path)
    
    # prepare data
    base_data = DataProcessor if model_args.stage == 'pretrain' else SenDataProcessor
    dataset_training, id2label, label2id  = base_data(image_processor, data_args).__call__()
    data_collator = DataCollator(image_processor)
    
    # prepare model
    if model_args.stage == 'pretrain':
        model = AutoModelForObjectDetection.from_pretrained(
            model_args.model_name_or_path,
            num_labels = len(id2label),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
    else:
        model = SenAGaze.from_pretrained(
            model_args.model_name_or_path,
            num_labels = len(id2label),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
            model_args=model_args
        )

    if training_args.do_train:
        train_dataset = dataset_training['train']
    if training_args.do_eval:
        eval_dataset = dataset_training['validation']

    trainer = TrainerWithLogs(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset = eval_dataset if training_args.do_eval else None,
        tokenizer=image_processor,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        trainer.train(resume_from_checkpoint=checkpoint)

        trainer.save_model()  # Saves the tokenizer too for easy upload

if __name__ == '__main__':
    main()