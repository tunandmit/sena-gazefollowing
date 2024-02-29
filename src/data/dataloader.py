import os
import glob
import torch
import logging
import numpy as np
import albumentations

from datasets import load_dataset, Dataset

from .postprocessing import (
    convert_obj_to_coco_format,
    prepare_finetune_format,
    prepare_evaluate_format
    )

logger = logging.info(__name__)



class DataProcessor:
    def __init__(self, image_processor, data_args) -> None:
        self.image_processor = image_processor
        self.data_args = data_args

        self.id2label = dict()
        with open(self.data_args.object_path, 'r') as file:
            for item in file.readlines():
                name, idx = item.strip().split('=')
                self.id2label.update({idx : name})

        self.label2id = {v: k for k, v in self.id2label.items()}

        self.transform = albumentations.Compose(
            [
                albumentations.HorizontalFlip(p=1.0),
                albumentations.RandomBrightnessContrast(p=1.0),
            ],
            bbox_params=albumentations.BboxParams(format="coco", label_fields=["category"]),
        )

    def __call__(self):
        datasets = {}
        # train set
        if self.data_args.train_dir is not None or self.data_args.dataset_name_train is not None:
            train_path = self.data_args.train_dir if self.data_args.train_dir is not None else self.data_args.dataset_name_train
            train_data = self.load_dataset(train_path, 'train')

            if self.data_args.max_train_samples is not None:
                train_data = train_data.select(range(self.data_args.max_train_samples))

            datasets['train'] = train_data.with_transform(self.transform_aug_ann)
            
        # validation set
        if self.data_args.valid_dir is not None or self.data_args.dataset_name_val is not None:
            valid_path = self.data_args.valid_dir if self.data_args.valid_dir is not None else self.data_args.dataset_name_val
            valid_data = self.load_dataset(valid_path, 'train')

            if self.data_args.max_valid_samples is not None:
                valid_data = valid_data.select(range(self.data_args.max_valid_samples))

            datasets['validation'] = valid_data.with_transform(self.transform_aug_ann)
            
        # test set
        if self.data_args.test_dir is not None or self.data_args.dataset_name_test is not None:
            test_path = self.data_args.test_dir if self.data_args.test_dir is not None else self.data_args.dataset_name_test
            test_data = self.load_dataset(test_path, 'train')

            if self.data_args.max_test_samples is not None:
                test_data = test_data.select(range(self.data_args.max_test_samples))

            datasets['test'] = test_data.with_transform(self.transform_aug_ann)

        return datasets, self.id2label, self.label2id

    def load_dataset(self, data_path:str=None, key:str='train') -> Dataset:
        """ Load datasets function

        Args:
            data_path (str, optional): folder contain list of input files name. Defaults to None.
            key (str, optional): help dataloader know is train file or test file.

                                Input file can be train/validation/test. Defaults to 'train'.

        Raises:
            Exception: _description_

        Returns:
            Datasets
        """
        if not os.path.exists(data_path):

            if self.data_args.streaming:
                dataset = load_dataset(data_path, streaming=True)[key]
            else:
                dataset = load_dataset(data_path, num_proc=self.data_args.num_workers)[key]

            # convert format to coco
            dataset = dataset.map(lambda example:convert_obj_to_coco_format(example),
                                  remove_columns=['bboxes', 'labels', 'seg'])

            return dataset
        else:
            files = glob.glob(os.path.join(data_path, '*'))
            extention = files[0].split('.')[-1]

            try:
                data_file = f"{data_path}/*.{extention}"

                if self.data_args.streaming:
                    dataset = load_dataset(
                        extention, data_files=data_file, split=key, streaming=self.data_args.streaming
                    )
                else:
                    dataset = load_dataset(
                        extention, data_files=data_file, split=key,
                        num_proc=self.data_args.num_workers
                    )

                # convert format to coco
                dataset = dataset.map(lambda example:convert_obj_to_coco_format(example),
                                      remove_columns=['bboxes', 'labels', 'seg'],
                                      num_proc=self.data_args.num_workers)
                return dataset
            except:
                logger.info(f'Error loading dataset {data_path} with {extention} extention')

    # transforming a batch
    def transform_aug_ann(self, examples):
        image_ids = examples["image_id"]
        images, bboxes, area, categories = [], [], [], []
        for image, objects in zip(examples["image"], examples["objects"]):
            image = np.array(image.convert("RGB"))[:, :, ::-1]
            out = self.transform(image=image, bboxes=objects["bbox"], category=objects["category"])

            area.append(objects["area"])
            images.append(out["image"])
            bboxes.append(out["bboxes"])
            categories.append(out["category"])

        targets = [
            {"image_id": id_, "annotations": self.formatted_anns(id_, cat_, ar_, box_)}
            for id_, cat_, ar_, box_ in zip(image_ids, categories, area, bboxes)
        ]

        result = self.image_processor(images=images, annotations=targets, return_tensors="pt")

        return result

    def formatted_anns(self, image_id, category, area, bbox):
        annotations = []
        for i in range(0, len(category)):
            new_ann = {
                "image_id": image_id,
                "category_id": category[i],
                "isCrowd": 0,
                "area": area[i],
                "bbox": list(bbox[i]),
            }
            annotations.append(new_ann)

        return annotations

class SenDataProcessor(DataProcessor):
    def load_dataset(self, data_path:str=None, key:str='train') -> Dataset:
        """ Load datasets function

        Args:
            data_path (str, optional): folder contain list of input files name. Defaults to None.
            key (str, optional): help dataloader know is train file or test file.

                                Input file can be train/validation/test. Defaults to 'train'.

        Raises:
            Exception: _description_

        Returns:
            Datasets
        """
        if not os.path.exists(data_path):

            if self.data_args.streaming:
                dataset = load_dataset(data_path, streaming=True)[key]
            else:
                dataset = load_dataset(data_path, num_proc=self.data_args.num_workers)[key]
            
            # convert format to coco
            dataset = dataset.map(lambda example:convert_obj_to_coco_format(example, self.data_args),
                                  #load_from_cache_file=False,
                                  desc="format coco")
          
            return dataset
        else:
            files = glob.glob(os.path.join(data_path, '*'))
            extention = files[0].split('.')[-1]

            try:
                data_file = f"{data_path}/*.{extention}"

                if self.data_args.streaming:
                    dataset = load_dataset(
                        extention, data_files=data_file, split=key, streaming=self.data_args.streaming
                    )
                else:
                    dataset = load_dataset(
                        extention, data_files=data_file, split=key,
                        num_proc=self.data_args.num_workers
                    )
                # convert format to coco
                dataset = dataset.map(lambda example:convert_obj_to_coco_format(example, self.data_args),
                                      num_proc=self.data_args.num_workers,
                                      desc="convert object to coco format")
                return dataset
            except:
                logger.info(f'Error loading dataset {data_path} with {extention} extention')
  
    # transforming a batch
    def transform_aug_ann(self, examples):
        image_ids = examples["image_id"]
        images, bboxes, area, categories = [], [], [], []

        for image, objects in zip(examples["image"], examples["objects"]):
            image = np.array(image.convert("RGB"))[:, :, ::-1]
            out = self.transform(image=image, bboxes=objects["bbox"], category=objects["category"])

            area.append(objects["area"])
            images.append(out["image"])
            bboxes.append(out["bboxes"])
            categories.append(out["category"])

        targets = [
            {"image_id": id_, "annotations": self.formatted_anns(id_, cat_, ar_, box_)}
            for id_, cat_, ar_, box_ in zip(image_ids, categories, area, bboxes)
        ] 

        result = self.image_processor(images=images, annotations=targets, return_tensors="pt")

        gazeformer = [prepare_finetune_format(example) for example in examples['gazeformer']]
        for idx, row in enumerate(result['labels']):
          row.update(gazeformer[idx])

        return result

class DataCollator:
    def __init__(self, image_processor):
        self.image_processor = image_processor

    def __call__(self, batch):

        pixel_values = [item["pixel_values"] for item in batch]
        encoding = self.image_processor.pad(pixel_values, return_tensors="pt")
        labels = [item["labels"] for item in batch]

        features = {}

        features["pixel_values"] = encoding["pixel_values"]
        features["pixel_mask"] = encoding["pixel_mask"]
        features["labels"] = labels

        return features
