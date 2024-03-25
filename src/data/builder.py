import json
import os
from functools import partial

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms.functional import to_pil_image
from transformers import CLIPProcessor, ViltProcessor

from .datasets import MmImdbCmmlDataset, UpmcFood101CmmlDataset

Image.MAX_IMAGE_PIXELS = 1000000000


def build_dataloaders(cfg, **kwargs):
    # Build processor
    processor_name = cfg.PROCESSOR.lower()
    if processor_name == "vilt" or processor_name == "rebq":
        processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
    elif processor_name == "clip":
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    else:
        raise ValueError(f"Cannot find processor named: {processor_name}")

    # Build dataset and data loader
    dataset_name = cfg.NAME.lower()
    assert dataset_name in ("upmc_food101_cmml", "mm_imdb_cmml")

    # Prepare all data
    task_data = prepare_task_data(cfg.DATA_DIR, dataset_name, cfg.NUM_TASKS)

    collate_fn = partial(our_collate_fn, dataset_name=dataset_name, training=True, processor=processor)

    if dataset_name == "upmc_food101_cmml":
        DatasetClass = UpmcFood101CmmlDataset
    elif dataset_name == "mm_imdb_cmml":
        DatasetClass = MmImdbCmmlDataset

    dataloaders = {"train": [], "val": []} # length: num_tasks
    for task_id in range(cfg.NUM_TASKS):
        train_dataset = DatasetClass(
            data=task_data[task_id]["train"],
            missing_params=cfg.missing_params,
            split="train",
            task_id=task_id,
            limited_data_ratio=cfg.LIMITED_TRAIN_DATA_RATIO,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.TRAIN_BATCH_SIZE,
            shuffle=None if kwargs["distributed"] else True,
            sampler=DistributedSampler(train_dataset, shuffle=True) if kwargs["distributed"] else None,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        dataloaders["train"].append(train_dataloader)

        # Construct val loaders from 0 to current task id
        val_dataloaders = []
        for tid in range(task_id+1):
            val_dataset = DatasetClass(
                data=task_data[tid]["val"],
                missing_params=cfg.missing_params,
                split="val",
                task_id=tid,
                limited_data_ratio=cfg.LIMITED_VAL_DATA_RATIO,
            )
            val_dataloaders.append(DataLoader(
                val_dataset,
                batch_size=cfg.TEST_BATCH_SIZE,
                sampler=DistributedSampler(val_dataset, shuffle=False) if kwargs["distributed"] else None,
                num_workers=cfg.NUM_WORKERS,
                pin_memory=True,
                collate_fn=collate_fn,
            ))
        dataloaders["val"].append(val_dataloaders)
    test_loaders = []
    for tid in range(cfg.NUM_TASKS):
        test_dataset = DatasetClass(
            data=task_data[tid]["test"],
            missing_params=cfg.missing_params,
            split="test",
            task_id=tid,
        )
        test_loaders.append(DataLoader(
            test_dataset,
            batch_size=cfg.TEST_BATCH_SIZE,
            sampler=DistributedSampler(test_dataset, shuffle=False) if kwargs["distributed"] else None,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=True,
            collate_fn=collate_fn,
        ))
    dataloaders["test"] = test_loaders
    
    return dataloaders


def prepare_task_data(data_dir, dataset_name, num_tasks):
    """ Prepare all task data.
    """
    if dataset_name == "upmc_food101_cmml":
        json_filename = "UPMC-Food101-CMML.json"
    elif dataset_name == "mm_imdb_cmml":
        json_filename = "MM-IMDB-CMML.json"
    else:
        raise ValueError(f"Cannot find dataset name: {dataset_name}")
    with open(os.path.join(data_dir, json_filename), "r") as f:
        json_data = json.load(f)
    
    task_data = []
    for _ in range(num_tasks):
        task_data.append({"train": [], "val": [], "test": []})
    
    # Error!!! All items will share a common memory!!!
    # task_data = [{"train": [], "val": [], "test": []}] * num_tasks

    for item in json_data:
        item_task_id = item["task_id"]
        if item_task_id < num_tasks:
            # For debug, we only use a subset and set num_tasks=1
            item_split = item["split"]
            task_data[item_task_id][item_split].append({
                "image": os.path.join(data_dir, "images", item["image"]),
                "text": ". ".join(item["text"]) if isinstance(item["text"], list) else item["text"],
                "label": item["label"],
            })

    return task_data


def our_collate_fn(data, dataset_name, training, processor):
    batch = {
        "images": [],
        "texts": [],
        "labels": [],
        "missing_types": [],
    }
    augmented_data = {
        "images": [],
        "texts": [],
        "labels": [],
        "missing_types": [],
        "rec_gts": [],
    }
    aug_cnt = 0
    for idx, item in enumerate(data):
        batch["images"].append(item["image"])
        batch["texts"].append(item["text"])
        if dataset_name == "upmc_food101_cmml":
            batch["labels"].append(item["label"])
        elif dataset_name == "mm_imdb_cmml":
            batch["labels"].append(torch.tensor(item["label"]))
        batch["missing_types"].append(item["missing_type"])

        if training and item["missing_type"] == 0:
            fake_text = ''
            fake_image = to_pil_image(torch.ones(item["image"].size)).convert("RGB")
            augmented_data["images"].append(item["image"])
            augmented_data["texts"].append(fake_text)
            augmented_data["missing_types"].append(3)
            augmented_data["rec_gts"].append(idx)
            if dataset_name == "upmc_food101_cmml":
                augmented_data["labels"].append(item["label"])
            elif dataset_name == "mm_imdb_cmml":
                augmented_data["labels"].append(torch.tensor(item["label"]))

            augmented_data["images"].append(fake_image)
            augmented_data["texts"].append(batch["texts"][idx])
            augmented_data["missing_types"].append(4)
            augmented_data["rec_gts"].append(idx)
            if dataset_name == "upmc_food101_cmml":
                augmented_data["labels"].append(item["label"])
            elif dataset_name == "mm_imdb_cmml":
                augmented_data["labels"].append(torch.tensor(item["label"]))

            aug_cnt += 2
    batch["images"].extend(augmented_data["images"])
    batch["texts"].extend(augmented_data["texts"])
    #batch["labels"].extend(augmented_data["labels"])
    if dataset_name == "upmc_food101_cmml":
        batch["labels"] = torch.tensor(batch["labels"])
    elif dataset_name == "mm_imdb_cmml":
        batch["labels"] = torch.vstack(batch["labels"]).float()
    batch["missing_types"].extend(augmented_data["missing_types"])
    batch["rec_gts"] = augmented_data["rec_gts"]
    batch["num_aug"] = aug_cnt
    
    inputs = processor(
        text=batch["texts"],
        images=batch["images"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    batch["inputs"] = inputs

    return batch
