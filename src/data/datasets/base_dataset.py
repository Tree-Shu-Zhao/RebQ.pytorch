import os
import random

import torch
from loguru import logger
from PIL import Image
from torchvision.transforms.functional import to_pil_image

Image.MAX_IMAGE_PIXELS = 1000000000


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        missing_params,
        dataset_name,
        split=None,
        task_id=None,
        limited_data_ratio=None,
    ):
        super(BaseDataset, self).__init__()

        self.data = data
        self.limited_data_ratio = limited_data_ratio

        # missing modality control        
        missing_ratio = missing_params.RATIO
        missing_type = missing_params.TYPE
        both_ratio = missing_params.BOTH_RATIO
        missing_table_root = missing_params.SAVE_ROOT
        os.makedirs(missing_table_root, exist_ok=True)
        missing_table_name = f'{dataset_name}_task{task_id}_split{split}_missing_{missing_type}_{missing_ratio}.pt'
        missing_table_path = os.path.join(missing_table_root, missing_table_name)
        
        # use image data to formulate missing table
        total_num = len(self.data)
        
        # Create or load a missing table
        if os.path.exists(missing_table_path):
            missing_table = torch.load(missing_table_path)
            if len(missing_table) != total_num:
                logger.info('missing table mismatched!')
                exit()
        else:
            missing_table = torch.zeros(total_num)
            
            if missing_ratio > 0:
                missing_index = random.sample(range(total_num), int(total_num*missing_ratio))

                if missing_type == 'text':
                    missing_table[missing_index] = 1
                elif missing_type == 'image':
                    missing_table[missing_index] = 2
                elif missing_type == 'both':
                    missing_table[missing_index] = 1
                    missing_index_image = random.sample(missing_index, int(len(missing_index)*both_ratio))
                    missing_table[missing_index_image] = 2
                    
                torch.save(missing_table, missing_table_path)

        self.missing_table = missing_table
        if limited_data_ratio is not None:
            logger.info(f"Limited data ratio: {limited_data_ratio}, size: {int(len(self.data)*self.limited_data_ratio)}")
        
    def __getitem__(self, index):
        data = self.data[index]
        image = data["image"]
        text = data["text"]
        label = data["label"]

        image = Image.open(image).convert("RGB")
        
        # missing image, dummy image is all-one image
        if self.missing_table[index] == 2:
            image = to_pil_image(torch.ones(image.size)).convert("RGB")
            
        # missing text, dummy text is 'None'
        if self.missing_table[index] == 1:
            text = ''
        
        return {
            "image": image,
            "text": text,
            "label": label,
            "missing_type": self.missing_table[index].item(),
        }

    def __len__(self):
        return len(self.data) if self.limited_data_ratio is None else int(len(self.data)*self.limited_data_ratio)
