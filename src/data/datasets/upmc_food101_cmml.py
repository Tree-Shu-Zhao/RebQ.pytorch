
from .base_dataset import BaseDataset


class UpmcFood101CmmlDataset(BaseDataset):
    def __init__(
        self,
        data,
        missing_params,
        split=None,
        task_id=None,
        limited_data_ratio=None,
    ):
        super(UpmcFood101CmmlDataset, self).__init__(data, missing_params, "upmc_food101_cmml", split, task_id, limited_data_ratio)