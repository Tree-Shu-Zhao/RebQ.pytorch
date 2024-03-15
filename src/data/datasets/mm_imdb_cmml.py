from .base_dataset import BaseDataset


class MmImdbCmmlDataset(BaseDataset):
    def __init__(
        self,
        data,
        missing_params,
        split=None,
        task_id=None,
        limited_data_ratio=None,
    ):
        super(MmImdbCmmlDataset, self).__init__(data, missing_params, "mm_imdb_cmml", split, task_id, limited_data_ratio)