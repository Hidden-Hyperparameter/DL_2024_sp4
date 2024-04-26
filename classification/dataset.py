from torch.utils.data import Dataset
import json
import os
import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))


class CLSDataset(Dataset):

    def __init__(self,
                 data_path=os.path.join(os.path.dirname(cur_dir),
                                        "Datasets/CLS/"),
                 split="train",
                 device="cpu"):

        self.filename = os.path.join(data_path, "{}.json".format(split))

        with open(self.filename, encoding="utf-8") as f:
            self.data = json.load(f)

        self.padding_idx = None
        self.cls_idx = self.bos_idx = None
        self.sep_idx = self.eos_idx = None

        self.cls_map = {"A": 0, "B": 1, "C": 2, "D": 3}
        self.pairs = []
        for article in self.data:
            content = article["Content"]
            for question in article['Questions']:
                q = question['Question']
                choices = question['Choices']
                label = self.cls_map[question['Answer']]
                self.pairs.append([content, q, choices, label])
        self.device = device
        self.split = split

    def __len__(self):
        return len(self.pairs)

    # @profile
    def __getitem__(self, index):
        """
        Get a data pair from the dataset by index.

        This method is used by PyTorch DataLoader to retrieve individual data pairs
        from the dataset. It should be implemented to return the data pair at the
        specified index.

        Args:
            index (int): Index of the data pair.

        Returns:
            your_data_pair(obj): Data pair at the specified index.
        """
        ##############################################################################
        #                  TODO: You need to complete the code here                  #
        ##############################################################################
        raise NotImplementedError()
        ##############################################################################
        #                              END OF YOUR CODE                              #
        ##############################################################################

    # @profile
    def collate_fn(self, samples):
        """
        Collate function for DataLoader.

        This method is used by PyTorch DataLoader to process and batch the samples
        returned by the __getitem__ method. It should be implemented to return a
        batch of data in the desired format.

        Args:
            samples (list): List of data pairs retrieved using the __getitem__ method.

        Returns:
            your_batch_data: Batch of data in your desired format.
        """
        ##############################################################################
        #                  TODO: You need to complete the code here                  #
        ##############################################################################
        raise NotImplementedError()
        ##############################################################################
        #                              END OF YOUR CODE                              #
        ##############################################################################
