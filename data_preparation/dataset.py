import torch
from pathlib import Path
from torch.utils.data import Dataset

class EncodedDataset(Dataset):
    """ Dataset that splits token ID tensors into input-target sequences for next-token prediction."""
    def __init__(self, file_path, block_size, stride=1):
        self.block_size = block_size
        self.stride = stride

        if not Path(file_path).exists():
            raise FileNotFoundError(f"Tokenized file not found: {file_path}")
        
        tokens = torch.load(file_path, weights_only=False)

        # Use unfold to create a sliding window view of the tokens
        self.sequences = tokens.unfold(0, block_size + 1, stride)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        input_ids = seq[:-1].clone().detach()
        target_ids = seq[1:].clone().detach()

        return {"input_ids": input_ids, "target_ids": target_ids}