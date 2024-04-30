from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
import torch

from dataset import *
from lstm import *
from transformer import *


@torch.no_grad()
def evaluate(model, dataset):
    model.eval()
    batch_size = 128
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            collate_fn=dataset.collate_fn)
    ppls = []
    losses = []
    for samples in dataloader:
        bsz = len(samples['lengths'])
        logits = model.logits(**samples)
        lprobs = F.log_softmax(logits, dim=-1).view(-1, logits.size(-1))
        entropy = F.nll_loss(lprobs,
                             samples["target"].view(-1),
                             ignore_index=dataset.padding_idx,
                             reduction="none").view(bsz, -1)
        ppl = np.exp((entropy.sum(dim=-1, keepdim=True) /
                      (samples["target"] != dataset.padding_idx).sum(
                          dim=-1, keepdim=True)).cpu())
        ppls.extend(ppl.tolist())
        losses.append(entropy.mean().item())
    print("%s: loss: %.3f, ppl: %.3f" %
          (dataset.split, np.mean(losses), np.mean(ppls)))


if __name__ == "__main__":
    basedir = os.path.dirname(os.path.abspath(__file__))
    for task in ["lm", "seq2seq"]:
        Dataset = LMDataset if task == "lm" else Seq2SeqDataset
        try:
            dataset = Dataset(split='test', device="cuda")
        except FileNotFoundError:
            dataset = Dataset(split="valid", device="cuda")
        for model_type in ["lstm", "transformer"]:
            model_name = "{}_{}.pt".format(model_type, task)

            try:
                model = torch.load(os.path.join(basedir, "models", model_name), map_location='cpu').to('cuda')
            except FileNotFoundError as e:
                print(e)
                continue
            print(task, model_type)
            evaluate(model, dataset)

            if hasattr(model, "generate"):
                if task == "lm":
                    print("好-->", model.generate("好", beam_size=3))
                elif task == "seq2seq":
                    print("改革春风吹满地-->", model.generate("改革春风吹满地", beam_size=2))
            print("-" * 50)
