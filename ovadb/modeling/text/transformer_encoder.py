from copy import deepcopy
import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from collections import OrderedDict


class TransformerText(nn.Module):
    def __init__(
        self,
        model_name,
        output_feature="pooler_output",
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        # model.eval()

        self.freeze()
        self.out_channels = self.model.config.hidden_size
        self.embeddings = self.model.embeddings.word_embeddings.weight
        self.mlm = False
        self.output_feature = output_feature

    @property
    def device(self):
        return self.embeddings.device

    def forward_model(self, text_list):
        tokenized_batch = self.tokenizer(
            text_list, padding="longest", return_tensors="pt"
        )
        if self.mlm:
            tokenized_batch["target_ids"] = deepcopy(tokenized_batch["input_ids"])
            tokenized_batch["mlm_mask"] = []
            for i, item in enumerate(tokenized_batch["input_ids"]):
                mlm_mask = []
                for j in range(len(item)):
                    if (
                        tokenized_batch["special_tokens_mask"][i][j]
                        or not tokenized_batch["attention_mask"][i][j]
                        or not (self.training or self.mlm_during_validation)
                    ):
                        mlm_mask.append(0)
                        continue
                    prob = np.random.rand()
                    if prob < self.mlm_prob:
                        mlm_mask.append(1)
                        prob /= self.mlm_prob
                        if prob < self.mlm_prob_mask:
                            item[j] = self.tokenizer.convert_tokens_to_ids(
                                self.tokenizer.mask_token
                            )
                            tokenized_batch["special_tokens_mask"][i][j] = 1
                        elif prob < self.mlm_prob_mask + self.mlm_prob_noise:
                            item[j] = np.random.randint(len(self.tokenizer))
                    else:
                        mlm_mask.append(0)
                tokenized_batch["mlm_mask"].append(mlm_mask)

        tokenized_batch = {
            k: torch.tensor(v).to(self.device) for k, v in tokenized_batch.items()
        }
        # text_output = self.model(
        #     input_ids=tokenized_batch['input_ids'],
        #     attention_mask=tokenized_batch['attention_mask'],
        # )
        text_output = self.model(**tokenized_batch)
        outputs = text_output.pooler_output
        tokenized_batch["pooler_output"] = outputs
        tokenized_batch["encoded_tokens"] = text_output[0]
        tokenized_batch["input_embeddings"] = self.embeddings[
            tokenized_batch["input_ids"]
        ]
        tokenized_batch["cls_token"] = text_output["last_hidden_state"][:, 0, :]
        return tokenized_batch

    def forward(self, text_list):
        tokenized_batch = self.forward_model(text_list)
        if self.output_feature == "pooler_output":
            return tokenized_batch["pooler_output"]
        elif self.output_feature == "cls_token":
            return tokenized_batch["cls_token"]
        elif self.output_feature in {"mean_encodings", "mean_embeddings"}:
            mask = tokenized_batch["attention_mask"]
            mask[:, 0] = 0
            bs = mask.shape[0]
            sep_idx = mask.sum(1)
            for idx in range(bs):
                mask[idx, sep_idx[idx]] = 0
            if self.output_feature == "mean_encodings":
                output = (tokenized_batch["encoded_tokens"] * mask[:, :, None]).sum(
                    1
                ) / mask.sum(1)[:, None]
            else:  # 'mean_embeddings'
                output = (tokenized_batch["input_embeddings"] * mask[:, :, None]).sum(
                    1
                ) / mask.sum(1)[:, None]
            return output
        else:
            assert False, "Output features not available {}".format(self.output_feature)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False


PRETRAINED_MODELS = {
    "bert": ("bert-base-uncased", 768),
    "bert-large": ("bert-large-uncased", 1024),
    # 'distilbert': ('distilbert-base-uncased', 768),
    # 'gpt2': ('gpt2', 768),
    # 'distilgpt2': ('distilgpt2', 768),
    "roberta": ("roberta-base", 768),
    "roberta-large": ("roberta-large", 1024),
    "distilroberta-base": ("distilroberta-base", 768),
}


def build_text_encoder(text_model, cfg=None):
    assert text_model in PRETRAINED_MODELS.keys(), "Model not available {}".format(
        text_model
    )
    text_encoder = TransformerText(
        PRETRAINED_MODELS[text_model][0], cfg.MODEL.TEXT_MODEL_OUTPUT_FEATURES
    )
    if cfg is not None:
        assert cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_DIM == text_encoder.out_channels
        assert (
            cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_DIM
            == PRETRAINED_MODELS[text_model][1]
        )
    return text_encoder

