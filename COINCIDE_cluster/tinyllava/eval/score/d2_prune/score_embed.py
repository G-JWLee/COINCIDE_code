import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import transformers
import copy
import math
import numpy as np

from typing import Dict, Sequence
from dataclasses import dataclass
from tinyllava import conversation as conversation_lib
from tinyllava.data.process import preprocess, preprocess_multimodal
from tinyllava.constants import IGNORE_INDEX
from tinyllava.model.builder import load_pretrained_model
from tinyllava.utils import disable_torch_init
from tinyllava.mm_utils import get_model_name_from_path
from torch.utils.data import Dataset, DataLoader


from PIL import Image


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


class LazySupervisedDataset_w_split(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 image_processor,
                 data_args):
        super(LazySupervisedDataset_w_split, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        np.random.seed(42)
        random_indices = np.arange(len(list_data_dict))
        np.random.shuffle(random_indices)
        list_data_dict = [list_data_dict[random_idx] for random_idx in random_indices]
        self.random_recov_indices = np.argsort(random_indices)

        self.list_data_dict = get_chunk(list_data_dict, data_args.num_chunks, data_args.chunk_idx)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]

        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        if self.data_args.is_multimodal:

            if 'image' in sources[0]:
                image_file = self.list_data_dict[i]['image']
                image_folder = self.data_args.image_folder
                image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
                if self.data_args.image_aspect_ratio == 'pad':
                    def expand2square(pil_img, background_color):
                        width, height = pil_img.size
                        if width == height:
                            return pil_img
                        elif width > height:
                            result = Image.new(pil_img.mode, (width, width), background_color)
                            result.paste(pil_img, (0, (width - height) // 2))
                            return result
                        else:
                            result = Image.new(pil_img.mode, (height, height), background_color)
                            result.paste(pil_img, ((height - width) // 2, 0))
                            return result
                    image = expand2square(image, tuple(int(x*255) for x in self.image_processor.image_mean))
                    image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                else:
                    image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                sources = preprocess_multimodal(
                    copy.deepcopy([e["conversations"] for e in sources]),
                    self.data_args)
            else:
                sources = copy.deepcopy([e["conversations"] for e in sources])
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=self.data_args.is_multimodal)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0],
                             data_idx=i)

        if self.data_args.is_multimodal:

            # image exist in the data
            if 'image' in self.list_data_dict[i]:
                data_dict['image'] = image
            else:
                # image does not exist in the data, but the model is multimodal
                crop_size = self.image_processor.crop_size
                data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]

        data_idx = torch.tensor([instance['data_idx'] for instance in instances], dtype=torch.int64)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            data_idx=data_idx,
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch

# DataLoader
def create_data_loader(tokenizer, image_processor, data_args, num_workers=4):
    """Make dataset and collator for supervised fine-tuning."""
    dataset = LazySupervisedDataset_w_split(tokenizer=tokenizer,
                                image_processor=image_processor,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_loader = DataLoader(dataset, batch_size=data_args.batch_size, num_workers=num_workers,
                             shuffle=False, drop_last=False, collate_fn=data_collator)

    return data_loader



def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    if args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    data_loader = create_data_loader(tokenizer=tokenizer, image_processor=image_processor,
                                     data_args=args)

    chunk_size = len(data_loader.dataset.list_data_dict)
    np.save(args.emb_memory_loc + '_recover_indices.npy', data_loader.dataset.random_recov_indices)
    embed_size = model.config.hidden_size
    emb_array = np.zeros((chunk_size, embed_size), dtype='float16')

    for batch in tqdm(data_loader):

        input_batch = {k: v.to(device='cuda', non_blocking=True) if k!="images" else v.to(device='cuda', dtype=torch.float16, non_blocking=True)
                 for k, v in batch.items() if k != 'data_idx'}

        with torch.inference_mode():
            outputs = model(
                **input_batch,
                use_cache=False,
                output_hidden_states=True,
            )
            last_layer_last_tok = outputs['hidden_states'] if isinstance(outputs, dict) else outputs[-1]

            if args.avg_embed:
                embed = last_layer_last_tok[-1].mean(dim=1)  # average pool the last layer token as done in D2-prune
            else:
                embed = last_layer_last_tok[-1][:,-1,:]  # use the last layer last token as done in SemDeDup

        emb_array[batch['data_idx']] = embed.detach().cpu().numpy()

    np.save(args.emb_memory_loc + f'_{args.chunk_idx}.npy', emb_array)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--emb_memory_loc", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--image_folder", type=str, default="")
    parser.add_argument("--image_aspect_ratio", type=str, default="pad")
    parser.add_argument("--is_multimodal", action="store_true")
    parser.add_argument("--mm_use_im_start_end", action="store_true")
    parser.add_argument("--version", type=str, default="plain")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--avg_embed", action='store_true', default=False)
    args = parser.parse_args()


    eval_model(args)
