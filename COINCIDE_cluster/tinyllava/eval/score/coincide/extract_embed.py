import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from tqdm import tqdm
import shortuuid
import transformers
import copy
import numpy as np
import time

from typing import Dict, Sequence
from dataclasses import dataclass
from tinyllava import conversation as conversation_lib
from tinyllava.data.process import preprocess, preprocess_multimodal
from tinyllava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from tinyllava.model.builder import load_pretrained_model
from tinyllava.utils import disable_torch_init
from tinyllava.mm_utils import get_model_name_from_path
from tinyllava.eval.score.coincide import autograd_hacks
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math


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

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0],
                             data_idx=i)

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    prompt_len: int

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
        prompt_length = self.prompt_len
        image_idx = (batch['input_ids'] == IMAGE_TOKEN_INDEX).sum(dim=-1)
        language_length = torch.tensor([batch['attention_mask'][idx][prompt_length+1:].sum() if image_idx[idx] != 0 else
                                        batch['attention_mask'][idx][prompt_length:].sum() for idx in range(input_ids.shape[0])])
        batch['language_length'] = language_length

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
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, prompt_len=data_args.prompt_len)
    data_loader = DataLoader(dataset, batch_size=data_args.batch_size, num_workers=num_workers,
                             shuffle=False, drop_last=False, collate_fn=data_collator)

    return data_loader



def prepare_calibration_input(model, batch):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    cache = {'inps': None, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            cache['inps'] = inp
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    try:
        with torch.no_grad():
            model(**batch, use_cache=False)
    except ValueError:
        pass
    layers[0] = layers[0].module

    inps = cache['inps']
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids


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
    prompt_len = args.prompt_len

    layers = model.model.layers
    for i in args.layer_list:
        autograd_hacks.add_hooks(layers[i].self_attn.o_proj)

    chunk_size = len(data_loader.dataset.list_data_dict)
    np.save(args.score_path + '_recover_indices.npy', data_loader.dataset.random_recov_indices)
    num_layers = len(layers)
    num_target_layers = len(args.layer_list)

    sim_act = nn.Tanh()

    msa_task_emb = np.zeros((chunk_size, model.config.hidden_size * num_target_layers * 2), dtype='float16')

    # We do not define the memory yet.
    for batch in tqdm(data_loader, total=len(data_loader)):
        input_batch = {k: v.to(device='cuda', non_blocking=True) if k!="images" else v.to(device='cuda', dtype=torch.float16, non_blocking=True)
                 for k, v in batch.items() if k != 'data_idx' and k != 'language_length'}

        with torch.no_grad():
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, input_batch)

        msa_embed_list = []

        for i_layer in range(num_layers):

            if i_layer > args.layer_list[-1]:
                break

            layer = layers[i_layer]
            with torch.no_grad():
                outs = layer(inps, attention_mask=attention_mask, position_ids=position_ids)[0]

            if hasattr(layer.self_attn.o_proj, "activations"):
                msa_act = layer.self_attn.o_proj.activations
                msa_act_v = torch.mean(msa_act[:, prompt_len:prompt_len+729], dim=1)
                msa_act_v_np = F.normalize(sim_act(msa_act_v), dim=-1).detach().cpu().numpy()  # We perform normalization layer-wise and modal-wise.

                msa_act_l_np = np.zeros_like(msa_act_v_np)
                for batch_idx in range(len(batch['data_idx'])):
                    i_lang_len = batch['language_length'][batch_idx]
                    if i_lang_len != 0:
                        msa_act_l = torch.mean(msa_act[batch_idx, prompt_len+729:prompt_len+729 + i_lang_len], dim=0)
                        msa_act_l_np[batch_idx] = F.normalize(sim_act(msa_act_l), dim=-1).detach().cpu().numpy()

                msa_embed_list.append(np.concatenate([msa_act_v_np, msa_act_l_np], axis=-1) / np.sqrt(2 * num_target_layers))

            inps, outs = outs, inps

        global_msa_mask = np.concatenate(msa_embed_list, axis=-1)

        msa_task_emb[batch['data_idx']] = global_msa_mask

    target_layer_str = [str(num) for num in args.layer_list]
    target_layer_str = ''.join(target_layer_str)
    msa_task_emb[np.isnan(msa_task_emb)] = 0

    np.save(os.path.join(args.score_path, f'tan_act_{target_layer_str}_msa_{args.chunk_idx}.npy'), msa_task_emb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--score_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--image_folder", type=str, default="")
    parser.add_argument("--image_aspect_ratio", type=str, default="pad")
    parser.add_argument("--is_multimodal", action="store_true")
    parser.add_argument("--mm_use_im_start_end", action="store_true")
    parser.add_argument("--version", type=str, default="plain")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--layer_list", type=int, nargs='+', default=[3, 4, 5])
    parser.add_argument("--prompt_len", type=int, default=31)
    args = parser.parse_args()

    eval_model(args)