import os
import sys

from dataclasses import dataclass
import numpy as np

import torch

import supervision as sv
from autodistill.detection import CaptionOntology, DetectionBaseModel

from PIL import Image
from transformers import TextStreamer
from collections import namedtuple

import subprocess

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AUTODISTILL_DIR = os.path.join(HOME, ".autodistill")
LLAVA_DIR = os.path.join(AUTODISTILL_DIR, "LLaVA")
MODEL = "liuhaotian/llava-v1.5-7b"

def run_command(cmd, directory=None):
    result = subprocess.run(cmd, cwd=directory, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    
    if result.returncode != 0:
        raise ValueError(f"Command '{' '.join(cmd)}' failed to run.")

def install_llava_dependencies():
    commands = [
        (["mkdir", "-p", AUTODISTILL_DIR], None),
        (["git", "clone", "https://github.com/haotian-liu/LLaVA.git"], AUTODISTILL_DIR),
        (["pip", "install", "-e", "."], LLAVA_DIR),
        
    ]
    
    for cmd, dir in commands:
        run_command(cmd, dir)

if not os.path.exists(LLAVA_DIR):
    install_llava_dependencies()

sys.path.insert(0, LLAVA_DIR)

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


@dataclass
class LLaVA(DetectionBaseModel):
    ontology: CaptionOntology
    
    def __init__(self, ontology: CaptionOntology):
        self.ontology = ontology
        
        disable_torch_init()

        print("x")

        model_name = get_model_name_from_path(MODEL)

        print(model_name)
        tokenizer, model, image_processor, context_len = load_pretrained_model(MODEL, None, model_name, True, False, device=DEVICE)

        if 'llama-2' in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        conv = conv_templates[conv_mode].copy()

        if "mpt" in model_name.lower():
            roles = ('user', 'assistant')
        else:
            roles = conv.roles

        print(image_processor)

        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.context_len = context_len
        self.roles = roles
        self.conv = conv

        pass

    def predict(self, input: str) -> sv.Detections:
        image = Image.open(input)

        ImageInfo = namedtuple('ImageInfo', ['image_aspect_ratio'])

        args = ImageInfo(image_aspect_ratio="pad")

        image_tensor = process_images([image], self.image_processor, args)

        if type(image_tensor) is list:
            image_tensor = [image.to(self.model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

        objects = self.ontology.prompts()
        
        result = []

        for object in objects:
            inp = f"""Detect all {object} in the image. Return coordinates in x0, y0, x1, y1 format like: object, 0, 0, 0, 0. There should be one object per line. If there are no detections, say STOP."""

            if image is not None:
                if self.model.config.mm_use_im_start_end:
                    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
                else:
                    inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
                self.conv.append_message(self.conv.roles[0], inp)
                image = None
            else:
                self.conv.append_message(self.conv.roles[0], inp)

            self.conv.append_message(self.conv.roles[1], None)

            prompt = self.conv.get_prompt()

            print(prompt)

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            stop_str = self.conv.sep if self.conv.sep_style != SeparatorStyle.TWO else self.conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
            streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=512,
                    streamer=streamer,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])

            outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            
            self.conv.messages[-1][-1] = outputs

            outputs = outputs.split('\n')

            print(outputs)

            for line in outputs:
                line = line.strip()

                # remove all characters that aren't numbers, a period, a comma
                line = ''.join([c for c in line if c in '0123456789.,'])

                if line == '':
                    continue

                line = line.split(',')

                # if response is malformed, skip to next prompt
                try:
                    result.append({
                        'label': object,
                        'x0': float(line[0]),
                        'y0': float(line[1]),
                        'x1': float(line[2]),
                        'y1': float(line[3])
                    })
                except Exception as e:
                    print(e)
                    continue

            if stop_str in self.conv.messages[-1][-1] or "STOP" in self.conv.messages[-1][-1]:
                continue

        if len(result) == 0:
            return sv.Detections.empty()
        
        print([[d['x0'], d['y0'], d['x1'], d['y1']] for d in result])

        detections = sv.Detections(
            xyxy=np.array([[d['x0'], d['y0'], d['x1'], d['y1']] for d in result]),
            class_id=np.array([self.ontology.prompts().index(d['label']) for d in result]),
            confidence=np.array([1.0 for _ in result]),
        )

        return detections