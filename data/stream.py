import torch, random, os
from transformers import PreTrainedTokenizer

from .utils import rand_bool
import logging

class StreamMixIn(torch.utils.data.Dataset):

    def __init__(self, is_training: bool, system_prompt: str, augmentation: bool, max_num_frames: int, tokenizer: PreTrainedTokenizer, **kwargs):
        super().__init__()
        self.is_training = is_training
        self.system_prompt = system_prompt
        self.augmentation = augmentation    
        self.tokenizer = tokenizer
        self.max_num_frames = max_num_frames  
        assert system_prompt is not None, 'Please add a system prompt'

    def augment(self, conversation): 
        if not self.augmentation or not self.is_training:
            return conversation

        assistant_messages = [(i, message) for i, message in enumerate(conversation) if message['role'] == 'assistant' and message.get('learn', False)]
        if len(assistant_messages) <= 1:
            return conversation
        i, assistant_message_i = random.choice(assistant_messages[:-1]) # do not choose the last one, since its meaningless to dependency
        real_content = assistant_message_i['content']
        
        fake_contents = list(set(message['content'] for _, message in assistant_messages if message['content'] != real_content)) + [''] + [None]
  
        fake_content = random.choice(fake_contents)
        fake_message_i = {'role': 'assistant', 'content': fake_content, 'learn': False} if fake_content is not None else None
        
        if rand_bool(): # fix the wrong content at the next frame
            
            if fake_message_i is not None and conversation[i+1]['role'] == 'stream' and conversation[i+1]['num_frames'] > 1: 
                conversation = conversation[:i] + [
                    fake_message_i,
                    {'role': 'stream', 'num_frames': 1, 'learn': True}, 
                    {'role': 'assistant', 'content': f'(Sorry, the last response is wrong) {real_content}', 'learn': True},
                    {'role': 'stream', 'num_frames': conversation[i+1]['num_frames'] - 1, 'learn': True}
                ] + conversation[i+2:]
            
            elif fake_message_i is None and conversation[i-1]['role'] == 'stream' and conversation[i+1]['role'] == 'stream' and conversation[i+1]['num_frames'] > 1: 
                conversation = conversation[:i-1] + [
                    {'role': 'stream', 'num_frames': conversation[i-1]['num_frames'] + 1, 'learn': conversation[i-1]['num_frames'] - 1},
                    {'role': 'assistant', 'content': real_content, 'learn': True},
                    {'role': 'stream', 'num_frames': conversation[i+1]['num_frames'] - 1, 'learn': True}
                ] + conversation[i+2:]
        
        
        else: # not fix 
            if fake_message_i is not None:
                if conversation[i+1]['role'] == 'stream': 
                    conversation = conversation[:i] + [
                        fake_message_i,
                        {'role': 'stream', 'num_frames': conversation[i+1]['num_frames'], 'learn': False}, 
                    ] + conversation[i+2:]
                else:
                    conversation = conversation[:i] + [fake_message_i] + conversation[i+1:]
            else: 
                if conversation[i-1]['role'] == 'stream':
                    if conversation[i+1]['role'] != 'stream':
                        conversation = conversation[:i-1] + [
                            {'role': 'stream', 'num_frames': conversation[i-1]['num_frames'], 'learn': conversation[i-1]['num_frames'] - 1},
                        ] + conversation[i+1:]
                    else:
                        conversation = conversation[:i-1] + [
                            {'role': 'stream', 'num_frames': conversation[i-1]['num_frames'] + conversation[i+1]['num_frames'], 'learn': conversation[i-1]['num_frames'] - 1}, 
                        ] + conversation[i+2:]
                else:
                    if conversation[i+1]['role'] == 'stream':
                        conversation = conversation[:i] + [
                            {'role': 'stream', 'num_frames': conversation[i+1]['num_frames'], 'learn': False}, 
                        ] + conversation[i+2:]
                    else:
                        conversation = conversation[:i] + conversation[i+1:]
        
        return conversation
    
    def max_frames_clip(self, conversation: list[dict], load_ranges: dict[str, range], max_num_frames: int):
        
        cum_num_frames = 0  
        for i, message in enumerate(conversation):   
            if message['role'] == 'stream':
                if cum_num_frames + message['num_frames'] > max_num_frames:
                    conversation = conversation[:i]
                    load_ranges = {path: range(ranger.start, ranger.start + cum_num_frames) for path, ranger in load_ranges.items()}
                    break
                cum_num_frames += message['num_frames']
        return conversation, load_ranges

    def __getitem__(self, *, conversation: list[dict], load_ranges: dict[str, range], load_ranges1: dict[str, range] | torch.Tensor = None, add_generation_prompt=False, **kwargs):
        
        conversation1 = conversation
        # 1.1 load visual encoding
        if isinstance(load_ranges, torch.Tensor):
            frames = load_ranges
        elif load_ranges is not None:
            conversation, load_ranges = self.max_frames_clip(conversation, load_ranges, self.max_num_frames)
            frames_list = []
            for path, ranger in load_ranges.items():
                if os.path.exists(path):
                    try:
                        frame = torch.load(path)[ranger]
                        frames_list.append(frame)
                    except Exception as e:
                        print(f"警告: 加载文件时出错 {path}: {e}")
                else:
                    print(f"警告: 文件不存在，跳过 {path}")
            
            if frames_list:
                frames = torch.cat(frames_list)
            else:
                frames = torch.tensor([])
        
        # 1.2 load new visual encoding
        if isinstance(load_ranges1, torch.Tensor):
            frames1 = load_ranges1
        elif load_ranges1 is not None:
            conversation1, load_ranges1 = self.max_frames_clip(conversation1, load_ranges1, self.max_num_frames)
            frames_list1 = []
            for path1, ranger1 in load_ranges1.items():
                if os.path.exists(path1):
                    try:
                        frame1 = torch.load(path1)[ranger1]
                        frames_list1.append(frame1)
                    except Exception as e:
                        print(f"警告: 加载文件时出错 {path1}: {e}")
                else:
                    print(f"警告: 文件不存在，跳过 {path1}")
            
            if frames_list1:
                frames1 = torch.cat(frames_list1)
            else:
                frames1 = torch.tensor([])

        # 2. prepare texts
        if self.augmentation:
            conversation = self.augment(conversation)
        conversation = [{"role": "system", "content": self.system_prompt}] + conversation


        text = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=add_generation_prompt)
        # 3. learn ranges
        learn_ranges = self.tokenizer.get_learn_ranges(conversation) if not add_generation_prompt else []
        
        return text, frames, frames1, learn_ranges

