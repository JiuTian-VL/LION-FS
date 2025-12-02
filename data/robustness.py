import torch
from .stream import StreamMixIn

class Robustness(StreamMixIn):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.annos = [ 
            {
                'conversation': [
                    {"role": "user", 'content': 'Please help me to narrate the video in real time.'},
                    {"role": "assistant", "content": 'Sure! Please provide me the video stream, and I will help you.', 'learn': True},
                ],
                'load_ranges': None,
            },
            {
                'conversation': [
                    {"role": "user", 'content': 'Concisely output what you have seen.'},
                    {"role": "assistant", "content": 'Sure! Please provide me the video inputs.', 'learn': True},
                ],
                'load_ranges': None,
            },
            {
                'conversation': [
                    {"role": "user", 'content': '...'},
                    {"role": "assistant", "content": 'Hi! How can I help you?', 'learn': True},
                ],
                'load_ranges': None,
            },
            {
                'conversation': [
                    {"role": "stream", 'num_frames': 10, 'learn': False},
                    {"role": "user", 'content': 'What is happening now?', 'learn': False},
                    {"role": "assistant", 'content': 'Sorry, it seems the input video frames are noise. Could you have a check?', 'learn': True},
                ],
                'load_ranges': torch.rand(1, 1024),  #此时的load_ranges表示 帧特征的1024个数值均为随机
            },
            {
                'conversation': [
                    {"role": "stream", 'num_frames': 1, 'learn': False},
                    {"role": "user", 'content': 'What is the next step?', 'learn': False},
                    {"role": "assistant", 'content': 'Sorry, I cannot see anything in the input video frame. Could you check it?', 'learn': True},
                ],
                'load_ranges': torch.zeros(1, 1024),   #此时的load_ranges表示 帧特征的1024个数值均为0
            },
        ]

    # 根据索引从 self.annos 中获取对应的注释数据，并调用父类的 __getitem__ 方法来处理这些数据，返回一个包含对话、加载范围、索引和一个空字典的元组
    def __getitem__(self, index):
        anno = self.annos[index]
        return *super().__getitem__(
            conversation=anno['conversation'],
            load_ranges=anno['load_ranges']
        ), index, {}

    # 返回 self.annos 的长度，即注释数据的数量
    def __len__(self):
        return len(self.annos)



# build_robustness 函数用于创建并返回一个 Robustness 类的实例。kwargs：关键字参数，用于初始化 Robustness 类。
def build_robustness(**kwargs):
    return Robustness(**kwargs)