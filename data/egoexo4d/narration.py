import os, torch, json, tqdm, collections, random
from transformers import EvalPrediction

from .egoexo4d import EgoExo4D

from ..stream import StreamMixIn
from ..utils import ceil_time_by_fps, DictWithTo



class Ego4DNarrationStream(EgoExo4D, StreamMixIn):
    instructions = [{"role": "user", "content": "Please concisely narrate the video in real time. Use the tag 'C' to denote the camera wearer, and other letter tags, such as 'X', to denote other individuals in the scene."}]
    evaluation_kwargs = DictWithTo(evaluator='stream_evaluate')


    def get_annos(self, split: str) -> dict[str, dict[str, list]]:
        anno_path = os.path.join(EgoExo4D.anno_root, f'refined_narration_stream_{split}.json')
        narration_streams = {}

        if os.path.exists(anno_path):
            print(f'File exists: {anno_path}')
            with open(anno_path, 'r', encoding='utf-8') as f:
                try:
                    narration_streams = json.load(f)
                    print(f"Loaded JSON content successfully")  
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                except Exception as e:
                    print(f"Error loading file: {e}")
        else:
            print(f'File does not exist: {anno_path}')
        
        if isinstance(narration_streams, dict):
            print(f'load {len(narration_streams)} from transformed_narrations_corrected_{split}.json')
        else:
            print(f"Loaded data is not a dictionary, but a {type(narration_streams)}")

        return narration_streams

   
    def __init__(self, *, split: str, frame_fps: int, is_training: bool, augmentation: bool, **kwargs):
        super().__init__(split=split, frame_fps=frame_fps, augmentation=augmentation, is_training=is_training, **kwargs)
        self.is_training = is_training
        self.frame_fps = frame_fps

        annos = self.get_annos(split)
        print(f'Successfully load {len(annos)} videos from transformed_narrations_corrected_{split}.json')
        self.annos = []

        for video_uid, _annotation_uid_narrations in tqdm.tqdm(annos.items(), desc=f'transformed_narrations_corrected_{split}...'):

            if video_uid not in self.metadata:
                print(f"Warning: '{video_uid}' not found in metadata, skipping.")
                continue 
            duration = self.metadata[video_uid]['duration'] 
            for narrations in _annotation_uid_narrations.values(): 
                if not narrations:
                    continue
                start_time = ceil_time_by_fps(narrations[0]['time'], frame_fps, min_time=0, max_time=duration) 
                conversation = []
                last_time = start_time - 1 / frame_fps 
                last_text = None 
                
                for narration in narrations: 
                    if last_time >= duration: 
                        break
                    text = narration['text'] 
                    if text == last_text: 
                        continue
                    time = ceil_time_by_fps(narration['time'], frame_fps, min_time=0, max_time=duration) 
                    if time == last_time: 
                        conversation[-1]['content'] = text 
                    else: 
                        num_frames = int((time - last_time) * frame_fps) 
                        conversation.extend([ 
                            {"role": "stream", 'num_frames': num_frames, 'learn': True},
                            {"role": "assistant", "content": text, 'learn': True},
                        ])
                    last_time = time 
                    last_text = text 
                
                if not conversation:
                    continue
                
                self.annos.append({
                    'conversation': conversation,
                    'load_ranges': {self.metadata[video_uid]['path']: range(int(start_time*frame_fps), int(last_time*frame_fps)+1)},
                    'load_ranges1': {self.metadata1[video_uid]['path']: range(int(start_time*frame_fps), int(last_time*frame_fps)+1)}
                })

    @staticmethod 
    def _clean_text(src: str):
        dst = src.replace('#C', '').replace('#c', '').replace('@c', '')
        dst = dst.replace('#O', '').replace('#o', '')
        dst = dst.replace('#Unsure', '').replace('#unsure', '')
        dst = dst.replace('#', '')

        dst = dst.strip('.,\n ') + '.'
        words = dst.split()
        words[0] = words[0].capitalize()
        dst = ' '.join(words)
        return dst

    def preprocess_conversation(self, conversation):
        assert conversation[0]['role'] == 'stream' and conversation[0]['num_frames'] == 1 
        conversation[0]['learn'] = False 
        return conversation[:1] + [random.choice(self.instructions)] + conversation[1:] 

    def __getitem__(self, index):
        anno = self.annos[index]
        return *super().__getitem__(  
            conversation=self.preprocess_conversation(anno['conversation']),
            load_ranges=anno['load_ranges'],
            load_ranges1=anno['load_ranges1'] 
        ), index, self.evaluation_kwargs  

    def compute_metrics(self, eval_predictions: EvalPrediction, *args, **kwargs):
        lm_ppl, frame_diff, fluency, lm_correctness = torch.from_numpy(eval_predictions.predictions).mean(dim=0).tolist()
        return {
            f'lm_ppl': lm_ppl,
            f'time_diff': frame_diff / self.frame_fps,
            f'fluency': fluency,
            f'lm_correctness': lm_correctness
        }

def build_transformed_narrations_corrected_train(**kwargs):
    return Ego4DNarrationStream(split='train', **kwargs)

def build_transformed_narrations_corrected_val(**kwargs):
    return Ego4DNarrationStream(split='val', **kwargs)

class Ego4DRefinedNarrationStream(Ego4DNarrationStream):
    instructions = [
        {"role": "user", "content": "Please concisely narrate the video in real time."},
        {"role": "user", "content": "Help me to illustrate my view in short."},
        {"role": "user", "content": "Please simply describe what do you see."},
        {"role": "user", "content": "Continuously answer what you observed with simple text."},
        {"role": "user", "content": "Do concise real-time narration."},
        {"role": "user", "content": "Hey assistant, do you know the current video content? Reply me concisely."},
        {"role": "user", "content": "Simply interpret the scene for me."},
        {"role": "user", "content": "What can you tell me about? Be concise."},
        {"role": "user", "content": "Use simple text to explain what is shown in front of me."},
        {"role": "user", "content": "What is the action now? Please response in short."},
    ]

    def get_annos(self, split: str) -> dict:
        anno_path = os.path.join(EgoExo4D.anno_root, f'refined_narration_stream_{split}.json')
        assert os.path.exists(anno_path)
        narration_streams = json.load(open(anno_path)) 
        return narration_streams


def build_refined_transformed_narrations_corrected_train(**kwargs):
    return Ego4DRefinedNarrationStream(split='train', **kwargs)


def build_refined_transformed_narrations_corrected_val(**kwargs):
    return Ego4DRefinedNarrationStream(split='val', **kwargs)

def build_refined_transformed_narrations_corrected_train(**kwargs):
    return Ego4DRefinedNarrationStream(split='train', **kwargs)

def build_refined_narration_stream_val(**kwargs):
    return Ego4DRefinedNarrationStream(split='val', **kwargs)

def build_refined_narration_stream_train(**kwargs):
    return Ego4DRefinedNarrationStream(split='train', **kwargs)


if __name__ == '__main__':
    build_refined_transformed_narrations_corrected_train(
        frame_fps=2, is_training=True, augmentation=True,
        system_prompt='', tokenizer=None,
        vision_pretrained='siglip-large-patch16-384',
        embed_mark='2fps_max384_2fps_384_1+3x3',
        max_num_frames = 1200
    )
    build_refined_transformed_narrations_corrected_val(
        frame_fps=2, is_training=True, augmentation=True,
        system_prompt='', tokenizer=None,
        vision_pretrained='siglip-large-patch16-384',
        embed_mark='2fps_max384_2fps_384_1+3x3',
        max_num_frames = 1200
    )
