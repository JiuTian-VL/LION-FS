import torch, os, json, tqdm

class EgoExo4D:
    root = 'your/path/to/egoexo4d'  
    video_root = os.path.join(root, 'takes')
    anno_root = os.path.join(root, 'takes_annotations')

    def __init__(self, vision_pretrained: str, embed_mark: str, frame_fps: int, **kwargs):
        super().__init__(**kwargs)
        self.embed_dir_siglip = "/your/path/to/ego4d/siglip_embeddings"
        self.embed_dir_egovlpv2 = "/your/path/to/ego4d/egovlpv2_embeddings"
        
        print(f'self.embed_dir_siglip: {self.embed_dir_siglip}')
        print(f'self.embed_dir_egovlpv2: {self.embed_dir_egovlpv2}')

        self.frame_fps = frame_fps
        self.metadata, self.metadata1 = self.get_metadata() # moe
        self.annos: list[dict]

    def __len__(self):
        return len(self.annos)


    def get_metadata(self, ):
        metadata_path = '/your/path/to/egoexo4d/metadata_siglip.json' # egoexo4d siglip
        metadata_path_egovlpv2 = '/your/path/to/egoexo4d/metadata_egovlpv2.json' # egoexo4d egovlpv2
        
        if os.path.exists(metadata_path):
            print(f'load {metadata_path}...')
            metadata = json.load(open(metadata_path))
            if isinstance(metadata, dict):
                print(f"Metadata is a valid dictionary.load {metadata_path} done.")
        else:
            print(f'prepare {metadata_path}...')
            metadata = {}
            for file in tqdm.tqdm(os.listdir(self.embed_dir_siglip), desc=f'prepare {metadata_path}...'):
                path = os.path.join(self.embed_dir_siglip, file)
                duration = (len(torch.load(path)) - 1) / self.frame_fps
                key = os.path.splitext(os.path.basename(path))[0]  
                metadata[key] = {'duration': duration, 'path': path}
            json.dump(metadata, open(metadata_path, 'w'), indent=4)

        if os.path.exists(metadata_path_egovlpv2):
            print(f'load {metadata_path_egovlpv2}...')
            metadata1 = json.load(open(metadata_path_egovlpv2))
            if isinstance(metadata1, dict):
                print(f"Metadata is a valid dictionary.load {metadata_path_egovlpv2} done.")        
        else:
            print(f'prepare {metadata_path_egovlpv2}...')
            metadata1 = {}
            for file in tqdm.tqdm(os.listdir(self.embed_dir_egovlpv2), desc=f'prepare {metadata_path_egovlpv2}...'):
                path = os.path.join(self.embed_dir_egovlpv2, file)
                duration = (len(torch.load(path)) - 1) / self.frame_fps
                key = os.path.splitext(os.path.basename(path))[0] 
                metadata1[key] = {'duration': duration, 'path': path}
            json.dump(metadata1, open(metadata_path_egovlpv2, 'w'), indent=4)

        return metadata, metadata1

