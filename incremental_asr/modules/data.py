import json
import torch
import torchaudio


class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, annotation_file_path: str) -> None:
        super().__init__()

        annotation_file = open(annotation_file_path, 'r')
        annotation_dict = dict(json.load(annotation_file))

        corrected_dict = {}
        for i, val in enumerate(annotation_dict.items()):
            corrected_dict[i] = {**{'id': val[0]}, **val[1]}
        self.annotation_dict = corrected_dict

    def __len__(self) -> int:
        return len(self.annotation_dict)

    def __getitem__(self, index: int) -> dict:
        audio_id = self.annotation_dict[index]['id']
        signal, _ = torchaudio.load(self.annotation_dict[index]['audio_path'])
        transcription = self.annotation_dict[index]['transcription']
        
        item = {
            'id': audio_id,
            'signal': signal,
            'transcription': transcription
        }
        return item