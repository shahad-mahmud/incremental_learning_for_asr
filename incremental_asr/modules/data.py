import json
import torch
import torchaudio
import sentencepiece


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

        return audio_id, signal, transcription


class SpeechDataLoader(torch.utils.data.DataLoader):
    def __init__(self, data_set_type: str, configs: dict,
                tokenizer: sentencepiece.SentencePieceProcessor) -> None:
        dataset = SpeechDataset(configs[f'{data_set_type}_annotation'])
        batch_size = configs[f'{data_set_type}_batch_size']
        super().__init__(dataset,
                        batch_size=batch_size,
                        shuffle=configs['shuffle'],
                        collate_fn=self.collate_function_padded,
                        drop_last=True)

        self.tokenizer = tokenizer

    def collate_function_padded(self, batch):
        ids, signals, transcriptions, tokens = [], [], [], []

        for id, signal, transcription in batch:
            ids.append(id)
            signals.append(signal.squeeze())
            transcriptions.append(transcription)
            tokens.append(
                torch.tensor(self.tokenizer.encode_as_ids(transcription)))

        signals = torch.nn.utils.rnn.pad_sequence(signals, batch_first=True)
        tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)

        return ids, signals, transcriptions, tokens
