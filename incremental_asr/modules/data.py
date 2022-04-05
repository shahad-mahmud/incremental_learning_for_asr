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
    def __init__(
        self, 
        data_set_type: str, 
        configs: dict,
        tokenizer: sentencepiece.SentencePieceProcessor
    ) -> None:
        self.configs = configs
        self.data_set_type = data_set_type
        dataset = SpeechDataset(configs[f'{data_set_type}_annotation'])
        batch_size = configs[f'{data_set_type}_batch_size']
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=configs['shuffle'],
            collate_fn=self.collate_function_padded,
            drop_last=True,
            num_workers=configs['num_workers'],
            pin_memory=True
        )

        self.tokenizer = tokenizer

    def collate_function_padded(self, batch):
        ids, signals, transcriptions, batch_tokens = [], [], [], []
        signal_lengths, token_lengths = [], []

        for id, signal, transcription in batch:
            ids.append(id)
            signals.append(signal.squeeze())
            signal_lengths.append(signal.shape[1])
            transcriptions.append(transcription)

            tokens = self.tokenizer.encode_as_ids(transcription)
            tokens.insert(0, self.tokenizer.bos_id())
            tokens.append(self.tokenizer.eos_id())
            token_lengths.append(len(tokens))
            batch_tokens.append(torch.tensor(tokens))

        signals = torch.nn.utils.rnn.pad_sequence(signals, batch_first=True)
        batch_tokens = torch.nn.utils.rnn.pad_sequence(batch_tokens, batch_first=True)
        signal_lengths = torch.tensor(signal_lengths)
        token_lengths = torch.tensor(token_lengths)

        return ids, signals, transcriptions, batch_tokens, signal_lengths, token_lengths
