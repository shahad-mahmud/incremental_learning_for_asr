import os
import json
import regex
import logging
import torchaudio
from tqdm import tqdm

from utils.io import *


def prepare_annotation_files(configs: dict) -> None:
    """Read the data directory from configurations and create
    annotation files for training, validation and testing. The
    annotation files are JSON files with the following format:
    {
        ...
        utterance_id: {
            audio_path: path_to_audio_file,
            duration: duration_of_audio_file,
            transcription: transcription_of_audio_file
        }
        ...
    }

    Args:
        configs (dict): The configurations dictionary.
    """
    if annotation_files_exist(configs):
        logging.warn("Annotation files already exist. Skipping preparation.")
        return

    audio_file_paths = get_files_with_extensions(configs['data_dir'],
                                                 configs['audio_extensions'])
    transcript_file_path = get_files_with_extensions(configs['data_dir'],
                                                     ['.tsv'])

    transcripts = get_transcription(transcript_file_path)
    train_paths, valid_paths, test_paths = get_sets(audio_file_paths)

    create_json(train_paths, transcripts, configs['train_annotation'])
    create_json(valid_paths, transcripts, configs['valid_annotation'])
    create_json(test_paths, transcripts, configs['test_annotation'])


def create_json(audio_paths: List[str],
                trans_dict: dict,
                json_file_path: str,
                streaming_dataset: bool = True,
                streaming_duration_limit: int = 15):
    """Create JSON files for a dataset

    Args:
        audio_paths (List[str]): Audio paths of the dataset
        trans_dict (dict): Transcription dictionary where the key if the audio ID
        json_file_path (str): JSON file path to dump the dir
        streaming_dataset (bool, optional): Is the dataset for streaming ASR. Defaults to False.
        streaming_duration_limit (int, optional): Maximum duration of audio to allow. Defaults to 10.
    """
    transcription_dict = {}
    for audio_path in tqdm(audio_paths, desc=f"Creating {json_file_path}"):
        try:
            signal, sample_rate = torchaudio.load(audio_path)
            duration = signal.shape[1] / sample_rate

            if streaming_dataset and duration > streaming_duration_limit:
                continue
            if duration < 0.15:
                continue

            path_parts = audio_path.split(os.path.sep)
            utt_id, _ = os.path.splitext(path_parts[-1])

            transcription_dict[utt_id] = {
                "audio_path": audio_path,
                "duration": duration,
                "transcription": trans_dict[utt_id],
            }
        except Exception as e:
            logging.error(f"Problem with wave file: {audio_path}")
            logging.error(e)

    dir_path = os.path.dirname(json_file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with open(json_file_path, mode="w+") as f:
        json.dump(transcription_dict, f, indent=2)

    logging.info(f"{json_file_path} successfully created!")


def prepare_text_file(configs: dict) -> None:
    """Create a text file with the transcriptions of the dataset

    Args:
        configs (dict): The configurations dictionary.
    """
    if text_file_exists(configs):
        logging.warn("Text file already exists. Skipping preparation.")
        return

    transcript_file_path = get_files_with_extensions(configs['data_dir'],
                                                     ['.tsv'])
    transcripts = get_transcription(transcript_file_path)

    create_text_file(configs, transcripts)


def create_text_file(configs: dict, transcripts: dict):
    """Create a text file with the transcriptions of the dataset

    Args:
        configs (dict): The configurations dictionary.
        transcripts (dict): Transcription dictionary
    """
    # get parents
    parent = os.path.dirname(configs['text_file'])
    if not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

    with open(configs['text_file'], mode="w+") as f:
        for _, value in transcripts.items():
            f.write(f"{value.strip()}\n")

    logging.info(f"{configs['text_file']} successfully created!")


def get_sets(paths: list):
    train_set = paths[:int(0.8 * len(paths))]
    valid_set = paths[int(0.8 * len(paths)):int(0.8 * len(paths)) +
                      int(0.1 * len(paths))]
    test_set = paths[int(0.8 * len(paths)) + int(0.1 * len(paths)):]

    return train_set, valid_set, test_set


def get_transcription(trans_file_list: List[str]) -> dict:
    """Read a TSV file and create transcription dictionary

    Args:
        trans_file_list (List[str]): Path of the transcription file

    Returns:
        dict: Transcription dictionary
    """
    trans_dict = {}
    for trans_file in trans_file_list:
        logging.info(f"Getting transcript from file: {trans_file}")
        with open(trans_file) as f:
            for line in f:
                try:
                    utt_id = line.split("\t")[0]
                    text = line.split("\t")[2]

                    trans_dict[utt_id] = clean_text(text)
                except Exception as e:
                    logging.error(f"Problem raised\nLine details: {line}")
                    logging.error(e)

    logging.info("Transcription files read!")
    return trans_dict


def clean_text(text):
    text = regex.sub(r'[-_/]', ' ', text)
    text = regex.sub(r"""[-,;:'"!?“”_.()—…–।\[\]{}]""", '', text)
    text = regex.sub(r'\s+', ' ', text)

    return text.strip()


def annotation_files_exist(configs: dict) -> bool:
    if not os.path.exists(configs['train_annotation']):
        return False
    if not os.path.exists(configs['valid_annotation']):
        return False
    if not os.path.exists(configs['test_annotation']):
        return False

    return True


def text_file_exists(configs: dict) -> bool:
    return os.path.exists(configs['text_file'])