import logging
import os
import csv
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple
import soundfile as sf

from tqdm.auto import tqdm
from speechbrain.dataio.dataio import (
    load_pkl,
    save_pkl,
)

from ami_splits import get_AMI_split



logger = logging.getLogger(__name__)
SAMPLERATE = 16000


class AnnotationEntry(NamedTuple):
    text: str
    speaker: str
    gender: str
    start: float
    end: float
    words: List[Tuple[float, float, str]]


def prepare_ami(
    data_folder,
    save_folder,
    max_words_per_segment=10,
    mic="ihm",
    split_type="full_corpus_asr",
    min_chars_per_segment=0,
    min_duration=0.0,
    merge_consecutive=False,
):
    """
    Prepares reference RTTM and JSON files for the AMI dataset.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original amicorpus is stored.
    save_folder : str
        The save directory in results.
    max_words_per_segment : int
        Maximum number of words per segment. If None, then no splitting is done.
    mic : str
        The microphone type. Can be either "ihm" or "array". Defaults to "ihm".
        `ihm` stands for "In Head Microphone" and `array` stands for "Array Microphone".
    split_type : str
        The standard split option. Options are "scenario_only", "full_corpus", "full_corpus_asr".
        The latter two options are the same except that the "full_corpus_asr" option
        includes the ASR transcripts for the train, dev, and eval sets (which is what we want).
    min_chars_per_segment : int
        Minimum number of characters per segment. Segments with less than this number
        of characters will be discarded.
    min_duration : float
        Minimum duration of segments in seconds. Segments with less than this duration
        will be discarded.
    merge_consecutive : bool
        If True, then consecutive segments with less than max_words_per_segment words
        will be merged together.

    Example
    -------
    >>> from recipes.AMI.ami_prepare import prepare_ami
    >>> data_folder = '/network/datasets/ami/amicorpus/'
    >>> manual_annot_folder = '/home/mila/d/dawalatn/nauman/ami_public_manual/'
    >>> save_folder = 'results/save/'
    >>> split_type = 'full_corpus_asr'
    >>> mic_type = 'Lapel'
    >>> prepare_ami(data_folder, manual_annot_folder, save_folder, split_type, mic_type)
    """

    # # Meta files
    # meta_files = [
    #     os.path.join(meta_data_dir, "ami_train.subsegs.json"),
    #     os.path.join(meta_data_dir, "ami_dev.subsegs.json"),
    #     os.path.join(meta_data_dir, "ami_eval.subsegs.json"),
    # ]

    # Create configuration for easily skipping data_preparation stage
    conf = {
        "data_folder": data_folder,
        "save_folder": save_folder,
        # "meta_data_dir": meta_data_dir,
        "max_words_per_segment": max_words_per_segment,
        "min_chars_per_segment": min_chars_per_segment,
        "mic": mic,
        "min_duration": min_duration,
        "merge_consecutive": merge_consecutive,
        # "meta_files": meta_files,
    }

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Setting output option files.
    opt_file = "opt_ami_prepare.pkl"

    # Check if this phase is already done (if so, skip it)
    if skip(save_folder, conf, opt_file):
        logger.info(
            "Skipping data preparation, as it was completed in previous run."
        )
        return

    msg = "\tCreating meta-data file for the AMI Dataset.."
    logger.debug(msg)

    # # Get the split
    # train_set, dev_set, eval_set = get_AMI_split(split_type)

    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    # # Create reference RTTM files
    # splits = [train_set, dev_set, eval_set]
    # for spl in splits:
    #     prepare_segs_for_RTTM(
    #         spl,
    #         rttm_file,
    #         data_folder,
    #         manual_annot_folder,
    #         i,
    #         skip_TNO,
    #     )
    data_folder = Path(data_folder)
    wav_dir = data_folder / "signals"
    audio_paths = (
        wav_dir.rglob("*Headset-?.wav")
        if mic == "ihm"
        else wav_dir.rglob("*Array?-0?.wav")
    )
    # E.g. ami_dir/signals/EN2001a/audio/EN2001a.Array1-01.wav
    # Will be grouped to {"EN2001a": [Path("EN2001a.Array1-01.wav"), ...]"}
    grouped_audio = groupby(lambda x: x.parts[-3], audio_paths)

    recordings = []
    for meet_id, audio_files in grouped_audio.items():
        for audio_file in audio_files:
            # channel = int(audio_file.stem.split("-")[1])
            # spk_id = audio_file.stem.split("-")[2]
            audio = sf.SoundFile(audio_file)
            if audio.channels != 1:
                logger.warning(
                    f"Audio file {audio_file} has {audio.channels} channels, "
                    "but AMI data_preparation expects single-channel audio."
                    "Skipping this file."
                )
                continue
            sample_rate = audio.samplerate
            duration = audio.frames / sample_rate  # seconds
            if duration < min_duration:
                logger.debug(
                    f"Audio file {audio_file} has duration {duration} seconds, "
                    f"which is less than the minimum duration of {min_duration} seconds. "
                    "Skipping this file."
                )
                continue
            recordings.append(
                {
                    "meeting_id": meet_id,
                    "wav": str(audio_file),
                    "duration": duration,
                    "num_samples": audio.frames,
                    "sample_rate": sample_rate,
                    # "spk_id": spk_id,
                    # "channel": channel,
                }
            )
    # Get the split
    train_set, dev_set, eval_set = get_AMI_split(split_type)

    annotations = get_annotations(
        data_folder,
        max_words_per_segment=max_words_per_segment,
        merge_consecutive=merge_consecutive,
        min_chars_per_segment=min_chars_per_segment,
        min_duration=min_duration,
    )
    
    def process_split(split_set, split_name):
        recordings_split = [rec for rec in recordings if rec["meeting_id"] in split_set]
        annotations_split = {
            key: anns
            for key, anns in annotations.items()
            if key[0] in split_set
        }
        output_filename = os.path.join(
            save_folder,
            split_name + ".csv"
        )
        logger.info(f"Creating {split_name} csv file...")
        create_csvs(
            recordings_split,
            annotations_split,
            output_filename
        )
    
    process_split(train_set, "train")
    process_split(dev_set, "dev")
    process_split(eval_set, "eval")


    # # Create meta_files for splits
    # meta_data_dir = meta_data_dir
    # if not os.path.exists(meta_data_dir):
    #     os.makedirs(meta_data_dir)

    # splits = ["train", "dev", "eval"]
    # for spl in splits:
    #     # rttm_file = ref_rttm_dir + "/fullref_ami_" + i + ".rttm"
    #     # meta_filename_prefix = "ami_" + i
    #     output_filename = os.path.join(
    #         save_folder,
    #         spl + ".csv"
    #     )
    #     get_annotations(
    #         data_folder, output_filename, max_words_per_segment=30, merge_consecutive=False
    #     )

    save_opt_file = os.path.join(save_folder, opt_file)
    save_pkl(conf, save_opt_file)



def get_annotations(
    ami_dir: Path,
    max_words_per_segment: Optional[int] = None,
    merge_consecutive: bool = False,
    min_chars_per_segment: int = 0,
    min_duration: float = 0.0,
) -> Dict[str, List[tuple]]:
    
    # First we get global speaker ids and channels
    global_spk_id = {}
    channel_id = {}
    with open(ami_dir / "corpusResources" / "meetings.xml") as f:
        tree = ET.parse(f)
        for meeting in tree.getroot():
            meet_id = meeting.attrib["observation"]
            for speaker in meeting:
                local_id = (meet_id, speaker.attrib["nxt_agent"])
                global_spk_id[local_id] = speaker.attrib["global_name"]
                channel_id[local_id] = int(speaker.attrib["channel"])

    # Get the speaker segment times from the segments file
    segments = {}
    for file in (ami_dir / "segments").iterdir():
        meet_id, local_spkid, _ = file.stem.split(".")
        if (meet_id, local_spkid) not in global_spk_id:
            logging.warning(
                f"No speaker {meet_id}.{local_spkid} found! Skipping" " annotation."
            )
            continue
        spk = global_spk_id[(meet_id, local_spkid)]
        channel = channel_id[(meet_id, local_spkid)]
        key = (meet_id, spk, channel)
        segments[key] = []
        with open(file) as f:
            tree = ET.parse(f)
            for seg in tree.getroot():
                if seg.tag != "segment":
                    continue
                start_time = float(seg.attrib["transcriber_start"])
                end_time = float(seg.attrib["transcriber_end"])
                segments[key].append((start_time, end_time))

    # Now we go through each speaker's word-level annotations and store them
    words = {}
    for file in (ami_dir / "words").iterdir():
        meet_id, local_spkid, _ = file.stem.split(".")
        if (meet_id, local_spkid) not in global_spk_id:
            continue
        spk = global_spk_id[(meet_id, local_spkid)]
        channel = channel_id[(meet_id, local_spkid)]
        key = (meet_id, spk, channel)
        if key not in segments:
            continue
        words[key] = []
        with open(file) as f:
            tree = ET.parse(f)
            for word in tree.getroot():
                if word.tag != "w" or "starttime" not in word.attrib:
                    continue
                start_time = float(word.attrib["starttime"])
                end_time = float(word.attrib["endtime"])
                words[key].append((start_time, end_time, word.text))

    # Now we create segment-level annotations by combining the word-level
    # annotations with the speaker segment times. We also normalize the text
    # and break-up long segments (if requested).
    annotations = defaultdict(list)

    for key, segs in tqdm(segments.items()):
        # Get the words for this speaker
        spk_words: List[Tuple[int, int, str]] = words[key]
        # Now iterate over the speaker segments and create segment annotations
        for seg_start, seg_end in segs:
            # Get the words that fall within this segment
            seg_words = list(
                filter(lambda w: w[0] >= seg_start and w[1] <= seg_end, spk_words)
            )
            # Split the segment into subsegments where each subsegment has at most
            # max_words_per_segment words. If max_words_per_segment is None, then
            # no splitting is done.
            # If merge_consecutive is True, then consecutive subsegments with less
            # than max_words_per_segment words will be merged together.
            subsegments: List[List[Tuple[float, float, str]]] = split_segment(
                seg_words, max_words_per_segment, merge_consecutive
            )
            # Each subsegment will be treated as an utterance
            for subseg in subsegments:
                start = subseg[0][0]
                end = subseg[-1][1]
                word_alignments = []
                for w in subseg:
                    w_start = max(start, round(w[0], ndigits=4))
                    w_end = min(end, round(w[1], ndigits=4))
                    w_dur = add_durations(w_end, -w_start, sampling_rate=16000)
                    w_symbol = w[2].upper().strip()
                    if len(w_symbol) == 0:
                        continue
                    if w_dur <= 0:
                        logging.debug(
                            f"Segment {key[0]}.{key[1]}.{key[2]} at time {start}-{end} "
                            f"has a word with zero or negative duration ({w_dur}). Skipping."
                        )
                        continue
                    word_alignments.append(
                        (w_start, w_dur, w_symbol)
                    )
                # Merge all symbols into a single string
                text = " ".join(w[-1] for w in word_alignments)
                text = postprocess_text(text)
                if len(text) <= min_chars_per_segment:
                    continue
                duration = add_durations(end, -start, sampling_rate=16000)
                if duration < min_duration:
                    logging.debug(
                        f"Segment {key[0]}.{key[1]}.{key[2]} at time {start}-{end} "
                        f"has duration {duration} seconds, which is less than the "
                        f"minimum duration of {min_duration} seconds. Skipping."
                    )
                    continue
                annotations[key].append(
                    AnnotationEntry(
                        text,
                        key[1],  # speaker
                        key[1][0],  # gender
                        start,
                        end,
                        word_alignments,  # words
                    )
                )
    return annotations


def postprocess_text(text: str) -> str:
    # A basic postprocessing function that deletes some special characters
    # and replaces some others with spaces. It uses regular expressions.
    # remove punctuations
    text = re.sub(r"[^A-Z0-9']+", " ", text)
    # remove multiple spaces
    text = re.sub(r"\s+", " ", text)
    # apply few exception for dashed phrases, Mm-Hmm, Uh-Huh, OK etc. those are frequent in AMI
    # and will be added to dictionary
    text = re.sub(r"MM HMM", "MM-HMM", text)
    text = re.sub(r"UH HUH", "UH-HUH", text)
    text = re.sub(r"(\b)O K(\b)", r"\g<1>OK\g<2>", text)
    text = re.sub(r"(\b)O_K(\b)", r"\g<1>OK\g<2>", text)
    return text


def create_csvs(
    recordings: List[Dict[str, str]],
    annotations: Dict[str, List[AnnotationEntry]],
    csv_file: str,
) -> None:
    """
    Creates a csv file containing the metadata for the AMI dataset.

    Arguments
    ---------
    recordings : List[Dict[str, str]]
        A list of dictionaries containing the metadata for each recording.
        Each dictionary should contain the following keys:
            - meeting_id: str
            - wav: str
            - duration: float
            - num_samples: int
            - sample_rate: int
            - spk_id: str
            - channel: int
        Example:
            [
                {
                    "meeting_id": "EN2001a",
                    "wav": "EN2001a.Array1-01.wav",
                    "duration": 3600.0,
                    "num_samples": 57600000,
                    "sample_rate": 16000,
                    "spk_id": "A",
                    "channel": 1,
                },
                ...
            ]
    annotations : Dict[str, List[AnnotationEntry]]
        A dictionary containing the annotations for each recording. It will be
        used to create the "wrd" and "alignments" columns in the csv file.
        It is of the form:
            {
                ("EN2001a", "A", 1): [
                    AnnotationEntry(
                        text="Hello, how are you?",
                        speaker="A",
                        gender="M",
                        start=0.0,
                        end=1.0,
                        words=[(0.0, 0.5, "Hello"), (0.5, 1.0, "how are you?")]
                    ),
                    ...
                ],
                ...
            }
    csv_file : str
        The path to the csv file to be created.
    """
    csv_lines = [
        ["ID", "duration", "start_time", "wav", "spk_id", "wrd"]#, "alignments"]
    ]
    # Each annotation is one utterance
    # This means that we need to create one csv line per annotation
    for key, anns in tqdm(annotations.items()):
        meet_id, spk_id, channel = key
        # Find the recording that corresponds to this key
        rec = next(
            (
                rec
                for rec in recordings
                if rec["meeting_id"] == meet_id
            ),
            None,
        )
        if rec is None:
            logging.warning(
                f"Recording {meet_id}.{spk_id}.{channel} not found! Skipping."
            )
            continue
        assert os.path.isfile(rec["wav"]), f"Audio file {rec['wav']} not found!"
        for i, ann in enumerate(anns):
            if ann.end - ann.start <= 0 or len(ann.words) == 0:
                logging.debug(
                    f"Segment {meet_id}.{spk_id}.{channel} at time {ann.start}-{ann.end} "
                    f"has a word with zero or negative duration and no words. Skipping."
                )
                continue
            csv_lines.append(
                [
                    f"{meet_id}.{spk_id}.{channel}-{i}",
                    str(round(ann.end - ann.start, ndigits=2)),  # seconds
                    str(round(ann.start, ndigits=2)),  # seconds
                    rec["wav"],
                    ann.speaker,
                    ann.text,
                    # str(ann.words),
                ]
            )
    with open(csv_file, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        for line in csv_lines:
            csv_writer.writerow(line)


def add_durations(*durations, sampling_rate: int) -> float:
    """
    Adds two durations in seconds and returns the result in seconds.
    """
    return sum(int(d * sampling_rate) for d in durations) / sampling_rate

def groupby(func, l):
    g = defaultdict(list)
    from itertools import groupby as gb2
    for out in gb2(l, func):
        key = out[0]
        list_of_instances = list(out[1])
        g[key] += list_of_instances
    return  g


def skip(save_folder, conf, opt_file):
    """
    Detects if the AMI data_preparation has been already done.
    If the preparation has been done, we can skip it.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    # Checking if meta (json) files are available
    skip = True
    # for file_path in meta_files:
    #     if not os.path.isfile(file_path):
    #         skip = False

    # Checking saved options
    save_opt_file = os.path.join(save_folder, opt_file)
    if skip is True:
        if os.path.isfile(save_opt_file):
            opts_old = load_pkl(save_opt_file)
            if opts_old == conf:
                skip = True
            else:
                skip = False
        else:
            skip = False

    return skip


def split_segment(
    words: List[Tuple[float, float, str]],
    max_words_per_segment: Optional[int] = None,
    merge_consecutive: bool = False,
) -> List[List[Tuple[float, float, str]]]:
    """
    Given a list of words, return a list of segments (each segment is a list of words)
    where each segment has at most max_words_per_segment words. If merge_consecutive
    is True, then consecutive segments with less than max_words_per_segment words
    will be merged together.
    """
    segments = []
    current_segment = []
    for word in words:
        current_segment.append(word)
        if max_words_per_segment and len(current_segment) >= max_words_per_segment:
            segments.append(current_segment)
            current_segment = []
        elif not max_words_per_segment and word[2] == '.':
            segments.append(current_segment)
            current_segment = []
    if current_segment:
        segments.append(current_segment)
    if merge_consecutive:
        merged_segments = []
        for i, segment in enumerate(segments):
            if i == 0:
                merged_segments.append(segment)
            elif len(segment) + len(merged_segments[-1]) <= max_words_per_segment:
                merged_segments[-1].extend(segment)
            else:
                merged_segments.append(segment)
        segments = merged_segments
    return segments

    def split_(sequence, sep):
        chunk = []
        for val in sequence:
            if val[-1] == sep:
                if len(chunk) > 0:
                    yield chunk
                chunk = []
            else:
                chunk.append(val)
        if len(chunk) > 0:
            yield chunk

    def split_on_fullstop_(sequence):
        subsegs = list(split_(sequence, "."))
        if len(subsegs) < 2:
            return subsegs
        # Set a large default value for max_words_per_segment if not provided
        max_segment_length = max_words_per_segment if max_words_per_segment else 100000
        if merge_consecutive:
            # Merge consecutive subsegments if their length is less than max_words_per_segment
            merged_subsegs = [subsegs[0]]
            for subseg in subsegs[1:]:
                if (
                    merged_subsegs[-1][-1][1] == subseg[0][0]
                    and len(merged_subsegs[-1]) + len(subseg) <= max_segment_length
                ):
                    merged_subsegs[-1].extend(subseg)
                else:
                    merged_subsegs.append(subseg)
            subsegs = merged_subsegs
        return subsegs

    def split_on_comma_(segment):
        # This function smartly splits a segment on commas such that the number of words
        # in each subsegment is as close to max_words_per_segment as possible.
        # First we create subsegments by splitting on commas
        subsegs = list(split_(segment, ","))
        if len(subsegs) < 2:
            return subsegs
        # Now we merge subsegments while ensuring that the number of words in each
        # subsegment is less than max_words_per_segment
        merged_subsegs = [subsegs[0]]
        for subseg in subsegs[1:]:
            if len(merged_subsegs[-1]) + len(subseg) <= max_words_per_segment:
                merged_subsegs[-1].extend(subseg)
            else:
                merged_subsegs.append(subseg)
        return merged_subsegs

    # First we split the list based on full-stops.
    subsegments = list(split_on_fullstop_(words))

    if max_words_per_segment is not None:
        # Now we split each subsegment based on commas to get at most max_words_per_segment
        # words per subsegment.
        subsegments = [
            list(split_on_comma_(subseg))
            if len(subseg) > max_words_per_segment
            else [subseg]
            for subseg in subsegments
        ]
        # flatten the list of lists
        subsegments = [item for sublist in subsegments for item in sublist]

    # Filter out empty subsegments
    subsegments = list(filter(lambda s: len(s) > 0, subsegments))
    return subsegments