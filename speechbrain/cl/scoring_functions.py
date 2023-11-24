# Credits to Christoph Minixhofer: https://github.com/MiniXC/speech-collator/blob/main/speech_collator/measures/snr.py

from typing import Callable, Dict, List, Union
import torch
import librosa
import numpy as np

from librosa.feature import rms
from speechbrain.utils.edit_distance import wer_details_by_utterance

def _curve_gt0(x, c1=-0.10511163, c2=-14.13272781, d1=0.05551931, d2=5.79780909, loc=1.63692793, scale1=3.31585597, scale2=-2.06433912):
  x = (scale1/(1 + np.exp(-c1*(x-c2)))) + (scale2*np.log(1 + np.exp(-d1*(x-d2))))
  x += loc
  return x

def _curve_lt0(x, loc=0.40890103, a=0.26238549, b=-2.87692077):
  return np.exp(x*a+b)+loc

def approx_gamma_curve(x, bound=1):
  result = np.zeros(x.shape)
  b_s = _curve_lt0(-bound)
  b_e = _curve_gt0(bound)
  b_k = (b_s - b_e)/2
  result[(x>-bound)&(x<bound)] = (b_s + b_e)/2 - b_k * (x[(x>-bound)&(x<bound)] / (np.abs(bound)))
  result[x<=-bound] = _curve_lt0(x[x<=-bound])
  result[x>=bound] = _curve_gt0(x[x>=bound])
  return result

def wada_snr(wav: Union[str, np.array]) -> float:
    """Direct blind estimation of the SNR of a speech signal.
    Paper on WADA SNR:
      http://www.cs.cmu.edu/~robust/Papers/KimSternIS08.pdf
    This function was adapted from this matlab code:
      https://labrosa.ee.columbia.edu/projects/snreval/#9
    Taken from Cristoph's code:
      https://github.com/MiniXC/speech-collator/blob/main/speech_collator/measures/snr.py

    Args:
        wav: str | np.array
            The wav file or a numpy array containing the wav file.
        
    Returns:
        wav_snr: float
            The estimated SNR of the wav file.
        wav_snr_t: float
            The estimated SNR of the wav file at each time step.
    """

    eps = 1e-10
    # next 2 lines define a fancy curve derived from a gamma distribution -- see paper
    db_vals = np.linspace(-20, 100, 10_000)
    g_vals = approx_gamma_curve(db_vals)

    # init
    if isinstance(wav, str):
       # Load the wav file
        wav, _ = librosa.load(wav, sr=None)

    # peak normalize, get magnitude, clip lower bound
    wav = np.array(wav)
    wav[wav==0] = eps
    wav = wav / np.abs(wav).max()
    abs_wav = np.abs(wav)

    # calcuate statistics
    # E[|z|]
    v1 = max(eps, abs_wav.mean())
    v1_t = rms(y=abs_wav, frame_length=1024, hop_length=256)[0]
    # E[log|z|]
    v2 = np.log(abs_wav).mean()
    v2_t = -rms(y=np.log(abs_wav), frame_length=1024, hop_length=256)[0]
    # log(E[|z|]) - E[log(|z|)]
    v3 = np.log(v1) - v2
    v3_t = np.log(v1_t) - v2_t

    wav_snr_idx = None
    if any(g_vals < v3):
        wav_snr_idx = np.where(g_vals < v3)[0].max()
    v3_t[0] = v3
    v3_t[-1] = v3
    t_idx = np.tile(g_vals, (len(v3_t),1)).T < v3_t
    t_idx = t_idx.sum(axis=0)-1
    t_idx[t_idx<0] = 0
    t_idx[t_idx>=len(db_vals)] = -1
    wav_snr_t = db_vals[t_idx]
    wav_snr_t[wav_snr_t==100] += v3_t[wav_snr_t==100] * 10
    wav_snr_t[wav_snr_t==-20] -= v3_t[wav_snr_t==-20] * 20
    
    if wav_snr_idx is None:
        wav_snr = db_vals[0]
        wav_snr -= v3 * 20
    elif wav_snr_idx == len(db_vals) - 1:
        wav_snr = db_vals[-1]
        wav_snr += v3 * 10
    else:
        wav_snr = db_vals[wav_snr_idx]

    return wav_snr, wav_snr_t

def speech_rate(sig: torch.Tensor, text: str, sr: int = 16000) -> float:
    """Calculate the speech rate of a wav file. This is the number of words per second.
    Args:
        sig: torch.Tensor
            The wav file.
        text: str
            The text of the wav file.
        sr: int
            The sample rate of the wav file.
    Returns:
        speech_rate: float
            The speech rate of the wav file.
    """
    # Calculate the number of words
    words = text.split()
    num_words = len(words) + 1  # +1 for the <eos> token

    # Calculate the length of the wav file
    length = len(sig) / sr

    # Calculate the speech rate
    speech_rate = num_words / length

    return speech_rate

def whisper_score(
        utt_info: Dict[str, str],
        model_type: str = "base",
        postprocess_text: Callable[[str], str] = lambda x: x,    
    ) -> float:
    """Use OpenAI's Whisper 3 model to load the wav file from utt_info
    and calculate the word error rate (given the ground truth).

    NOTE: This function requires the whisper package to be installed.
    NOTE 2: This function converts all text to lowercase before computing
    the WER.

    Arguments
    ---------
    utt_info : Dict[[str], str]
        The utterance information dictionary. It must contain the
        following keys:
            - "wav" : The path to the wav file.
            - "wrd" : The ground truth text.
    model_type : str
        The type of the model to use. It can be either "base" or "large".
    postprocess_text : Callable[[str], str]
        A function to postprocess the text before computing the WER.
        This is useful if e.g. the text contains special tokens that need
        to be removed, or you need to convert the text to lowercase.

    Returns
    -------
    wer : float
        The word error rate.
    """
    try:
        import whisper
    except ImportError:
        raise ImportError("Please install the whisper package to use this scoring function.")
    # Load the wav file
    wav = utt_info["wav"]
    # Load the ground truth text
    wrd = utt_info["wrd"].strip().lower().split()
    
    # Use the Whisper model to calculate the word error rate
    model = whisper.load_model(model_type)
    result = model.transcribe(wav)
    hyp = postprocess_text(
        result["text"].strip()
    ).lower().split()
    # Get WER score
    ref_dict = {"dummy_utt_id": [wrd]}
    hyp_dict = {"dummy_utt_id": [hyp]}
    metric_score: List[Dict[str, float]] = wer_details_by_utterance(
        ref_dict,
        hyp_dict,
        compute_alignments=False
    )
    # Final WER is in the first element of the list, under the key "WER"
    wer = metric_score[0]["WER"]
    
    return wer

def dummy_score(utt_info: Dict[str, str]) -> float:
    """A dummy scoring function that always returns 0.5.

    Arguments
    ---------
    utt_info : Dict[[str], str]
        The utterance information dictionary. It must contain the
        following keys:
            - "wav" : The path to the wav file.
            - "wrd" : The ground truth text.

    Returns
    -------
    score : float
        The score.
    """
    import time
    time.sleep(0.1)
    return 0.5