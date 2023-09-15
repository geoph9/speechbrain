#!/usr/bin/env/python3
"""Recipe for training a conformer model with wav2vec2 encoder on LibriSpeech.
The system employs an encoder-decoder architecture, where the encoder is a
pre-trained wav2vec2 model and the decoder is a conformer model.
The loss function is LF-MMI. For more details, please refer to the original
paper: https://arxiv.org/pdf/2005.08100.pdf
To run this recipe, do the following:
> python train_with_wav2vec_k2.py hparams/hparams/train_hf_wav2vec_k2_mmi.yaml --data_folder=/path

Authors
 * Georgios Karakasidis 2023
 * Zeyu Zhao 2023
 * Rudolf A Braun 2022
 * Titouan Parcollet 2022
 * Sung-Lin Yeh 2021
 * Ju-Chieh Chou 2020
 * Mirco Ravanelli 2020
 * Abdel Heba 2020
 * Peter Plantinga 2020
 * Samuele Cornell 2020
"""

import os
import sys
from typing import List, Union
import torch
import logging
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main, if_main_process
from hyperpyyaml import load_hyperpyyaml
from pathlib import Path

from speechbrain.k2_integration.prepare_lang import prepare_lang
from speechbrain.k2_integration.lexicon import Lexicon
from speechbrain.k2_integration.make_kn_lm import make_kn_lm
from mmi_graph_compiler import MmiTrainingGraphCompiler

logger = logging.getLogger(__name__)


# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        # Downsample the inputs if specified
        if hasattr(self.modules, "downsampler"):
            wavs = self.modules.downsampler(wavs)
        # Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            if hasattr(self.modules, "env_corrupt"):
                wavs_noise = self.modules.env_corrupt(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])

            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)

        # Forward pass

        # Handling SpeechBrain vs HuggingFance pretrained models
        if hasattr(self.modules, "extractor"):  # SpeechBrain pretrained model
            latents = self.modules.extractor(wavs)
            feats = self.modules.encoder_wrapper(latents, wav_lens=wav_lens)[
                "embeddings"
            ]
        else:  # HuggingFace pretrained model
            feats = self.modules.wav2vec2(wavs, wav_lens)
        # print(f"feats: {feats.shape} -- {feats.shape[1]/batch.duration} -- id: {batch.id[0]}")
        x = self.modules.enc(feats)

        # Compute outputs
        logits = self.modules.ctc_lin(x)

        # Upsample the inputs if they have been highly downsampled
        if hasattr(self.hparams, "upsampling") and self.hparams.upsampling:
            logits = logits.view(
                logits.shape[0], -1, self.hparams.output_neurons
            )

        p_ctc = self.hparams.log_softmax(logits)
        return p_ctc, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        p_ctc, wav_lens = predictions

        ids = batch.id
        # tokens, tokens_lens = batch.tokens

        if hasattr(self.modules, "env_corrupt") and stage == sb.Stage.TRAIN:
            raise NotImplementedError(
                "Env. corruption is not implemented for models trained with k2"
            )
            
        # Sort batch to be descending by length of wav files, which is demanded by k2
        if self.hparams.sorting == "ascending":
            p_ctc = torch.flip(p_ctc, (0,))
            wav_lens = torch.flip(wav_lens, (0,))
            texts = [batch.wrd[i] for i in reversed(range(len(batch.wrd)))]
        elif self.hparams.sorting == "descending":
            texts = batch.wrd
        else:
            raise NotImplementedError("Only ascending or descending sorting is implemented, but got {}".format(self.hparams.sorting))

        is_training = (stage == sb.Stage.TRAIN)
        loss_ctc = self.hparams.mmi_cost(
            log_probs=p_ctc, 
            input_lens=wav_lens, 
            graph_compiler=self.graph_compiler,
            texts=texts,
            is_training=is_training,
        )

        loss = loss_ctc
        if loss.item() > 1000:
            logger.warning(f"Loss exploded to {loss.item()} on id {ids[0]}.")
        # print(f"loss: {loss.item()}, {loss.requires_grad=}, duration: {batch.duration}")


        if stage == sb.Stage.VALID:
            # Decode token terms to words
            predicted_texts = self.graph_compiler.decode(
                p_ctc,
                wav_lens,
                ac_scale=self.hparams.ac_scale,
                stage=stage,
            ) # list of strings
            predicted_words = [wrd.split(" ") for wrd in predicted_texts]
            target_words = [wrd.split(" ") for wrd in texts]
            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)
        if stage == sb.Stage.TEST:  # Language model decoding only used for test
            if self.hparams.use_language_modelling:
                raise NotImplementedError(
                    "Language modelling is not implemented for models trained with k2"
                )
            else:
                # If the decoding method is 1best then the metric stats will be
                # saved in a single file, otherwise, a new directory will be created
                # for each lm_scale used in whole lattice rescoring.
                decode_output: Union[dict, List[str]] = self.graph_compiler.decode(
                    p_ctc,
                    wav_lens,
                    search_beam=self.hparams.test_search_beam,
                    output_beam=self.hparams.test_output_beam,
                    ac_scale=self.hparams.ac_scale,
                    max_active_states=self.hparams.test_max_active_state,
                    is_test=True,
                    lm_scale_list=self.hparams.lm_scale_list,
                    stage=stage,
                ) # list of strings
                target_words: List[List[str]] = [wrd.split(" ") for wrd in texts]
                if self.graph_compiler.decoding_method == "1best":
                    predicted_words: List[List[str]] = [wrd.split(" ") for wrd in decode_output]
                    self.wer_metric.append(ids, predicted_words, target_words)
                    self.cer_metric.append(ids, predicted_words, target_words)
                else:
                    for i, lm_scale in enumerate(self.hparams.lm_scale_list):
                        predicted_texts: List[str] = decode_output[f"lm_scale_{lm_scale:.1f}"]
                        predicted_words: List[List[str]] = [wrd.split(" ") for wrd in predicted_texts]
                        self.wer_metric[i].append(ids, predicted_words, target_words)
                        self.cer_metric[i].append(ids, predicted_words, target_words)
        return loss

    def fit_batch(self, batch):
        should_step = self.step % self.grad_accumulation_factor == 0

        # Managing automatic mixed precision
        if self.auto_mix_prec:
            self.wav2vec_optimizer.zero_grad()
            self.model_optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                with self.no_sync():
                    outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            with self.no_sync(not should_step):
                self.scaler.scale(
                    loss / self.grad_accumulation_factor
                ).backward()
            if should_step:
                if not self.hparams.freeze_wav2vec:
                    self.scaler.unscale_(self.wav2vec_optimizer)
                self.scaler.unscale_(self.model_optimizer)
                if self.check_gradients(loss):
                    self.scaler.step(self.wav2vec_optimizer)
                    self.scaler.step(self.model_optimizer)
                self.scaler.update()
                self.optimizer_step += 1
        else:
            with self.no_sync():
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
            loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            (loss / self.grad_accumulation_factor).backward()
            if should_step:
                if self.check_gradients(loss):
                    self.wav2vec_optimizer.step()
                    self.model_optimizer.step()
                self.wav2vec_optimizer.zero_grad()
                self.model_optimizer.zero_grad()
                self.optimizer_step += 1

        return loss.detach().cpu()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch. In this case,
        it initializes the wer and cer metric watchers. If the decoding
        method is whole-lattice-rescoring then a list of wer/cer metrics
        will be initialized (for each lm scale). Otherwise, a single class
        will be initialized for wer and cer, respectively.
        """
        if stage != sb.Stage.TRAIN:
            if stage == sb.Stage.VALID or self.hparams.decoding_method == "1best":
                self.cer_metric = self.hparams.cer_computer()
                self.wer_metric = self.hparams.error_rate_computer()
            else:  # stage is TEST and dec-method is whole-lattice or nbest rescoring
                self.cer_metric = []
                self.wer_metric = []
                for _ in range(len(self.hparams.lm_scale_list)):
                    self.cer_metric.append(self.hparams.cer_computer())
                    self.wer_metric.append(self.hparams.error_rate_computer())            

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch. During testing, its primary goal
        is to summarize the WER/CER stats and save them in a txt file.
        If the decoding method is whole-lattice-rescoring then we will
        print the WER/CER score of the best lm_scale. In addition, we will
        save the WER/CER scores of all lm_scales in separate txt files.
        If the decoding method is 1best then we will print the WER/CER score
        and save the results in a txt file.
        """
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        elif stage == sb.Stage.VALID or self.hparams.decoding_method == "1best":
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")
        else:
            best_wer = 100
            best_lm_scale = -1
            best_cer = 100
            for i, lm_scale in enumerate(self.hparams.lm_scale_list):
                if self.wer_metric[i].summarize("error_rate") < best_wer:
                    best_wer = self.wer_metric[i].summarize("error_rate")
                    best_lm_scale = lm_scale
                    best_cer = self.cer_metric[i].summarize("error_rate")
            stage_stats[f"CER-lm_scale_{best_lm_scale:.1f}"] = best_cer
            stage_stats[f"WER-lm_scale_{best_lm_scale:.1f}"] = best_wer

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr_model, new_lr_model = self.hparams.lr_annealing_model(
                stage_stats["loss"]
            )
            old_lr_wav2vec, new_lr_wav2vec = self.hparams.lr_annealing_wav2vec(
                stage_stats["loss"]
            )
            sb.nnet.schedulers.update_learning_rate(
                self.model_optimizer, new_lr_model
            )
            sb.nnet.schedulers.update_learning_rate(
                self.wav2vec_optimizer, new_lr_wav2vec
            )
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr_model": old_lr_model,
                    "lr_wav2vec": old_lr_wav2vec,
                },
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]}, min_keys=["WER"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if if_main_process():
                if self.hparams.decoding_method == "1best":
                    with open(self.hparams.wer_file, "w") as w:
                        self.wer_metric.write_stats(w)
                else:
                    metrics_dir = asr_brain.hparams.metrics_dir
                    os.makedirs(metrics_dir, exist_ok=True)
                    for i, lm_scale in enumerate(self.hparams.lm_scale_list):
                        with open(
                            os.path.join(
                                metrics_dir, f"wer_lm_scale_{lm_scale:.1f}.txt"
                            ),
                            "w",
                        ) as w:
                            self.wer_metric[i].write_stats(w)

    def init_optimizers(self):
        "Initializes the wav2vec2 optimizer and model optimizer"
        # Handling SpeechBrain vs HuggingFance pretrained models
        if hasattr(self.modules, "extractor"):  # SpeechBrain pretrained model
            self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(
                self.modules.encoder_wrapper.parameters()
            )

        else:  # HuggingFace pretrained model
            self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(
                self.modules.wav2vec2.parameters()
            )

        self.model_optimizer = self.hparams.model_opt_class(
            self.hparams.model.parameters()
        )

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable(
                "wav2vec_opt", self.wav2vec_optimizer
            )
            self.checkpointer.add_recoverable("modelopt", self.model_optimizer)

    def zero_grad(self, set_to_none=False):
        self.wav2vec_optimizer.zero_grad(set_to_none)
        self.model_optimizer.zero_grad(set_to_none)


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    # test is separate
    test_datasets = {}
    for csv_file in hparams["test_csv"]:
        name = Path(csv_file).stem
        test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_file, replacements={"data_root": data_folder}
        )
        test_datasets[name] = test_datasets[name].filtered_sorted(
            sort_key="duration"
        )

    datasets = [train_data, valid_data] + [i for k, i in test_datasets.items()]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "char_list"
    )
    def text_pipeline(wrd):
        yield wrd
        char_list = list(wrd)
        yield char_list

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "wrd", "char_list", "duration"],
    )

    return train_data, valid_data, test_datasets

def get_lexicon(lang_dir, 
                csv_files, 
                extra_vocab_files, 
                add_word_boundary=True):
    '''
    Read csv_files to generate a $lang_dir/lexicon.txt for k2 training.
    This usually includes the csv files of the training set and the dev set in the output_folder.
    During training, we need to make sure that the lexicon.txt contains all (or the majority of) 
    the words in the training set and the dev set.

    Args:
    lang_dir: the directory to store the lexicon.txt
    csv_files: a list of csv file paths 
    extra_vocab_files: a list of extra vocab files, librispeech-vocab.txt is an example
    add_word_boundary: whether to add word boundary symbols <eow> at the end of each line to the lexicon for every word

    Note that in each csv_file, the first line is the header, and the remaining lines are in the following format:

    ID, duration, wav, spk_id, wrd (transcription)

    We only need the transcription in this function.

    Returns:
    None

    Writes out $lang_dir/lexicon.txt

    Note that the lexicon.txt is a text file with the following format:
    word1 phone1 phone2 phone3 ...
    word2 phone1 phone2 phone3 ...

    In this code, we simply use the characters in the word as the phones.
    You can use other phone sets, e.g., phonemes, BPEs, to train a better model.
    '''
    # Read train.csv, dev-clean.csv to generate a lexicon.txt for k2 training
    lexicon = dict()
    for file in csv_files:
        with open(file) as f:
            # Omit the first line
            f.readline()
            # Read the remaining lines
            for line in f:
                # Split the line 
                try:
                    _, _, _, _, _, trans = line.strip().split(",")
                except ValueError as e:
                    print(line.strip().split(","))
                    raise e
                # Split the transcription into words
                words = trans.split()
                for word in words:
                    if word not in lexicon:
                        if add_word_boundary:
                            lexicon[word] = list(word) + ["<eow>"]
                        else:
                            lexicon[word] = list(word)

    for file in extra_vocab_files:
        with open(file) as f:
            for line in f:
                # Split the line 
                word = line.strip().split()[0]
                # Split the transcription into words
                if word not in lexicon:
                    if add_word_boundary:
                        lexicon[word] = list(word) + ["<eow>"]
                    else:
                        lexicon[word] = list(word)
    # Write the lexicon to lang_dir/lexicon.txt
    os.makedirs(lang_dir, exist_ok=True)
    with open(os.path.join(lang_dir, "lexicon.txt"), "w") as f:
        fc = "<UNK> <unk>\n"
        for word in lexicon:
            fc += word + " " + " ".join(lexicon[word]) + "\n"
        f.write(fc)

def arpa_to_fst(
        arpa_dir: Path,
        output_dir: Path,
        words_txt: Path,
        disambig_symbol: str = "#0",
        convert_4gram: bool = True
    ):
    """ Use kaldilm to convert an ARPA LM to FST. For example, in librispeech
    you can find a 3-gram (pruned) and a 4-gram ARPA LM in the openslr
    website (https://www.openslr.org/11/). You can use this function to
    convert them to FSTs. The resulting FSTs can then be used to create a
    decoding graph (HLG) for k2 decoding.

    If `convert_4gram` is True, then we will convert the 4-gram ARPA LM to
    FST. Otherwise, we will only convert the 3-gram ARPA LM to FST.
    It is worth noting that if the fsts already exist in the output_dir,
    then we will not convert them again (so you may need to delete them
    by hand if you, at any point, change your ARPA model).

    Args:
        arpa_dir: Path to the directory containing the ARPA LM (we expect
            a file named 3-gram.pruned.1e-7.arpa to exist, and if
            `convert_4gram` is True, then "4-gram.arpa" should also exist).
        output_dir: Path to the directory where the FSTs will be saved.
        words_txt: Path to the words.txt file created by prepare_lang.
        disambig_symbol: The disambiguation symbol to use.
        convert_4gram: If True, then we will convert the 4-gram ARPA LM to
            FST. Otherwise, we will only convert the 3-gram ARPA LM to FST.
    
    Raises:
        ImportError: If kaldilm is not installed.
    """
    assert arpa_dir.is_dir()
    assert output_dir.is_dir()
    try:
        from kaldilm.arpa2fst import arpa2fst
    except ImportError:
        # This error will occur when there is fst LM in the provided lm_dir
        # and we are trying to create it by converting an ARPA LM to FST.
        # For this, we need to install kaldilm.
        raise ImportError(
            "Optional dependencies must be installed to use kaldilm.\n"
            "Install using `pip install kaldilm`."
        )
    def _arpa_to_fst_single(arpa_path: Path, out_fst_path: Path, max_order: int):
        """Convert a single ARPA LM to FST."""
        if out_fst_path.exists():
            return
        if not arpa_path.exists():
            raise FileNotFoundError(f"{arpa_path} not found while trying to create the {max_order} FST.")
        try:
            logger.info("Converting {} to FST".format(arpa_path))
            s = arpa2fst(
                input_arpa=str(arpa_path),
                disambig_symbol=disambig_symbol,
                read_symbol_table=str(words_txt),
                max_order=max_order,
            )
        except Exception as e:
            logger.info(f"Failed to create {max_order}-gram FST from input={arpa_path}, disambig_symbol={disambig_symbol}, read_symbol_table={words_txt}")
            raise e
        logger.info(f"Writing {out_fst_path}")
        with open(out_fst_path, "w") as f:
            f.write(s)
    # 3-gram arpa to fst conversion...
    arpa_path = arpa_dir / "3-gram.pruned.1e-7.arpa"
    fst_path = output_dir / "G_3_gram.fst.txt"
    _arpa_to_fst_single(arpa_path, fst_path, max_order=3)
    # Optionnal 4-gram arpa to fst conversion
    if convert_4gram:
        # arpa_path = arpa_dir / "4-gram.arpa"
        arpa_path = arpa_dir / "4-gram.arpa"
        fst_path = output_dir / "G_4_gram.fst.txt"
        _arpa_to_fst_single(arpa_path, fst_path, max_order=4)

def create_P_fst(
        lexicon: Lexicon,
        csv_path: str,
        output_dir: str,
        tokens_txt: str,
        disambig_symbol: str = "#0",
    ):
    """Create the P.fst.txt for LF-MMI. The reason we don't use `arpa_to_fst`
    is because this is a token-level LM (e.g. phone/word-piece/character LM).
    TODO: This could still be merged in `arpa_to_fst` by adding an extra
          `read_symbol_table` argument.

    Args:
        lexicon: The lexicon object used to get the tokenized version of 
            each transcript
        csv_path: The path to the csv file.
        output_dir: The directory where the P.arpa will be saved.
        tokens_txt: The path to the tokens.txt file created by prepare_lang.
        disambig_symbol: The disambiguation symbol to use.        
    """
    arpa_path = Path(output_dir) / "P.arpa"
    fst_path = Path(output_dir) / "P.fst.txt"
    if fst_path.exists():
        return
    if not arpa_path.exists():
        logger.info(f"Creating {arpa_path}")
        with open(csv_path) as f:
            texts = [line.strip().split(",")[-1] for line in f.readlines()[1:]]
        tokenized_transcripts = list(lexicon.generate_transcript_chars(texts))
        with open(Path(tokens_txt).parent / "tokenized_transcripts.txt", "w") as f:
            f.write("\n".join(tokenized_transcripts))
        # Generate kneser-ney language model as arpa format. By default,
        # it will read the corpus from standard input, and output to 
        # standard output. Adapted from icefall's librispeech recipe.
        make_kn_lm(
            text=tokenized_transcripts,
            lm=arpa_path,
            ngram_order=2,
        )
    logger.info(f"Creating {fst_path}")
    try:
        from kaldilm.arpa2fst import arpa2fst
        s = arpa2fst(
            input_arpa=str(arpa_path),
            disambig_symbol=disambig_symbol,
            read_symbol_table=str(tokens_txt),
            max_order=2,
            bos_symbol="<s>",
            eos_symbol="</s>",
        )
    except Exception as e:
        logger.info(f"Failed to create 2-gram FST from input={arpa_path}, disambig_symbol={disambig_symbol}, read_symbol_table={words_txt}")
        raise e
    logger.info(f"Writing {fst_path}")
    with open(fst_path, "w") as f:
        f.write(s)

if __name__ == "__main__":

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset prep (parsing Librispeech)
    from librispeech_prepare import prepare_librispeech  # noqa

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_librispeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "tr_splits": hparams["train_splits"],
            "dev_splits": hparams["dev_splits"],
            "te_splits": hparams["test_splits"],
            "save_folder": hparams["output_folder"],
            "merge_lst": hparams["train_splits"],
            "merge_name": "train.csv",
            "skip_prep": hparams["skip_prep"],
        },
    )

    # here we create the datasets objects as well as tokenization and encoding
    train_data, valid_data, test_datasets = dataio_prepare(hparams)

    # Create the lexicon.txt for k2 training
    run_on_main(
        get_lexicon,
        kwargs={
            "lang_dir": hparams["lang_dir"],
            "csv_files": [hparams["output_folder"] + "/train.csv"],
            "extra_vocab_files": [hparams["vocab_file"]],
            "add_word_boundary": hparams["add_word_boundary"],
        },
    )

    # Create the lang directory for k2 training
    run_on_main(
        prepare_lang,
        kwargs={
            "lang_dir": hparams["lang_dir"],
            "sil_prob": hparams["sil_prob"],
        },
    )


    lexicon = Lexicon(hparams["lang_dir"])

    # Loading the labels for the LM decoding and the CTC decoder
    if hasattr(hparams, "use_language_modelling"):
        raise NotImplementedError("use_language_modelling is not implemented yet")
    else:
        hparams["use_language_modelling"] = False

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    need_G = False
    rescoring_lm_path = None
    if getattr(asr_brain.hparams, "use_HLG", False) in [True, "True"]:
        G_path = Path(asr_brain.hparams.lm_dir) / "G_3_gram.fst.txt"
        logger.info(f"Will load LM from {G_path}")
        need_G = True
    else:
        G_path = None

    need_4gram = (asr_brain.hparams.decoding_method == "whole-lattice-rescoring")
    # NOTE: This means that even if the 3gram G is not needed, but we still plan to rescore,
    #       then G_3_gram.fst.txt will still be created (i.e. if HLG is False but the decoding
    #       method is whole-lattice-rescoring, then G_3_gram.fst.txt will still be created).
    if need_G or need_4gram:
        # Create the G_3_gram.fst.txt for k2 decoding and G_4_gram.fst.txt for k2 rescoring
        run_on_main(
            arpa_to_fst,
            kwargs={
                "arpa_dir": Path(asr_brain.hparams.lm_dir),
                "output_dir": Path(asr_brain.hparams.lm_dir),
                "words_txt": Path(asr_brain.hparams.lang_dir) / "words.txt",
                "convert_4gram": need_4gram,
            },
        )
        rescoring_lm_path = Path(asr_brain.hparams.lm_dir) / "G_4_gram.fst.txt"
    if need_G:
        assert G_path.is_file(), f"{G_path} does not exist"

    run_on_main(
        create_P_fst,
        kwargs={
            "lexicon": lexicon,
            "csv_path": Path(hparams["output_folder"]) / "train.csv",
            "output_dir": asr_brain.hparams.lm_dir,
            "tokens_txt": Path(asr_brain.hparams.lang_dir) / "tokens.txt",
            "disambig_symbol": "#0",
        },
    )
    
    graph_compiler = MmiTrainingGraphCompiler(
        lexicon=lexicon,
        device=asr_brain.device,
        G_path=G_path,
        P_path=Path(asr_brain.hparams.lm_dir) / "P.fst.txt",
        rescoring_lm_path=rescoring_lm_path,
        decoding_method=asr_brain.hparams.decoding_method,
    )

    # Add attributes to asr_brain
    setattr(asr_brain, "graph_compiler", graph_compiler)

    # We load the pretrained wav2vec2 model
    if "pretrainer" in hparams.keys():
        run_on_main(hparams["pretrainer"].collect_files)
        hparams["pretrainer"].load_collected(asr_brain.device)

    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Testing
    for k in test_datasets.keys():  # keys are test_clean, test_other etc
        asr_brain.hparams.wer_file = os.path.join(
            hparams["output_folder"], "wer_{}.txt".format(k)
        )
        if asr_brain.hparams.decoding_method != "1best":
            # define the metrics directory for whole-lattice rescoring
            asr_brain.hparams.metrics_dir = os.path.join(
                hparams["output_folder"], f"test_metrics_{k}"
            )
        asr_brain.evaluate(
            test_datasets[k], test_loader_kwargs=hparams["test_dataloader_opts"]
        )
