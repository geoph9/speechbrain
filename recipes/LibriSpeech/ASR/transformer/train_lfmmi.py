#!/usr/bin/env python3
"""Recipe for training a Transformer ASR system with librispeech.
The system employs an encoder, a decoder, and an attention mechanism
between them. Decoding is performed with (CTC/Att joint) beamsearch coupled with a neural
language model.

The loss function used is LF-MMI.

To run this recipe, do the following:
> python train_lfmmi.py hparams/transformer_lfmmi.yaml

With the default hyperparameters, the system employs a convolutional frontend and a transformer.
The decoder is based on a Transformer decoder. Beamsearch coupled with a Transformer
language model is used  on the top of decoder probabilities.


Authors
 * Georgios Karakasidis 2023
 * Jianyuan Zhong 2020
 * Mirco Ravanelli 2020
 * Peter Plantinga 2020
 * Samuele Cornell 2020, 2021, 2022
 * Titouan Parcollet 2021, 2022
"""

import os
import sys
from typing import List, Union
import torch
import logging
from pathlib import Path

import speechbrain as sb
from tqdm import tqdm
from hyperpyyaml import load_hyperpyyaml
from speechbrain.dataio.dataloader import LoopedLoader
from torch.utils.data import DataLoader
from speechbrain.lobes.models.transformer.TransformerASR import EncoderWrapper
from speechbrain.utils.distributed import run_on_main, if_main_process
from speechbrain.k2_integration.prepare_lang import prepare_lang
from speechbrain.k2_integration.prepare_lang_bpe import prepare_lang as prepare_lang_bpe
from speechbrain.k2_integration.lexicon import Lexicon
from speechbrain.k2_integration.graph_compiler import MmiTrainingGraphCompiler

from utils import get_lexicon, arpa_to_fst, create_P_fst, get_bpe_tokenizer


logger = logging.getLogger(__name__)


# Define training procedure
class ASR(sb.core.Brain):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = EncoderWrapper(self.modules.Transformer)

    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        # tokens_bos, _ = batch.tokens_bos  # TODO
        
        # TODO: Delete the following, this is only meant for debugging
        assert self.hparams.num_decoder_layers == 0

        # Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            if hasattr(self.modules, "env_corrupt"):
                raise NotImplementedError("env_corrupt not implemented yet")

        # compute features
        feats = self.hparams.compute_features(wavs)
        current_epoch = self.hparams.epoch_counter.current
        feats = self.modules.normalize(feats, wav_lens, epoch=current_epoch)

        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                feats = self.hparams.augmentation(feats)

        # forward modules
        src = self.modules.CNN(feats)
        enc_out = self.encoder(
            src, wav_lens, pad_idx=self.hparams.pad_index,
        )
        # del dec_output  # should be None anyway since num_decoder_layers is 0

        # output layer for log-probabilities
        logits = self.modules.lin(enc_out)
        log_probs = self.hparams.log_softmax(logits)

        # # Compute outputs
        # hyps = None
        # if stage == sb.Stage.TRAIN:
        #     hyps = None
        # elif stage == sb.Stage.VALID:
        #     hyps = None
        #     current_epoch = self.hparams.epoch_counter.current
        #     if current_epoch % self.hparams.valid_search_interval == 0:
        #         # for the sake of efficiency, we only perform beamsearch with limited capacity
        #         # and no LM to give user some idea of how the AM is doing
        #         hyps, _ = self.hparams.valid_search(enc_out.detach(), wav_lens)
        #     raise NotImplementedError("valid_search not implemented yet")
        # elif stage == sb.Stage.TEST:
        #     hyps, _ = self.hparams.test_search(enc_out.detach(), wav_lens)
        #     raise NotImplementedError("test_search not implemented yet")

        return log_probs, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        log_probs, wav_lens = predictions

        ids = batch.id
        # tokens_eos, tokens_eos_lens = batch.tokens_eos
        # tokens, tokens_lens = batch.tokens

        if hasattr(self.modules, "env_corrupt") and stage == sb.Stage.TRAIN:
            raise NotImplementedError("env_corrupt not implemented yet")
        
        texts = batch.wrd
        # # Sort batch to be descending by length of wav files, which is demanded by k2
        # if self.hparams.sorting == "ascending":
        #     log_probs = torch.flip(log_probs, (0,))
        #     wav_lens = torch.flip(wav_lens, (0,))
        #     texts = [batch.wrd[i] for i in reversed(range(len(batch.wrd)))]
        # elif self.hparams.sorting == "descending":
        #     texts = batch.wrd
        # else:
        #     raise NotImplementedError("Only ascending or descending sorting is implemented, but got {}".format(self.hparams.sorting))

        is_training = (stage == sb.Stage.TRAIN)
        current_epoch = self.hparams.epoch_counter.current
        if is_training or (
            stage == sb.Stage.VALID and current_epoch % self.hparams.validate_every == 0
        ):
            loss = self.hparams.mmi_cost(
                log_probs=log_probs, 
                input_lens=wav_lens, 
                graph_compiler=self.graph_compiler,
                texts=texts,
                is_training=is_training,
            )
        else:
            loss = torch.empty(0, device=self.device)

        if stage == sb.Stage.VALID:
            # Decode token terms to words
            if current_epoch % self.hparams.validate_every == 0:
                predicted_texts = self.graph_compiler.decode(
                    log_probs,
                    wav_lens,
                    ac_scale=self.hparams.ac_scale,
                    stage=stage,
                ) # list of strings
                predicted_words = [wrd.split(" ") for wrd in predicted_texts]
                target_words = [wrd.split(" ") for wrd in texts]
                self.wer_metric.append(ids, predicted_words, target_words)
                self.cer_metric.append(ids, predicted_words, target_words)
                # cleanup graph compiler to save memory
                # del self.graph_compiler.decoding_graph, self.graph_compiler.rescoring_graph
                # self.graph_compiler.rescoring_graph = None
                # self.graph_compiler.decoding_graph = None
        elif stage == sb.Stage.TEST:  # Language model decoding only used for test
            if self.hparams.use_language_modelling:
                raise NotImplementedError(
                    "Language modelling is not implemented for models trained with k2"
                )
            else:
                # If the decoding method is 1best then the metric stats will be
                # saved in a single file, otherwise, a new directory will be created
                # for each lm_scale used in whole lattice rescoring.
                decode_output: Union[dict, List[str]] = self.graph_compiler.decode(
                    log_probs,
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
            # # compute the accuracy of the one-step-forward prediction
            # self.acc_metric.append(log_probs, tokens_eos, tokens_eos_lens)
        return loss
    
    def evaluate(
        self,
        test_set,
        max_key=None,
        min_key=None,
        progressbar=None,
        test_loader_kwargs={},
    ):
        """Iterate test_set and evaluate brain performance. By default, loads
        the best-performing checkpoint (as recorded using the checkpointer).

        Arguments
        ---------
        test_set : Dataset, DataLoader
            If a DataLoader is given, it is iterated directly. Otherwise passed
            to ``self.make_dataloader()``.
        max_key : str
            Key to use for finding best checkpoint, passed to
            ``on_evaluate_start()``.
        min_key : str
            Key to use for finding best checkpoint, passed to
            ``on_evaluate_start()``.
        progressbar : bool
            Whether to display the progress in a progressbar.
        test_loader_kwargs : dict
            Kwargs passed to ``make_dataloader()`` if ``test_set`` is not a
            DataLoader. NOTE: ``loader_kwargs["ckpt_prefix"]`` gets
            automatically overwritten to ``None`` (so that the test DataLoader
            is not added to the checkpointer).

        Returns
        -------
        average test loss
        """
        if progressbar is None:
            progressbar = not self.noprogressbar

        if not (
            isinstance(test_set, DataLoader)
            or isinstance(test_set, LoopedLoader)
        ):
            test_loader_kwargs["ckpt_prefix"] = None
            test_set = self.make_dataloader(
                test_set, sb.Stage.TEST, **test_loader_kwargs
            )
        self.on_evaluate_start(max_key=max_key, min_key=min_key)
        self.on_stage_start(sb.Stage.TEST, epoch=None)
        self.modules.eval()
        with torch.no_grad():
            for batch in tqdm(
                test_set,
                dynamic_ncols=True,
                disable=not progressbar,
                colour=self.tqdm_barcolor["test"],
            ):
                self.step += 1
                _ = self.evaluate_batch(batch, stage=sb.Stage.TEST)

                # Profile only if desired (steps allow the profiler to know when all is warmed up)
                if self.profiler is not None:
                    if self.profiler.record_steps:
                        self.profiler.step()

                # Debug mode only runs a few batches
                if self.debug and self.step == self.debug_batches:
                    break

            self.on_stage_end(sb.Stage.TEST, None, None)
        self.step = 0
        return

    def _fit_valid(self, valid_set, epoch, enable):
        if epoch % self.hparams.validate_every != 0:
            return
        # Validation stage
        if valid_set is not None:
            self.on_stage_start(sb.Stage.VALID, epoch)
            self.modules.eval()
            avg_valid_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(
                    valid_set,
                    dynamic_ncols=True,
                    disable=not enable,
                    colour=self.tqdm_barcolor["valid"],
                ):
                    self.step += 1
                    loss = self.evaluate_batch(batch, stage=sb.Stage.VALID)
                    avg_valid_loss = self.update_average(loss, avg_valid_loss)

                    # Profile only if desired (steps allow the profiler to know when all is warmed up)
                    if self.profiler is not None:
                        if self.profiler.record_steps:
                            self.profiler.step()

                    # Debug mode only runs a few batches
                    if self.debug and self.step == self.debug_batches:
                        break

                self.step = 0
                self.on_stage_end(sb.Stage.VALID, avg_valid_loss, epoch)

    def fit_batch(self, batch):

        should_step = self.step % self.grad_accumulation_factor == 0
        # Managing automatic mixed precision
        if self.auto_mix_prec:
            with torch.autocast(torch.device(self.device).type):
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)

            # Losses are excluded from mixed precision to avoid instabilities
            loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            with self.no_sync(not should_step):
                self.scaler.scale(
                    loss / self.grad_accumulation_factor
                ).backward()
            if should_step:
                self.scaler.unscale_(self.optimizer)
                if self.check_gradients(loss):
                    self.scaler.step(self.optimizer)
                self.scaler.update()
                self.zero_grad()
                self.optimizer_step += 1
                self.hparams.noam_annealing(self.optimizer)
        else:
            if self.bfloat16_mix_prec:
                with torch.autocast(
                    device_type=torch.device(self.device).type,
                    dtype=torch.bfloat16,
                ):
                    outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                    loss = self.compute_objectives(
                        outputs, batch, sb.Stage.TRAIN
                    )
            else:
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            with self.no_sync(not should_step):
                (loss / self.grad_accumulation_factor).backward()
            if should_step:
                if self.check_gradients(loss):
                    self.optimizer.step()
                self.zero_grad()
                self.optimizer_step += 1
                self.hparams.noam_annealing(self.optimizer)

        self.on_fit_batch_end(batch, outputs, loss, should_step)
        return loss.detach().cpu()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            current_epoch = self.hparams.epoch_counter.current
            self.acc_metric = self.hparams.acc_computer()
            if (
                    stage == sb.Stage.TEST and self.hparams.decoding_method == "1best"
                ) or (
                    stage == sb.Stage.VALID and current_epoch % self.hparams.validate_every == 0
                ):
                self.cer_metric = self.hparams.cer_computer()
                self.wer_metric = self.hparams.error_rate_computer()
            else:  # stage is TEST and dec-method is whole-lattice or nbest rescoring
                self.cer_metric = []
                self.wer_metric = []
                for _ in range(len(self.hparams.lm_scale_list)):
                    self.cer_metric.append(self.hparams.cer_computer())
                    self.wer_metric.append(self.hparams.error_rate_computer())   

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        current_epoch = self.hparams.epoch_counter.current
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        elif (
                stage == sb.Stage.TEST and self.hparams.decoding_method == "1best"
            ) or (
                stage == sb.Stage.VALID and current_epoch % self.hparams.validate_every == 0
            ):
            # stage_stats["ACC"] = self.acc_metric.summarize()  # TODO: Implement in compute_objectives
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")
        else:
            # stage_stats["ACC"] = self.acc_metric.summarize()
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

        # log stats and save checkpoint at end-of-epoch
        if stage == sb.Stage.VALID:
            del self.graph_compiler.decoding_graph, self.graph_compiler.rescoring_graph
            self.graph_compiler.decoding_graph = None
            self.graph_compiler.rescoring_graph = None
            if sb.utils.distributed.if_main_process():
                lr = self.hparams.noam_annealing.current_lr
                steps = self.optimizer_step
                optimizer = self.optimizer.__class__.__name__

                epoch_stats = {
                    "epoch": epoch,
                    "lr": lr,
                    "steps": steps,
                    "optimizer": optimizer,
                }
                self.hparams.train_logger.log_stats(
                    stats_meta=epoch_stats,
                    train_stats=self.train_stats,
                    valid_stats=stage_stats,
                )
                # self.checkpointer.save_and_keep_only(
                #     meta={"ACC": stage_stats["ACC"], "epoch": epoch},
                #     max_keys=["ACC"],
                #     num_to_keep=10,
                # )  # TODO: Check if acc_computer needs to be kept in favour of wer
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

            # save the averaged checkpoint at the end of the evaluation stage
            # delete the rest of the intermediate checkpoints
            # ACC is set to 1.1 so checkpointer only keeps the averaged checkpoint
            # self.checkpointer.save_and_keep_only(
            #     meta={"ACC": 1.1, "epoch": epoch},
            #     max_keys=["ACC"],
            #     num_to_keep=1,
            # )  # TODO: Check if this should be kept

    def on_evaluate_start(self, max_key=None, min_key=None):
        """perform checkpoint averge if needed"""
        super().on_evaluate_start()

        ckpts = self.checkpointer.find_checkpoints(
            max_key=max_key, min_key=min_key
        )
        ckpt = sb.utils.checkpoints.average_checkpoints(
            ckpts, recoverable_name="model", device=self.device
        )

        self.hparams.model.load_state_dict(ckpt, strict=True)
        self.hparams.model.eval()
        print("Loaded the average")

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        with torch.no_grad():
            predictions = self.compute_forward(batch, stage=stage)
            loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()


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
    valtest_datasets = [valid_data] + [i for k, i in test_datasets.items()]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(valtest_datasets, audio_pipeline)

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline_train(wav):
        # Speed Perturb is done here so it is multi-threaded with the
        # workers of the dataloader (faster).
        if "speed_perturb" in hparams:
            sig = sb.dataio.dataio.read_audio(wav)

            sig = hparams["speed_perturb"](sig.unsqueeze(0)).squeeze(0)
        else:
            sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item([train_data], audio_pipeline_train)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "char_list"
    )
    def text_pipeline(wrd):
        yield wrd
        char_list = list(wrd)
        yield char_list

    # # 3. Define text pipeline:
    # @sb.utils.data_pipeline.takes("wrd")
    # @sb.utils.data_pipeline.provides(
    #     "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    # )
    # def text_pipeline(wrd):
    #     yield wrd
    #     tokens_list = tokenizer.encode_as_ids(wrd)
    #     yield tokens_list
    #     tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
    #     yield tokens_bos
    #     tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
    #     yield tokens_eos
    #     tokens = torch.LongTensor(tokens_list)
    #     yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # # 4. Set output:
    # sb.dataio.dataset.set_output_keys(
    #     datasets, ["id", "sig", "wrd", "tokens_bos", "tokens_eos", "tokens"],
    # )

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "wrd", "char_list", "duration"],
    )

    return train_data, valid_data, test_datasets

    # # 5. If Dynamic Batching is used, we instantiate the needed samplers.
    # train_batch_sampler = None
    # valid_batch_sampler = None
    # if hparams["dynamic_batching"]:
    #     from speechbrain.dataio.sampler import DynamicBatchSampler  # noqa

    #     dynamic_hparams = hparams["dynamic_batch_sampler"]
    #     num_buckets = dynamic_hparams["num_buckets"]

    #     train_batch_sampler = DynamicBatchSampler(
    #         train_data,
    #         dynamic_hparams["max_batch_len"],
    #         num_buckets=num_buckets,
    #         length_func=lambda x: x["duration"],
    #         shuffle=dynamic_hparams["shuffle_ex"],
    #         batch_ordering=dynamic_hparams["batch_ordering"],
    #         max_batch_ex=dynamic_hparams["max_batch_ex"],
    #     )

    #     valid_batch_sampler = DynamicBatchSampler(
    #         valid_data,
    #         dynamic_hparams["max_batch_len_val"],
    #         num_buckets=num_buckets,
    #         length_func=lambda x: x["duration"],
    #         shuffle=dynamic_hparams["shuffle_ex"],
    #         batch_ordering=dynamic_hparams["batch_ordering"],
    #     )

    # return (
    #     train_data,
    #     valid_data,
    #     test_datasets,
    #     tokenizer,
    #     train_batch_sampler,
    #     valid_batch_sampler,
    # )


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If --distributed_launch then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # 1.  # Dataset prep (parsing Librispeech)
    from librispeech_prepare import prepare_librispeech  # noqa

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

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

    tokenizer = None
    if hparams["token_type"] == "bpe":
        tokenizer = get_bpe_tokenizer(hparams)

    # # here we create the datasets objects as well as tokenization and encoding
    # (
    #     train_data,
    #     valid_data,
    #     test_datasets,
    #     tokenizer,
    #     train_bsampler,
    #     valid_bsampler,
    # ) = dataio_prepare(hparams)

    # here we create the datasets objects as well as tokenization and encoding
    train_data, valid_data, test_datasets = dataio_prepare(hparams)

    # # We download the pretrained LM from HuggingFace (or elsewhere depending on
    # # the path given in the YAML file). The tokenizer is loaded at the same time.
    # run_on_main(hparams["pretrainer"].collect_files)
    # hparams["pretrainer"].load_collected(device=run_opts["device"])

    # Create the lexicon.txt for k2 training
    run_on_main(
        get_lexicon,
        kwargs={
            "lang_dir": hparams["lang_dir"],
            "csv_files": [hparams["output_folder"] + "/train.csv"],
            "extra_vocab_files": [hparams["vocab_file"]],
            "add_word_boundary": hparams["add_word_boundary"],
            "tokenizer": tokenizer,
            "unit_type": hparams["token_type"],
        },
    )

    if hparams["token_type"] == "char":
        # Create the lang directory for k2 training
        run_on_main(
            prepare_lang,
            kwargs={
                "lang_dir": hparams["lang_dir"],
                "sil_prob": hparams["sil_prob"],
            },
        )
    else:
        # Create the lang directory for k2 training with BPE
        run_on_main(
            prepare_lang_bpe,
            kwargs={
                "lang_dir": hparams["lang_dir"],
                "tokenizer": tokenizer,
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
        opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    need_G = False
    rescoring_lm_path = None
    fsts_suffix = hparams.get("fsts_suffix", "")
    if getattr(asr_brain.hparams, "use_HLG", False) in [True, "True"]:
        G_path = Path(asr_brain.hparams.lm_dir) / f"G_3_gram{fsts_suffix}.fst.txt"
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
                "suffix": fsts_suffix
            },
        )
        rescoring_lm_path = Path(asr_brain.hparams.lm_dir) / f"G_4_gram{fsts_suffix}.fst.txt"
    if need_G:
        assert G_path.is_file(), f"{G_path} does not exist"

    P_model_name = hparams.get("P_model_name", "P")
    run_on_main(
        create_P_fst,
        kwargs={
            "lexicon": lexicon,
            "csv_path": Path(hparams["output_folder"]) / "train.csv",
            "output_dir": asr_brain.hparams.lm_dir,
            "tokens_txt": Path(asr_brain.hparams.lang_dir) / "tokens.txt",
            "disambig_symbol": "#0",
            "max_order": hparams.get("unit_level_max_order", 2),
            "model_name": P_model_name,
        },
    )
    
    graph_compiler = MmiTrainingGraphCompiler(
        lexicon=lexicon,
        device=asr_brain.device,
        G_path=G_path,
        P_path=Path(asr_brain.hparams.lm_dir) / f"{P_model_name}.fst.txt",
        rescoring_lm_path=rescoring_lm_path,
        decoding_method=asr_brain.hparams.decoding_method,
        subsampling_factor=asr_brain.hparams.subsample_factor,
    )

    # Add attributes to asr_brain
    setattr(asr_brain, "graph_compiler", graph_compiler)

    # We load the pretrained wav2vec2 model
    if "pretrainer" in hparams.keys():
        run_on_main(hparams["pretrainer"].collect_files)
        hparams["pretrainer"].load_collected(asr_brain.device)

    with torch.autograd.detect_anomaly():
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


    # # adding objects to trainer:
    # asr_brain.tokenizer = hparams["tokenizer"]
    # train_dataloader_opts = hparams["train_dataloader_opts"]
    # valid_dataloader_opts = hparams["valid_dataloader_opts"]

    # if train_bsampler is not None:
    #     collate_fn = None
    #     if "collate_fn" in train_dataloader_opts:
    #         collate_fn = train_dataloader_opts["collate_fn"]

    #     train_dataloader_opts = {
    #         "batch_sampler": train_bsampler,
    #         "num_workers": hparams["num_workers"],
    #     }

    #     if collate_fn is not None:
    #         train_dataloader_opts["collate_fn"] = collate_fn

    # if valid_bsampler is not None:
    #     collate_fn = None
    #     if "collate_fn" in valid_dataloader_opts:
    #         collate_fn = valid_dataloader_opts["collate_fn"]

    #     valid_dataloader_opts = {"batch_sampler": valid_bsampler}

    #     if collate_fn is not None:
    #         valid_dataloader_opts["collate_fn"] = collate_fn

    # # Training
    # asr_brain.fit(
    #     asr_brain.hparams.epoch_counter,
    #     train_data,
    #     valid_data,
    #     train_loader_kwargs=train_dataloader_opts,
    #     valid_loader_kwargs=valid_dataloader_opts,
    # )

    # # Testing
    # if not os.path.exists(hparams["output_wer_folder"]):
    #     os.makedirs(hparams["output_wer_folder"])

    # for k in test_datasets.keys():  # keys are test_clean, test_other etc
    #     asr_brain.hparams.test_wer_file = os.path.join(
    #         hparams["output_wer_folder"], f"wer_{k}.txt"
    #     )
    #     asr_brain.evaluate(
    #         test_datasets[k],
    #         max_key="ACC",
    #         test_loader_kwargs=hparams["test_dataloader_opts"],
    #     )
