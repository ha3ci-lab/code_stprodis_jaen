# code_stprodis_jaen

This is the implementation of the paper [Low-Resource Japanese-English Speech-to-Text Translation Leveraging Speech-Text Unified-model Representation Learning](https://sigul-2023.ilc.cnr.it/wp-content/uploads/2023/08/29_Paper.pdf). Our code is based on the implementation of [SpeechT5 model](https://github.com/microsoft/SpeechT5/tree/main/SpeechT5). Since pretraining SpeechT5 model requires the outputs of the first iteration HuBERT BASE model, we also provide the dockerfile for environment settings of pre-training HuBERT and training our model.
# Data preparation
In term of pre-training SpeechT5 model for Japanese, we combine datasets from [JSUT](https://sites.google.com/site/shinnosuketakamichi/publication/jsut), [KoKoro](https://github.com/kaiidams/Kokoro-Speech-Dataset), [CoVoST 2](https://github.com/facebookresearch/covost), and our [own dataset](https://github.com/ha3ci-lab/data_stprodis_jaen). To process the data, please visit the [data preparation](https://github.com/microsoft/SpeechT5/tree/main/SpeechT5#speech-data-and-s2t-data) for pre-training SpeechT5.
In term of fine-tuning the model for Ja-En text translation, we use [JESC](https://nlp.stanford.edu/projects/jesc/index_ja.html).
Finally, we evaluate our model on Speech-to-text translation from Ja-En using [CoVoST 2](https://github.com/facebookresearch/covost) and our [own dataset](https://github.com/ha3ci-lab/data_stprodis_jaen) recorder by speaker F01.

We use [sentencepiece](https://github.com/google/sentencepiece) to create two different 32K universal BPE vocabulary files for Japanese and English.
# Pre-training SpeechT5 for Japanese
```
DATA_ROOT=
SAVE_DIR=
LABEL_DIR=
TRAIN_SET="speech_train|text_train"
VALID_SET="speech_valid|text_valid"
fairseq-train ${DATA_ROOT} \
  --save-dir ${SAVE_DIR} \
  --tensorboard-logdir ${SAVE_DIR} \
  --train-subset ${TRAIN_SET} \
  --valid-subset ${VALID_SET} \
  --hubert-label-dir ${LABEL_DIR} \
  --distributed-world-size 1 \
  --distributed-port 0 \
  --ddp-backend legacy_ddp \
  --load-checkpoint-on-all-dp-ranks \
  --user-dir speecht5 \
  --log-format json \
  --seed 1337 \
  --fp16 \
  --task speecht5 \
  --t5-task pretrain \
  --label-rates 50 \
  --sample-rate 16000 \
  --random-crop \
  --num-workers 0 \
  --max-tokens 1400000 \
  --max-speech-sample-size 250000 \
  --update-freq 8 \
  --batch-ratio "[1,0.0086]" \
  --criterion speecht5 \
  --optimizer adam \
  --reset-optimizer \
  --adam-betas "(0.9, 0.98)" \
  --adam-eps 1e-06 \
  --weight-decay 0.01 \
  --power 1 \
  --clip-norm 5.0 \
  --lr 0.0002 \
  --lr-scheduler polynomial_decay \
  --max-update 80000 \
  --warmup-updates 6400 \
  --total-num-update 80000 \
  --save-interval-updates 500 \
  --keep-interval-updates 2 \
  --skip-invalid-size-inputs-valid-test \
  --required-batch-size-multiple 1 \
  --arch t5_transformer_base \
  --find-unused-parameters \
  --bert-init \
  --relative-position-embedding \
  --use-codebook \
  --codebook-prob 0.1 \
  --loss-weights="[10,0.1]" \
  --max-text-positions 600 \
```
# Ja-En Text Translation
```
DATA_ROOT=
SAVE_DIR=
USER_DIR=
PT_CHECKPOINT_PATH=
fairseq-train ${DATA_ROOT} \
  --save-dir ${SAVE_DIR} \
  --tensorboard-logdir ${SAVE_DIR} \
  --distributed-world-size 1 \
  --distributed-port 0 \
  --ddp-backend no_c10d \
  --user-dir ${USER_DIR} \
  --log-format simple \
  --seed 1 \
  --num-workers 0 \
  --max-tokens 20000 \
  --update-freq 32 \
  --criterion speecht5 \
  --label-smoothing 0.1 \
  --optimizer adam \
  --adam-betas "(0.9, 0.98)" \
  --weight-decay 0.0 \
  --clip-norm 1.0 \
  --lr 0.001 \
  --lr-scheduler inverse_sqrt \
  --warmup-updates 4000 \
  --dropout 0.3 \
  --max-update 40000 \
  --no-epoch-checkpoints \
  --eval-bleu \
  --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
  --eval-bleu-detok moses \
  --eval-bleu-remove-bpe "@@"\
  --eval-bleu-print-samples \
  --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
  --warmup-init-lr 1e-07 \
  --stop-min-lr 1e-09 \
  --log-interval 5 \
  --validate-interval 1000 \
  --save-interval-updates 1000 \
  --keep-interval-updates 8 \
  --arch t5_transformer_base \
  --task speecht5 \
  --t5-task text_translation \
  --max-text-positions 1024 \
  --freeze-encoder-updates 40000 \
  --fp16 \
  --finetune-from-model ${PT_CHECKPOINT_PATH} \
```
# Ja-En Speech-to-Text Translation
```
DATA_ROOT=
SAVE_DIR=
TRAIN_SET=
VALID_SET=
LABEL_DIR=
BPE_TOKENIZER=
USER_DIR=
PT_CHECKPOINT_PATH=
MAX_UPDATE = # 20000 for CoVoST 2 and 8000 for own data
WARMUP_UPDATES = # 2500 for CoVoST 2 and 800 for own data
fairseq-train ${DATA_ROOT} \
  --save-dir ${SAVE_DIR} \
  --tensorboard-logdir ${SAVE_DIR} \
  --train-subset ${TRAIN_SET} \
  --valid-subset ${VALID_SET} \
  --hubert-label-dir ${LABEL_DIR} \
  --distributed-world-size 1 \
  --distributed-port 0 \
  --ddp-backend legacy_ddp \
  --user-dir ${USER_DIR} \
  --log-format json \
  --seed 1 \
  --fp16 \
  \
  --task speecht5 \
  --t5-task s2t \
  --sample-rate 16000 \
  --num-workers 0 \
  --max-tokens 1000000 \
  --update-freq 4 \
  --bpe-tokenizer ${BPE_TOKENIZER} \
  --max-tokens-valid 1000000 \
  --skip-invalid-size-inputs-valid-test \
  \
  --criterion speecht5 \
  --label-smoothing 0.1 \
  --report-accuracy \
  --sentence-avg \
  \
  --optimizer adam \
  --adam-betas "(0.9, 0.98)" \
  --weight-decay 0.0 \
  --clip-norm 10.0 \
  --lr 0.0002 \
  --lr-scheduler inverse_sqrt \
  --warmup-updates ${WARMUP_UPDATES} \
  --feature-grad-mult 1.0 \
  \
  --max-update ${MAX_UPDATE} \
  --max-text-positions 600 \
  --min-speech-sample-size 1056 \
  --max-speech-sample-size 480256 \
  --max-speech-positions 1876 \
  --required-batch-size-multiple 1 \
  --skip-invalid-size-inputs-valid-test \
  \
  --arch t5_transformer_base_asr \
  --share-input-output-embed \
  --find-unused-parameters \
  --bert-init \
  --relative-position-embedding \
  --freeze-encoder-updates 0 \
  --mask-prob 0.5 \
  --mask-channel-prob 0.5 \
  \
  --keep-last-epochs 10 \
  --eval-bleu \
  --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
  --eval-bleu-detok moses \
  --eval-bleu-remove-bpe "@@"\
  --eval-bleu-print-samples \
```
# Citation
If you find our work is useful in your research, please cite the following paper:
```
@inproceedings{tranlow,
  title={Low-Resource Japanese-English Speech-to-Text Translation Leveraging Speech-Text Unified-model Representation Learning},
  author={Tran, Tu Dinh and Sakti, Sakriani},
  booktitle={Proceedings of SIGUL 2023 Workshop at INTERSPEECH 2023},
  year={2023}
}
```
```
@article{Ao2021SpeechT5,
  title   = {SpeechT5: Unified-Modal Encoder-Decoder Pre-training for Spoken Language Processing},
  author  = {Junyi Ao and Rui Wang and Long Zhou and Chengyi Wang and Shuo Ren and Yu Wu and Shujie Liu and Tom Ko and Qing Li and Yu Zhang and Zhihua Wei and Yao Qian and Jinyu Li and Furu Wei},
  eprint={2110.07205},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  year={2021}
}
```
