# code_stprodis_jaen

This is the implementation of the paper [Low-Resource Japanese-English Speech-to-Text Translation Leveraging Speech-Text Unified-model Representation Learning](https://sigul-2023.ilc.cnr.it/wp-content/uploads/2023/08/29_Paper.pdf). Our code is based on the implementation of [SpeechT5 model](https://github.com/microsoft/SpeechT5/tree/main/SpeechT5). Since pretraining SpeechT5 model requires the outputs of the first iteration HuBERT BASE model, we also provide the dockerfile for environment settings of pre-training HuBERT and training our model.
# Data preparation
In term of pre-training SpeechT5 model for Japanese, we combined datasets from [JSUT](https://sites.google.com/site/shinnosuketakamichi/publication/jsut), [KoKoro](https://github.com/kaiidams/Kokoro-Speech-Dataset), [CoVoST 2](https://github.com/facebookresearch/covost), and our [own dataset](https://github.com/ha3ci-lab/data_stprodis_jaen). To process the data, please visit the [data preparation](https://github.com/microsoft/SpeechT5/tree/main/SpeechT5#speech-data-and-s2t-data) for pre-training SpeechT5.
In term of fine-tuning the model for Ja-En text translation, we used [JESC](https://nlp.stanford.edu/projects/jesc/index_ja.html).
Finally, we evaluate our model on Speech-to-text translation from Ja-En using [CoVoST 2](https://github.com/facebookresearch/covost) and our [own dataset](https://github.com/ha3ci-lab/data_stprodis_jaen) recorder by speaker F01.

# Citation
If you find our work is useful in your research, please cite the following paper:
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
