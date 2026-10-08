# Temporal-AVSR

## Overview

Temporal-AVSR combines pretrained speech encoders with the LLaMA language model for
visual speech recognition (VSR) and audio-visual speech recognition (AVSR).

The main components are:

1. **Speech encoders:** AV-HuBERT Large encodes video, and Whisper medium.en encodes audio.
2. **Temporal token processing:** Temporal attention, Q-Former compression, and projection map features into the language model's embedding space.
3. **Language model:** Llama-3.2-3B is loaded with 4-bit quantization and fine-tuned using LoRA.
   The training objective combines the language modeling loss with an auxiliary CTC loss.

## Installation

The existing environment uses **Python 3.9**, **PyTorch 2.1.2**, and **Transformers 4.47.1**.
The commands below use CUDA 11.8 packages.

```bash
git clone https://github.com/LyongW/Temporal-AVSR.git
cd Temporal-AVSR

conda create -n temporal_avsr python=3.9 -y
conda activate temporal_avsr
python -m pip install 'pip<24.1'
python -m pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation -e ./fairseq
```

Run the installation commands from the project root. The repository includes the Fairseq
source code, and installation compiles native extensions. Choose the appropriate PyTorch
package index for your CUDA environment.

## Preparation

### 1. Pretrained Models

Prepare the following components before training or evaluation:

| Component | Source | Default Path or Model ID |
|---|---|---|
| AV-HuBERT Large | [AV-HuBERT](https://github.com/facebookresearch/av_hubert) | `pretrained_models/avhubert/large_vox_iter5.pt` |
| Speech rate predictor | [MMS-LLaMA pretrained models](https://github.com/JeongHun0716/MMS-LLaMA#pretrained-models) | `pretrained_models/sr_predictor/checkpoint.pt` |
| Llama-3.2-3B | [Hugging Face](https://huggingface.co/meta-llama/Llama-3.2-3B) | `meta-llama/Llama-3.2-3B` |
| Whisper medium.en | [Hugging Face](https://huggingface.co/openai/whisper-medium.en) | `openai/whisper-medium.en` |
| Q-Former configuration | [BERT Large](https://huggingface.co/google-bert/bert-large-uncased) | `bert-large-uncased` |

Prepare the pretrained models listed above and configure their paths. For evaluation,
set `MODEL_PATH` to the trained checkpoint.

### 2. Data Preprocessing

For speech datasets such as LRS3 and VoxCeleb2, prepare:

- Mouth-region videos at **25 fps**;
- Mono audio at **16 kHz**;
- Transcripts corresponding to each video and audio pair.

For preprocessing instructions, refer to
[Auto-AVSR](https://github.com/mpc001/auto_avsr/tree/main/preparation) and the
[MMS-LLaMA data preparation guide](https://github.com/JeongHun0716/MMS-LLaMA#preparation).

### 3. Data Manifests

Create paired TSV/WRD files for training, validation, and evaluation:

```text
data/manifest/
├── train.tsv
├── train.wrd
├── valid.tsv
├── valid.wrd
├── test.tsv
└── test.wrd
```

Check the data manifests before running experiments:

```bash
python scripts/check_manifest.py /path/to/manifests/train.tsv

# Check the format example included in the repository; no actual media files are required
python scripts/check_manifest.py manifest/example/test.tsv --skip-media-check
```

`manifest/example/` illustrates the file format and does not contain actual evaluation samples.
If text labels are stored separately, use `--labels /path/to/train.wrd` when checking manifests
and set `LABEL_DIR` for training and evaluation.

## Training

### Configure Paths

Set the following variables in the terminal where you will run the experiments:

```bash
export DATA_DIR=/path/to/manifests
export AVHUBERT_PATH=/path/to/large_vox_iter5.pt
export SR_PREDICTOR_PATH=/path/to/sr_predictor.pt
```

### Visual Speech Recognition

```bash
MODALITIES='[video]' OUT_PATH=exp/vsr bash scripts/train.sh
```

### Audio-Visual Speech Recognition

```bash
MODALITIES='[video,audio]' OUT_PATH=exp/avsr bash scripts/train.sh
```

### Multiple GPUs and Configuration

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NGPUS=4 MODALITIES='[video]' \
OUT_PATH=exp/vsr_4gpu bash scripts/train.sh
```

Hyperparameters can be configured in `src/conf/temporal-avsr.yaml`. Some parameters are
overridden by the launch script; adjust them by editing the script or appending Hydra
arguments to the command.

## Evaluation

The commands below use the paths configured in the training section. Choose a checkpoint
that matches the input modalities and model architecture:

```bash
export MODEL_PATH=/path/to/checkpoint.pt
```

### Clean Evaluation

```bash
# VSR
MODALITIES='[video]' OUT_PATH=results/vsr_clean bash scripts/eval.sh

# AVSR
MODALITIES='[video,audio]' OUT_PATH=results/avsr_clean bash scripts/eval.sh
```

### Audio-Visual Evaluation under Noisy Conditions

```bash
SNR=0 OUT_PATH=results/avsr_snr0 bash scripts/eval_snr.sh
```

## Citation and Acknowledgments

If you use Temporal-AVSR, please cite **Bridging the Temporal Gap in Multimodal LLMs:
Deeply Stacking Temporal Tokens for Audio-Visual Speech Recognition**.

```bibtex
@inproceedings{wang2026bridging,
  title={Bridging the Temporal Gap in Multimodal {LLMs}: Deeply Stacking Temporal Tokens for Audio-Visual Speech Recognition},
  author={Wang, Liyong and Xing, Junliang and Hu, Tianyu and Jiang, Jianfei and Zhao, Jihuai and Ma, Huimin},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2026},
  year={2026},
  pages={27748--27759}
}
```

This implementation is based on
[MMS-LLaMA](https://github.com/JeongHun0716/MMS-LLaMA),
[AV-HuBERT](https://github.com/facebookresearch/av_hubert), and
[Fairseq](https://github.com/facebookresearch/fairseq). Data preparation draws on resources from
[Auto-AVSR](https://github.com/mpc001/auto_avsr).

We thank the authors of these projects for their solid work and open-source contributions.

## License

See [LICENSE](LICENSE) for the project license and
[fairseq/LICENSE](fairseq/LICENSE) for the Fairseq license. Upstream copyright notices
are retained in the source code.
