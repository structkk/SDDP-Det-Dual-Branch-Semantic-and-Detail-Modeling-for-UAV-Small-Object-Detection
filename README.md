# SDDP-Det

**Dual-Branch Semantic and Detail Modeling for UAV Small-Object Detection**

Chunmei Wang, Zhikang Xu, Xiaobao Yang, and Junyan Hu  
Xi'an University of Posts and Telecommunications

Manuscript ID: **appliedmath-4641695**  
Correspondence: **xuzhikang@stu.xupt.edu.cn**

SDDP-Det is a detector based on DEIM-Nano for small objects in UAV imagery. It maintains semantic and detail-processing branches and combines their outputs at corresponding pyramid levels.

## Release status

This README documents the method and results reported in the revised manuscript. Source files, the matching training and evaluation configurations, and trained checkpoints should be distributed together. Checkpoint download links are pending upload and must be added to the Model Zoo when the files are publicly available.

The paths in the command examples are placeholders. Replace them with the configuration and checkpoint paths included in the release.

## Architecture

The encoder contains four main components:

- **FCPM — Frequency-Contour Perception Module:** processes projected features using contour-guided spatial and Fourier-domain operations.
- **FASM — Feature-Aligned Semantic Module:** models global context and propagates information across pyramid levels.
- **SDFM — Structural Detail-Focused Module:** updates a detail pyramid through successive stages and historical-feature aggregation.
- **CAFM — Channel-Adaptive Fusion Module:** combines semantic and detail features using input-dependent channel weights.

The manuscript configuration uses backbone features at strides 8, 16, and 32. Each branch projects its inputs to 128 channels and applies three independently parameterized FCPM instances. CAFM fuses the corresponding semantic and detail outputs before the detection decoder.

The reported FCPM configuration uses FFC mode, a Fourier-channel ratio of 0.25, a derivative-modulation gain of 0.5, and MMCV modulated deformable convolution. Ordinary-convolution fallback and alternative module versions are different configurations and should be identified separately if included in the repository.

## Reported results

The following results are reported in the revised manuscript. AP values are percentages. GFLOPs and parameter counts refer to the corresponding reported configurations.

### VisDrone2019-DET

| Model | Split | Input | GFLOPs | Params (M) | AP | AP50 | AP75 | APS | APM | APL |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DEIM-Nano | test-dev | 640 x 640 | 7.124 | 3.73 | 20.5 | 36.4 | 19.6 | 12.2 | 30.2 | 39.9 |
| Expanded DEIM-Nano | test-dev | 640 x 640 | 34.032 | 8.93 | 25.6 | 43.2 | 25.8 | 16.9 | 37.0 | 39.2 |
| SDDP-Det | test-dev | 640 x 640 | 34.364 | 8.92 | 27.3 | 46.2 | 27.4 | 18.7 | 38.6 | 46.2 |
| SDDP-Det | val | 640 x 640 | 34.364 | 8.92 | 29.1 | 48.0 | 28.9 | 20.4 | 40.8 | 48.5 |

Expanded DEIM-Nano and SDDP-Det were trained for 160 epochs under the same training protocol. Expanded DEIM-Nano uses original DEIM components with increased encoder width and depth; it does not contain the proposed modules.

### Controlled encoder replacement

| Configuration | GFLOPs | Params (M) | AP | APS |
|---|---:|---:|---:|---:|
| FCPM + HybridEncoder + SDFM + CAFM | 34.018 | 8.68 | 26.9 | 18.1 |
| FCPM + FASM + SDFM + CAFM | 34.364 | 8.92 | 27.3 | 18.7 |

This comparison retains FCPM, SDFM, CAFM, and the remaining architecture and training settings while replacing the semantic encoder.

### Additional benchmarks

| Dataset | Model | AP | APS |
|---|---|---:|---:|
| MS COCO | DEIM-Nano baseline | 43.0 | 24.5 |
| MS COCO | SDDP-Det | 45.6 | 27.4 |
| DOTA-v1.0 | DEIM-Nano baseline | 45.4 | 26.4 |
| DOTA-v1.0 | SDDP-Det | 48.2 | 28.4 |

Consult the manuscript and the matching dataset configuration for each experiment's protocol.

## Environment

The implementation uses Python and PyTorch. Core dependencies include:

- PyTorch and TorchVision;
- MMCV with the compiled modulated deformable convolution operator;
- NumPy, SciPy, PyYAML, Pillow, and OpenCV;
- faster-coco-eval, tqdm, and TensorBoard;
- calflops for the complexity measurements used in this project.

Use mutually compatible PyTorch, TorchVision, CUDA, and MMCV versions. A successful import of MMCV alone does not establish that its compiled CUDA operator is available.

The following command checks the required Python imports:

```bash
python -c "import torch, torchvision; from mmcv.ops import ModulatedDeformConv2d; print('PyTorch:', torch.__version__); print('TorchVision:', torchvision.__version__); print('CUDA available:', torch.cuda.is_available())"
```

The exact package versions used for a distributed checkpoint should accompany that checkpoint in its release notes or environment file.

## Dataset preparation

Dataset sources:

- VisDrone: https://github.com/VisDrone/VisDrone-Dataset
- MS COCO: https://cocodataset.org/
- DOTA: https://captain-whu.github.io/DOTA/

VisDrone2019-DET contains 6,471 training images, 548 validation images, 1,610 test-dev images, and 1,580 test-challenge images. Validation and test-dev are separate evaluation splits. The main VisDrone comparisons in the manuscript use test-dev; the validation result is reported separately for comparison with published validation results.

Set the image and annotation paths in the selected YAML configuration. For the current training entry point, these fields are:

```yaml
train_dataloader:
  dataset:
    img_folder: /path/to/train/images
    ann_file: /path/to/train_annotations.json

val_dataloader:
  dataset:
    img_folder: /path/to/evaluation/images
    ann_file: /path/to/evaluation_annotations.json
```

The annotation files must be in the COCO-style format expected by the dataset loader, with category IDs matching the selected configuration. Set `val_dataloader` to the intended evaluation split. Do not combine val, test-dev, and test-challenge results under a single split name.

## Training

Run commands from the repository root. Always pass the intended configuration explicitly rather than relying on local defaults in `train.py`.

```bash
TRAIN_CONFIG="path/to/released_train_config.yml"

python train.py -c "$TRAIN_CONFIG" --output-dir outputs/sddp_det --seed 0
```

The main VisDrone training settings reported in the manuscript are:

| Setting | Value |
|---|---|
| Input resolution | 640 x 640 |
| Training duration | 160 epochs |
| Training batch size | 8 |
| Optimizer | AdamW |
| AdamW betas | 0.9, 0.999 |
| Weight decay | 0.0001 |
| Learning-rate schedule | Flat-cosine |
| Backbone initialization | `PPHGNetV2_B0_stage1.pth` |
| Training GPU | NVIDIA GeForce RTX 5080 |

Use the matching released YAML for the remaining optimization, augmentation, and checkpoint-selection settings. Backbone initialization weights and a trained detection checkpoint serve different purposes.

## Evaluation

Download a trained checkpoint from the Model Zoo after its link is available, and use its matching evaluation configuration.

```bash
EVAL_CONFIG="path/to/released_testdev_config.yml"
CHECKPOINT="path/to/released_checkpoint.pth"

python train.py -c "$EVAL_CONFIG" -r "$CHECKPOINT" --test-only --output-dir outputs/evaluation
```

For validation evaluation, use the corresponding val configuration. Report the selected split together with the resulting metrics. In this entry point, `--test-only` runs evaluation using the dataset specified by `val_dataloader`; the flag does not select test-dev automatically.

## Model Zoo

| Checkpoint | Matching configuration | Download |
|---|---|---|
| SDDP-Det, VisDrone2019-DET | To be listed with the released checkpoint | Pending upload |

When publishing a checkpoint, include its exact configuration, evaluation command, software environment, and checksum. Add a public download link here or in the repository's Releases page. Replace the pending entries before describing the trained weights as publicly available in the manuscript.

## Speed and complexity

The manuscript reports 17 FPS for SDDP-Det on an NVIDIA GeForce RTX 5080 with batch size 1 and FP32 precision. Timing includes preprocessing, CPU-GPU data transfer, forward inference, and postprocessing.

Compare throughput only when the device, batch size, numerical precision, input resolution, and timing scope are specified. GFLOPs and parameter counts should also be measured with a consistent deployment state and profiling procedure. Record whether custom operators are supported by the selected profiler.

## Qualitative visualizations

Feature-response and Grad-CAM maps are qualitative visualizations. Some manuscript visualizations are independently normalized or renormalized inside predicted bounding boxes. These display operations should be stated when reproducing the figures.

In Figures 10 and 11, red arrows identify detected objects and green arrows identify missed objects. Arrow colours are annotations and do not encode detection confidence.

## Citation

The manuscript is under review. No publication DOI, volume, or page range is assigned in this README.

```bibtex
@misc{wang2026sddpdet,
  title = {SDDP-Det: Dual-Branch Semantic and Detail Modeling for UAV Small-Object Detection},
  author = {Wang, Chunmei and Xu, Zhikang and Yang, Xiaobao and Hu, Junyan},
  year = {2026},
  note = {Manuscript under review at AppliedMath, manuscript ID appliedmath-4641695}
}
```

## Acknowledgements

This work builds on DEIM and D-FINE and uses established operators including BiFormer-derived attention and ECA. We acknowledge the authors of these methods and the maintainers of the underlying software and datasets. Retain the applicable upstream copyright notices and license terms when distributing derived source files.

## Contact

Questions about the method and experimental configurations can be directed to **xuzhikang@stu.xupt.edu.cn**.
