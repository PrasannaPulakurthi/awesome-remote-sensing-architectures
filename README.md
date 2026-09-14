# Awesome Remote Sensing Architectures [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> Deep learning architectures for Earth observation — **2021-2026**, top venues, code-first.

[![License: CC0-1.0](https://img.shields.io/badge/License-CC0_1.0-lightgrey.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Last commit](https://img.shields.io/github/last-commit/PrasannaPulakurthi/awesome-remote-sensing-architectures)](https://github.com/PrasannaPulakurthi/awesome-remote-sensing-architectures/commits/main)
[![Stars](https://img.shields.io/github/stars/PrasannaPulakurthi/awesome-remote-sensing-architectures?style=social)](https://github.com/PrasannaPulakurthi/awesome-remote-sensing-architectures/stargazers)

Most remote sensing lists are organised by **task** — here is everything on change
detection, here is everything on segmentation. That is useful when you already know
what you are building. It is much less useful when you are trying to answer the
question researchers actually ask: *what is the current state of architectural
design in this field, and what should I build on?*

This list is organised by **architecture family** instead. Convolutional networks,
vision transformers, state-space models, diffusion models, foundation models,
vision-language models. Within each family, entries are grouped by what they were
designed to do. A task index at the end maps back the other way.

**Three rules keep this list small enough to be useful:**

1. **2021 or later.** This is the modern era — post-ViT, post-MAE, post-Mamba.
2. **Top venues only.** See the [venue policy](#venue-policy) below.
3. **Code strongly preferred.** ~90% of entries link a verified public
   implementation. Entries without one are marked `no code` and are included only
   when genuinely foundational.

Every venue below was verified against a primary source, and every code link was
checked to resolve. Corrections are the most welcome kind of contribution — see
[CONTRIBUTING.md](CONTRIBUTING.md).

---

## How to read an entry

```
- **ModelName** — What is structurally new about it, in one sentence.
  [`paper`](…) [`code`](…) `TGRS'24` `segmentation` `644★`
```

| Marker | Meaning |
|---|---|
| `644★` | Approximate GitHub stars at time of indexing. Indicative only. |
| `no code` | No public implementation found. Included on significance alone. |
| `preprint` | Not peer-reviewed at a listed venue, but widely adopted **and** ships code or weights. |
| `weights: gated` | Checkpoints exist but require a request, or are non-commercial only. |
| ⚠ | Venue sits outside the policy below; kept because the work is canonical for its sub-area. |

### Venue policy

**Machine learning and computer vision** — CVPR, ICCV, ECCV, NeurIPS, ICLR, ICML,
AAAI, IJCAI, TPAMI, IJCV.

**Remote sensing and geoscience** — IEEE TGRS, ISPRS Journal of Photogrammetry and
Remote Sensing, Remote Sensing of Environment, IEEE JSTARS, IEEE GRSL, IEEE
Geoscience and Remote Sensing Magazine.

**Adjacent journals accepted where the work is central to a sub-area** — IEEE TIP,
IEEE TMM, IEEE TCSVT, Information Fusion, Nature Machine Intelligence.

Anything else is marked ⚠ and included only when excluding it would leave a
sub-area without its most-cited work. IGARSS, MDPI journals and workshop papers
are generally out of scope, with a small number of flagged exceptions where the
model has become a de-facto baseline.

---

## Contents

**Architecture families**

- [Foundation Models & Self-Supervised Pretraining](#foundation-models--self-supervised-pretraining)
- [Vision Transformers](#vision-transformers)
- [Mamba & State-Space Models](#mamba--state-space-models)
- [Diffusion & Generative Models](#diffusion--generative-models)
- [Vision-Language Models & Multimodal LLMs](#vision-language-models--multimodal-llms)
- [Efficient Architectures](#efficient-architectures)

**By modality**

- [Hyperspectral](#hyperspectral)
- [SAR](#sar)
- [3D, LiDAR & Neural Fields](#3d-lidar--neural-fields)
- [Satellite Image Time Series](#satellite-image-time-series)

**Resources**

- [Surveys](#surveys)
- [Benchmarks & Datasets](#benchmarks--datasets)
- [Libraries & Tooling](#libraries--tooling)
- [Task Index](#task-index)

---

## Foundation Models & Self-Supervised Pretraining

The centre of gravity in remote sensing deep learning since 2022. The field has
moved from "pretrain on ImageNet and hope" to domain-specific pretraining at
scale, and more recently to models that accept arbitrary sensors as input.

**If you want a strong default to build on:** [Prithvi-EO-2.0](#agency--open-release-models)
or [Galileo](#multi-modal-foundation-models) for open weights and permissive
licences, [DOFA](#multi-modal-foundation-models) or [TerraMind](#multi-modal-foundation-models)
if your sensor set is unusual.

### Masked image modeling

- **SatMAE** — Extends MAE with temporal embeddings and independent per-timestep
  masking, plus spectral band grouping with distinct positional encodings. The
  reference formulation that later work benchmarks against.
  [`paper`](https://arxiv.org/abs/2207.08051) [`code`](https://github.com/sustainlab-group/SatMAE) `NeurIPS'22` `pretraining` `265★`
- **Scale-MAE** — Conditions ViT positional encoding on ground sample distance and
  decodes through a Laplacian-pyramid bandpass decoder, making representations
  explicitly scale-aware rather than resolution-agnostic.
  [`paper`](https://arxiv.org/abs/2212.14532) [`code`](https://github.com/bair-climate-initiative/scale-mae) `ICCV'23` `pretraining` `171★`
- **GFM** — Continual pretraining with a frozen ImageNet-22k teacher distilled
  alongside in-domain masked image modeling, matching large-scale in-domain
  pretraining on a compact 600K corpus.
  [`paper`](https://arxiv.org/abs/2302.04476) [`code`](https://github.com/mmendiet/GFM) `ICCV'23` `pretraining` `79★`
- **Cross-Scale MAE** — Scale augmentation with cross-scale consistency
  constraints, removing the need for aligned multi-GSD image pairs at pretraining
  time.
  [`paper`](https://arxiv.org/abs/2401.15855) [`code`](https://github.com/aicip/Cross-Scale-MAE) `NeurIPS'23` `pretraining` `55★`
- **SatMAE++** — Adds multi-scale hierarchical reconstruction with convolutional
  upsampling, exploiting the scale information that SatMAE discards.
  [`paper`](https://arxiv.org/abs/2403.05419) [`code`](https://github.com/techmn/satmae_pp) `CVPR'24` `pretraining` `133★`
- **SelectiveMAE** — Progressive semantic token selection encodes and reconstructs
  only semantically dense patches, skipping redundant background for a 2.2-2.7×
  pretraining speedup. Ships the OpticalRS-13M corpus.
  [`paper`](https://arxiv.org/abs/2406.11933) [`code`](https://github.com/MiliLab/SelectiveMAE) `ICCV'25` `pretraining` `139★`

### Contrastive & geo-aware self-supervision

- **SeCo** — Treats seasonal variation at a fixed location as a free augmentation
  axis, with multi-head embeddings that separate season-invariant from
  season-varying factors.
  [`paper`](https://arxiv.org/abs/2103.16607) [`code`](https://github.com/ServiceNow/seasonal-contrast) `ICCV'21` `pretraining` `184★`
- **GASSL** — Adds temporal positive pairs and a geo-location prediction pretext
  task to MoCo-v2, first closing the contrastive-versus-supervised gap on RS
  benchmarks.
  [`paper`](https://openaccess.thecvf.com/content/ICCV2021/papers/Ayush_Geography-Aware_Self-Supervised_Learning_ICCV_2021_paper.pdf) [`code`](https://github.com/sustainlab-group/geography-aware-ssl) `ICCV'21` `pretraining` `45★`
- **CACo** — Contrasts long-term against short-term temporal differences so the
  objective learns what *change* looks like, with change-driven geographic
  sampling.
  [`paper`](https://openaccess.thecvf.com/content/CVPR2023/html/Mall_Change-Aware_Sampling_and_Contrastive_Learning_for_Satellite_Images_CVPR_2023_paper.html) [`code`](https://github.com/utkarshmall13/CACo) `CVPR'23` `pretraining` `59★`
- **CROMA** — Joint contrastive alignment and masked reconstruction over
  co-registered SAR/optical pairs, with X- and 2D-ALiBi spatial biases replacing
  learned positional encodings.
  [`paper`](https://arxiv.org/abs/2311.00566) [`code`](https://github.com/antofuller/CROMA) `NeurIPS'23` `multimodal pretraining` `52★`

### Multi-modal foundation models

The most active frontier. The design problem is no longer "how do we pretrain on
satellite data" but "how does one model accept optical, SAR, elevation and
climate inputs — including sensors it never saw during training".

- **SkySense** — Billion-parameter factorised multi-modal spatiotemporal encoder
  over high-resolution optical plus Sentinel-1/2 time series, trained with
  multi-granularity contrastive learning and geo-context prototypes.
  [`paper`](https://arxiv.org/abs/2312.10115) [`code`](https://github.com/Jack-bo1220/SkySense) `CVPR'24` `multimodal` `weights: gated` `85★`
- **msGFM** — Cross-sensor masked image modeling that handles paired *and*
  unpaired sensor data, covering RGB, Sentinel-2, SAR and DSM.
  [`paper`](https://arxiv.org/abs/2404.01260) [`code`](https://github.com/boranhan/Geospatial_Foundation_Models) `CVPR'24` `multimodal` `weights: no`
- **OmniSat** — Exploits the natural spatial alignment between EO modalities for
  label-free fusion of VHR aerial, Sentinel-1 and Sentinel-2 time series. Also
  contributes TreeSatAI-TS and PASTIS-HD.
  [`paper`](https://arxiv.org/abs/2404.08351) [`code`](https://github.com/gastruc/OmniSat) `ECCV'24` `multimodal` `99★`
- **DOFA** — A wavelength-conditioned dynamic hypernetwork generates patch-embedding
  weights, so a single ViT serves any sensor, including ones absent from
  pretraining. Neural-plasticity framing.
  [`paper`](https://arxiv.org/abs/2403.15356) [`code`](https://github.com/zhu-xlab/DOFA) `preprint` `multimodal` `213★`
- **Galileo** — Dual global and local self-supervised objectives with different
  masking strategies across nine modalities, letting an 85M generalist beat
  specialist SOTA on 11 benchmarks. Nano variant is 0.8M parameters.
  [`paper`](https://arxiv.org/abs/2502.09356) [`code`](https://github.com/nasaharvest/galileo) `ICML'25` `multimodal` `200★`
- **AnySat** — JEPA with resolution-adaptive spatial encoders sharing 75% of
  parameters across 0.2m-500m GSD, 3-12 channels and 11 sensors, trained jointly
  on datasets with incompatible geometry.
  [`paper`](https://arxiv.org/abs/2412.14123) [`code`](https://github.com/gastruc/AnySat) `CVPR'25` `multimodal` `210★`
- **Copernicus-FM** — Extends dynamic hypernetworks to *non-spectral* sensors and
  adds flexible metadata encoding, covering surface-to-atmosphere Copernicus data.
  [`paper`](https://arxiv.org/abs/2503.11849) [`code`](https://github.com/zhu-xlab/Copernicus-FM) `ICCV'25` `multimodal` `150★`
- **TerraMind** — First any-to-any *generative* EO foundation model, with
  "Thinking-in-Modalities" inference that synthesises missing modalities as
  intermediate reasoning steps. Apache-2.0 across four sizes.
  [`paper`](https://arxiv.org/abs/2504.11171) [`code`](https://github.com/IBM/terramind) `ICCV'25` `multimodal` `319★`
- **SkySense V2** — Replaces per-modality backbones with one shared transformer
  using adaptive patch merging, modality prompt tokens and mixture-of-experts.
  [`paper`](https://arxiv.org/abs/2507.13812) `ICCV'25` `multimodal` `weights: no`
- **SkySense++** — Two-stage progressive pretraining, contrastive then masked
  *semantic* learning, giving strong few-shot transfer across 12 EO tasks in 7
  application domains.
  [`paper`](https://doi.org/10.1038/s42256-025-01078-8) [`code`](https://github.com/kang-wu/SkySensePlusPlus) `Nat. Mach. Intell.'25` `multimodal` `weights: gated` `242★`

### Plain-ViT backbones & parameter scaling

- **RSP** — First systematic study of pretraining CNN, Swin and ViTAEv2 backbones
  on MillionAID rather than ImageNet, establishing RS-domain weights as a standard
  baseline.
  [`paper`](https://arxiv.org/abs/2204.02825) [`code`](https://github.com/ViTAE-Transformer/RSP) `TGRS'22` `pretraining` `159★`
- **RingMo** — The first RS-specific masked image modeling foundation model, with a
  masking strategy that avoids destroying dense small objects during
  reconstruction.
  [`paper`](https://doi.org/10.1109/TGRS.2022.3194732) `TGRS'22` `pretraining` `no code`
- **RVSA** — Replaces full attention in a ~100M plain ViT with rotated varied-size
  window attention, giving orientation-aware attention at reduced cost.
  [`paper`](https://arxiv.org/abs/2208.03987) [`code`](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA) `TGRS'23` `backbone` `469★`
- **SatlasPretrain** — Large-scale *supervised* multi-task pretraining corpus
  (856K tiles, 302M labels, 137 categories) across Sentinel-1/2, Landsat and
  aerial imagery.
  [`paper`](https://arxiv.org/abs/2211.15660) [`code`](https://github.com/allenai/satlas) `ICCV'23` `pretraining` `284★`
- **MTP** — Supervised multi-task pretraining over semantic segmentation, instance
  segmentation and rotated detection simultaneously, closing the
  pretext-to-downstream task gap.
  [`paper`](https://arxiv.org/abs/2403.13430) [`code`](https://github.com/ViTAE-Transformer/MTP) `JSTARS'24` `pretraining` `254★`
- **Billion-scale RSFM** — First systematic study of parameter scaling for RS
  foundation models, from 86M to 2.4B parameters.
  [`paper`](https://arxiv.org/abs/2304.05215) `JSTARS'24` `scaling` `weights: no`

### Agency & open-release models

Models released primarily as *artefacts* rather than papers — usually with
permissive licences and real fine-tuning tooling. In practice these are what most
applied work actually builds on.

- **Prithvi-EO-1.0** — NASA/IBM's first open geospatial foundation model, 100M ViT
  MAE over Harmonized Landsat-Sentinel. Established the open-weights plus
  fine-tuning-recipe release pattern for EO.
  [`paper`](https://arxiv.org/abs/2310.18660) [`weights`](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M) `preprint` `Apache-2.0`
- **Prithvi-EO-2.0** — 300M/600M with 3D patch and positional embeddings plus
  optional temporal and location metadata encoding. The most widely fine-tuned
  open geospatial FM, with full TerraTorch integration.
  [`paper`](https://arxiv.org/abs/2412.02732) [`code`](https://github.com/NASA-IMPACT/Prithvi-EO-2.0) `preprint` `Apache-2.0` `309★`
- **Clay** — Community-governed MAE with a DINOv2 teacher and a dynamic embedding
  block that adapts to arbitrary sensor band sets. 70M global chips, no paper.
  [`docs`](https://clay-foundation.github.io/model/) [`code`](https://github.com/Clay-foundation/model) `model release` `Apache-2.0` `614★`
- **SSL4EO-S12** — The de-facto standard global SSL pretraining corpus for
  Sentinel-1/2 (251K locations × 4 seasons, 1.5 TB), shipped with reference
  weights for four SSL methods. ⚠ `IEEE GRSM`
  [`paper`](https://arxiv.org/abs/2211.07044) [`code`](https://github.com/zhu-xlab/SSL4EO-S12) `corpus` `300★`
- **Major TOM** — A grid and metadata standard that makes terabyte-scale EO
  datasets interoperable and mergeable; now the default distribution format on
  Hugging Face. ⚠ `IGARSS`
  [`paper`](https://arxiv.org/abs/2402.12095) [`code`](https://github.com/ESA-PhiLab/Major-TOM) `corpus` `230★`
- **AlphaEarth Foundations** — Continuous "embedding field" formulation producing
  analysis-ready global annual 64-D embeddings at 10m, 2017-2024. Embeddings are
  public via Earth Engine; the model is not.
  [`paper`](https://arxiv.org/abs/2507.22291) `preprint` `weights: no`
- **SARATR-X** — First SAR target recognition foundation model: two-step
  self-supervision with multi-scale gradient features, competitive with fully
  supervised baselines in few-shot settings.
  [`paper`](https://arxiv.org/abs/2405.09365) [`code`](https://github.com/waterdisappear/SARATR-X) `IEEE TIP'25` `SAR` `270★`
- **HyperSIGMA** — First billion-parameter hyperspectral foundation model, using
  sparse sampling attention to counter spatial-spectral redundancy. See
  [Hyperspectral](#hyperspectral).
  [`paper`](https://arxiv.org/abs/2406.11519) [`code`](https://github.com/WHU-Sigma/HyperSIGMA) `TPAMI'25` `hyperspectral` `390★`

### Benchmarks for foundation models

Evaluating a GFM is genuinely hard, and early comparisons were unreliable. These
three are the current standards.

- **GEO-Bench** — The first standardised expert-curated GFM evaluation suite:
  6 classification and 6 segmentation tasks with a defined aggregation
  methodology. 20 models benchmarked.
  [`paper`](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a0644215d9cff6646fa334dfa5d29c5a-Abstract-Datasets_and_Benchmarks.html) [`code`](https://github.com/ServiceNow/geo-bench) `NeurIPS'23` `194★`
- **PANGAEA** — Corrects the North-America/Europe geographic bias and task
  triviality of earlier GFM evaluations, with 12 datasets across 7 task types
  under one protocol.
  [`paper`](https://arxiv.org/abs/2412.04204) [`code`](https://github.com/VMarsocci/pangaea-bench) `preprint` `284★`
- **Copernicus-Bench** — Mission-structured benchmark spanning preprocessing
  through specialised applications, 15 hierarchical tasks across three levels.
  [`paper`](https://arxiv.org/abs/2503.11849) [`code`](https://github.com/zhu-xlab/Copernicus-FM) `ICCV'25` `150★`

---

## Vision Transformers

Backbone pretraining lives under [Foundation Models](#foundation-models--self-supervised-pretraining).
This section covers transformer architectures designed around specific remote
sensing *problems* — rotation, scale variance, bi-temporal reasoning, and the
extreme size of a single scene.

### Efficient backbones

- **LeMeViT** — Learnable sparse "meta tokens" exchange information with image
  tokens through dual cross-attention, cutting attention cost on redundant RS
  imagery for a ~1.7× speedup.
  [`paper`](https://arxiv.org/abs/2405.09789) [`code`](https://github.com/ViTAE-Transformer/LeMeViT) `IJCAI'24` `backbone` `55★`

### Semantic segmentation

- **DC-Swin** — Swin encoder paired with a densely connected feature aggregation
  decoder to restore the multi-scale spatial detail that window attention loses.
  [`paper`](https://arxiv.org/abs/2104.12137) [`code`](https://github.com/WangLibo1995/GeoSeg) `GRSL'22` `segmentation` `1102★`
- **UNetFormer** — ResNet18 encoder with a transformer decoder built on an
  efficient global-local attention block, reaching 322 FPS at competitive mIoU.
  The canonical RS hybrid, and GeoSeg is the de-facto RS segmentation model zoo.
  [`paper`](https://arxiv.org/abs/2109.08937) [`code`](https://github.com/WangLibo1995/GeoSeg) `ISPRS J.'22` `segmentation` `1102★`
- **BuildFormer** — Dual-path window transformer keeping a high-resolution spatial
  branch alongside the global context branch, for sharp building boundaries.
  [`paper`](https://ieeexplore.ieee.org/document/9808187) [`code`](https://github.com/WangLibo1995/BuildFormer) `TGRS'22` `building extraction` `110★`
- **CMTFNet** — Encoder-decoder fusing CNN local features with a multiscale
  multihead self-attention decoder plus channel-wise feature fusion.
  [`paper`](https://ieeexplore.ieee.org/document/10247595) [`code`](https://github.com/DrWuHonglin/CMTFNet) `TGRS'23` `segmentation` `42★`
- **RSPrompter** — Learns category-aware prompt embeddings that drive a frozen SAM
  decoder, turning SAM into an end-to-end automatic instance segmenter rather than
  an interactive one.
  [`paper`](https://arxiv.org/abs/2306.16269) [`code`](https://github.com/KyanChen/RSPrompter) `TGRS'24` `instance segmentation` `667★`
- **CrossEarth** — Earth-style injection augmentation combined with multi-task
  DINOv2 backbone adaptation for cross-domain segmentation without target-domain
  data.
  [`paper`](https://arxiv.org/abs/2410.22629) [`code`](https://github.com/Cuzyoung/CrossEarth) `TPAMI'25` `domain generalization` `189★`

### Change detection

- **BIT** — Expresses each image as a handful of semantic tokens and refines them
  with a small transformer, beating pure-CNN baselines at roughly a third of the
  cost. The reference transformer CD baseline.
  [`paper`](https://arxiv.org/abs/2103.00208) [`code`](https://github.com/justchenhao/BIT_CD) `TGRS'21` `change detection` `513★`
- **ChangeStar** — A ChangeMixin module converts any single-temporal segmentation
  network into a change detector, removing the need for paired bi-temporal labels
  entirely.
  [`paper`](https://arxiv.org/abs/2108.07002) [`code`](https://github.com/Z-Zheng/ChangeStar) `ICCV'21` `weakly-supervised CD` `191★`
- **ICIF-Net** — Parallel CNN and transformer branches communicate at equal
  resolution, then fuse across scales, avoiding the local-global misalignment of
  late fusion.
  [`paper`](https://ieeexplore.ieee.org/document/9759285) [`code`](https://github.com/ZhengJianwei2/ICIF-Net) `TGRS'22` `change detection` `42★`
- **TransUNetCD** — UNet skeleton where a transformer encodes tokenized CNN feature
  maps, with skip connections restoring the localisation a pure-transformer encoder
  loses.
  [`paper`](https://ieeexplore.ieee.org/document/9761892) `TGRS'22` `change detection` `no code`
- **Changer** — A meta-architecture that inserts bi-temporal feature-interaction
  layers *inside* the extractor rather than after it, including a parameter-free
  feature exchange variant.
  [`paper`](https://arxiv.org/abs/2209.08290) [`code`](https://github.com/likyoo/open-cd) `TGRS'23` `change detection` `890★`
- **SCanNet** — Jointly models the triple-branch spatio-temporal token set instead
  of fusing semantic and change branches post hoc.
  [`paper`](https://ieeexplore.ieee.org/document/10443352) [`code`](https://github.com/ggsDing/SCanNet) `TGRS'24` `semantic CD` `64★`
- **BAN** — Bi-temporal adapter network that freezes a foundation model (CLIP or
  SAM) and bridges it to any existing CD head with few learnable parameters.
  [`paper`](https://arxiv.org/abs/2312.01163) [`code`](https://github.com/likyoo/BAN) `TGRS'24` `change detection, PEFT` `102★`
- **Changen2** — Resolution-scalable generative change process model that
  synthesises labelled multi-temporal sequences, yielding zero-shot-capable
  pretrained CD weights.
  [`paper`](https://arxiv.org/abs/2406.17998) [`code`](https://github.com/Z-Zheng/pytorch-change-models) `TPAMI'24` `generative CD` `263★`
- **AnyChange** — Training-free bi-temporal latent matching over SAM's latent
  space, giving SAM zero-shot change detection with no CD training at all.
  [`paper`](https://arxiv.org/abs/2402.01188) [`code`](https://github.com/Z-Zheng/pytorch-change-models) `NeurIPS'24` `zero-shot CD` `263★`

### Oriented object detection

Objects in overhead imagery have no canonical "up", so the whole detection stack —
anchors, NMS, loss, receptive field — has to be rebuilt around rotation.

- **Oriented R-CNN** — An oriented RPN generates high-quality rotated proposals
  nearly cost-free via a 6-parameter midpoint-offset representation, removing the
  two-stage oriented-proposal bottleneck.
  [`paper`](https://arxiv.org/abs/2108.05699) [`code`](https://github.com/jbwang1997/OBBDetection) `ICCV'21` `oriented detection` `611★`
- **LSKNet** — Decomposes large-kernel convolutions into a depth-wise sequence with
  growing kernel and dilation, plus spatial kernel selection to size the receptive
  field per object. The standard modern RS backbone.
  [`paper`](https://arxiv.org/abs/2303.09030) [`code`](https://github.com/zcablii/LSKNet) `ICCV'23` `oriented detection` `700★`
  · journal extension generalising it to a lightweight all-purpose backbone:
  [`paper`](https://arxiv.org/abs/2403.11735) `IJCV'24`
- **ARC** — Convolution kernels rotate adaptively per input through a conditional
  computation routing mechanism, handling multiple object orientations within a
  single image.
  [`paper`](https://arxiv.org/abs/2303.07820) [`code`](https://github.com/LeapLabTHU/ARC) `ICCV'23` `oriented detection` `148★`
- **ARS-DETR** — Aspect-ratio-aware circular smooth label, rotated deformable
  attention and an aspect-ratio-weighted angle loss make a DETR competitive at
  high-IoU oriented detection.
  [`paper`](https://arxiv.org/abs/2303.04989) [`code`](https://github.com/httle/ARS-DETR) `TGRS'24` `oriented detection, DETR` `75★`
- **PKINet** — Parallel multi-scale non-dilated kernels for local context plus a
  context anchor attention module for long-range context, avoiding the background
  noise that large kernels pull in.
  [`paper`](https://arxiv.org/abs/2403.06258) [`code`](https://github.com/PKINet/PKINet) `CVPR'24` `oriented detection` `89★`
- **MSFA / SARDet-100K** — Multi-stage filter-augmentation pretraining bridges the
  RGB-to-SAR domain and structure gap; ships the first COCO-scale multi-class SAR
  detection dataset.
  [`paper`](https://arxiv.org/abs/2403.06534) [`code`](https://github.com/zcablii/SARDet_100K) `NeurIPS'24` `SAR detection` `782★`
- **Point2RBox-v2** — First to exploit inter-instance spatial layout for
  point-supervised oriented detection, via Gaussian overlap and Voronoi watershed
  losses that bound the object extent from above and below.
  [`paper`](https://arxiv.org/abs/2502.04268) [`code`](https://github.com/VisionXLab/point2rbox-v2) `CVPR'25` `weakly-supervised detection` `45★`

### Segmentation datasets generated by foundation models

- **SAMRS** — Prompts SAM with existing RS detection boxes to auto-generate 105K
  images and 1.67M instances, enabling *segmentation*-task pretraining instead of
  classification pretraining.
  [`paper`](https://arxiv.org/abs/2305.02034) [`code`](https://github.com/ViTAE-Transformer/SAMRS) `NeurIPS'23` `segmentation pretraining` `385★`

---

## Mamba & State-Space Models

Remote sensing adopted Mamba faster than almost any other vision domain, for a
concrete reason: a single satellite scene is enormous, and attention is quadratic
in sequence length. Linear-complexity sequence modelling means you can process a
whole VHR tile without cropping it into patches and stitching the seams back
together.

The recurring design problem is that Mamba is **causal and one-dimensional** while
images are neither. Almost every architecture below is, at heart, a different
answer to "in what order do we scan the pixels?"

> Hyperspectral Mamba models are covered in [Hyperspectral](#hyperspectral),
> where the spectral scan order is its own design axis.

### Backbones & classification

- **RSMamba** — Dynamic multi-path activation (forward, reverse and shuffled scan
  paths) lets a causal SSM model non-causal 2D image tokens. The canonical entry
  point for this family.
  [`paper`](https://arxiv.org/abs/2403.19654) [`code`](https://github.com/KyanChen/RSMamba) `GRSL'24` `classification` `306★`
- **CE-VSS** — Injects explicit contour and edge priors into the 2D selective scan
  so the state-space backbone retains the object boundaries that pure sequential
  scanning erodes.
  [`paper`](https://ieeexplore.ieee.org/document/10810482) [`code`](https://github.com/yanliyue/Contour-enhanced-Visual-State-Space-Model) `TGRS'24` `classification` `7★`
- **CMSI-Mamba** — Cross-modal Mamba interaction blocks align spatial and spectral
  token streams before fusion, avoiding quadratic cross-attention.
  [`paper`](https://ieeexplore.ieee.org/document/10829637) [`code`](https://github.com/ru-willow/CMSI-Mamba) `TGRS'25` `multimodal classification` `4★`

### Dense prediction & segmentation

- **RS³Mamba** — Dual-branch encoder where a visual state-space auxiliary branch
  supplies global context to a CNN main branch, fused by a collaborative
  completion module. First vision-Mamba segmenter for RS.
  [`paper`](https://arxiv.org/abs/2404.02457) [`code`](https://github.com/sstary/SSRS) `GRSL'24` `segmentation` `806★`
- **RS-Mamba** — An omnidirectional selective scan lets a linear-complexity SSM
  ingest entire large VHR images without cropping, which is the whole practical
  argument for Mamba in this field.
  [`paper`](https://arxiv.org/abs/2404.02668) [`code`](https://github.com/walking-shadow/Official_Remote_Sensing_Mamba) `TGRS'24` `segmentation, dense prediction` `348★`
- **PPMamba** — Pyramid-pooling CNN branch interleaved with 2D selective scan
  blocks to recover local detail at multiple scales.
  [`paper`](https://ieeexplore.ieee.org/document/10769411) [`code`](https://github.com/Jerrymo59/PPMambaSeg) `TGRS'24` `segmentation` `2★`
- **CM-UNet** — CNN encoder with a CSMamba decoder block and multi-scale attention
  fusion, giving UNet a linear-cost global decoder.
  [`paper`](https://arxiv.org/abs/2405.10530) [`code`](https://github.com/XiaoBuL/CM-UNet) `preprint` `segmentation` `214★`
- **Samba** — Samba-block encoder with a UperNet decoder; the first SSM
  encoder-decoder benchmark for RS segmentation. ⚠ `Heliyon`
  [`paper`](https://doi.org/10.1016/j.heliyon.2024.e38495) [`code`](https://github.com/zhuqinfeng1999/Samba) `2024` `segmentation` `159★`
- **RSMTMamba** — Mamba-based cross-task feature learning in a shared-encoder
  design for joint segmentation, height estimation and boundary detection.
  [`paper`](https://ieeexplore.ieee.org/document/10879310) [`code`](https://github.com/sycs-2024/RSMultitaskMamba) `TGRS'25` `multi-task` `3★`

### Change detection

- **ChangeMamba** — VMamba encoder plus three spatio-temporal relation mechanisms
  (sequential, cross and parallel scan) covering binary CD, semantic CD and
  building damage assessment in one framework. ESI Highly Cited.
  [`paper`](https://arxiv.org/abs/2404.03425) [`code`](https://github.com/ChenHongruixuan/ChangeMamba) `TGRS'24` `change detection` `644★`
- **MF-Mamba** — Multi-level SSM fusion aggregating state representations across
  encoder stages rather than only at the bottleneck.
  [`paper`](https://ieeexplore.ieee.org/document/10756674) [`code`](https://github.com/121zzy/MF-Mamba) `TGRS'24` `change detection` `5★`
- **RSCaMa** — Stacked layers combining a spatial-difference SSM with a
  temporal-traversing SSM that scans bi-temporal features cross-wise, for change
  *captioning*.
  [`paper`](https://arxiv.org/abs/2404.18895) [`code`](https://github.com/Chen-Yang-Liu/RSCaMa) `GRSL'24` `change captioning` `81★`
- **LCCDMamba** — Visual state-space encoder with a land-cover-aware difference
  aggregation decoder tuned for VHR land-cover transitions.
  [`paper`](https://ieeexplore.ieee.org/document/10845192) [`code`](https://github.com/juncyan/lccdmamba) `JSTARS'25` `land-cover CD` `3★`
- **CDMamba** — A scaled residual ConvMamba block recovers the fine local detail
  Mamba discards, plus adaptive global-local guided fusion for bi-temporal
  interaction.
  [`paper`](https://arxiv.org/abs/2406.04207) [`code`](https://github.com/zmoka-zht/CDMamba) `TGRS'25` `change detection` `116★`

### Fusion, super-resolution & detection

- **FusionMamba** — Extends the single-input Mamba block into a plug-and-play
  *dual-input* block for arbitrary two-source fusion, covering pansharpening and
  hyperspectral-multispectral fusion.
  [`paper`](https://arxiv.org/abs/2404.07932) [`code`](https://github.com/PSRben/FusionMamba) `TGRS'24` `pansharpening, fusion` `139★`
- **SDMSPan** — Detail-branch supervision guides a multi-scale SSM so
  high-frequency panchromatic detail is explicitly injected rather than implicitly
  learned.
  [`paper`](https://ieeexplore.ieee.org/document/10812822) [`code`](https://github.com/zhaomengjiao123/SDMSPan) `TGRS'24` `pansharpening` `1★`
- **MiM-ISTD** — Nested outer/inner Mamba over patches and sub-patches, making
  large-image infrared small-target detection viable at a fraction of transformer
  GPU cost.
  [`paper`](https://ieeexplore.ieee.org/document/10740056) [`code`](https://github.com/txchen-USTC/MiM-ISTD) `TGRS'24` `infrared detection` `179★`
- **TrackingMamba** — Single-stream visual state-space tracking backbone replacing
  transformer relation modelling for satellite video.
  [`paper`](https://ieeexplore.ieee.org/document/10678881) [`code`](https://github.com/KustTeamWQW/TrackingMamba) `JSTARS'24` `video tracking` `27★`
- **FMambaIR** — Couples SSM spatial modelling with an explicit frequency-domain
  branch to restore both structure and texture.
  [`paper`](https://ieeexplore.ieee.org/document/10834441) [`code`](https://github.com/mickoluan/FMambaIR) `TGRS'25` `restoration, dehazing` `23★`
- **Pan-Mamba** — Channel-swapping Mamba for cheap cross-modal exchange plus
  cross-modal Mamba for PAN/MS relation modelling. The canonical Mamba
  pansharpening paper. ⚠ `Information Fusion`
  [`paper`](https://arxiv.org/abs/2402.12192) [`code`](https://github.com/alexhe101/Pan-Mamba) `2025` `pansharpening` `140★`
- **FreMamba** — Frequency selection, vision state-space and hybrid gate modules;
  the first Mamba for RS super-resolution, beating HAT-L at ~28% of its memory.
  ⚠ `IEEE TMM`
  [`paper`](https://arxiv.org/abs/2405.04964) [`code`](https://github.com/XY-boy/FreMamba) `2024` `super-resolution` `99★`

### Mamba foundation models

- **RoMA** — Auto-regressive Mamba pretraining with rotation-aware adaptive
  cropping, angular embeddings and multi-scale token prediction; beats ViT-based
  RSFMs while cutting GPU memory roughly 80% on high-resolution data.
  [`paper`](https://arxiv.org/abs/2503.10392) [`code`](https://github.com/MiliLab/RoMA) `NeurIPS'25` `foundation model` `135★`
- **SatMamba** — Masked autoencoder whose encoder *and* decoder are multi-way
  Mamba blocks rather than transformer blocks, giving linear scaling in sequence
  length.
  [`paper`](https://arxiv.org/abs/2502.00435) [`code`](https://github.com/mdchuc/HRSFM) `preprint` `foundation model` `4★`

---

## Efficient Architectures

Attention and state-space models are not the only options. This section collects
architectures whose primary contribution is a different **cost** profile.

- **RS-vHeat** — Replaces attention with a heat conduction operator (O(N^1.5) with
  a global receptive field) guided by object structure, pretrained via
  frequency-domain hierarchical masking. 84% less memory and 2.7× throughput
  versus attention-based RSFMs.
  [`paper`](https://arxiv.org/abs/2411.17984) [`code`](https://github.com/iecashhy/RS-vHeat) `ICCV'25` `foundation model, efficient` `16★`
- **DeepKANSeg** — A DeepKAN refinement module and global-local decoder built from
  Kolmogorov-Arnold linear layers, decomposing high-dimensional features into
  univariate learnable transforms.
  [`paper`](https://arxiv.org/abs/2501.07390) [`code`](https://github.com/sstary/SSRS) `TGRS'26` `segmentation, KAN` `806★`
- **RSRWKV** — 2D-WKV scanning in four directions removes RWKV's one-dimensional
  anisotropy, plus multi-view convolutional shift and efficient channel attention.
  [`paper`](https://arxiv.org/abs/2503.20382) `2025` `linear attention` `no code`

---

## Hyperspectral

The deepest section in this list. Hyperspectral imaging has its own architectural
logic: the spectral axis is a *sequence* with physical meaning, bands are
correlated and redundant, and labelled pixels are scarce and spatially clustered.
Architectures that ignore this and treat an HSI cube as a 200-channel RGB image
tend to score well on the classic benchmarks and generalise poorly — see the
[methodological note](#a-note-on-hsi-benchmarking) below, which matters more than
any individual entry here.

### Transformers for HSI classification

- **SpectralFormer** — Group-wise spectral embedding over adjacent band groups plus
  cross-layer adaptive fusion, treating the spectrum itself as the transformer's
  sequence dimension. The reference HSI transformer.
  [`paper`](https://doi.org/10.1109/TGRS.2021.3130716) [`code`](https://github.com/danfenghong/IEEE_TGRS_SpectralFormer) `TGRS'22` `classification` `338★`
- **SSFTT** — A 3D/2D CNN shallow spectral-spatial extractor feeds a
  Gaussian-weighted feature tokenizer into a transformer encoder. The most-copied
  hybrid CNN-transformer HSI baseline.
  [`paper`](https://ieeexplore.ieee.org/document/9684381) [`code`](https://github.com/zgr6010/HSI_SSFTT) `TGRS'22` `classification` `137★`
- **morphFormer** — Injects learnable spectral and spatial morphological
  (erosion/dilation) operations into the attention block to capture shape and
  structure cues that plain attention misses.
  [`paper`](https://doi.org/10.1109/TGRS.2023.3242346) [`code`](https://github.com/mhaut/morphFormer) `TGRS'23` `classification` `46★`
- **SSTFormer** — Extends spectral-spatial tokenization with a temporal transformer
  branch for bi-temporal hyperspectral change detection.
  [`paper`](https://doi.org/10.1109/TGRS.2022.3203075) [`code`](https://github.com/yanhengwang-heu/IEEE_TGRS_SSTFormer) `TGRS'22` `change detection` `37★`

### State-space models for HSI

Mamba's linear scaling is a natural fit for long spectral sequences, and the
scan-order question becomes genuinely two-dimensional here: you are choosing an
order across *space* and across *wavelength* simultaneously.

- **MambaHSI** — The first *image-level* rather than patch-level Mamba HSI
  classifier, with separate spatial and spectral Mamba blocks and an adaptive
  spatial-spectral fusion module. The canonical HSI-Mamba paper.
  [`paper`](https://ieeexplore.ieee.org/document/10604894) [`code`](https://github.com/li-yapeng/MambaHSI) `TGRS'24` `classification` `165★`
- **3DSS-Mamba** — Pixel-wise 3D selective scanning across spectral *and* spatial
  axes, operating on tokens from a spectral-spatial token generator.
  [`paper`](https://arxiv.org/abs/2405.12487) [`code`](https://github.com/IIP-Team/3DSS-Mamba) `TGRS'24` `classification` `18★`
- **IGroupSS-Mamba** — Interval grouping of spectral bands with group-wise
  multi-directional scanning, cutting the redundancy of scanning all bands jointly.
  [`paper`](https://arxiv.org/abs/2410.05100) [`code`](https://github.com/IIP-Team/IGroupSS-Mamba) `TGRS'24` `classification` `23★`
- **HyperMamba** — Spectral-adaptive state transition conditioning the SSM
  parameters on local spectral statistics.
  [`paper`](https://ieeexplore.ieee.org/document/10720896) [`code`](https://github.com/chiangliu/HyperMamba) `TGRS'24` `classification` `39★`
- **GraphMamba** — Learns a graph over superpixel nodes and orders the Mamba scan
  along graph structure rather than a raster path.
  [`paper`](https://ieeexplore.ieee.org/document/10746459) [`code`](https://github.com/ahappyyang/GraphMamba) `TGRS'24` `classification` `36★`
- **MambaLG** — Local-global dual-scan coupling a local patch scan with a global
  scene scan in one state-space encoder.
  [`paper`](https://ieeexplore.ieee.org/document/10812905) [`code`](https://github.com/danfenghong/IEEE_TGRS_MambaLG) `TGRS'24` `classification` `57★`
- **DualMamba** — Parallel lightweight Mamba and convolution branches with dynamic
  gated fusion, for a very small parameter budget.
  [`paper`](https://ieeexplore.ieee.org/document/10798573) `TGRS'24` `classification, lightweight` `no code`
- **S²Mamba** — Two parallel selective scans — patch-cross-scan for space,
  bi-directional scan for spectrum — merged by a learnable spatial-spectral mixture
  gate.
  [`paper`](https://ieeexplore.ieee.org/document/10844849) [`code`](https://github.com/PURE-melo/S2Mamba) `TGRS'25` `classification` `64★`
- **STMamba** — Replaces raster patches with learned semantic tokens before
  state-space modelling, shortening the sequence the SSM has to traverse.
  [`paper`](https://ieeexplore.ieee.org/document/10838328) [`code`](https://github.com/AlanLowell/STMamba) `JSTARS'25` `classification` `5★`
- **MambaHSI+** — Simplifies MambaHSI into a multidirectional state-propagation
  scheme with higher accuracy at lower cost.
  [`paper`](https://ieeexplore.ieee.org/document/11023867) [`code`](https://github.com/RockAilab/MambaHSI_Plus) `TGRS'25` `classification, efficient` `19★`

### Hyperspectral foundation models

The most consequential recent shift. Until 2023 there was no meaningful pretraining
story for HSI — everything was trained from scratch on a few hundred labelled
pixels. These models change what the starting point looks like.

- **SpectralGPT** — 3D spatial-spectral token generation with multi-target
  reconstruction; over 600M parameters trained on ~1M spectral images. The first
  large spectral-native foundation model.
  [`paper`](https://arxiv.org/abs/2311.07113) [`code`](https://github.com/danfenghong/IEEE_TPAMI_SpectralGPT) `TPAMI'24` `foundation model` `282★`
- **HyperSIGMA** — First billion-parameter HSI foundation model, with separate
  spatial and spectral MAEs and sparse sampling attention to counter
  spatial-spectral redundancy. Evaluated across 16 datasets and 7 tasks.
  [`paper`](https://arxiv.org/abs/2406.11519) [`code`](https://github.com/WHU-Sigma/HyperSIGMA) `TPAMI'25` `foundation model` `390★`
- **DOFA** — Not HSI-specific, but its wavelength-conditioned hypernetwork accepts
  hyperspectral input directly alongside other sensors, which makes it a useful
  cross-sensor baseline. See [Foundation Models](#multi-modal-foundation-models).
  [`paper`](https://arxiv.org/abs/2403.15356) [`code`](https://github.com/zhu-xlab/DOFA) `preprint` `multimodal` `213★`

### Fusion, detection, denoising & restoration

- **SSUMamba** — Alternating spatial-spectral continuous scanning inside a U-shaped
  SSM so noise modelling sees full 3D context at linear cost.
  [`paper`](https://arxiv.org/abs/2405.01726) [`code`](https://github.com/lronkitty/SSUMamba) `TGRS'24` `denoising` `57★`
- **HLMamba** — Dual-stream Mamba with a cross-modal state-space fusion block
  bridging HSI spectra and LiDAR elevation.
  [`paper`](https://ieeexplore.ieee.org/document/10679212) [`code`](https://github.com/Dilingliao/HLMamba) `TGRS'24` `HSI+LiDAR fusion` `23★`
- **FusionMamba** — Plug-and-play dual-input Mamba block for arbitrary two-source
  fusion, covering hyperspectral pansharpening and HSI-MSI fusion. See
  [Mamba](#fusion-super-resolution--detection).
  [`paper`](https://arxiv.org/abs/2404.07932) [`code`](https://github.com/PSRben/FusionMamba) `TGRS'24` `fusion` `139★`
- **HTD-Mamba** — Self-supervised spectrally contrastive learning over group-wise
  spectral embeddings with a pyramid SSM backbone, for target detection.
  ESI Highly Cited.
  [`paper`](https://arxiv.org/abs/2407.06841) [`code`](https://github.com/shendb2022/HTD-Mamba) `TGRS'25` `target detection` `51★`
- **MSFMamba** — Three-block design (multi-scale spatial Mamba, spectral Mamba,
  dual-input fusion Mamba) extending Mamba to two heterogeneous sources.
  [`paper`](https://arxiv.org/abs/2408.14255) [`code`](https://github.com/oucailab/MSFMamba) `TGRS'25` `HSI+LiDAR/SAR fusion` `51★`

### Hyperspectral datasets

**The classic scenes.** Small, single-scene, and near-saturated. Still the default
comparison set, which is a problem — see the note below.

| Dataset | Sensor | Bands | Classes | Notes |
|---|---|---|---|---|
| Indian Pines | AVIRIS | 224 (200 after water-band removal) | 16 | 145×145. Heavily imbalanced; the most over-used benchmark in the field. |
| Pavia University | ROSIS | 103 | 9 | 610×340. Urban, high spatial detail. |
| Pavia Centre | ROSIS | 102 | 9 | Companion scene to Pavia University. |
| Salinas | AVIRIS | 224 (204 usable) | 16 | 512×217. Agricultural, spectrally clean, very high reported accuracies. |
| Kennedy Space Center | AVIRIS | 176 usable | 13 | Wetland vegetation. |
| Botswana | Hyperion (EO-1) | 145 usable | 14 | Spaceborne rather than airborne. |
| Houston 2013 | CASI | 144 | 15 | 349×1905. IEEE GRSS Data Fusion Contest; has an official train/test split. |
| Houston 2018 | CASI + LiDAR | 48 | 20 | GRSS DFC 2018; multimodal. |
| WHU-Hi (Longkou / Hanchuan / Honghu) | UAV-borne | 270-274 | 9-22 | Low-altitude UAV, fine crop classes; more realistic than the AVIRIS scenes. |

**Modern large-scale corpora.** These are what pretraining actually needs, and the
reason foundation models became possible for HSI.

- **HySpecNet-11k** — EnMAP spaceborne patches at 202 bands, built specifically as
  a large-scale corpus for learned compression and pretraining rather than
  pixel classification.
- **SpectralEarth** — Large EnMAP-derived corpus assembled for hyperspectral
  foundation model pretraining, with global geographic coverage.
- **HyperGlobal-450K** — The pretraining corpus behind HyperSIGMA; global, and by a
  wide margin the largest used for an HSI foundation model to date.
- **EnMAP / PRISMA / EMIT archives** — Operational spaceborne imaging spectrometer
  missions. The practical source of at-scale modern HSI data, and where new
  benchmarks should be coming from.
- **ARAD-1K, ICVL, CAVE, Harvard** — Natural-scene hyperspectral sets used mainly
  for spectral reconstruction from RGB, not Earth observation classification.

### A note on HSI benchmarking

Worth reading before you trust any accuracy number in this sub-field.

The standard protocol on Indian Pines, Pavia and Salinas draws training and test
pixels **randomly from the same scene**. Because neighbouring pixels are spatially
correlated and most architectures consume a spatial patch around the target pixel,
training patches physically overlap test pixels. The result is information leakage,
and it inflates reported accuracy substantially.

This is why nearly every paper on these scenes reports 98-99%+ overall accuracy,
why differences between methods are often within noise, and why gains frequently
fail to transfer to new scenes. The literature on **disjoint sampling** and spatially
separated train/test splits addresses this directly, and the Houston 2013 benchmark
is more trustworthy precisely because it ships a fixed, spatially disjoint split.

Practical guidance:

- Treat single-scene random-split results as a sanity check, not evidence.
- Prefer benchmarks with official disjoint splits, or construct spatially separated
  folds yourself.
- For any claim about generalisation, evaluate cross-scene or cross-sensor —
  the domain adaptation literature exists because this gap is large and real.
- When comparing to published numbers, confirm the sampling protocol matches
  before concluding anything. It frequently does not.

---

## Vision-Language Models & Multimodal LLMs

> **Partial section.** See [Coverage status](#coverage-status) — entries here are
> verified, but this family is under-covered relative to its current activity.

- **GeoChat** — The first grounded large vision-language model for remote sensing,
  supporting region-level grounding, referring detection and visually grounded
  conversation rather than whole-image captioning alone.
  [`paper`](https://openaccess.thecvf.com/content/CVPR2024/html/Kuckreja_GeoChat_Grounded_Large_Vision-Language_Model_for_Remote_Sensing_CVPR_2024_paper.html) [`code`](https://github.com/mbzuai-oryx/GeoChat) `CVPR'24` `MLLM, grounding`
- **EarthGPT** — Universal multimodal LLM for multi-sensor comprehension, spanning
  optical, SAR and infrared in a single instruction-following model.
  [`paper`](https://ieeexplore.ieee.org/document/10547418) `TGRS'24` `MLLM, multi-sensor`
- **LHRS-Bot** — Uses volunteered geographic information (OpenStreetMap) paired with
  imagery to build instruction data at scale, rather than relying on human
  annotation.
  [`paper`](https://arxiv.org/abs/2402.02544) [`code`](https://github.com/NJU-LHRS/LHRS-Bot) `ECCV'24` `MLLM, instruction tuning`
- **RSGPT** — Remote sensing vision-language model released alongside a
  human-annotated captioning and VQA benchmark.
  [`paper`](https://arxiv.org/abs/2307.15266) [`code`](https://github.com/Lavender105/RSGPT) `ISPRS J.'25` `MLLM, captioning, VQA`
- **RMSIN** — Intra-scale and cross-scale interaction with adaptive rotated
  convolution for *referring* segmentation, where the query is free text. Introduces
  the RRSIS-D benchmark.
  [`paper`](https://arxiv.org/abs/2312.12470) [`code`](https://github.com/Lsan2401/RMSIN) `CVPR'24` `referring segmentation`
- **RSPrompter** — Learns category-aware prompts that drive a frozen SAM decoder,
  converting an interactive foundation model into an automatic instance segmenter.
  [`paper`](https://arxiv.org/abs/2306.16269) [`code`](https://github.com/KyanChen/RSPrompter) `TGRS'24` `instance segmentation, SAM`
- **AnyChange** — Training-free bi-temporal latent matching in SAM's latent space,
  giving zero-shot change detection with no change-detection training at all.
  [`paper`](https://arxiv.org/abs/2402.01188) [`code`](https://github.com/Z-Zheng/pytorch-change-models) `NeurIPS'24` `zero-shot CD, SAM`
- **BAN** — Freezes a CLIP or SAM backbone and bridges it to any existing change
  detection head through a bi-temporal adapter, with few trainable parameters.
  [`paper`](https://arxiv.org/abs/2312.01163) [`code`](https://github.com/likyoo/BAN) `TGRS'24` `change detection, PEFT`
- **RSCaMa** — State-space model for change *captioning*, combining a
  spatial-difference SSM with a temporal-traversing SSM.
  [`paper`](https://arxiv.org/abs/2404.18895) [`code`](https://github.com/Chen-Yang-Liu/RSCaMa) `GRSL'24` `change captioning`

---

## Diffusion & Generative Models

> **Partial section.** See [Coverage status](#coverage-status).

- **DiffusionSat** — Conditions generation on numerical satellite metadata
  (location, GSD, timestamp) alongside text, with conditioning modules trainable
  for super-resolution, inpainting and temporal generation.
  [`paper`](https://proceedings.iclr.cc/paper_files/paper/2024/file/16c3c941409d0581286eff49b180930f-Paper-Conference.pdf) [`code`](https://github.com/samar-khanna/DiffusionSat) `ICLR'24` `generation, super-resolution`
- **CRS-Diff** — Supports text, metadata and image conditioning simultaneously,
  giving ControlNet-style fine-grained control over remote sensing generation.
  [`paper`](https://arxiv.org/abs/2403.11614) [`code`](https://github.com/Sonettoo/CRS-Diff) `preprint` `controllable generation`
- **MetaEarth** — Resolution-conditioned generative model aimed at global-scale
  image generation across zoom levels rather than single-tile synthesis.
  [`paper`](https://arxiv.org/abs/2405.13570) `preprint` `generation`
- **Changen2** — Generative change process model that synthesises labelled
  multi-temporal sequences, producing pretrained weights with zero-shot change
  detection capability. Cross-listed from
  [Vision Transformers](#change-detection).
  [`paper`](https://arxiv.org/abs/2406.17998) [`code`](https://github.com/Z-Zheng/pytorch-change-models) `TPAMI'24` `generative CD`
- **TerraMind** — Any-to-any generative multimodal model that synthesises missing
  modalities as an intermediate reasoning step. Cross-listed from
  [Foundation Models](#multi-modal-foundation-models).
  [`paper`](https://arxiv.org/abs/2504.11171) [`code`](https://github.com/IBM/terramind) `ICCV'25` `generative, multimodal`

---

## Libraries & Tooling

Frameworks that make the models above practical to train, fine-tune and deploy.
Star counts are deliberately omitted here pending verification.

| Library | What it is |
|---|---|
| [TorchGeo](https://github.com/microsoft/torchgeo) | PyTorch datasets, samplers, transforms and pretrained weights for geospatial data. The most common starting point. |
| [TerraTorch](https://github.com/IBM/terratorch) | IBM's fine-tuning toolkit for geospatial foundation models; the reference path for Prithvi and TerraMind. |
| [Raster Vision](https://github.com/azavea/raster-vision) | End-to-end pipeline framework for chipping, training and prediction on large rasters. |
| [GeoSeg](https://github.com/WangLibo1995/GeoSeg) | Segmentation model zoo built around UNetFormer and DC-Swin; a de-facto RS segmentation baseline suite. |
| [Open-CD](https://github.com/likyoo/open-cd) | Change detection toolbox on MMSegmentation, home of Changer and BAN. |
| [torchange](https://github.com/Z-Zheng/pytorch-change-models) | Change model library covering ChangeStar, Changen2 and AnyChange. |
| [MMRotate](https://github.com/open-mmlab/mmrotate) | OpenMMLab toolbox for rotated object detection; most oriented detectors here ship as MMRotate configs. |
| [segment-geospatial](https://github.com/opengeos/segment-geospatial) | SAM applied to geospatial rasters with a practical, notebook-friendly API. |
| [eo-learn](https://github.com/sentinel-hub/eo-learn) | Earth observation workflow framework bridging satellite archives and ML pipelines. |
| [DeepForest](https://github.com/weecology/DeepForest) | Tree crown detection from airborne imagery, with pretrained ecological models. |
| [GEO-Bench](https://github.com/ServiceNow/geo-bench) | Standardised foundation model evaluation suite. |
| [PANGAEA](https://github.com/VMarsocci/pangaea-bench) | Geographically inclusive GFM benchmark across 7 task types. |

---

## Task Index

The list is organised by architecture family. This maps back the other way.

| Task | Where to look |
|---|---|
| **Scene classification** | [Mamba backbones](#backbones--classification) · [MIM pretraining](#masked-image-modeling) · [HSI classification](#transformers-for-hsi-classification) |
| **Semantic segmentation** | [ViT segmentation](#semantic-segmentation) · [Mamba dense prediction](#dense-prediction--segmentation) · [DeepKANSeg](#efficient-architectures) |
| **Change detection** | [ViT change detection](#change-detection) · [Mamba change detection](#change-detection-1) · [AnyChange / BAN](#vision-language-models--multimodal-llms) |
| **Object detection (oriented)** | [Oriented object detection](#oriented-object-detection) |
| **Instance segmentation** | [RSPrompter](#semantic-segmentation) · [SAMRS](#segmentation-datasets-generated-by-foundation-models) |
| **Super-resolution** | [FreMamba](#fusion-super-resolution--detection) · [DiffusionSat](#diffusion--generative-models) |
| **Pansharpening / fusion** | [FusionMamba, Pan-Mamba, SDMSPan](#fusion-super-resolution--detection) · [HSI fusion](#fusion-detection-denoising--restoration) |
| **Image generation** | [Diffusion & generative](#diffusion--generative-models) |
| **Captioning / VQA / chat** | [Vision-language models](#vision-language-models--multimodal-llms) |
| **Referring segmentation** | [RMSIN](#vision-language-models--multimodal-llms) |
| **Hyperspectral (all tasks)** | [Hyperspectral](#hyperspectral) |
| **SAR** | [SARATR-X](#agency--open-release-models) · [SARDet-100K](#oriented-object-detection) · [CROMA](#contrastive--geo-aware-self-supervision) |
| **Denoising / restoration** | [SSUMamba](#fusion-detection-denoising--restoration) · [FMambaIR](#fusion-super-resolution--detection) |
| **Target / small object detection** | [MiM-ISTD](#fusion-super-resolution--detection) · [HTD-Mamba](#fusion-detection-denoising--restoration) |
| **Tracking** | [TrackingMamba](#fusion-super-resolution--detection) |
| **Multi-task** | [MTP](#plain-vit-backbones--parameter-scaling) · [RSMTMamba](#dense-prediction--segmentation) |

---

## Coverage status

This list is under active construction. Being explicit about what is and is not
covered is more useful than implying uniform depth.

| Section | Status |
|---|---|
| Foundation models & SSL | **Complete** — venues and code links verified |
| Vision transformers | **Complete** — venues and code links verified |
| Mamba & state-space | **Complete** — venues and code links verified |
| Hyperspectral | **Complete** — the deepest section, with dataset guidance |
| Efficient architectures | **Complete** |
| Vision-language models | **Partial** — verified entries only; many recent models not yet indexed |
| Diffusion & generative | **Partial** — verified entries only |
| SAR | **Planned** — currently only cross-references; despeckling, ATR and InSAR missing |
| 3D, LiDAR & neural fields | **Planned** — Sat-NeRF, Gaussian splatting, height estimation missing |
| Satellite image time series | **Planned** — U-TAE, TSViT, Presto and crop mapping missing |
| Surveys | **Planned** |
| Benchmarks & datasets | **Partial** — hyperspectral only; segmentation, CD, detection and SITS benchmarks missing |

Planned sections are genuinely absent rather than thin — they are not yet
researched to the verification standard the rest of the list meets. Contributions
toward them are especially welcome.

### On verification

Every venue in the completed sections was checked against a primary source
(publisher page, proceedings listing, or the arXiv comments field), and every code
link was confirmed to resolve. Star counts were captured at indexing time and
drift; treat them as indicative.

Where a venue could not be confirmed, the entry is marked `preprint` rather than
given a plausible-looking guess. A wrong citation in a list like this propagates
into other people's bibliographies, so absence is preferred to invention.

### Related lists

This list is deliberately narrow — architectures, recent, top venues. These cover
adjacent ground well:

- [satellite-image-deep-learning/techniques](https://github.com/satellite-image-deep-learning/techniques) — the broadest applied catalogue of techniques and tooling.
- [satellite-image-deep-learning/datasets](https://github.com/satellite-image-deep-learning/datasets) — the exhaustive dataset catalogue.
- [awesome-remote-sensing-change-detection](https://github.com/wenhwu/awesome-remote-sensing-change-detection) — deep coverage of change detection specifically.
- [Awesome-Mamba-in-Remote-Sensing](https://github.com/BaoBao0926/Awesome-Mamba-in-Remote-Sensing) — companion to the Vision Mamba in RS survey.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Corrections to venues, years, links and
attributions are the most valuable contribution — please open an issue or PR.

## License

[CC0-1.0](LICENSE). The curation is public domain; linked papers, datasets and
code remain under their own licences.
