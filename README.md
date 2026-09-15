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
  [`paper`](…) [`code`](…) `TGRS'24` `segmentation` [![Stars](…)](…)
```

Star badges are generated from the code link, so never type a star count by hand
— a number written into the file is wrong the moment the repository changes.

| Marker | Meaning |
|---|---|
| ![Stars](https://img.shields.io/github/stars/KyanChen/RSMamba?style=social) | Live GitHub star count, updated automatically. Indicates adoption, not quality. |
| `no code` | No public implementation found. Included on significance alone. |
| `preprint` | Not peer-reviewed at a listed venue, but widely adopted **and** ships code or weights. |
| `weights: gated` | Checkpoints exist but require a request, or are non-commercial only. |
| ⚠ | Venue sits outside the policy below; kept because the work is canonical for its sub-area. |
| † | Venue reported by a secondary source; not yet confirmed against a primary source. |

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
  [`paper`](https://arxiv.org/abs/2207.08051) [`code`](https://github.com/sustainlab-group/SatMAE) `NeurIPS'22` `pretraining` [![Stars](https://img.shields.io/github/stars/sustainlab-group/SatMAE?style=social)](https://github.com/sustainlab-group/SatMAE/stargazers)
- **Scale-MAE** — Conditions ViT positional encoding on ground sample distance and
  decodes through a Laplacian-pyramid bandpass decoder, making representations
  explicitly scale-aware rather than resolution-agnostic.
  [`paper`](https://arxiv.org/abs/2212.14532) [`code`](https://github.com/bair-climate-initiative/scale-mae) `ICCV'23` `pretraining` [![Stars](https://img.shields.io/github/stars/bair-climate-initiative/scale-mae?style=social)](https://github.com/bair-climate-initiative/scale-mae/stargazers)
- **GFM** — Continual pretraining with a frozen ImageNet-22k teacher distilled
  alongside in-domain masked image modeling, matching large-scale in-domain
  pretraining on a compact 600K corpus.
  [`paper`](https://arxiv.org/abs/2302.04476) [`code`](https://github.com/mmendiet/GFM) `ICCV'23` `pretraining` [![Stars](https://img.shields.io/github/stars/mmendiet/GFM?style=social)](https://github.com/mmendiet/GFM/stargazers)
- **Cross-Scale MAE** — Scale augmentation with cross-scale consistency
  constraints, removing the need for aligned multi-GSD image pairs at pretraining
  time.
  [`paper`](https://arxiv.org/abs/2401.15855) [`code`](https://github.com/aicip/Cross-Scale-MAE) `NeurIPS'23` `pretraining` [![Stars](https://img.shields.io/github/stars/aicip/Cross-Scale-MAE?style=social)](https://github.com/aicip/Cross-Scale-MAE/stargazers)
- **SatMAE++** — Adds multi-scale hierarchical reconstruction with convolutional
  upsampling, exploiting the scale information that SatMAE discards.
  [`paper`](https://arxiv.org/abs/2403.05419) [`code`](https://github.com/techmn/satmae_pp) `CVPR'24` `pretraining` [![Stars](https://img.shields.io/github/stars/techmn/satmae_pp?style=social)](https://github.com/techmn/satmae_pp/stargazers)
- **SelectiveMAE** — Progressive semantic token selection encodes and reconstructs
  only semantically dense patches, skipping redundant background for a 2.2-2.7×
  pretraining speedup. Ships the OpticalRS-13M corpus.
  [`paper`](https://arxiv.org/abs/2406.11933) [`code`](https://github.com/MiliLab/SelectiveMAE) `ICCV'25` `pretraining` [![Stars](https://img.shields.io/github/stars/MiliLab/SelectiveMAE?style=social)](https://github.com/MiliLab/SelectiveMAE/stargazers)

### Contrastive & geo-aware self-supervision

- **SeCo** — Treats seasonal variation at a fixed location as a free augmentation
  axis, with multi-head embeddings that separate season-invariant from
  season-varying factors.
  [`paper`](https://arxiv.org/abs/2103.16607) [`code`](https://github.com/ServiceNow/seasonal-contrast) `ICCV'21` `pretraining` [![Stars](https://img.shields.io/github/stars/ServiceNow/seasonal-contrast?style=social)](https://github.com/ServiceNow/seasonal-contrast/stargazers)
- **GASSL** — Adds temporal positive pairs and a geo-location prediction pretext
  task to MoCo-v2, first closing the contrastive-versus-supervised gap on RS
  benchmarks.
  [`paper`](https://openaccess.thecvf.com/content/ICCV2021/papers/Ayush_Geography-Aware_Self-Supervised_Learning_ICCV_2021_paper.pdf) [`code`](https://github.com/sustainlab-group/geography-aware-ssl) `ICCV'21` `pretraining` [![Stars](https://img.shields.io/github/stars/sustainlab-group/geography-aware-ssl?style=social)](https://github.com/sustainlab-group/geography-aware-ssl/stargazers)
- **CACo** — Contrasts long-term against short-term temporal differences so the
  objective learns what *change* looks like, with change-driven geographic
  sampling.
  [`paper`](https://openaccess.thecvf.com/content/CVPR2023/html/Mall_Change-Aware_Sampling_and_Contrastive_Learning_for_Satellite_Images_CVPR_2023_paper.html) [`code`](https://github.com/utkarshmall13/CACo) `CVPR'23` `pretraining` [![Stars](https://img.shields.io/github/stars/utkarshmall13/CACo?style=social)](https://github.com/utkarshmall13/CACo/stargazers)
- **CROMA** — Joint contrastive alignment and masked reconstruction over
  co-registered SAR/optical pairs, with X- and 2D-ALiBi spatial biases replacing
  learned positional encodings.
  [`paper`](https://arxiv.org/abs/2311.00566) [`code`](https://github.com/antofuller/CROMA) `NeurIPS'23` `multimodal pretraining` [![Stars](https://img.shields.io/github/stars/antofuller/CROMA?style=social)](https://github.com/antofuller/CROMA/stargazers)

### Multi-modal foundation models

The most active frontier. The design problem is no longer "how do we pretrain on
satellite data" but "how does one model accept optical, SAR, elevation and
climate inputs — including sensors it never saw during training".

- **SkySense** — Billion-parameter factorised multi-modal spatiotemporal encoder
  over high-resolution optical plus Sentinel-1/2 time series, trained with
  multi-granularity contrastive learning and geo-context prototypes.
  [`paper`](https://arxiv.org/abs/2312.10115) [`code`](https://github.com/Jack-bo1220/SkySense) `CVPR'24` `multimodal` `weights: gated` [![Stars](https://img.shields.io/github/stars/Jack-bo1220/SkySense?style=social)](https://github.com/Jack-bo1220/SkySense/stargazers)
- **msGFM** — Cross-sensor masked image modeling that handles paired *and*
  unpaired sensor data, covering RGB, Sentinel-2, SAR and DSM.
  [`paper`](https://arxiv.org/abs/2404.01260) [`code`](https://github.com/boranhan/Geospatial_Foundation_Models) `CVPR'24` `multimodal` `weights: no` [![Stars](https://img.shields.io/github/stars/boranhan/Geospatial_Foundation_Models?style=social)](https://github.com/boranhan/Geospatial_Foundation_Models/stargazers)
- **OmniSat** — Exploits the natural spatial alignment between EO modalities for
  label-free fusion of VHR aerial, Sentinel-1 and Sentinel-2 time series. Also
  contributes TreeSatAI-TS and PASTIS-HD.
  [`paper`](https://arxiv.org/abs/2404.08351) [`code`](https://github.com/gastruc/OmniSat) `ECCV'24` `multimodal` [![Stars](https://img.shields.io/github/stars/gastruc/OmniSat?style=social)](https://github.com/gastruc/OmniSat/stargazers)
- **DOFA** — A wavelength-conditioned dynamic hypernetwork generates patch-embedding
  weights, so a single ViT serves any sensor, including ones absent from
  pretraining. Neural-plasticity framing.
  [`paper`](https://arxiv.org/abs/2403.15356) [`code`](https://github.com/zhu-xlab/DOFA) `preprint` `multimodal` [![Stars](https://img.shields.io/github/stars/zhu-xlab/DOFA?style=social)](https://github.com/zhu-xlab/DOFA/stargazers)
- **Galileo** — Dual global and local self-supervised objectives with different
  masking strategies across nine modalities, letting an 85M generalist beat
  specialist SOTA on 11 benchmarks. Nano variant is 0.8M parameters.
  [`paper`](https://arxiv.org/abs/2502.09356) [`code`](https://github.com/nasaharvest/galileo) `ICML'25` `multimodal` [![Stars](https://img.shields.io/github/stars/nasaharvest/galileo?style=social)](https://github.com/nasaharvest/galileo/stargazers)
- **AnySat** — JEPA with resolution-adaptive spatial encoders sharing 75% of
  parameters across 0.2m-500m GSD, 3-12 channels and 11 sensors, trained jointly
  on datasets with incompatible geometry.
  [`paper`](https://arxiv.org/abs/2412.14123) [`code`](https://github.com/gastruc/AnySat) `CVPR'25` `multimodal` [![Stars](https://img.shields.io/github/stars/gastruc/AnySat?style=social)](https://github.com/gastruc/AnySat/stargazers)
- **Copernicus-FM** — Extends dynamic hypernetworks to *non-spectral* sensors and
  adds flexible metadata encoding, covering surface-to-atmosphere Copernicus data.
  [`paper`](https://arxiv.org/abs/2503.11849) [`code`](https://github.com/zhu-xlab/Copernicus-FM) `ICCV'25` `multimodal` [![Stars](https://img.shields.io/github/stars/zhu-xlab/Copernicus-FM?style=social)](https://github.com/zhu-xlab/Copernicus-FM/stargazers)
- **TerraMind** — First any-to-any *generative* EO foundation model, with
  "Thinking-in-Modalities" inference that synthesises missing modalities as
  intermediate reasoning steps. Apache-2.0 across four sizes.
  [`paper`](https://arxiv.org/abs/2504.11171) [`code`](https://github.com/IBM/terramind) `ICCV'25` `multimodal` [![Stars](https://img.shields.io/github/stars/IBM/terramind?style=social)](https://github.com/IBM/terramind/stargazers)
- **SkySense V2** — Replaces per-modality backbones with one shared transformer
  using adaptive patch merging, modality prompt tokens and mixture-of-experts.
  [`paper`](https://arxiv.org/abs/2507.13812) `ICCV'25` `multimodal` `weights: no`
- **SkySense++** — Two-stage progressive pretraining, contrastive then masked
  *semantic* learning, giving strong few-shot transfer across 12 EO tasks in 7
  application domains.
  [`paper`](https://doi.org/10.1038/s42256-025-01078-8) [`code`](https://github.com/kang-wu/SkySensePlusPlus) `Nat. Mach. Intell.'25` `multimodal` `weights: gated` [![Stars](https://img.shields.io/github/stars/kang-wu/SkySensePlusPlus?style=social)](https://github.com/kang-wu/SkySensePlusPlus/stargazers)

### Plain-ViT backbones & parameter scaling

- **RSP** — First systematic study of pretraining CNN, Swin and ViTAEv2 backbones
  on MillionAID rather than ImageNet, establishing RS-domain weights as a standard
  baseline.
  [`paper`](https://arxiv.org/abs/2204.02825) [`code`](https://github.com/ViTAE-Transformer/RSP) `TGRS'22` `pretraining` [![Stars](https://img.shields.io/github/stars/ViTAE-Transformer/RSP?style=social)](https://github.com/ViTAE-Transformer/RSP/stargazers)
- **RingMo** — The first RS-specific masked image modeling foundation model, with a
  masking strategy that avoids destroying dense small objects during
  reconstruction.
  [`paper`](https://doi.org/10.1109/TGRS.2022.3194732) `TGRS'22` `pretraining` `no code`
- **RVSA** — Replaces full attention in a ~100M plain ViT with rotated varied-size
  window attention, giving orientation-aware attention at reduced cost.
  [`paper`](https://arxiv.org/abs/2208.03987) [`code`](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA) `TGRS'23` `backbone` [![Stars](https://img.shields.io/github/stars/ViTAE-Transformer/Remote-Sensing-RVSA?style=social)](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA/stargazers)
- **SatlasPretrain** — Large-scale *supervised* multi-task pretraining corpus
  (856K tiles, 302M labels, 137 categories) across Sentinel-1/2, Landsat and
  aerial imagery.
  [`paper`](https://arxiv.org/abs/2211.15660) [`code`](https://github.com/allenai/satlas) `ICCV'23` `pretraining` [![Stars](https://img.shields.io/github/stars/allenai/satlas?style=social)](https://github.com/allenai/satlas/stargazers)
- **MTP** — Supervised multi-task pretraining over semantic segmentation, instance
  segmentation and rotated detection simultaneously, closing the
  pretext-to-downstream task gap.
  [`paper`](https://arxiv.org/abs/2403.13430) [`code`](https://github.com/ViTAE-Transformer/MTP) `JSTARS'24` `pretraining` [![Stars](https://img.shields.io/github/stars/ViTAE-Transformer/MTP?style=social)](https://github.com/ViTAE-Transformer/MTP/stargazers)
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
  [`paper`](https://arxiv.org/abs/2412.02732) [`code`](https://github.com/NASA-IMPACT/Prithvi-EO-2.0) `preprint` `Apache-2.0` [![Stars](https://img.shields.io/github/stars/NASA-IMPACT/Prithvi-EO-2.0?style=social)](https://github.com/NASA-IMPACT/Prithvi-EO-2.0/stargazers)
- **Clay** — Community-governed MAE with a DINOv2 teacher and a dynamic embedding
  block that adapts to arbitrary sensor band sets. 70M global chips, no paper.
  [`docs`](https://clay-foundation.github.io/model/) [`code`](https://github.com/Clay-foundation/model) `model release` `Apache-2.0` [![Stars](https://img.shields.io/github/stars/Clay-foundation/model?style=social)](https://github.com/Clay-foundation/model/stargazers)
- **SSL4EO-S12** — The de-facto standard global SSL pretraining corpus for
  Sentinel-1/2 (251K locations × 4 seasons, 1.5 TB), shipped with reference
  weights for four SSL methods.
  [`paper`](https://arxiv.org/abs/2211.07044) [`code`](https://github.com/zhu-xlab/SSL4EO-S12) `IEEE GRSM'23†` `corpus` [![Stars](https://img.shields.io/github/stars/zhu-xlab/SSL4EO-S12?style=social)](https://github.com/zhu-xlab/SSL4EO-S12/stargazers)
- **Major TOM** — A grid and metadata standard that makes terabyte-scale EO
  datasets interoperable and mergeable; now the default distribution format on
  Hugging Face. ⚠ `IGARSS`
  [`paper`](https://arxiv.org/abs/2402.12095) [`code`](https://github.com/ESA-PhiLab/Major-TOM) `corpus` [![Stars](https://img.shields.io/github/stars/ESA-PhiLab/Major-TOM?style=social)](https://github.com/ESA-PhiLab/Major-TOM/stargazers)
- **AlphaEarth Foundations** — Continuous "embedding field" formulation producing
  analysis-ready global annual 64-D embeddings at 10m, 2017-2024. Embeddings are
  public via Earth Engine; the model is not.
  [`paper`](https://arxiv.org/abs/2507.22291) `preprint` `weights: no`
- **SARATR-X** — First SAR target recognition foundation model: two-step
  self-supervision with multi-scale gradient features, competitive with fully
  supervised baselines in few-shot settings.
  [`paper`](https://arxiv.org/abs/2405.09365) [`code`](https://github.com/waterdisappear/SARATR-X) `IEEE TIP'25` `SAR` [![Stars](https://img.shields.io/github/stars/waterdisappear/SARATR-X?style=social)](https://github.com/waterdisappear/SARATR-X/stargazers)
- **HyperSIGMA** — First billion-parameter hyperspectral foundation model, using
  sparse sampling attention to counter spatial-spectral redundancy. See
  [Hyperspectral](#hyperspectral).
  [`paper`](https://arxiv.org/abs/2406.11519) [`code`](https://github.com/WHU-Sigma/HyperSIGMA) `TPAMI'25` `hyperspectral` [![Stars](https://img.shields.io/github/stars/WHU-Sigma/HyperSIGMA?style=social)](https://github.com/WHU-Sigma/HyperSIGMA/stargazers)

### Benchmarks for foundation models

Evaluating a GFM is genuinely hard, and early comparisons were unreliable. These
three are the current standards.

- **GEO-Bench** — The first standardised expert-curated GFM evaluation suite:
  6 classification and 6 segmentation tasks with a defined aggregation
  methodology. 20 models benchmarked.
  [`paper`](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a0644215d9cff6646fa334dfa5d29c5a-Abstract-Datasets_and_Benchmarks.html) [`code`](https://github.com/ServiceNow/geo-bench) `NeurIPS'23` [![Stars](https://img.shields.io/github/stars/ServiceNow/geo-bench?style=social)](https://github.com/ServiceNow/geo-bench/stargazers)
- **PANGAEA** — Corrects the North-America/Europe geographic bias and task
  triviality of earlier GFM evaluations, with 12 datasets across 7 task types
  under one protocol.
  [`paper`](https://arxiv.org/abs/2412.04204) [`code`](https://github.com/VMarsocci/pangaea-bench) `preprint` [![Stars](https://img.shields.io/github/stars/VMarsocci/pangaea-bench?style=social)](https://github.com/VMarsocci/pangaea-bench/stargazers)
- **Copernicus-Bench** — Mission-structured benchmark spanning preprocessing
  through specialised applications, 15 hierarchical tasks across three levels.
  [`paper`](https://arxiv.org/abs/2503.11849) [`code`](https://github.com/zhu-xlab/Copernicus-FM) `ICCV'25` [![Stars](https://img.shields.io/github/stars/zhu-xlab/Copernicus-FM?style=social)](https://github.com/zhu-xlab/Copernicus-FM/stargazers)

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
  [`paper`](https://arxiv.org/abs/2405.09789) [`code`](https://github.com/ViTAE-Transformer/LeMeViT) `IJCAI'24` `backbone` [![Stars](https://img.shields.io/github/stars/ViTAE-Transformer/LeMeViT?style=social)](https://github.com/ViTAE-Transformer/LeMeViT/stargazers)

### Semantic segmentation

- **DC-Swin** — Swin encoder paired with a densely connected feature aggregation
  decoder to restore the multi-scale spatial detail that window attention loses.
  [`paper`](https://arxiv.org/abs/2104.12137) [`code`](https://github.com/WangLibo1995/GeoSeg) `GRSL'22` `segmentation` [![Stars](https://img.shields.io/github/stars/WangLibo1995/GeoSeg?style=social)](https://github.com/WangLibo1995/GeoSeg/stargazers)
- **UNetFormer** — ResNet18 encoder with a transformer decoder built on an
  efficient global-local attention block, reaching 322 FPS at competitive mIoU.
  The canonical RS hybrid, and GeoSeg is the de-facto RS segmentation model zoo.
  [`paper`](https://arxiv.org/abs/2109.08937) [`code`](https://github.com/WangLibo1995/GeoSeg) `ISPRS J.'22` `segmentation` [![Stars](https://img.shields.io/github/stars/WangLibo1995/GeoSeg?style=social)](https://github.com/WangLibo1995/GeoSeg/stargazers)
- **BuildFormer** — Dual-path window transformer keeping a high-resolution spatial
  branch alongside the global context branch, for sharp building boundaries.
  [`paper`](https://ieeexplore.ieee.org/document/9808187) [`code`](https://github.com/WangLibo1995/BuildFormer) `TGRS'22` `building extraction` [![Stars](https://img.shields.io/github/stars/WangLibo1995/BuildFormer?style=social)](https://github.com/WangLibo1995/BuildFormer/stargazers)
- **CMTFNet** — Encoder-decoder fusing CNN local features with a multiscale
  multihead self-attention decoder plus channel-wise feature fusion.
  [`paper`](https://ieeexplore.ieee.org/document/10247595) [`code`](https://github.com/DrWuHonglin/CMTFNet) `TGRS'23` `segmentation` [![Stars](https://img.shields.io/github/stars/DrWuHonglin/CMTFNet?style=social)](https://github.com/DrWuHonglin/CMTFNet/stargazers)
- **RSPrompter** — Learns category-aware prompt embeddings that drive a frozen SAM
  decoder, turning SAM into an end-to-end automatic instance segmenter rather than
  an interactive one.
  [`paper`](https://arxiv.org/abs/2306.16269) [`code`](https://github.com/KyanChen/RSPrompter) `TGRS'24` `instance segmentation` [![Stars](https://img.shields.io/github/stars/KyanChen/RSPrompter?style=social)](https://github.com/KyanChen/RSPrompter/stargazers)
- **CrossEarth** — Earth-style injection augmentation combined with multi-task
  DINOv2 backbone adaptation for cross-domain segmentation without target-domain
  data.
  [`paper`](https://arxiv.org/abs/2410.22629) [`code`](https://github.com/Cuzyoung/CrossEarth) `preprint` `domain generalization` [![Stars](https://img.shields.io/github/stars/Cuzyoung/CrossEarth?style=social)](https://github.com/Cuzyoung/CrossEarth/stargazers)

### Change detection

- **BIT** — Expresses each image as a handful of semantic tokens and refines them
  with a small transformer, beating pure-CNN baselines at roughly a third of the
  cost. The reference transformer CD baseline.
  [`paper`](https://arxiv.org/abs/2103.00208) [`code`](https://github.com/justchenhao/BIT_CD) `TGRS'21` `change detection` [![Stars](https://img.shields.io/github/stars/justchenhao/BIT_CD?style=social)](https://github.com/justchenhao/BIT_CD/stargazers)
- **ChangeStar** — A ChangeMixin module converts any single-temporal segmentation
  network into a change detector, removing the need for paired bi-temporal labels
  entirely.
  [`paper`](https://arxiv.org/abs/2108.07002) [`code`](https://github.com/Z-Zheng/ChangeStar) `ICCV'21` `weakly-supervised CD` [![Stars](https://img.shields.io/github/stars/Z-Zheng/ChangeStar?style=social)](https://github.com/Z-Zheng/ChangeStar/stargazers)
- **ICIF-Net** — Parallel CNN and transformer branches communicate at equal
  resolution, then fuse across scales, avoiding the local-global misalignment of
  late fusion.
  [`paper`](https://ieeexplore.ieee.org/document/9759285) [`code`](https://github.com/ZhengJianwei2/ICIF-Net) `TGRS'22` `change detection` [![Stars](https://img.shields.io/github/stars/ZhengJianwei2/ICIF-Net?style=social)](https://github.com/ZhengJianwei2/ICIF-Net/stargazers)
- **TransUNetCD** — UNet skeleton where a transformer encodes tokenized CNN feature
  maps, with skip connections restoring the localisation a pure-transformer encoder
  loses.
  [`paper`](https://ieeexplore.ieee.org/document/9761892) `TGRS'22` `change detection` `no code`
- **Changer** — A meta-architecture that inserts bi-temporal feature-interaction
  layers *inside* the extractor rather than after it, including a parameter-free
  feature exchange variant.
  [`paper`](https://arxiv.org/abs/2209.08290) [`code`](https://github.com/likyoo/open-cd) `TGRS'23` `change detection` [![Stars](https://img.shields.io/github/stars/likyoo/open-cd?style=social)](https://github.com/likyoo/open-cd/stargazers)
- **SCanNet** — Jointly models the triple-branch spatio-temporal token set instead
  of fusing semantic and change branches post hoc.
  [`paper`](https://ieeexplore.ieee.org/document/10443352) [`code`](https://github.com/ggsDing/SCanNet) `TGRS'24` `semantic CD` [![Stars](https://img.shields.io/github/stars/ggsDing/SCanNet?style=social)](https://github.com/ggsDing/SCanNet/stargazers)
- **BAN** — Bi-temporal adapter network that freezes a foundation model (CLIP or
  SAM) and bridges it to any existing CD head with few learnable parameters.
  [`paper`](https://arxiv.org/abs/2312.01163) [`code`](https://github.com/likyoo/BAN) `TGRS'24` `change detection, PEFT` [![Stars](https://img.shields.io/github/stars/likyoo/BAN?style=social)](https://github.com/likyoo/BAN/stargazers)
- **Changen2** — Resolution-scalable generative change process model that
  synthesises labelled multi-temporal sequences, yielding zero-shot-capable
  pretrained CD weights.
  [`paper`](https://arxiv.org/abs/2406.17998) [`code`](https://github.com/Z-Zheng/pytorch-change-models) `TPAMI'24` `generative CD` [![Stars](https://img.shields.io/github/stars/Z-Zheng/pytorch-change-models?style=social)](https://github.com/Z-Zheng/pytorch-change-models/stargazers)
- **AnyChange** — Training-free bi-temporal latent matching over SAM's latent
  space, giving SAM zero-shot change detection with no CD training at all.
  [`paper`](https://arxiv.org/abs/2402.01188) [`code`](https://github.com/Z-Zheng/pytorch-change-models) `NeurIPS'24` `zero-shot CD` [![Stars](https://img.shields.io/github/stars/Z-Zheng/pytorch-change-models?style=social)](https://github.com/Z-Zheng/pytorch-change-models/stargazers)

### Oriented object detection

Objects in overhead imagery have no canonical "up", so the whole detection stack —
anchors, NMS, loss, receptive field — has to be rebuilt around rotation.

- **Oriented R-CNN** — An oriented RPN generates high-quality rotated proposals
  nearly cost-free via a 6-parameter midpoint-offset representation, removing the
  two-stage oriented-proposal bottleneck.
  [`paper`](https://arxiv.org/abs/2108.05699) [`code`](https://github.com/jbwang1997/OBBDetection) `ICCV'21` `oriented detection` [![Stars](https://img.shields.io/github/stars/jbwang1997/OBBDetection?style=social)](https://github.com/jbwang1997/OBBDetection/stargazers)
- **LSKNet** — Decomposes large-kernel convolutions into a depth-wise sequence with
  growing kernel and dilation, plus spatial kernel selection to size the receptive
  field per object. The standard modern RS backbone.
  [`paper`](https://arxiv.org/abs/2303.09030) [`code`](https://github.com/zcablii/LSKNet) `ICCV'23` `oriented detection` [![Stars](https://img.shields.io/github/stars/zcablii/LSKNet?style=social)](https://github.com/zcablii/LSKNet/stargazers)
  · journal extension generalising it to a lightweight all-purpose backbone:
  [`paper`](https://arxiv.org/abs/2403.11735) `IJCV'24`
- **ARC** — Convolution kernels rotate adaptively per input through a conditional
  computation routing mechanism, handling multiple object orientations within a
  single image.
  [`paper`](https://arxiv.org/abs/2303.07820) [`code`](https://github.com/LeapLabTHU/ARC) `ICCV'23` `oriented detection` [![Stars](https://img.shields.io/github/stars/LeapLabTHU/ARC?style=social)](https://github.com/LeapLabTHU/ARC/stargazers)
- **ARS-DETR** — Aspect-ratio-aware circular smooth label, rotated deformable
  attention and an aspect-ratio-weighted angle loss make a DETR competitive at
  high-IoU oriented detection.
  [`paper`](https://arxiv.org/abs/2303.04989) [`code`](https://github.com/httle/ARS-DETR) `TGRS'24` `oriented detection, DETR` [![Stars](https://img.shields.io/github/stars/httle/ARS-DETR?style=social)](https://github.com/httle/ARS-DETR/stargazers)
- **PKINet** — Parallel multi-scale non-dilated kernels for local context plus a
  context anchor attention module for long-range context, avoiding the background
  noise that large kernels pull in.
  [`paper`](https://arxiv.org/abs/2403.06258) [`code`](https://github.com/PKINet/PKINet) `CVPR'24` `oriented detection` [![Stars](https://img.shields.io/github/stars/PKINet/PKINet?style=social)](https://github.com/PKINet/PKINet/stargazers)
- **MSFA / SARDet-100K** — Multi-stage filter-augmentation pretraining bridges the
  RGB-to-SAR domain and structure gap; ships the first COCO-scale multi-class SAR
  detection dataset.
  [`paper`](https://arxiv.org/abs/2403.06534) [`code`](https://github.com/zcablii/SARDet_100K) `NeurIPS'24` `SAR detection` [![Stars](https://img.shields.io/github/stars/zcablii/SARDet_100K?style=social)](https://github.com/zcablii/SARDet_100K/stargazers)
- **Point2RBox-v2** — First to exploit inter-instance spatial layout for
  point-supervised oriented detection, via Gaussian overlap and Voronoi watershed
  losses that bound the object extent from above and below.
  [`paper`](https://arxiv.org/abs/2502.04268) [`code`](https://github.com/VisionXLab/point2rbox-v2) `CVPR'25` `weakly-supervised detection` [![Stars](https://img.shields.io/github/stars/VisionXLab/point2rbox-v2?style=social)](https://github.com/VisionXLab/point2rbox-v2/stargazers)

### Segmentation datasets generated by foundation models

- **SAMRS** — Prompts SAM with existing RS detection boxes to auto-generate 105K
  images and 1.67M instances, enabling *segmentation*-task pretraining instead of
  classification pretraining.
  [`paper`](https://arxiv.org/abs/2305.02034) [`code`](https://github.com/ViTAE-Transformer/SAMRS) `NeurIPS'23` `segmentation pretraining` [![Stars](https://img.shields.io/github/stars/ViTAE-Transformer/SAMRS?style=social)](https://github.com/ViTAE-Transformer/SAMRS/stargazers)

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
  [`paper`](https://arxiv.org/abs/2403.19654) [`code`](https://github.com/KyanChen/RSMamba) `GRSL'24` `classification` [![Stars](https://img.shields.io/github/stars/KyanChen/RSMamba?style=social)](https://github.com/KyanChen/RSMamba/stargazers)
- **CE-VSS** — Injects explicit contour and edge priors into the 2D selective scan
  so the state-space backbone retains the object boundaries that pure sequential
  scanning erodes.
  [`paper`](https://ieeexplore.ieee.org/document/10810482) [`code`](https://github.com/yanliyue/Contour-enhanced-Visual-State-Space-Model) `TGRS'24` `classification` [![Stars](https://img.shields.io/github/stars/yanliyue/Contour-enhanced-Visual-State-Space-Model?style=social)](https://github.com/yanliyue/Contour-enhanced-Visual-State-Space-Model/stargazers)
- **CMSI-Mamba** — Cross-modal Mamba interaction blocks align spatial and spectral
  token streams before fusion, avoiding quadratic cross-attention.
  [`paper`](https://ieeexplore.ieee.org/document/10829637) [`code`](https://github.com/ru-willow/CMSI-Mamba) `TGRS'25` `multimodal classification` [![Stars](https://img.shields.io/github/stars/ru-willow/CMSI-Mamba?style=social)](https://github.com/ru-willow/CMSI-Mamba/stargazers)

### Dense prediction & segmentation

- **RS³Mamba** — Dual-branch encoder where a visual state-space auxiliary branch
  supplies global context to a CNN main branch, fused by a collaborative
  completion module. First vision-Mamba segmenter for RS.
  [`paper`](https://arxiv.org/abs/2404.02457) [`code`](https://github.com/sstary/SSRS) `GRSL'24` `segmentation` [![Stars](https://img.shields.io/github/stars/sstary/SSRS?style=social)](https://github.com/sstary/SSRS/stargazers)
- **RS-Mamba** — An omnidirectional selective scan lets a linear-complexity SSM
  ingest entire large VHR images without cropping, which is the whole practical
  argument for Mamba in this field.
  [`paper`](https://arxiv.org/abs/2404.02668) [`code`](https://github.com/walking-shadow/Official_Remote_Sensing_Mamba) `TGRS'24` `segmentation, dense prediction` [![Stars](https://img.shields.io/github/stars/walking-shadow/Official_Remote_Sensing_Mamba?style=social)](https://github.com/walking-shadow/Official_Remote_Sensing_Mamba/stargazers)
- **PPMamba** — Pyramid-pooling CNN branch interleaved with 2D selective scan
  blocks to recover local detail at multiple scales.
  [`paper`](https://ieeexplore.ieee.org/document/10769411) [`code`](https://github.com/Jerrymo59/PPMambaSeg) `TGRS'24` `segmentation` [![Stars](https://img.shields.io/github/stars/Jerrymo59/PPMambaSeg?style=social)](https://github.com/Jerrymo59/PPMambaSeg/stargazers)
- **CM-UNet** — CNN encoder with a CSMamba decoder block and multi-scale attention
  fusion, giving UNet a linear-cost global decoder.
  [`paper`](https://arxiv.org/abs/2405.10530) [`code`](https://github.com/XiaoBuL/CM-UNet) `preprint` `segmentation` [![Stars](https://img.shields.io/github/stars/XiaoBuL/CM-UNet?style=social)](https://github.com/XiaoBuL/CM-UNet/stargazers)
- **Samba** — Samba-block encoder with a UperNet decoder; the first SSM
  encoder-decoder benchmark for RS segmentation. ⚠ `Heliyon`
  [`paper`](https://doi.org/10.1016/j.heliyon.2024.e38495) [`code`](https://github.com/zhuqinfeng1999/Samba) `2024` `segmentation` [![Stars](https://img.shields.io/github/stars/zhuqinfeng1999/Samba?style=social)](https://github.com/zhuqinfeng1999/Samba/stargazers)
- **RSMTMamba** — Mamba-based cross-task feature learning in a shared-encoder
  design for joint segmentation, height estimation and boundary detection.
  [`paper`](https://ieeexplore.ieee.org/document/10879310) [`code`](https://github.com/sycs-2024/RSMultitaskMamba) `TGRS'25` `multi-task` [![Stars](https://img.shields.io/github/stars/sycs-2024/RSMultitaskMamba?style=social)](https://github.com/sycs-2024/RSMultitaskMamba/stargazers)

### Change detection

- **ChangeMamba** — VMamba encoder plus three spatio-temporal relation mechanisms
  (sequential, cross and parallel scan) covering binary CD, semantic CD and
  building damage assessment in one framework. ESI Highly Cited.
  [`paper`](https://arxiv.org/abs/2404.03425) [`code`](https://github.com/ChenHongruixuan/ChangeMamba) `TGRS'24` `change detection` [![Stars](https://img.shields.io/github/stars/ChenHongruixuan/ChangeMamba?style=social)](https://github.com/ChenHongruixuan/ChangeMamba/stargazers)
- **MF-Mamba** — Multi-level SSM fusion aggregating state representations across
  encoder stages rather than only at the bottleneck.
  [`paper`](https://ieeexplore.ieee.org/document/10756674) [`code`](https://github.com/121zzy/MF-Mamba) `TGRS'24` `change detection` [![Stars](https://img.shields.io/github/stars/121zzy/MF-Mamba?style=social)](https://github.com/121zzy/MF-Mamba/stargazers)
- **RSCaMa** — Stacked layers combining a spatial-difference SSM with a
  temporal-traversing SSM that scans bi-temporal features cross-wise, for change
  *captioning*.
  [`paper`](https://arxiv.org/abs/2404.18895) [`code`](https://github.com/Chen-Yang-Liu/RSCaMa) `GRSL'24` `change captioning` [![Stars](https://img.shields.io/github/stars/Chen-Yang-Liu/RSCaMa?style=social)](https://github.com/Chen-Yang-Liu/RSCaMa/stargazers)
- **LCCDMamba** — Visual state-space encoder with a land-cover-aware difference
  aggregation decoder tuned for VHR land-cover transitions.
  [`paper`](https://ieeexplore.ieee.org/document/10845192) [`code`](https://github.com/juncyan/lccdmamba) `JSTARS'25` `land-cover CD` [![Stars](https://img.shields.io/github/stars/juncyan/lccdmamba?style=social)](https://github.com/juncyan/lccdmamba/stargazers)
- **CDMamba** — A scaled residual ConvMamba block recovers the fine local detail
  Mamba discards, plus adaptive global-local guided fusion for bi-temporal
  interaction.
  [`paper`](https://arxiv.org/abs/2406.04207) [`code`](https://github.com/zmoka-zht/CDMamba) `TGRS'25` `change detection` [![Stars](https://img.shields.io/github/stars/zmoka-zht/CDMamba?style=social)](https://github.com/zmoka-zht/CDMamba/stargazers)

### Fusion, super-resolution & detection

- **FusionMamba** — Extends the single-input Mamba block into a plug-and-play
  *dual-input* block for arbitrary two-source fusion, covering pansharpening and
  hyperspectral-multispectral fusion.
  [`paper`](https://arxiv.org/abs/2404.07932) [`code`](https://github.com/PSRben/FusionMamba) `TGRS'24` `pansharpening, fusion` [![Stars](https://img.shields.io/github/stars/PSRben/FusionMamba?style=social)](https://github.com/PSRben/FusionMamba/stargazers)
- **SDMSPan** — Detail-branch supervision guides a multi-scale SSM so
  high-frequency panchromatic detail is explicitly injected rather than implicitly
  learned.
  [`paper`](https://ieeexplore.ieee.org/document/10812822) [`code`](https://github.com/zhaomengjiao123/SDMSPan) `TGRS'24` `pansharpening` [![Stars](https://img.shields.io/github/stars/zhaomengjiao123/SDMSPan?style=social)](https://github.com/zhaomengjiao123/SDMSPan/stargazers)
- **MiM-ISTD** — Nested outer/inner Mamba over patches and sub-patches, making
  large-image infrared small-target detection viable at a fraction of transformer
  GPU cost.
  [`paper`](https://ieeexplore.ieee.org/document/10740056) [`code`](https://github.com/txchen-USTC/MiM-ISTD) `TGRS'24` `infrared detection` [![Stars](https://img.shields.io/github/stars/txchen-USTC/MiM-ISTD?style=social)](https://github.com/txchen-USTC/MiM-ISTD/stargazers)
- **TrackingMamba** — Single-stream visual state-space tracking backbone replacing
  transformer relation modelling for satellite video.
  [`paper`](https://ieeexplore.ieee.org/document/10678881) [`code`](https://github.com/KustTeamWQW/TrackingMamba) `JSTARS'24` `video tracking` [![Stars](https://img.shields.io/github/stars/KustTeamWQW/TrackingMamba?style=social)](https://github.com/KustTeamWQW/TrackingMamba/stargazers)
- **FMambaIR** — Couples SSM spatial modelling with an explicit frequency-domain
  branch to restore both structure and texture.
  [`paper`](https://ieeexplore.ieee.org/document/10834441) [`code`](https://github.com/mickoluan/FMambaIR) `TGRS'25` `restoration, dehazing` [![Stars](https://img.shields.io/github/stars/mickoluan/FMambaIR?style=social)](https://github.com/mickoluan/FMambaIR/stargazers)
- **Pan-Mamba** — Channel-swapping Mamba for cheap cross-modal exchange plus
  cross-modal Mamba for PAN/MS relation modelling. The canonical Mamba
  pansharpening paper. ⚠ `Information Fusion`
  [`paper`](https://arxiv.org/abs/2402.12192) [`code`](https://github.com/alexhe101/Pan-Mamba) `2025` `pansharpening` [![Stars](https://img.shields.io/github/stars/alexhe101/Pan-Mamba?style=social)](https://github.com/alexhe101/Pan-Mamba/stargazers)
- **FreMamba** — Frequency selection, vision state-space and hybrid gate modules;
  the first Mamba for RS super-resolution, beating HAT-L at ~28% of its memory.
  ⚠ `IEEE TMM`
  [`paper`](https://arxiv.org/abs/2405.04964) [`code`](https://github.com/XY-boy/FreMamba) `2024` `super-resolution` [![Stars](https://img.shields.io/github/stars/XY-boy/FreMamba?style=social)](https://github.com/XY-boy/FreMamba/stargazers)

### Mamba foundation models

- **RoMA** — Auto-regressive Mamba pretraining with rotation-aware adaptive
  cropping, angular embeddings and multi-scale token prediction; beats ViT-based
  RSFMs while cutting GPU memory roughly 80% on high-resolution data.
  [`paper`](https://arxiv.org/abs/2503.10392) [`code`](https://github.com/MiliLab/RoMA) `NeurIPS'25` `foundation model` [![Stars](https://img.shields.io/github/stars/MiliLab/RoMA?style=social)](https://github.com/MiliLab/RoMA/stargazers)
- **SatMamba** — Masked autoencoder whose encoder *and* decoder are multi-way
  Mamba blocks rather than transformer blocks, giving linear scaling in sequence
  length.
  [`paper`](https://arxiv.org/abs/2502.00435) [`code`](https://github.com/mdchuc/HRSFM) `preprint` `foundation model` [![Stars](https://img.shields.io/github/stars/mdchuc/HRSFM?style=social)](https://github.com/mdchuc/HRSFM/stargazers)

---

## Efficient Architectures

Attention and state-space models are not the only options. This section collects
architectures whose primary contribution is a different **cost** profile.

- **RS-vHeat** — Replaces attention with a heat conduction operator (O(N^1.5) with
  a global receptive field) guided by object structure, pretrained via
  frequency-domain hierarchical masking. 84% less memory and 2.7× throughput
  versus attention-based RSFMs.
  [`paper`](https://arxiv.org/abs/2411.17984) [`code`](https://github.com/iecashhy/RS-vHeat) `ICCV'25` `foundation model, efficient` [![Stars](https://img.shields.io/github/stars/iecashhy/RS-vHeat?style=social)](https://github.com/iecashhy/RS-vHeat/stargazers)
- **DeepKANSeg** — A DeepKAN refinement module and global-local decoder built from
  Kolmogorov-Arnold linear layers, decomposing high-dimensional features into
  univariate learnable transforms.
  [`paper`](https://arxiv.org/abs/2501.07390) [`code`](https://github.com/sstary/SSRS) `preprint` `segmentation, KAN` [![Stars](https://img.shields.io/github/stars/sstary/SSRS?style=social)](https://github.com/sstary/SSRS/stargazers)
- **RSRWKV** — 2D-WKV scanning in four directions removes RWKV's one-dimensional
  anisotropy, plus multi-view convolutional shift and efficient channel attention.
  [`paper`](https://arxiv.org/abs/2503.20382) `preprint` `linear attention` `no code`

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
  [`paper`](https://doi.org/10.1109/TGRS.2021.3130716) [`code`](https://github.com/danfenghong/IEEE_TGRS_SpectralFormer) `TGRS'22` `classification` [![Stars](https://img.shields.io/github/stars/danfenghong/IEEE_TGRS_SpectralFormer?style=social)](https://github.com/danfenghong/IEEE_TGRS_SpectralFormer/stargazers)
- **SSFTT** — A 3D/2D CNN shallow spectral-spatial extractor feeds a
  Gaussian-weighted feature tokenizer into a transformer encoder. The most-copied
  hybrid CNN-transformer HSI baseline.
  [`paper`](https://ieeexplore.ieee.org/document/9684381) [`code`](https://github.com/zgr6010/HSI_SSFTT) `TGRS'22` `classification` [![Stars](https://img.shields.io/github/stars/zgr6010/HSI_SSFTT?style=social)](https://github.com/zgr6010/HSI_SSFTT/stargazers)
- **morphFormer** — Injects learnable spectral and spatial morphological
  (erosion/dilation) operations into the attention block to capture shape and
  structure cues that plain attention misses.
  [`paper`](https://doi.org/10.1109/TGRS.2023.3242346) [`code`](https://github.com/mhaut/morphFormer) `TGRS'23` `classification` [![Stars](https://img.shields.io/github/stars/mhaut/morphFormer?style=social)](https://github.com/mhaut/morphFormer/stargazers)
- **SSTFormer** — Extends spectral-spatial tokenization with a temporal transformer
  branch for bi-temporal hyperspectral change detection.
  [`paper`](https://doi.org/10.1109/TGRS.2022.3203075) [`code`](https://github.com/yanhengwang-heu/IEEE_TGRS_SSTFormer) `TGRS'22` `change detection` [![Stars](https://img.shields.io/github/stars/yanhengwang-heu/IEEE_TGRS_SSTFormer?style=social)](https://github.com/yanhengwang-heu/IEEE_TGRS_SSTFormer/stargazers)

### State-space models for HSI

Mamba's linear scaling is a natural fit for long spectral sequences, and the
scan-order question becomes genuinely two-dimensional here: you are choosing an
order across *space* and across *wavelength* simultaneously.

- **MambaHSI** — The first *image-level* rather than patch-level Mamba HSI
  classifier, with separate spatial and spectral Mamba blocks and an adaptive
  spatial-spectral fusion module. The canonical HSI-Mamba paper.
  [`paper`](https://ieeexplore.ieee.org/document/10604894) [`code`](https://github.com/li-yapeng/MambaHSI) `TGRS'24` `classification` [![Stars](https://img.shields.io/github/stars/li-yapeng/MambaHSI?style=social)](https://github.com/li-yapeng/MambaHSI/stargazers)
- **3DSS-Mamba** — Pixel-wise 3D selective scanning across spectral *and* spatial
  axes, operating on tokens from a spectral-spatial token generator.
  [`paper`](https://arxiv.org/abs/2405.12487) [`code`](https://github.com/IIP-Team/3DSS-Mamba) `TGRS'24` `classification` [![Stars](https://img.shields.io/github/stars/IIP-Team/3DSS-Mamba?style=social)](https://github.com/IIP-Team/3DSS-Mamba/stargazers)
- **IGroupSS-Mamba** — Interval grouping of spectral bands with group-wise
  multi-directional scanning, cutting the redundancy of scanning all bands jointly.
  [`paper`](https://arxiv.org/abs/2410.05100) [`code`](https://github.com/IIP-Team/IGroupSS-Mamba) `TGRS'24` `classification` [![Stars](https://img.shields.io/github/stars/IIP-Team/IGroupSS-Mamba?style=social)](https://github.com/IIP-Team/IGroupSS-Mamba/stargazers)
- **HyperMamba** — Spectral-adaptive state transition conditioning the SSM
  parameters on local spectral statistics.
  [`paper`](https://ieeexplore.ieee.org/document/10720896) [`code`](https://github.com/chiangliu/HyperMamba) `TGRS'24` `classification` [![Stars](https://img.shields.io/github/stars/chiangliu/HyperMamba?style=social)](https://github.com/chiangliu/HyperMamba/stargazers)
- **GraphMamba** — Learns a graph over superpixel nodes and orders the Mamba scan
  along graph structure rather than a raster path.
  [`paper`](https://ieeexplore.ieee.org/document/10746459) [`code`](https://github.com/ahappyyang/GraphMamba) `TGRS'24` `classification` [![Stars](https://img.shields.io/github/stars/ahappyyang/GraphMamba?style=social)](https://github.com/ahappyyang/GraphMamba/stargazers)
- **MambaLG** — Local-global dual-scan coupling a local patch scan with a global
  scene scan in one state-space encoder.
  [`paper`](https://ieeexplore.ieee.org/document/10812905) [`code`](https://github.com/danfenghong/IEEE_TGRS_MambaLG) `TGRS'24` `classification` [![Stars](https://img.shields.io/github/stars/danfenghong/IEEE_TGRS_MambaLG?style=social)](https://github.com/danfenghong/IEEE_TGRS_MambaLG/stargazers)
- **DualMamba** — Parallel lightweight Mamba and convolution branches with dynamic
  gated fusion, for a very small parameter budget.
  [`paper`](https://ieeexplore.ieee.org/document/10798573) `TGRS'24` `classification, lightweight` `no code`
- **S²Mamba** — Two parallel selective scans — patch-cross-scan for space,
  bi-directional scan for spectrum — merged by a learnable spatial-spectral mixture
  gate.
  [`paper`](https://ieeexplore.ieee.org/document/10844849) [`code`](https://github.com/PURE-melo/S2Mamba) `TGRS'25` `classification` [![Stars](https://img.shields.io/github/stars/PURE-melo/S2Mamba?style=social)](https://github.com/PURE-melo/S2Mamba/stargazers)
- **STMamba** — Replaces raster patches with learned semantic tokens before
  state-space modelling, shortening the sequence the SSM has to traverse.
  [`paper`](https://ieeexplore.ieee.org/document/10838328) [`code`](https://github.com/AlanLowell/STMamba) `JSTARS'25` `classification` [![Stars](https://img.shields.io/github/stars/AlanLowell/STMamba?style=social)](https://github.com/AlanLowell/STMamba/stargazers)
- **MambaHSI+** — Simplifies MambaHSI into a multidirectional state-propagation
  scheme with higher accuracy at lower cost.
  [`paper`](https://ieeexplore.ieee.org/document/11023867) [`code`](https://github.com/RockAilab/MambaHSI_Plus) `TGRS'25` `classification, efficient` [![Stars](https://img.shields.io/github/stars/RockAilab/MambaHSI_Plus?style=social)](https://github.com/RockAilab/MambaHSI_Plus/stargazers)

### Hyperspectral foundation models

The most consequential recent shift. Until 2023 there was no meaningful pretraining
story for HSI — everything was trained from scratch on a few hundred labelled
pixels. These models change what the starting point looks like.

- **SpectralGPT** — 3D spatial-spectral token generation with multi-target
  reconstruction; over 600M parameters trained on ~1M spectral images. The first
  large spectral-native foundation model.
  [`paper`](https://arxiv.org/abs/2311.07113) [`code`](https://github.com/danfenghong/IEEE_TPAMI_SpectralGPT) `TPAMI'24` `foundation model` [![Stars](https://img.shields.io/github/stars/danfenghong/IEEE_TPAMI_SpectralGPT?style=social)](https://github.com/danfenghong/IEEE_TPAMI_SpectralGPT/stargazers)
- **HyperSIGMA** — First billion-parameter HSI foundation model, with separate
  spatial and spectral MAEs and sparse sampling attention to counter
  spatial-spectral redundancy. Evaluated across 16 datasets and 7 tasks.
  [`paper`](https://arxiv.org/abs/2406.11519) [`code`](https://github.com/WHU-Sigma/HyperSIGMA) `TPAMI'25` `foundation model` [![Stars](https://img.shields.io/github/stars/WHU-Sigma/HyperSIGMA?style=social)](https://github.com/WHU-Sigma/HyperSIGMA/stargazers)
- **DOFA** — Not HSI-specific, but its wavelength-conditioned hypernetwork accepts
  hyperspectral input directly alongside other sensors, which makes it a useful
  cross-sensor baseline. See [Foundation Models](#multi-modal-foundation-models).
  [`paper`](https://arxiv.org/abs/2403.15356) [`code`](https://github.com/zhu-xlab/DOFA) `preprint` `multimodal` [![Stars](https://img.shields.io/github/stars/zhu-xlab/DOFA?style=social)](https://github.com/zhu-xlab/DOFA/stargazers)

### Fusion, detection, denoising & restoration

- **SSUMamba** — Alternating spatial-spectral continuous scanning inside a U-shaped
  SSM so noise modelling sees full 3D context at linear cost.
  [`paper`](https://arxiv.org/abs/2405.01726) [`code`](https://github.com/lronkitty/SSUMamba) `TGRS'24` `denoising` [![Stars](https://img.shields.io/github/stars/lronkitty/SSUMamba?style=social)](https://github.com/lronkitty/SSUMamba/stargazers)
- **HLMamba** — Dual-stream Mamba with a cross-modal state-space fusion block
  bridging HSI spectra and LiDAR elevation.
  [`paper`](https://ieeexplore.ieee.org/document/10679212) [`code`](https://github.com/Dilingliao/HLMamba) `TGRS'24` `HSI+LiDAR fusion` [![Stars](https://img.shields.io/github/stars/Dilingliao/HLMamba?style=social)](https://github.com/Dilingliao/HLMamba/stargazers)
- **FusionMamba** — Plug-and-play dual-input Mamba block for arbitrary two-source
  fusion, covering hyperspectral pansharpening and HSI-MSI fusion. See
  [Mamba](#fusion-super-resolution--detection).
  [`paper`](https://arxiv.org/abs/2404.07932) [`code`](https://github.com/PSRben/FusionMamba) `TGRS'24` `fusion` [![Stars](https://img.shields.io/github/stars/PSRben/FusionMamba?style=social)](https://github.com/PSRben/FusionMamba/stargazers)
- **HTD-Mamba** — Self-supervised spectrally contrastive learning over group-wise
  spectral embeddings with a pyramid SSM backbone, for target detection.
  ESI Highly Cited.
  [`paper`](https://arxiv.org/abs/2407.06841) [`code`](https://github.com/shendb2022/HTD-Mamba) `TGRS'25` `target detection` [![Stars](https://img.shields.io/github/stars/shendb2022/HTD-Mamba?style=social)](https://github.com/shendb2022/HTD-Mamba/stargazers)
- **MSFMamba** — Three-block design (multi-scale spatial Mamba, spectral Mamba,
  dual-input fusion Mamba) extending Mamba to two heterogeneous sources.
  [`paper`](https://arxiv.org/abs/2408.14255) [`code`](https://github.com/oucailab/MSFMamba) `TGRS'25` `HSI+LiDAR/SAR fusion` [![Stars](https://img.shields.io/github/stars/oucailab/MSFMamba?style=social)](https://github.com/oucailab/MSFMamba/stargazers)

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

Language entered remote sensing through two distinct doors, and they are worth
keeping separate. **Contrastive models** (CLIP-style) align images and text in a
shared embedding space, which buys zero-shot classification and cross-modal
retrieval. **Multimodal LLMs** attach a vision encoder to a language model, which
buys conversation, grounding and question answering. The first is a
representation; the second is an interface.

The binding constraint in both cases is data. There is no web-scale corpus of
captioned satellite imagery, so nearly every model below is really a proposal for
how to *manufacture* image-text pairs — from OpenStreetMap tags, from existing
detection boxes, or from a captioning model run over unlabelled archives.

### Contrastive vision-language models

- **RemoteCLIP** — Manufactures training pairs by converting existing annotations
  (box-to-caption, mask-to-box) rather than scraping captions, yielding a 12×
  larger pretraining set than was previously available.
  [`paper`](https://arxiv.org/abs/2306.11029) [`code`](https://github.com/ChenDelong1999/RemoteCLIP) `TGRS'24` `retrieval, zero-shot classification` [![Stars](https://img.shields.io/github/stars/ChenDelong1999/RemoteCLIP?style=social)](https://github.com/ChenDelong1999/RemoteCLIP/stargazers)
- **GeoRSCLIP / RS5M** — Builds a 5M image-text corpus by filtering public
  paired datasets and captioning label-only RS datasets with a pretrained VLM,
  then fine-tunes CLIP on it with full and parameter-efficient variants.
  [`paper`](https://arxiv.org/abs/2306.11300) [`code`](https://github.com/om-ai-lab/RS5M) `TGRS'24†` `retrieval, zero-shot classification` [![Stars](https://img.shields.io/github/stars/om-ai-lab/RS5M?style=social)](https://github.com/om-ai-lab/RS5M/stargazers)
- **SkyCLIP / SkyScript** — Pairs unlabelled imagery with OpenStreetMap semantics
  through shared geo-coordinates, giving 5.2M pairs over 29K semantic tags with
  no human captioning at all.
  [`paper`](https://arxiv.org/abs/2312.12856) [`code`](https://github.com/wangzhecheng/SkyScript) `AAAI'24` `zero-shot classification, retrieval` [![Stars](https://img.shields.io/github/stars/wangzhecheng/SkyScript?style=social)](https://github.com/wangzhecheng/SkyScript/stargazers)
- **PriorCLIP** — Injects visual priors into the alignment objective to handle the
  scale and orientation variance that generic CLIP embeddings blur together.
  [`paper`](https://arxiv.org/abs/2405.10160) `preprint` `retrieval`

### Multimodal LLMs

- **GeoChat** — The first *grounded* large vision-language model for remote
  sensing: region-level grounding and referring detection rather than whole-image
  captioning alone.
  [`paper`](https://openaccess.thecvf.com/content/CVPR2024/html/Kuckreja_GeoChat_Grounded_Large_Vision-Language_Model_for_Remote_Sensing_CVPR_2024_paper.html) [`code`](https://github.com/mbzuai-oryx/GeoChat) `CVPR'24` `MLLM, grounding` [![Stars](https://img.shields.io/github/stars/mbzuai-oryx/GeoChat?style=social)](https://github.com/mbzuai-oryx/GeoChat/stargazers)
- **EarthGPT** — Unifies optical, SAR and infrared in one instruction-following
  model via a visual-enhanced perception mechanism and cross-modal mutual
  comprehension.
  [`paper`](https://arxiv.org/abs/2401.16822) `TGRS'24` `MLLM, multi-sensor`
- **LHRS-Bot** — Derives instruction data at scale from volunteered geographic
  information paired with imagery, sidestepping human annotation entirely.
  [`paper`](https://arxiv.org/abs/2402.02544) [`code`](https://github.com/NJU-LHRS/LHRS-Bot) `ECCV'24` `MLLM, instruction tuning` [![Stars](https://img.shields.io/github/stars/NJU-LHRS/LHRS-Bot?style=social)](https://github.com/NJU-LHRS/LHRS-Bot/stargazers)
- **SkyEyeGPT** — Unifies RS vision-language tasks through instruction tuning on a
  manually verified 968K-sample instruction-following corpus.
  [`paper`](https://arxiv.org/abs/2401.09712) [`code`](https://github.com/ZhanYang-nwpu/SkyEyeGPT) `preprint` `MLLM, instruction tuning` [![Stars](https://img.shields.io/github/stars/ZhanYang-nwpu/SkyEyeGPT?style=social)](https://github.com/ZhanYang-nwpu/SkyEyeGPT/stargazers)
- **RSGPT** — Released alongside RSICap and RSIEval, a small human-annotated
  captioning and VQA benchmark built to evaluate RS VLMs properly.
  [`paper`](https://arxiv.org/abs/2307.15266) [`code`](https://github.com/Lavender105/RSGPT) `ISPRS J.'25` `MLLM, captioning, VQA` [![Stars](https://img.shields.io/github/stars/Lavender105/RSGPT?style=social)](https://github.com/Lavender105/RSGPT/stargazers)
- **VHM** — Trains on both factual and deliberately deceptive questions so the
  model learns to decline unanswerable ones, targeting hallucination rather than
  capability.
  [`paper`](https://arxiv.org/abs/2403.20213) [`code`](https://github.com/opendatalab/VHM) `preprint` `MLLM, honesty` [![Stars](https://img.shields.io/github/stars/opendatalab/VHM?style=social)](https://github.com/opendatalab/VHM/stargazers)
- **GeoGround** — Unifies the three grounding output formats (box, mask, text
  coordinates) in one model rather than training a separate head per format.
  [`paper`](https://arxiv.org/abs/2411.11904) [`code`](https://github.com/zytx121/GeoGround) `preprint` `visual grounding` [![Stars](https://img.shields.io/github/stars/zytx121/GeoGround?style=social)](https://github.com/zytx121/GeoGround/stargazers)

### Promptable & referring models

Adapting Segment Anything to overhead imagery is its own design problem: SAM is
interactive by construction, and remote sensing needs it to run unattended over
millions of tiles.

- **RMSIN** — Intra-scale and cross-scale interaction with adaptive rotated
  convolution for referring segmentation from a free-text query. Introduces
  RRSIS-D.
  [`paper`](https://arxiv.org/abs/2312.12470) [`code`](https://github.com/Lsan2401/RMSIN) `CVPR'24` `referring segmentation` [![Stars](https://img.shields.io/github/stars/Lsan2401/RMSIN?style=social)](https://github.com/Lsan2401/RMSIN/stargazers)

Language- and SAM-conditioned models indexed under their primary family:

- [RSPrompter](#semantic-segmentation) — learned prompts driving a frozen SAM
  decoder for automatic instance segmentation. `TGRS'24`
- [AnyChange](#change-detection) — training-free zero-shot change detection in
  SAM's latent space. `NeurIPS'24`
- [BAN](#change-detection) — freezes CLIP or SAM and bridges it to any change
  detection head. `TGRS'24`
- [RSCaMa](#change-detection-1) — state-space model for change captioning.
  `GRSL'24`

### Benchmarks

- **VRSBench** — 29,614 human-verified captions, 52,472 object references and
  123,221 QA pairs in one benchmark, covering captioning, grounding and VQA
  together rather than one task in isolation.
  [`paper`](https://arxiv.org/abs/2406.12384) [`code`](https://github.com/lx709/VRSBench) `NeurIPS'24` `benchmark` [![Stars](https://img.shields.io/github/stars/lx709/VRSBench?style=social)](https://github.com/lx709/VRSBench/stargazers)

---

## Diffusion & Generative Models

Diffusion arrived in remote sensing with an unusual advantage: the conditioning
signal is unusually rich. A satellite image comes with geo-coordinates, ground
sample distance, acquisition timestamp and sensor metadata, all of which are
natural conditioning variables that a natural-image model simply does not have.
Much of the work below is about exploiting that.

The second thread is restoration. Clouds, speckle and low resolution are
*physical* degradations with known structure, which makes them a better fit for a
learned prior than generic inpainting.

### Controllable generation

- **DiffusionSat** — Conditions on numerical satellite metadata (location, GSD,
  timestamp) alongside text, with conditioning modules trainable for
  super-resolution, inpainting and temporal generation.
  [`paper`](https://proceedings.iclr.cc/paper_files/paper/2024/file/16c3c941409d0581286eff49b180930f-Paper-Conference.pdf) [`code`](https://github.com/samar-khanna/DiffusionSat) `ICLR'24` `generation, super-resolution` [![Stars](https://img.shields.io/github/stars/samar-khanna/DiffusionSat?style=social)](https://github.com/samar-khanna/DiffusionSat/stargazers)
- **CRS-Diff** — Accepts text, metadata and image conditioning simultaneously,
  giving ControlNet-style fine-grained control over the generation process.
  [`paper`](https://arxiv.org/abs/2403.11614) [`code`](https://github.com/Sonettoo/CRS-Diff) `preprint` `controllable generation` [![Stars](https://img.shields.io/github/stars/Sonettoo/CRS-Diff?style=social)](https://github.com/Sonettoo/CRS-Diff/stargazers)
- **MetaEarth** — Resolution-conditioned generation aimed at global-scale synthesis
  across zoom levels rather than single-tile output.
  [`paper`](https://arxiv.org/abs/2405.13570) `preprint` `generation`
- **Text2Earth** — Text-driven generation trained on a global-scale corpus, framed
  as a generative foundation model rather than a single-task generator.
  [`paper`](https://arxiv.org/abs/2501.00895) [`code`](https://github.com/Chen-Yang-Liu/Text2Earth) `preprint` `text-to-image` [![Stars](https://img.shields.io/github/stars/Chen-Yang-Liu/Text2Earth?style=social)](https://github.com/Chen-Yang-Liu/Text2Earth/stargazers)

### Restoration & super-resolution

- **EDiffSR** — Efficient conditional diffusion for super-resolution, using a
  lightweight prior encoder to avoid the over-smoothing and training instability
  of earlier diffusion SR.
  [`paper`](https://arxiv.org/abs/2310.19288) [`code`](https://github.com/XY-boy/EDiffSR) `TGRS'24` `super-resolution` [![Stars](https://img.shields.io/github/stars/XY-boy/EDiffSR?style=social)](https://github.com/XY-boy/EDiffSR/stargazers)
- **DiffCR** — Fast conditional diffusion for cloud removal, reaching
  state-of-the-art with roughly 5% of the parameters and compute of the prior
  best method.
  [`paper`](https://arxiv.org/abs/2308.04417) [`code`](https://github.com/XavierJiezou/DiffCR) `TGRS'24` `cloud removal` [![Stars](https://img.shields.io/github/stars/XavierJiezou/DiffCR?style=social)](https://github.com/XavierJiezou/DiffCR/stargazers)

### Generative models as label factories

The most consequential use of generative models here is not making pictures — it
is manufacturing *supervised training data* for tasks where labels are scarce.

- **Changen2** — Generative change process model that synthesises labelled
  multi-temporal sequences, yielding pretrained weights with zero-shot change
  detection capability. Cross-listed from
  [Vision Transformers](#change-detection).
  [`paper`](https://arxiv.org/abs/2406.17998) [`code`](https://github.com/Z-Zheng/pytorch-change-models) `TPAMI'24` `generative CD` [![Stars](https://img.shields.io/github/stars/Z-Zheng/pytorch-change-models?style=social)](https://github.com/Z-Zheng/pytorch-change-models/stargazers)
- **TerraMind** — Any-to-any generative multimodal model that synthesises missing
  modalities as an intermediate reasoning step. Cross-listed from
  [Foundation Models](#multi-modal-foundation-models).
  [`paper`](https://arxiv.org/abs/2504.11171) [`code`](https://github.com/IBM/terramind) `ICCV'25` `generative, multimodal` [![Stars](https://img.shields.io/github/stars/IBM/terramind?style=social)](https://github.com/IBM/terramind/stargazers)

---

## Satellite Image Time Series

Optical time series are irregularly sampled, cloud-interrupted and seasonally
structured. The architectural question is how to encode a variable-length,
unevenly-spaced temporal axis without pretending it is a video.

- **U-TAE** — Convolutional temporal attention encoder giving the first
  end-to-end single-stage panoptic segmentation of satellite image time series.
  Introduces PASTIS, the first open SITS dataset with panoptic annotations.
  [`paper`](https://arxiv.org/abs/2107.07933) [`code`](https://github.com/VSainteuf/utae-paps) [`PASTIS`](https://github.com/VSainteuf/pastis-benchmark) `ICCV'21` `panoptic segmentation, crop mapping` [![Stars](https://img.shields.io/github/stars/VSainteuf/utae-paps?style=social)](https://github.com/VSainteuf/utae-paps/stargazers)
- **Exchanger** — Reformulates SITS processing as set prediction rather than
  sequence modelling, targeting irregular acquisition times directly.
  [`paper`](https://arxiv.org/abs/2305.02086) `preprint` `segmentation, crop mapping`
- **TSViT** — Factorises attention into temporal-then-spatial over
  spatiotemporal patch tokens, with acquisition-date positional encoding rather
  than sequence index, so irregular revisit intervals are represented directly.
  [`paper`](https://arxiv.org/abs/2301.04944) `CVPR'23` `segmentation, crop mapping`

Time-series capability also appears in several models listed elsewhere:

- [Galileo](#multi-modal-foundation-models) — handles both image and pixel
  time-series tasks across nine modalities. `ICML'25`
- [AnySat](#multi-modal-foundation-models) — joint training across sensors and
  resolutions including time series. `CVPR'25`
- [OmniSat](#multi-modal-foundation-models) — fuses VHR aerial with Sentinel-1/2
  time series; contributes PASTIS-HD. `ECCV'24`
- [SatMAE](#masked-image-modeling) — temporal embeddings with independent
  per-timestep masking. `NeurIPS'22`
- [TESSERA / Presto](#coverage-status) — pixel-timeseries embedding models,
  pending venue verification.

---

## SAR

Speckle is multiplicative, not additive, so denoising architectures built for
optical noise do not transfer. The absence of clean reference images also makes
supervised training impossible, which is why self-supervision dominates here.

### Despeckling

Speckle has no clean ground truth — you cannot photograph the same scene without
it. Every method below is therefore a different answer to the same question: how
do you train a denoiser when no noise-free target exists?

- **SAR2SAR** — Adapts noise2noise to multi-temporal stacks, learning to restore
  from pairs of *noisy* acquisitions of the same scene rather than from clean
  references.
  [`paper`](https://arxiv.org/abs/2006.15037) [`code`](https://github.com/emanueledalsasso/SAR2SAR) `JSTARS'21` `despeckling, self-supervised` [![Stars](https://img.shields.io/github/stars/emanueledalsasso/SAR2SAR?style=social)](https://github.com/emanueledalsasso/SAR2SAR/stargazers)
- **Speckle2Void** — Blind-spot convolutional network whose receptive field is
  shaped to exclude a tunable neighbourhood, accounting for spatially correlated
  speckle rather than assuming independence.
  [`paper`](https://arxiv.org/abs/2007.02075) [`code`](https://github.com/diegovalsesia/speckle2void) `TGRS'22` `despeckling, self-supervised` [![Stars](https://img.shields.io/github/stars/diegovalsesia/speckle2void?style=social)](https://github.com/diegovalsesia/speckle2void/stargazers)
- **MERLIN** — Exploits the statistical independence of the real and imaginary
  parts of single-look complex SAR, so a single acquisition supplies both input
  and target.
  [`paper`](https://arxiv.org/abs/2110.13148) `TGRS'22` `despeckling, self-supervised`

### Interferometry

InSAR phase is measured modulo 2π, so recovering absolute deformation means
resolving how many whole cycles were lost — an integer problem wrapped inside a
continuous one, which is why it resists a plain regression formulation.

- **Unwrap-Net** — Treats phase unwrapping as semantic segmentation over wrap
  counts, fusing gradient information and using airborne LiDAR as supervision to
  limit the error propagation that defeats classical path-following methods.
  [`paper`](https://doi.org/10.1016/j.isprsjprs.2024.11.009) [`code`](https://github.com/yangwangyangzi48/UNWRAPNETV1) `ISPRS J.'24` `phase unwrapping` [![Stars](https://img.shields.io/github/stars/yangwangyangzi48/UNWRAPNETV1?style=social)](https://github.com/yangwangyangzi48/UNWRAPNETV1/stargazers)

Most deep-learning InSAR work — phase filtering, coherence estimation,
deformation time series — is published in venues outside this list's policy,
chiefly MDPI *Remote Sensing* and the ISPRS Annals. That is a property of where
the sub-field publishes rather than a judgement on the work, and it is the reason
this subsection is short.

### Recognition, detection & fusion

SAR-relevant architectures listed elsewhere:

- [SARATR-X](#agency--open-release-models) — first SAR target recognition
  foundation model. `IEEE TIP'25`
- [SARDet-100K / MSFA](#oriented-object-detection) — COCO-scale multi-class SAR
  detection benchmark with filter-augmentation pretraining. `NeurIPS'24`
- [CROMA](#contrastive--geo-aware-self-supervision) — contrastive radar-optical
  masked autoencoding. `NeurIPS'23`
- [MSFMamba](#fusion-detection-denoising--restoration) — multi-source fusion
  including SAR. `TGRS'25`
- [TerraMind](#multi-modal-foundation-models) — generates SAR modalities
  as intermediate reasoning steps. `ICCV'25`

---

## 3D, LiDAR & Neural Fields

Reconstructing geometry from satellite imagery has to contend with pushbroom RPC
camera models, multi-date acquisitions with moving shadows, and transient objects
between captures — none of which standard multi-view pipelines assume.

### Neural fields & Gaussian splatting

Satellite photogrammetry breaks the assumptions of standard multi-view pipelines:
the camera is a pushbroom RPC model rather than a pinhole, acquisitions are months
apart, and the sun moves between them, so shadows are a *time-varying* part of the
scene rather than fixed geometry.

- **Sat-NeRF** — Adapts neural radiance fields to RPC camera models with explicit
  shadow and transient-object modelling for multi-date imagery. ⚠ `CVPRW`
  [`paper`](https://arxiv.org/abs/2203.08896) `CVPRW'22` `3D reconstruction, novel view synthesis`
- **EO-NeRF** — Renders shadows geometrically from the reconstructed surface
  rather than learning them per-image, which makes the resulting digital surface
  models markedly more accurate. ⚠ `CVPRW`
  [`paper`](https://doi.org/10.1109/cvprw59228.2023.00197) `CVPRW'23` `DSM generation, 3D reconstruction`
- **EOGS** — First Gaussian splatting method for digital elevation modelling,
  carrying over EO-NeRF's radiometric and shadow modelling while reaching
  comparable accuracy roughly 300× faster — minutes rather than a day.
  [`paper`](https://arxiv.org/abs/2412.13047) `CVPR'25` `3D reconstruction, DSM`

### Airborne LiDAR

Worth being direct about this one: airborne LiDAR segmentation is mostly done
with **general-purpose 3D architectures** rather than remote-sensing-specific
ones. The field's workhorses — KPConv (ICCV'19), RandLA-Net (CVPR'20) — predate
this list's 2021 cutoff, and their successors are general 3D vision papers that
happen to be applied to aerial data.

- **Point Transformer V3** — Trades sophisticated attention mechanisms for scale,
  using serialised point ordering to process far larger point clouds; the current
  default backbone for large-scale LiDAR segmentation, aerial included. Not
  remote-sensing-specific.
  [`paper`](https://doi.org/10.1109/cvpr52733.2024.00463) [`code`](https://github.com/Pointcept/PointTransformerV3) `CVPR'24` `point cloud segmentation` [![Stars](https://img.shields.io/github/stars/Pointcept/PointTransformerV3?style=social)](https://github.com/Pointcept/PointTransformerV3/stargazers)

Remote-sensing-specific airborne LiDAR architectures are published almost
entirely in the ISPRS Annals, MDPI *Remote Sensing* and similar venues outside
this list's policy. If you work in this area, the ISPRS 3D Semantic Labeling
benchmark and the FRACTAL dataset are the standard evaluation targets, and
[Pointcept](https://github.com/Pointcept/Pointcept) is the most practical
codebase.

### Cross-modal 3D

Work listed under other families:

- [HLMamba](#fusion-detection-denoising--restoration) — hyperspectral and LiDAR
  fusion via cross-modal state-space blocks. `TGRS'24`
- [RSMTMamba](#dense-prediction--segmentation) — joint segmentation, height
  estimation and boundary detection. `TGRS'25`

---

## Surveys

Start here if you are new to a sub-area. A good survey saves weeks; these are the
ones that are current enough to be worth the time.

- **Vision Foundation Models in Remote Sensing: A Survey** — Covers foundation
  models from June 2021 to June 2024, organised by pretraining strategy and by
  task level (image, pixel, region, spatiotemporal). The most complete single
  reference for this list's largest section.
  [`paper`](https://arxiv.org/abs/2408.03464) `IEEE GRSM'25`
- **Foundation Models for Remote Sensing and Earth Observation: A Survey** —
  Complementary coverage with a stronger emphasis on Earth observation
  applications rather than architecture taxonomy.
  [`paper`](https://arxiv.org/abs/2410.16602) `preprint`
- **A Survey on Remote Sensing Foundation Models: From Vision to Multimodality** —
  Traces the shift from single-sensor vision models to the multimodal and
  any-sensor designs that now dominate.
  [`paper`](https://arxiv.org/abs/2503.22081) `preprint`
- **Vision Mamba in Remote Sensing: A Comprehensive Survey** — The reference
  survey for the state-space family, with a companion repository tracking new
  work.
  [`paper`](https://arxiv.org/abs/2505.00630) `preprint`
- **Remote Sensing SpatioTemporal Vision-Language Models: A Comprehensive
  Survey** — Covers the vision-language literature with attention to the
  temporal dimension, which most VLM surveys skip.
  [`paper`](https://arxiv.org/abs/2412.02573) `preprint`

---

## Benchmarks & Datasets

This is not an exhaustive catalogue — [satellite-image-deep-learning/datasets](https://github.com/satellite-image-deep-learning/datasets)
already does that well. The purpose here is narrower: these are the benchmarks the
architectures in this list actually report on, so you can tell which numbers are
comparable to which.

Two warnings before you use any of them.

**Check the split protocol before comparing numbers.** Several of these ship an
official split and several do not, and papers using self-constructed splits are
not comparable with papers using the official one even on the same dataset. The
hyperspectral case is severe enough to have [its own note](#a-note-on-hsi-benchmarking).

**Saturation is common.** On several long-standing benchmarks the spread between
the top ten methods is smaller than the variance from training seed and
augmentation choices. A new state of the art on Potsdam means considerably less
than a new state of the art on a large, geographically diverse benchmark.

### Semantic segmentation

| Dataset | Scale | Notes |
|---|---|---|
| **ISPRS Potsdam / Vaihingen** | 38 / 33 tiles, 6 classes | The long-standing default. Small, urban, European, and effectively saturated — treat as a sanity check. Used by [UNetFormer](#semantic-segmentation), [DC-Swin](#semantic-segmentation), [DeepKANSeg](#efficient-architectures). |
| **LoveDA** | 5,987 patches at 1024², 0.3 m, 7 classes | Explicitly built for *domain adaptive* segmentation, with an urban/rural split that creates a real distribution shift. [`paper`](https://arxiv.org/abs/2110.08733) |
| **iSAID** | 2,806 images, 655,451 instances, 15 classes | Instance segmentation on very large aerial images. The instance counts per image are far higher than in natural-image benchmarks. |
| **OpenEarthMap** | 5,000 images, 0.25-0.5 m, 8 classes | Global coverage across 97 regions and 44 countries, which makes it a much better generalisation test than the European city benchmarks. [`paper`](https://arxiv.org/abs/2210.10732) |
| **Five-Billion-Pixels** | 150 Gaofen-2 images, 24 classes | >5 billion labelled pixels with a fine-grained 24-category system; useful when you need category depth rather than image count. |

### Change detection

| Dataset | Scale | Notes |
|---|---|---|
| **LEVIR-CD** | 637 pairs at 1024², 0.5 m | Building change across 20 Texas regions, 2002-2018. The near-universal default. Used by [BIT](#change-detection), [Changer](#change-detection), [ChangeMamba](#change-detection-1), [CDMamba](#change-detection-1). |
| **S2Looking** | 5,000 pairs at 1024², 0.5-0.8 m | Deliberately harder: side-looking off-nadir imagery across five continents, so registration error and parallax are part of the problem. [`paper`](https://arxiv.org/abs/2107.09244) |
| **WHU-CD** | 1 pair, 32,507×15,354 | Building change over Christchurch after the 2011 earthquake. |
| **SECOND** | 4,662 pairs | *Semantic* change detection — what the land became, not merely that it changed. Used by [SCanNet](#change-detection), [ChangeMamba](#change-detection-1). |
| **xBD / xView2** | 22,068 images | Building damage assessment with four-level damage grading across 19 disasters. The damage grading makes it ordinal, not binary. |

### Object detection

| Dataset | Scale | Notes |
|---|---|---|
| **DOTA (v1.0 / v2.0)** | 2,806 / 11,268 images, 15 / 18 classes | The standard oriented detection benchmark. Images run to tens of thousands of pixels and are normally tiled to 1024². Used by [Oriented R-CNN](#oriented-object-detection), [LSKNet](#oriented-object-detection), [ARC](#oriented-object-detection), [PKINet](#oriented-object-detection), [RVSA](#plain-vit-backbones--parameter-scaling). |
| **DIOR / DIOR-R** | 23,463 images, 192,472 instances, 20 classes | Broad geographic coverage across 80+ countries; DIOR-R adds oriented boxes. |
| **FAIR1M** | 40,000+ images, 1M+ instances | Fine-grained: 5 categories and 37 subcategories, so it tests discrimination rather than localisation. |
| **SARDet-100K** | 116,598 images | The first COCO-scale multi-class SAR detection benchmark. Ships with [MSFA](#oriented-object-detection) pretraining. |

### Scene classification & pretraining corpora

| Dataset | Scale | Notes |
|---|---|---|
| **NWPU-RESISC45** | 31,500 images, 45 classes | Standard classification benchmark; saturated but still universally reported. |
| **AID** | 10,000 images, 30 classes | Aerial scene classification, commonly paired with RESISC45. |
| **BigEarthNet / -v2** | 590,326 patches, Sentinel-1+2 | Multi-label land cover at continental scale; v2 corrects label noise in the original. The default multispectral benchmark. |
| **fMoW / fMoW-Sentinel** | 1M+ images, temporal | Functional Map of the World — the pretraining corpus behind [SatMAE](#masked-image-modeling), [Scale-MAE](#masked-image-modeling) and [SatMAE++](#masked-image-modeling). |
| **MillionAID** | 1M images | Large-scale classification corpus used for [RSP](#plain-vit-backbones--parameter-scaling) and [RVSA](#plain-vit-backbones--parameter-scaling) pretraining. |
| **SSL4EO-S12** | 251K locations × 4 seasons, 1.5 TB | The de-facto Sentinel-1/2 self-supervised pretraining corpus. See [Agency & open-release models](#agency--open-release-models). |
| **SatlasPretrain** | 856K tiles, 302M labels | Large-scale *supervised* multi-task pretraining. See [Plain-ViT backbones](#plain-vit-backbones--parameter-scaling). |
| **Major TOM** | 2.2M+ patches, ~40 TB | A grid and metadata standard rather than a fixed dataset; now the common distribution format on Hugging Face. |

### Time series & multimodal

| Dataset | Scale | Notes |
|---|---|---|
| **PASTIS / PASTIS-HD** | 2,433 Sentinel-2 time series | Panoptic crop mapping; introduced with [U-TAE](#satellite-image-time-series) and extended by [OmniSat](#multi-modal-foundation-models). The reference SITS benchmark. |
| **Sen1Floods11** | 4,831 chips | Flood mapping from paired Sentinel-1 and Sentinel-2, a standard foundation-model downstream task. |
| **SEN12MS** | 180,662 triplets | Sentinel-1 SAR, Sentinel-2 optical and land cover, aligned. Common for SAR-optical fusion. |

### Evaluation suites

Prefer these to single-dataset comparisons when assessing a foundation model —
they exist because per-dataset numbers proved unreliable for that purpose.

- **GEO-Bench** — 6 classification and 6 segmentation tasks with a defined
  aggregation methodology. See [Benchmarks for foundation models](#benchmarks-for-foundation-models).
- **PANGAEA** — 12 datasets across 7 task types, built to correct geographic bias.
- **Copernicus-Bench** — 15 hierarchical tasks structured by Sentinel mission.

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
| Vision-language models | **Complete** — contrastive, MLLM, promptable and benchmarks |
| Diffusion & generative | **Complete** — controllable generation, restoration, synthetic labels |
| SAR | **Complete** — despeckling, interferometry, recognition, detection and fusion |
| 3D, LiDAR & neural fields | **Complete** — neural fields, Gaussian splatting and airborne LiDAR |
| Satellite image time series | **Complete** — U-TAE, TSViT and Exchanger, plus cross-references |
| Surveys | **Complete** — five current surveys covering the main families |
| Benchmarks & datasets | **Complete** — segmentation, CD, detection, classification, SITS and evaluation suites |

Every section marked complete contains only entries verified against a primary
source, plus cross-references to relevant models indexed under other families.
Complete means the main architectural threads are covered, not that the section
is exhaustive — the gaps below are named rather than papered over, and
contributions toward them are especially welcome.

**Known entries awaiting verification.** These are real papers that belong in the
list, held back only because their venue could not be confirmed against a primary
source during indexing:

- **Presto** and **TESSERA** — widely used pixel-timeseries models with released
  weights. Crossref holds no record for either, and the strongest evidence found
  for Presto is an ICLR 2024 *submission* header, which is not an acceptance.
  Both stay out until a venue can be confirmed.
- **InSAR** and **airborne LiDAR** are now covered, but thinly, and deliberately
  so. Both sub-fields publish mainly in venues outside this list's policy, and
  airborne LiDAR is dominated by general-purpose 3D architectures that predate
  the 2021 cutoff. Each section says so rather than padding itself with
  out-of-policy entries.

### On verification

Every venue in the completed sections was checked against a primary source
(publisher page, proceedings listing, or the arXiv comments field), and every code
link was confirmed to resolve. Star counts are live badges rather than recorded
numbers, so they cannot go stale.

Where a venue could not be confirmed, the entry is marked `preprint` rather than
given a plausible-looking guess. Where a venue is reported by a secondary source
but not yet confirmed against a publisher or proceedings page, it carries a
† marker. A wrong citation in a list like this propagates into other people's
bibliographies, so absence is preferred to invention.

Known verification gaps, stated plainly:

- The three †-marked vision-language entries have confirmed arXiv identifiers and
  titles, but their journal/conference venues rest on secondary sources.
- Star counts were captured at indexing time and are not re-checked on every commit.

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
