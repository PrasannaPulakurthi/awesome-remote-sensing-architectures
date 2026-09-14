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
