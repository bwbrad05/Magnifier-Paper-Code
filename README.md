<h1 align="center">
Magnifier: A Multi-grained Neural Network-based Architecture for Burned Area Delineation
</h1>

<div align="center">
<br />

[![Project license](https://img.shields.io/github/license/DarthReca/magnifier-california)](LICENSE)
[![code with love by DarthReca](https://img.shields.io/badge/%3C%2F%3E%20with%20%E2%99%A5%20by-DarthReca-ff1414.svg?style=flat-square)](https://github.com/DarthReca)

</div>

<details open="open">
<summary>Table of Contents</summary>
  
- [About](#about)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [License](#license)
- [Authors & contributors](#authors--contributors)

</details>

---

## About

This is the source code for the paper "Magnifier: A Multi-grained Neural Network-based Architecture for Burned Area Delineation".
In this paper, we propose a novel methodology, namely Magnifier, to improve segmentation performance with limited data availability. The Magnifier methodology is applicable to any existing encoder-decoder architecture, as it extends a model by merging information at different contextual levels through a dual-encoder approach: a local and global encoder. **Magnifier analyzes the input data twice using the dual-encoder approach. In particular, the local and global encoders extract information from the same input at different granularities.** This allows Magnifier to extract more information than the other approaches, given the same set of input images. Magnifier improves the quality of the results of +2.65% on average IoU while leading to a restrained increase in terms of the number of trainable parameters compared to the original model. We evaluated our proposed approach with state-of-the-art burned area segmentation models, demonstrating, on average, comparable or better performances in less than half of the GFLOPs.

You can find it on [Arxiv](https://arxiv.org/abs/2504.19589v1) or on [IEEE](https://ieeexplore.ieee.org/document/10980409).

## Getting Started

All required packages can be found in _requirements.txt_

## Usage

Run _main.py_ with a configuration file, that can be selected from the ones in _configs_ folder.

## License

This project is licensed under the **Apache-2.0 license**. See [LICENSE](LICENSE) for more information.

Some files have different licenses:

- unified_focal_loss.py: [Apache-2.0](licenses/UNIFIED_FOCAL)
- hf_segformer.py, hf_segformer_backbone.py, hf_segformer_head.py: [Apache-2.0](licenses/HF_LICENSE)

## Authors & contributors

The original setup of this repository is by [Daniele Rege Cambrin](https://github.com/DarthReca).

For a full list of all authors and contributors, see [the contributors page](https://github.com/DarthReca/magnifier-california/contributors).

```bibtex
@ARTICLE{10980409,
  author={Cambrin, Daniele Rege and Colomba, Luca and Garza, Paolo},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={Magnifier: A Multi-grained Neural Network-based Architecture for Burned Area Delineation}, 
  year={2025},
  volume={},
  number={},
  pages={1-16},
  keywords={Computer architecture;Deep learning;Semantic segmentation;Neural networks;Data models;Computer vision;Transformers;Computational modeling;Remote sensing;Machine learning;Earth Observation;Natural Hazard Management;Deep Learning;Semantic Segmentation;Post-Wildfire Segmentation},
  doi={10.1109/JSTARS.2025.3565819}}
```
