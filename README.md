# Introduction
- Paper name "Semantically Enhanced Attention Map-Driven Occluded Person Re-identification" . Our network named AMD-Net
- This article was submitted for publication in “IEEE Signal processing letters”. The authors are Yiyuan Ge, M.S. and Mingxin Yu, Ph.
- We are currently providing the key code for this paper, and we promise that the full code will be released after acceptance of the article

# Abstract

- Occluded person Re-identification (Re-ID)  aims to identify a particular person when the person's body parts are occluded. However, challenges remain in enhancing effective information representation and suppressing background clutter when considering occlusion scenes. In this paper, we propose a novel Attention Map-Driven Net-work (AMD-Net) for occluded person Re-ID. In AMD-Net, human parsing labels are introduced to supervise the generation of partial attention maps, while we suggest a Fourier-based Spatial-fequency Interaction Module (FSIM) to complement the higher-order semantic information from the frequency domain. Furthermore, we propose a Taylor-inspired Global-partial Feature Filter (TGFF) for mitigating background disturbance and extracting fine-grained features. Moreover, we also design a part-soft triplet loss, which is robust to non-discriminative body partial features. We compare the proposed AMD-Net on four standard datasets: Occluded-duke, Occluded-reid, Market-1501, and Duke-MTMC. Experimental results show that our method outperforms existing state-of-the-art methods. 


# Dependencies

- Python 3.6

- PyTorch 1.6.0

- torchreid


# Arithmetic Support

- The model is trained on one RTX 4090 GPU
