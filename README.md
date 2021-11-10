# SGNs-master
This is a project for a TKDE 2019 paper "Subgraph Networks With Application to Structural Feature Space Expansion". Please stay tuned for future updates!

**Update:** This paper has been published in IEEE TKDE, Volume: 33, Issue: 6, June 1 2021. [PaperLink](https://ieeexplore.ieee.org/document/8924759) 

Real-world networks exhibit prominent hierarchical and modular structures, with various subgraphs as building blocks. Most existing studies simply consider distinct subgraphs as motifs and use only their numbers to characterize the underlying network. Although such statistics can be used to describe a network model, or even to design some network algorithms, the role of subgraphs in such applications can be further explored so as to improve the results. In this article, the concept of subgraph network (SGN) is introduced and then applied to network models, with algorithms designed for constructing the 1st-order and 2nd-order SGNs, which can be easily extended to build higher-order ones. Furthermore, these SGNs are used to expand the structural feature space of the underlying network, beneficial for network classification. Numerical experiments demonstrate that the network classification model based on the structural features of the original network together with the 1st-order and 2nd-order SGNs always performs the best as compared to the models based only on one or two of such networks. In other words, the structural features of SGNs can complement that of the original network for better network classification, regardless of the feature extraction method used, such as the handcrafted, network embedding and kernel-based methods.

## SGN model
![SGN](https://user-images.githubusercontent.com/26339035/125916703-f8d29f71-adae-42ac-a374-967e4ed6e402.png)


## Cite
Please cite our paper if you use this code in your own work:

```
@article{xuan2021subgraph,
  title={Subgraph Networks With Application to Structural Feature Space Expansion},
  author={Xuan, Qi and Wang, Jinhuan and Zhao, Minghao and Yuan, Junkun and Fu, Chenbo and Ruan, Zhongyuan and Chen, Guanrong},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  volume={33},
  number={6},
  pages={2776--2789},
  year={2021},
  publisher={IEEE COMPUTER SOC}
}
```
