# End-to-End Learning of  Probabilistic Hierarchies on Graphs

<p align="center">
<img src="https://www.in.tum.de/fileadmin/w00bws/daml/fph/model_overview.png" width="400">
</p>

Implementation of the paper:   
**[End-to-End Learning of Probabilistic Hierarchies on Graphs](https://openreview.net/forum?id=g2LCQwG7Of)**

by Daniel Zügner, Bertrand Charpentier, Morgane Ayle, Sascha Geringer, Stephan Günnemann.   
Published at ICLR'22.

Copyright (C) 2022   
Daniel Zügner   
Technical University of Munich    

## Additional resources
[[Paper](https://openreview.net/forum?id=g2LCQwG7Of) | [Poster](https://www.in.tum.de/fileadmin/w00bws/daml/fph/iclr_2022_poster.pdf) | [Slides](https://www.in.tum.de/fileadmin/w00bws/daml/fph/presentation.pdf)]

## Run the code
 
The fastest way to try our code is to use the Jupyter notebook `notebooks/demo.ipynb`.

In order to reproduce our results, refer to `notebooks/experiments.ipynb` as well as the hyperparameter configurations in `configs/`.


## Installation
### With GPU support
```bash
conda create -f env.yml
pip install -e .
```

### CPU only
```bash
conda create -f env.cpu.yml
pip install -e .
```
 
## Contact
Please contact zuegnerd@in.tum.de in case you have any questions.

## References
### Datasets
In the `data` folder we provide the following datasets originally published by   
#### Cora
McCallum, Andrew Kachites, Nigam, Kamal, Rennie, Jason, and Seymore, Kristie.  
*Automating the construction of internet portals with machine learning.*   
Information Retrieval, 3(2):127–163, 2000.

and the graph was extracted by

Bojchevski, Aleksandar, and Stephan Günnemann. *"Deep gaussian embedding of   
attributed graphs: Unsupervised inductive learning via ranking."* ICLR 2018.

#### Citeseer
Sen, Prithviraj, Namata, Galileo, Bilgic, Mustafa, Getoor, Lise, Galligher, Brian, and Eliassi-Rad, Tina.   
*Collective classification in network data.*   
AI magazine, 29(3):93, 2008.

#### PubMed
Sen, Prithviraj, Namata, Galileo, Bilgic, Mustafa, Getoor, Lise, Galligher, Brian, and Eliassi-Rad, Tina.   
*Collective classification in network data.*   
AI magazine, 29(3):93, 2008.

#### PolBlogs
Adamic, Lada A., and Natalie Glance. *The political blogosphere and the 2004 US election: divided they blog.* Proceedings of the 3rd international workshop on Link discovery. 2005.

#### Brain
Amunts, Katrin, et al. *BigBrain: an ultrahigh-resolution 3D human brain model.* Science 340.6139 (2013): 1472-1475.

#### Genes
Cho, Ara, et al. *WormNet v3: a network-assisted hypothesis-generating server for Caenorhabditis elegans.* Nucleic acids research 42.W1 (2014): W76-W82.

#### WikiPhysics
Aspert, Nicolas, et al. *A graph-structured dataset for Wikipedia research.* Companion Proceedings of The 2019 World Wide Web Conference. 2019.

#### OpenFlights
Jani Patokallio. *Openflight*. online https://openflights.org.

#### ogbn-products, ogbn-arxiv, ogbl-collab
Hu, Weihua, et al. *Open graph benchmark: Datasets for machine learning on graphs.* Advances in neural information processing systems 33 (2020): 22118-22133.

#### DBLP
Yang, Jaewon, and Jure Leskovec. *Defining and evaluating network communities based on ground-truth.* Knowledge and Information Systems 42.1 (2015): 181-213.

## Cite
Please cite our paper if you use the model or this code in your own work:

```
@inproceedings{
zugner2022endtoend,
title={End-to-End Learning of Probabilistic Hierarchies on Graphs},
author={Daniel Z{\"u}gner and Bertrand Charpentier and Morgane Ayle and Sascha Geringer and Stephan G{\"u}nnemann},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=g2LCQwG7Of}
}

```
