# FACT: Towards Visually Explaining Variational Autoencoders

## Authors
TA: Christos Athanasiadis

| Name           | Email                            |
|----------------|:-------------------------:       |
| André Jesus    | andre.fialhojesus@student.uva.nl |
| Carmen Veenker | carmen.veenker@student.uva.nl    |
| Kevin Waller   | kevin.waller@student.uva.nl      |
| Qiao Ren       | qiao.ren@student.uva.nl          |

## Summary
TODO

## Requirements
The dependencies to run this projects are described in ```experiments.txt```;
TODO create and define the procedure for a conda environment in ubuntu/mac;
### For Attention Disentanglement
Download the 2D Shapes(dsprites) Dataset for the correct folder with ``` sh scripts/prepare_data.sh dsprites ```

## References
### Reference
1. Disentangling by Factorising, Kim et al.([http://arxiv.org/abs/1802.05983])
[http://arxiv.org/abs/1802.05983]: http://arxiv.org/abs/1802.05983

### References for Attention Disentanglement Replication
* The baseline codebase for this project is provided by the authors at [1] (https://github.com/liuem607/expVAE)
* The FactorVAE implementation used in the Attention Disentanglement section is copied from [2] (https://github.com/1Konny/FactorVAE)
* To implement the proposed disentanglement metric by the FactorVAE authors, the code from the following repository was adapted [3] (https://github.com/nicolasigor/FactorVAE/blob/f27136ef944b5fded7cc49ecaeb398f6909cc312/vae_dsprites_v2.py#L377)
