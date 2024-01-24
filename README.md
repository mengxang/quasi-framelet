Quasi-Framelets: Robust Graph Neural Networks via Adaptive Framelet Convolution

The work aims to provide a novel design of a multiscale framelet convolution for spectral graph neural networks. 
The new framelet convolution decomposes graph data into low-pass and high-pass spectra in a fine-tuned multiscale manner and directly designs filtering functions in the spectral domain. 
These new designs offer great advantages in filtering out unwanted spectral information and effectively mitigating the negative effect of noisy graph signals.  
Also, extensive experiments on real-world graphs validate that our new framelet convolution can achieve superior node classification performance, demonstrating strong robustness on noisy graphs and under adversarial attacks. 

For adversarial attacks experiments, we employed attacked datasets provided by the work, Elastic GNN. They provided the data available at https://github.com/lxiaorui/ElasticGNN/tree/master/data/attacked_graph.
