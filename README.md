# Guiding directed protein evolution with Bayesian Optimization

In this project we looked into a possibility of improving the current direct evolution workflow by reducing the experimental effort associated with directed protein evolution. By incorporating Variational Autoencoders, a novel unsupervised deep generative model and Bayesian Optimization we seek to reduce the number of expensive experiments needed to find the target protein with desired properties. Starting with a discrete sequence of amino acids of our protein of choice (the wildtype) and it’s multiple-sequence-aligned neighboring sequences (being proteins with the same function found elsewhere in nature), we use a VAE to learn a non-linear mapping from discrete sequence of aminoacids into a latent continuous space. We then use Bayesian Optimization in the latent space to propose promising changes to the wildtype protein (generating mutants). The proposed approach is validated on a large published emperical fitness landscape for all 4997 single mutations in TEM-1 β-lactamase selecting for the wild-type function (Stiffler et al., 2015). We learned that our VAE+BO approach significantly outperforms random mutant selection as it easily finds at least two of the best 9 mutants in less than 200 steps.

[:page_facing_up: Full paper](paper.pdf)

## Running the code
```
main.py <number_of_latent_dimensions: 2, 5, 8, 16, 30> 
