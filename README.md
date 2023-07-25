# **Deep Learning Course**
## Introduction
> 3 layers cloud computing, First, we use vargrant to enable three virtual machines as front-end, back-end, and data base.

${\Large\color{orange}A0 \ \ classfication \ \  \Large\color{pink}try}$
- try classfication model with `CIFAR10`
- Learned
    - write dataset class, dataloader
    - data transform with pytorch
    - build CNN model
    - simple train, test part

${\Large\color{orange}A1 \ \ classfication \ \  \Large\color{pink}practice}$
- practice classfication model with `Mnist, KMnist, Flower102`
- learned
    - dynamic add transform when training phase
    - use `PrintLayer` to debug model layer
    - Comparing the effects of different activation finction `relu, leakyRelu, elu`
    - Accelerating training with `amp.GradScaler()`

${\Large\color{orange}A2  \ \  Feed  \ \ forward  \ \ NN \ \  \Large\color{pink}Implement}$
- Implement FN from scratch
- learned
    - how NN work  (forward, backward)

${\Large\color{orange}A3  \ \  LSTM \ \ \Large\color{pink}Implement}$
- Solving simple NLP Problems with LSTM
- learned
    - how LSTM worked (`EOS, SOS, inference`)
    - `embedding layer, position encoding`
    - Impact of `teacher forcing`

${\Large\color{orange}A4  \ \  Generative \ \ Model \ \ \Large\color{pink}train}$
 - preporcess binary dataset `EMnist`, and try 3 generative model `DCGAN, Cycle GAN, Conditional diffusion model`
 - learned
    - preporcess binary dataset `EMnist` to npy file
    - implement `residual Block`
    - implement `resNet`
    - Compare cycle gan different training methods
    - try `Conditional diffusion model`

${\Large\color{orange}A5  \ \  Unet \ \ Model \ \ \Large\color{pink}segmentation}$
- implment `Unet` from scratch, and try to segment, classify `CCAgT` dataset
- learned
    - design an `Unet model`
    - data preprocess `process two img with same transform`, `find interested part and crop`
    - implement `attention Unet model`, `attention gate`
    - use `dice loss`
