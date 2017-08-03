StackGAN in Tensorflow
======================

Tensorflow implementation of text to image synthesis with [Stacked Generative Adversarial Network](https://arxiv.org/abs/1606.03657) for MNIST handwritten digit dataset.

Prerequisites
-------------

This code requires [Tensorflow](https://www.tensorflow.org/) and [OpenCV](http://opencv.org). The MNIST dataset is stored in the 'MNIST_data' directory. The files will be automatically downloaded if the dataset does not exist.
    
If you want to use `--show_progress True` option, you need to install python package `progress`.

    $ pip install progress

Usage
-----

To train a vanilla InfoGAN with z dimension 14, stage1 generator input code dimension 2, stage 2 generator input code dimension 10, run the following command:

    $ python main.py --z_dim 14 --c0_dim 2 --c1_dim 10

To see all training options, run:

    $ python main.py --help

which will print:

    usage: main.py [-h] [--X0_dim X0_DIM] [--X1_dim X1_DIM] [--nwords NWORDS]
               [--vocab_size VOCAB_SIZE] [--z_dim Z_DIM] [--c0_dim C0_DIM]
               [--c1_dim C1_DIM] [--e_dim E_DIM] [--d_update D_UPDATE]
               [--g0_nepoch G0_NEPOCH] [--g1_nepoch G1_NEPOCH]
               [--batch_size BATCH_SIZE] [--lmda LMDA] [--lr LR]
               [--checkpoint_dir CHECKPOINT_DIR] [--image_dir IMAGE_DIR]
               [--retrain_stage1 [RETRAIN_STAGE1]] [--noretrain_stage1]
               [--retrain_stage2 [RETRAIN_STAGE2]] [--noretrain_stage2]
               [--use_adam [USE_ADAM]] [--nouse_adam]
               [--show_progress [SHOW_PROGRESS]] [--noshow_progress]

    optional arguments:
      -h, --help            show this help message and exit
      --X0_dim X0_DIM       dimension of the small image [196]
      --X1_dim X1_DIM       dimension of the original image [784]
      --nwords NWORDS       number of words in the input sentence (e.g. "thin
                            number one with left skew") [6]
      --vocab_size VOCAB_SIZE
                            size of the vocabulary [19]
      --z_dim Z_DIM         dimension of the generator input noise variable z
                            [20]
      --c0_dim C0_DIM       dimension of stage1 generator input code variable c0
                            [2]
      --c1_dim C1_DIM       dimension of stage2 generator input code variable c1
                            [10]
      --e_dim E_DIM         dimension of the word embedding phi [20]
      --d_update D_UPDATE   update the discriminator weights [d_update] times per
                            generator update [5]
      --g0_nepoch G0_NEPOCH
                            number of epochs to use during training of stage-1 GAN
                            [51]
      --g1_nepoch G1_NEPOCH
                            number of epochs to use during training of stage-2 GAN
                            [51]
      --batch_size BATCH_SIZE
                            batch size to use during training [128]
      --lmda LMDA           the regularization term that parameterizes Kullback-
                            Leibler divergence loss term [1.]
      --lr LR               learning rate of the optimizer to use during training
                            [0.001]
      --checkpoint_dir CHECKPOINT_DIR
                            checkpoint directory [.\checkpoints]
      --image_dir IMAGE_DIR
                            directory to save generated images to [.\images]
      --retrain_stage1 [RETRAIN_STAGE1]
                            whether to retrain stage1 GAN [True]
      --noretrain_stage1
      --retrain_stage2 [RETRAIN_STAGE2]
                            whether to retrain stage2 GAN [True]
      --noretrain_stage2
      --use_adam [USE_ADAM]
                            if True, use Adam optimizer; otherwise, use SGD [True]
      --nouse_adam
      --show_progress [SHOW_PROGRESS]
                            print progress [False]
      --noshow_progress

(Optional) If you want to see a progress bar, install `progress` with `pip`:

    $ pip install progress
    $ python main.py main.py --z_dim 14 --c0_dim 2 --c1_dim 10 --show_progress True

Notes
-----

The Annotated_MNIST.py is a thickness and skew labeler for MNIST handwritten digit dataset, intended to be used for toy text to image generation tasks or specific classification tasks. More details can be found in [this](https://github.com/1202kbs/Annotated_MNIST) repository.