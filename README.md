
<h1>A Vocoder Based Method For Singing Voice Extraction</h1>

<h2>Pritish Chandna, Merlijn Blaauw, Jordi Bonada, Emilia Gómez</h2>

<h2>Music Technology Group, Universitat Pompeu Fabra, Barcelona</h2>

This repository contains the source code for the paper with the same title. The main code is in the *train_tf.py* file.

To use the file, you will have to download the model weights (to be uploaded shortly) and place it in the *log_dir_m1* directory, defined in *config.py*. Wave files to be tested should be placed in the *wav_dir*, as defined in *config.py*. You will also require <a href="http://www.tensorflow.org" rel="nofollow">TensorFlow</a> to be installed on the machine. 

<h3>Data pre-processing</h3>

Once the *iKala* files have been put in the *wav_dir*, you can run <pre><code>python prep_data_ikala.py</code></pre> to carry out the data pre-processing step.

<h3>Training and inference</h3>


Once setup, you can run the command <pre><code>python train_tf.py -t</code></pre> to train or <pre><code>python train_tf.py -s &lt;filename&gt; -p (optional, for plots)</code></pre> to synthesize the output.The output will be saved in the *val_dir* specified in the *config.py* file. Note that plots are only supported for *iKala* songs as the ground truth is available for these songs. 
  
We are currently working on future applications for the methodology and the rest of the files in the repository are for this purpose, please ignore. We will further update the repository in the coming months. 


<h2>Acknowledgments</h2>
The TITANX used for this research was donated by the NVIDIA Corporation. This work is partially supported by the Towards Richer Online Music Public-domain Archives <a href="https://trompamusic.eu/" rel="nofollow">(TROMPA)</a> project.

