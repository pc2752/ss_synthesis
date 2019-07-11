
<h1>A Vocoder Based Method For Singing Voice Extraction</h1>

<h2>Pritish Chandna, Merlijn Blaauw, Jordi Bonada, Emilia Gómez</h2>

<h2>Music Technology Group, Universitat Pompeu Fabra, Barcelona</h2>

This repository contains the source code for the paper with the same title.
Please note that the model presented here is currently configured just for the iKala dataset, as published in the corresponding paper, but can also be used for other commerical songs. For examples of the output of the system, please visit: <i>https://pc2752.github.io/singing_voice_sep/</i>

<h3>Installation</h3>
To install, clone the repository and use <pre><code>pip install requirements.txt </code></pre> to install the packages required.

 The main code is in the *train_tf.py* file.  To use the file, you will have to download the <a href="https://drive.google.com/file/d/11ReUgbp1veEDWEBbt30YjvFe2mT34C0G/view?usp=sharing" rel="nofollow"> model weights</a> and place it in the *log_dir_m1* directory, defined in *config.py*. Wave files to be tested should be placed in the *wav_dir*, as defined in *config.py*. You will also require <a href="http://www.tensorflow.org" rel="nofollow">TensorFlow</a> to be installed on the machine. 

<h3>Data pre-processing</h3>

Once the *iKala* files have been put in the *wav_dir*, you can run <pre><code>python prep_data_ikala.py</code></pre> to carry out the data pre-processing step.

<h3>Training and inference</h3>


Once setup, you can run the command <pre><code>python main.py -t</code></pre> to train or <pre><code>python main.py -e &lt;filename&gt;</code></pre> to synthesize the output from an hdf5 file or <pre><code>python main.py -v &lt;filename&gt;</code></pre> for a .wav file. The output will be saved in the *val_dir* specified in the *config.py* file. The plots show the ground truth and output values for the vocoder features as well as the f0 and the accuracy. Note that plots are only supported for *iKala* songs as the ground truth is available for these songs.  
 

 
We are currently working on future applications for the methodology and the rest of the files in the repository are for this purpose, please ignore. We will further update the repository in the coming months. 


<h2>Acknowledgments</h2>
The TITANX used for this research was donated by the NVIDIA Corporation. This work is partially supported by the Towards Richer Online Music Public-domain Archives <a href="https://trompamusic.eu/" rel="nofollow">(TROMPA)</a> project.

