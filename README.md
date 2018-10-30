
<h1>A Vocoder Based Method For Singing Voice Extraction</h1>

<h2>Pritish Chandna, Merlijn Blaauw, Jordi Bonada, Emilia Gómez</h2>

<h2>Music Technology Group, Universitat Pompeu Fabra, Barcelona</h2>

This repository contains the source code for the paper with the same title. The main code is in the train_tf.py file.

To use the file, you will have to download the model weights (to be uploaded shortly) and place it in the "log_dir_m1" directory, defined in config.py. Wave files to be tested should be placed in the wav_dir, as defined in config.py. You will also require TensorFlow to be installed on the machine. 

Once setup, you can run the command <pre><code>python train_tf.py -t</code></pre> to train or python train_tf.py -s <filename> -p (optional, for plots) to synthesize the output. Note that plots are only supported for iKala songs as the gorund truth is available for these songs. 
  
  
