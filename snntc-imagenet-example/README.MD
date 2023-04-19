This is an example showing how to use SNN-TC to construct and train Imagenet models. 

Because the train_label.txt file is over the 25MB limit of GitHub, we provide the compressed zip file. Please run

       gunzip train_label.txt.gz

to decompress it before using.

To try the code, make sure you have downloaded ImageNet dataset, and change the imagenet data folder in config.py to your imagenet data folder.
Then run

       python snntc-imagent.py

