# ML_for_CMS_particle_quality

Author: Claire Savard  
Email: claire.savard@colorado.edu

In run_ML_analysis.ipynb, I create and train a neural network
and a gradient boosted decision tree for the CMS particle
quality clsssification task. This script shows a few metrics
that I use to compare the performances of these 2 classifiers
and against a set of physics cuts used by some of the CMS
community.

Before running, you will need to install:
1. *jupyter notebook (https://jupyter.org/)
2. *scikit-learn (https://scikit-learn.org/stable/install.html)
3. *keras (https://keras.io/#installation)
4. uproot (https://pypi.org/project/uproot/)  
*I suggest you install anaconda 
(https://www.anaconda.com/distribution/) which will install
all packages 1-3 necessary from python.

You can also run this as a python (.py) file if your prefer
that to a jupyter notebook. To do that, you need to create a
.py file and copy and paste the code into it, then you can
run it using "python <filename>.py".
