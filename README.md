Pipeline
========

Pipeline is a modular machine learning library used in eonum's products and services.

Using modular design principles, a data pipeline suited for the problem at hand can be built and evaluated. A data pipeline is composed of readers, tranformers, classifiers, regressors, normalizers, parameter optimizers, output generators and other modules.
The software can process simple vectors, sequences / time series data and graphs. Structured as well as unstructured data can be processed with a corresponding reader.

Below you can find a selection of pipeline modules.

##### Data structures
* Vectors / Instances, sparse and dense implementations
* Data Sets
* Sequences / Time series, sparse and dense implentations
* Graphs

##### Classifiers / Regressors
* Random Forests / Decision Trees / CART
* Neural Nets
* Recurrent Neural Nets (Long Short Term Memory)
* Support Vector Machines
* Linear Regression
* Gradient Boosting
* Nearest Neighbor
* Ensemble Methods (Bagging / Boosting)

##### Distance metrics
* Euclid
* Cosine Distance
* Minkowski Distance
* Graph Edit Distance
* Polynomial Kernel
* Dynamic Time Warping

##### Optimization
* Genetic Algorithms
* Gradient descent

##### Clustering
* Self organizing maps / Kohonen map
* Gaussian Mixture Models
* K-Means Clustering
* EM Fuzzy Clustering

##### Transformer
* Principal Component Analysis
* Feature extraction and selection

##### Validation
* k-fold cross validation
* evaluation metrics (RMSE, AUC, Recognition Rate, LogLoss)
* Validation of meta parameters of entire data pipelines.


## Getting started
The easiest way is to import the code into Eclipse as an existing Project after having invoked `git clone git@github.com:eonum/pipeline.git`. The code is configured as a Maven Java Project and can also be used without Eclipse.
The only dependecy which is not resolved automatically by Maven is gnuplot. In order to produce accuracy and validation plots during training and validation the command line tool 'gnuplot' is required.
Your application will not break without the gnuplot dependency. You will just have some error messages and no plots.

On Ubuntu/Debian gnuplot is installed as follows.
```bash
sudo apt-get install gnuplot
```
