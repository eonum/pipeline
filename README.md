pipeline
========

Pipeline is a modular machine learning library used in eonum's products and services.

Using modular design principles, a data pipeline suited for the problem at hand can be built and evaluated. A data pipeline is composed of readers, tranformers, classifiers, regressors, normalizer, parameter optimizer, output generators and other modules.
The software can process simple vectors and sequences / time series data. Structured as well as unstructured data can be processed with a corresponding reader.

Below you can find a selection of pipeline modules.

#### Data structures
* Vectors / Instances, sparse and dense implementations
* Data Sets
* Sequences / Time series, sparse and dense implentations
* Graphs

#### Classifier / Regressors
* Random Forests / Decision Trees / CART
* Neural Nets
* Recurrent Neural Nets (Long Short Term Memory)
* Support Vector Machines
* Linear Regression
* Gradient Boosting
* Nearest Neighbor
* Ensemble Methods (Bagging / Boosting)

#### Distance metrics
* Euclid
* Cosine Distance
* Minkowski Distance
* Graph Edit Distance
* Polynomial Kernel

#### Optimization
* Genetic Algorithms
* Gradient descent

#### Clustering / Data Mining
* Self organizing maps / Kohonen map
* Gaussian Mixture Models
* K-Means Clustering
* EM Fuzzy Clustering

#### Transformators
* Principal Component Analysis
* Feature extraction and selection
* Dynamic Time Warping

#### Validation
* k-fold cross validation
* evaluation metrics (RMSE, AUC, Recognition Rate, LogLoss)
* Validation of meta parameters of entire data pipelines.
