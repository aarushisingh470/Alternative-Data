AltDataGA2049\_2024.as20499\_sa8220\_nks8871

Team Information  
Names \- NET IDs:  
Aarushi Singh \- as20499  
Srujitha Ambati \- sa8220  
Neil Shah \- nks8871

Team Folder Name: AltDataGA2049\_2024.as20499\_sa8220\_nks8871

Project Overview  
This project involves text processing, clustering, and classification on a dataset of merchant transactions. We use various Python libraries for text vectorization, dimensionality reduction, clustering, and classification, with visualizations for data interpretation.

Data  
Location: data folder  
Description:

* Contains the 1Q20-PR.pdf file.  
* Contains the 2Q20-PR.pdf file.  
* Contains the 3Q20-PR.pdf file.  
* Contains the 4Q20-PR.pdf file.  
* Contains the facteus\_10k\_user\_panel.zip file.  
* Contains the fiscal\_calander.csv file.  
* Contains the mapped\_fiscal\_quarter\_data.csv file.  
* Contains the revenues\_kpis.csv file.

Notes: 

* The 1Q20-PR.pdf, 2Q20-PR.pdf, 3Q20-PR.pdf, and 4Q20-PR.pdf files contain the quarterly revenue segments used in homework 3\. The link to where these files are found online is in the homework 3 ipynb, and the data was entered manually.  
* When unzipped, facteus\_10k\_user\_panel.csv is in data. This csv contains the column account, date, merchant, merchant\_string\_example, merchant\_ticker, merchant\_exchange, transactions, spend, spend\_min, and spend\_max. This csv is used for data extraction, specifically merchant\_string\_example to get the merchant name to compare with the column merchant.  
* fiscal\_calander.csv contains the columns (ignoring capitalization) period\_name, period\_type, period\_start\_date, period\_end\_date, company\_id1, company\_id2, and period\_name\_standardized. It provides information about fiscal periods and their standardization.  
* mapped\_fiscal\_quarter\_data.csv (ignoring capitalization) contains the columns account, date, merchant, merchant\_string\_example, merchant\_ticker, merchant\_exchange, transactions, spend, spend\_min, spend\_max, month, year, fiscal\_quarter, and mapped\_fiscal\_quarter. It is used for mapping fiscal quarters to the transaction data.  
* revenues\_kpis.csv (ignoring capitalization) contains the columns company\_id, merchant\_ticker, merchant\_exchange, merchant\_name, fiscal\_quarter, kpiname, kpivalue, lastconsenses, is\_primary\_kpi, and include\_ticker\_in\_panel\_total. It provides key performance indicator (kpi) data for merchants, including financial metrics and inclusion flags.

Code and Notebooks  
Location: src folder  
Description: Jupyter notebooks are located here, as well as the libraries imported in the notebooks.  
Notes:  
Questions refer to the parts

1. alt\_data\_hw1.ipynb  
   1. Run the notebook cell by cell in chronological order (instructions are in the comments). Question 1 and 2 can be run independently. Question 3 includes parts of Question 2 to set it up, but it takes a sample of 500 rows from the dataframe since there is not enough memory to run Question 3 on all of the data.  
2. alt\_data\_hw2.ipynb  
   1. Run the notebook cell by cell in chronological order (instructions are in the comments). Run Question 1 before Question 2\.  
3. alt\_data\_hw3.ipynb  
   1. Run the notebook cell by cell in chronological order (instructions are in the comments). Each question relies on each other so run it in chronological order.

Libraries and Dependencies  
Location: src/libraries (used in the ipynb files)  
Instructions:

1. Install Required Libraries  
2. Load Data  
   1. Use pandas to load dataset as a DataFrame  
   2. df \= pd.read\_csv('facteus\_10k\_user\_panel.csv')  
3. Text Preprocessing  
   1. Use re to clean and preprocess text  
   2. CountVectorizer and TfidfVectorizer convert text data into numeric vectors  
4. Data Scaling and Dimensionality Reduction  
   1. Normalize the TF-IDF vectors with StandardScaler  
   2. Reduce dimensions to 2D using TSNE for visualization  
5. Clustering  
   1. Perform clustering using KMeans with k=10  
   2. Generate a Voronoi diagram using Voronoi from scipy.spatial and plot it using voronoi\_plot\_2d   
6. Classification  
   1. Split data into training and testing sets using train\_test\_split  
   2. Use LogisticRegression wrapped in OneVsRestClassifier for multi-class classification  
   3. Evaluate model performance using cross\_val\_score and classification\_report   
7. Visualization  
   1. Use matplotlib to create plots, including scatter plots for visualizing clusters and Voronoi diagrams for cluster boundaries.

Libraries Description:

1. Pandas  
   1. Usage \- import pandas as pd  
   2. Purpose  
      1. Used for data manipulation and analysis  
      2. We load and handle the dataset using Pandas DataFrames  
   3. Installation \- pip install pandas  
2. NumPy  
   1. Usage \- import numpy as np  
   2. Purpose  
      1. Provides support for large, multi-dimensional arrays and matrices  
      2. Has mathematical functions to operate on these arrays  
      3. Used for numerical computations and random data generation  
   3. Installation \- pip install numpy  
3. Matplotlib  
   1. Usage \- import matplotlib.pyplot as plt  
   2. Purpose  
      1. Used for data visualization  
      2. We created scatter plots, Voronoi diagrams, and other visualizations to interpret the data  
   3. Installation \- pip install matplotlib  
4. SciPy  
   1. Usage \- import individual modules as needed from scipy  
   2. Purpose  
      1. SciPy provides scientific and mathematical functions  
      2. SciPy sub libraries provide various plot and diagram functions  
   3. Installation \- pip install scipy  
   4. Modules  
      1. Voronoi, voronoi\_plot\_2d  
         1. Usage  
            1. from scipy.spatial import Voronoi  
            2. from scipy.spatial import voronoi\_plot\_2d  
         2. Used for creating Voronoi diagrams on K-means cluster centers  
         3. We use the spatial module for the Voronoi plot  
      2. Pdist, Squareform  
         1. Usage  
            1. from scipy.spatial.distance import pdist  
            2. from scipy.spatial.distance import squareform  
         2. Compute pairwise distances between data points.  
         3. Useful for clustering and distance-based computations.  
5.  Random  
   1. Usage: import random  
   2. Purpose:  
      1. Used for generating random numbers.  
      2. Useful for simulating random processes or creating synthetic datasets.  
   3. Installation: Built-in library (no installation required).  
6. FancyImpute  
   1. Usage: from fancyimpute import KNN  
   2. Purpose:  
      1. Impute missing data using machine learning algorithms.  
      2. KNN imputes missing values based on nearest neighbors.  
   3. Installation: pip install fancyimpute  
7. Seaborn  
   1. Usage: import seaborn as sns  
   2. Purpose:  
      1. Provides a high-level interface for creating attractive statistical graphics.  
      2. Simplifies visualization of complex data trends (e.g., heatmaps, pair plots).  
   3. Installation: pip install seaborn  
8. Regular Expressions (re)  
   1. Usage \- import re  
   2. Purpose  
      1. Used for text processing  
      2. Can clean and tokenize text data (e.g., removing non-alphanumeric characters)  
   3. Installation \- pip install re  
9. Scikit-Learn (sklearn)  
   1. Usage \- Import individual modules as needed from sklearn  
   2. Purpose  
      1. Scikit-Learn is a comprehensive machine learning library  
      2. We use it for vectorization, preprocessing, dimensionality reduction, clustering, and classification.   
   3. Installation \- pip install scikit-learn  
   4. Modules  
      1. CountVectorizer and TfidfVectorizer  
         1. Usage  
            1. from sklearn.feature\_extraction.text import CountVectorizer  
            2. from sklearn.feature\_extraction.text import TfidfVectorizer  
         2. Convert text data into numeric vectors for model input  
         3. CountVectorizer generates simple word count vectors  
         4. TfidfVectorizer generates weighted vectors based on term frequency-inverse document frequency  
      2. StandardScaler  
         1. Usage  
            1. from sklearn.preprocessing import StandardScaler  
         2. Standardizes features by removing the mean and scaling to unit variance  
         3. Used to normalize TF-IDF vectors for t-SNE processing  
      3. TSNE  
         1. Usage  
            1. from sklearn.manifold import TSNE  
         2. Reduces high-dimensional data to 2-D for visualization  
         3. Commonly used for visualizing clusters in a lower dimensional space  
      4. KMeans  
         1. Usage  
            1. from sklearn.cluster import KMeans  
         2. Clustering algorithm that partitions data into k clusters  
         3. Used to segment merchant data into groups  
      5. LogisticRegression and OneVsRestClassifier  
         1. Usage  
            1. from sklearn.linear\_model import LogisticRegression  
            2. from sklearn.multiclass import OneVsRestClassifier  
         2. LogisticRegression is used for classification  
         3. OneVsRestClassifier allows it to handle multi-class problems by fitting one classifier per class  
      6. train\_test\_split and cross\_val\_score  
         1. Usage  
            1. from sklearn.model\_selection import train\_test\_split  
            2. from sklearn.model\_selection import cross\_val\_score  
         2. Used for splitting data into training and test sets by train\_test\_split  
         3. Evaluates the model using cross-validation by cross\_val\_score  
      7. classification\_report and metrics  
         1. Usage  
            1. from sklearn.metrics import classification\_report  
            2. from sklearn.metrics import metrics  
         2. Provides performance metrics for evaluating the classification model  
         3. The metrics include precision, recall, and F1 score  
      8. classification\_report and metrics  
         1. Usage  
            1. from sklearn.metrics import classification\_report  
            2. from sklearn.metrics import metrics  
         2. Provides performance metrics for evaluating the classification model  
         3. The metrics include precision, recall, and F1 score  
      9. cosine\_similarity  
         1. Usage  
            1. from sklearn.metrics.pairwise import cosine\_similarity  
         2. Measures the cosine of the angle between two non-zero vectors in a multi-dimensional space.  
         3. Often used for determining the similarity between text documents or feature vectors.  
      10. euclidean\_distances  
          1. Usage  
             1. from sklearn.metrics.pairwise import euclidean\_distances  
          2. Computes the Euclidean distance between points in multi-dimensional space.  
          3. Used to measure dissimilarity or spatial distances in datasets.