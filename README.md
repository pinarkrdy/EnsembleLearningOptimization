Ensemble Pruning by Disciplined Convex Concave Programming
===============================
To cite:

[1] Akyüz, S.  &  Otar, B. C. (2017). Ensemble clustering selection by optimization of accuracy-diversity trade off. Paper presented at the 2017 25th Signal Processing and Communications Applications Conference, SIU 2017, doi:10.1109/SIU.2017.7960636 

[2] Ali, M. A., Ucuncu, D., Karadayi Atas, P., & Özöğür-Akyüz, S. (2019). Classification of motor imagery task by using novel ensemble pruning approach. IEEE Transactions on Fuzzy Systems, doi:10.1109/TFUZZ.2019.2900859

[3] Karadayı Ataş, P. & Özöğür-Akyüz, S. Ensemble Learning Based Feature Selection by Using Convex Concave Programming, under review in European Journal of Operational Research, 2019. 


This project mainly has two part:

- Ensemble Clustering Selection By Optimization of Accuracy Diverstiy Trade Off (.\scikit-feature-master\EnsembleMethodsByOptimization)
- Ensemble Feature Selection By   Optimization of Accuracy Diverstiy Trade Off (.\scikit-feature-master\skfeature\function,.scikit-feature-master\EnsembleMethodsByOptimization)
- Datasets used in the experiments can be found from the directory (.\skfeature\data)

##Ensemble Clustering Selection By Optimization of Accuracy Diversity Trade-Off
1.  Datasets that exist under the data folder can be used as source dataset in that project
2.    The number of clustering solutions in the subset of the ensemble are obtained automatically by the optimization model called DCCP.
3.    The proposed model optimizes accuracy and diversity simultaneously.
4.    The result of the proposed model can be compared by the literature methods Joint Criteria, Cluster Select (Fern et al 2008), and Full Ensemble Library, and random selection methods.
5.    EnsembleSize (default is 50), number of folds in cross-validation, normalization status, RHO content in DCCP values are optional according to user demand.

##Ensemble Feature Selection By   Optimization of Accuracy Diversity Trade-Off
The proposed technique is adapted/applied to feature selection problem and the proposed approach can be  implemented as follows:
1. Datasets that exist under the data folder can be used as source dataset in that project.
    mat = scipy.io.loadmat('...\scikit-feature-master\skfeature\data\Yale.mat')
    X = mat['X']  # data
    X = X.astype(float)
    y = mat['Y']  # label
2. In this project, 28 feature selection algorithm was applied, the number of feature selection algoritm can be changed accoridng to user demand.
3.The number of Faeture in the subset of the ensemble will be obtained automatically by the optimization model called DCCP.
4. All results are saved under the SQLite,Example code is :    sq.write('FeatureSelectionResults'+str(s), 'features',line)
5. Number of fold in cross validation, normalization status, Types of feature selection algorithm, rho parameter in DCCP are optional according to user demand.
6. Examples for each code exits at the end of each .py file as a command under the ClusteringSelectionOyptimzationPart

Feature selection repository scikit-feature in Python (DMML Lab@ASU).
scikit-feature is an open-source feature selection repository in Python developed by Data Mining and Machine Learning Lab at Arizona State University. It is built upon one widely used machine learning package scikit-learn and two scientific computing packages Numpy and Scipy. scikit-feature contains around 40 popular feature selection algorithms, including traditional feature selection algorithms and some structural and streaming feature selection algorithms.
It serves as a platform for facilitating feature selection application, research, and comparative study. It is designed to share widely used feature selection algorithms developed in the feature selection research, and offer convenience for researchers and practitioners to perform the empirical evaluation in developing new feature selection algorithms.

## Requirements
Install scikit-feature package
install Cluster_Ensembles 1.16 package
### Prerequisites:
Python 2.7 *and Python 3*
NumPy
SciPy
Scikit-learn
SQLite
scipy.io

### Steps:
After you download  the packages
 scikit-feature-1.0.0.zip from the project website (http://featureselection.asu.edu/), unzip the file.

For Linux users, you can install the repository by the following command:

    python setup.py install

For Windows users, you can also install the repository by the following command:

    setup.py install

References:

[1] Agrawal, A., Verschueren, R., Diamond, S., & Boyd, S. (2018). A rewriting system for convex optimization problems. Journal of Control and Decision, 5 , 42-60.

[2] Diamond, S., & Boyd, S. (2016). CVXPY: A Python-embedded modeling language for convex optimization. Journal of Machine Learning Research, 17 , 1-5.

[3] Xiaoli Z. Fern and Wei Lin. 2008. Cluster Ensemble Selection. Stat. Anal. Data Min. 1, 3 (November 2008), 128-141. DOI=http://dx.doi.org/10.1002/sam.v1:3




## Contact
Assoc. Prof. Dr. Sureyya Akyuz (Bahcesehir University, Istanbul Turkey)

E-mail: sureyya.akyuz@eng.bau.edu.tr

Pınar Karadayı Ataş (Bahcesehir University, Istanbul Turkey)

E-mail: pinar.karadayiatas@bahcesehir.edu.tr
"# Ensemble-Clustering-Selection-by-Optimization-of-Accuracy-Diversity-Tradeoff"
