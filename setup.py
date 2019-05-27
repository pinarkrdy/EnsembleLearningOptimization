from distutils.core import setup

NAME = "skfeature"

DESCRIPTION = "Ensemble Clustering Selection by Optimization of Accuracy-Diversity Trade off"

KEYWORDS = "Ensemble Clustering Selection by Optimization of Accuracy-Diversity Trade off"

AUTHOR = "Süreyya Akyüz , Pinar Karadayi Atas,"

AUTHOR_EMAIL = "sureyya.akyuz@eng.bahcesehir.edu.tr, pinar.karadayiatas@bahcesehir.edu.tr, "

URL = "https://github.com/pinarkrdy/EnsembleLearningOptimization"

VERSION = "1.0.0"


setup(
    name = NAME,
    version = VERSION,
    description = DESCRIPTION,
    keywords = KEYWORDS,
    author = AUTHOR,
    author_email = AUTHOR_EMAIL,
    url = URL,
    packages =['skfeature', 'skfeature.utility','skfeature.function','skfeature.function.information_theoretical_based','skfeature.function.similarity_based','skfeature.function.sparse_learning_based','skfeature.function.statistical_based','skfeature.function.streaming','skfeature.function.structure','skfeature.function.wrapper','EnsembleMethodsByOptimization',],
    requires=['ensemble']
)
