import sklearn
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier
from matplotlib import pyplot


# create classification dataset
def get_dataset():
	X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
	return(X, y)

# get the list of base models
def get_models():
	"""
		get the base models for stacking
	"""
	models = {}
	models['lr'] = LogisticRegression()
	models['knn'] = KNeighborsClassifier()
	models['cart'] = DecisionTreeClassifier()
	models['svm'] = SVC()
	models['bayes'] = GaussianNB()

	return(models)

def evaluate_models(model):
	"""
		cross validation and model evaluation
	"""
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(model,X, y, scoring='accuracy', cv=cv,n_jobs=-1, error_score='raise')

	return(scores)

def get_stacking():
	"""
		pile the base models into a stack of base models
		
	"""
	level0 = []
	level0.append(('lr', LogisticRegression()))
	level0.append(('knn', KNeighborsClassifier()))
	level0.append(('cart', DecisionTreeClassifier()))
	level0.append(('svm', SVC()))
	level0.append(('bayes', GaussianNB()))

	# define meta learner model
	meta = LogisticRegression()
	# define stacking ensemble
	model = StackingClassifier(estimators=level0, final_estimator=meta, cv=5)

	return(model)


def get_models_with_stacking():
	"""
		get the base models and the stacking model
	"""
	models = dict()
	models['lr'] = LogisticRegression()
	models['knn'] = KNeighborsClassifier()
	models['cart'] = DecisionTreeClassifier()
	models['svm'] = SVC()
	models['bayes'] = GaussianNB()
	models['stacking'] = get_stacking()
	
	return(models)



if __name__ == '__main__':
	X, y = get_dataset()
	# get the models to evaluate
	models = get_models_with_stacking()
	# evaluate models and store results
	results, names =  list(), list()

	for name, model in models.items():
		scores = evaluate_models(model)
		results.append(scores)
		names.append(name)
		print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
	
	# plot model performancr for comparism
	pyplot.boxplot(results, labels=names, showmeans=True)
	pyplot.show()
