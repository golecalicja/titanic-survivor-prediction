import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn import model_selection

def evaluate(models, x_train, y_train, x_test, y_test):
    for name, model in models:
        model.fit(x_train, y_train) 
        x_train_prediction = model.predict(x_train)
        x_test_prediction = model.predict(x_test)
        train_accuracy = accuracy_score(y_train, x_train_prediction)
        test_accuracy = accuracy_score(y_test, x_test_prediction)
        clf_report= classification_report(y_test, x_test_prediction)
        print(name)
        print(f'Train set accuracy: {train_accuracy:.2f}')
        print(f'Test set accuracy: {test_accuracy:.2f}' + '\n')
        print(clf_report + '\n')
    

def compare_on_boxplot(models, x_train, y_train):
    results = []
    names = []
    scoring = 'accuracy'
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10)
        cv_results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

    fig = plt.figure()
    fig.suptitle('Model comparison')
    ax = fig.add_subplot(111)
    ax.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()
  
    
def perform_grid_search_cv(model, grid, x_train, y_train):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
    grid_result = grid_search.fit(x_train, y_train)
    
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
   
    return grid_result.best_params_


def create_confusion_matrix(y_test, y_pred):
    cf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(cf_matrix / np.sum(cf_matrix), linewidths=1,
                cmap='RdBu', linecolor='white', annot=True,
                fmt='.1%')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')