from sklearn.model_selection import StratifiedKFold , GridSearchCV , cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from mlxtend.classifier import StackingClassifier

'''
def kf_cv_test(X_df , y_df , *model) : 
    kf = StratifiedKFold(n_splits = 3)
    for i , j in kf.split(X_df , y_df) : 
        train_X = X_df.iloc[i]
        train_y = y_df.iloc[i]
        val_X = X_df.iloc[j]
        val_y = y_df.iloc[j]
'''


def model_test(train_X , train_y , *model) : 
    for i in model : 
        cv_score(train_X , train_y , i)


def cv_score(train_X , train_y , clf) : 
    score = cross_val_score(clf , train_X , train_y , cv = 3 , scoring = 'accuracy')
    mean_score = 0
    for i in score : 
        mean_score += i
    print(mean_score / len(score))


def lr(train_X , train_y , test_X) : 
    lr = LogisticRegression(solver = 'lbfgs')
    lr.fit(train_X , train_y)
    get_y = lr.predict_proba(test_X)
    pred_y = []
    for i in get_y : 
        pred_y.append(i[1]) 
    return pred_y

def rf(train_X , train_y , test_X) : 
    rf = RandomForestClassifier(max_depth = 15 , min_samples_leaf = 3)
    rf.fit(train_X , train_y)
    get_y = rf.predict_proba(test_X)
    pred_y = []
    for i in get_y : 
        pred_y.append(i[1]) 
    return pred_y

def gdbt(train_X , train_y , test_X) : 
    gdbt = GradientBoostingClassifier()
    gdbt.fit(train_X , train_y)
    get_y = gdbt.predict_proba(test_X)
    pred_y = []
    for i in get_y : 
        pred_y.append(i[1]) 
    return pred_y

def search(model , train_X , train_y) : 
    if model == 'rf' : 
        #best : 15 , 3
        rf = RandomForestClassifier()
        max_depth = [10 , 15 , 20]
        min_samples_leaf = [3 , 4 , 5]
        param_grid = dict(max_depth = max_depth , min_samples_leaf = min_samples_leaf)
        grid_search = GridSearchCV(rf , param_grid , scoring = 'accuracy')
        grid_result = grid_search.fit(train_X , train_y)
        best_param = grid_result.best_params_
        best_score = grid_result.best_score_

    elif model == 'gdbt' : 
        #best : 0.1 , 5
        gdbt = GradientBoostingClassifier(subsample = 0.8)
        learning_rate = [0.05 , 0.1 , 1.5]
        n_estimators = [5 , 6 , 7]
        param_grid = dict(learning_rate = learning_rate , n_estimators = n_estimators)
        grid_search = GridSearchCV(gdbt , param_grid , scoring = 'accuracy')
        grid_result = grid_search.fit(train_X , train_y)
        best_param = grid_result.best_params_
        best_score = grid_result.best_score_

    return best_param , best_score

def blending(train_X , train_y , test_X , *param) : 
    pred_rf = rf(train_X , train_y , test_X)
    pred_gdbt = gdbt(train_X , train_y , test_X)
    
    pred_y = []

    for i in range(len(pred_gdbt)) : 
        pred_y.append(param[0] * pred_rf[i] + param[1] * pred_gdbt[i])
    return pred_y

def stacking(train_X , train_y , test_X) : 
    lr = LogisticRegression(solver = 'lbfgs')
    rf = RandomForestClassifier(max_depth = 15 , min_samples_leaf = 3)
    gdbt = GradientBoostingClassifier(subsample = 0.8 , learning_rate = 0.1 , n_estimators = 5)
    meta_estimator = GradientBoostingClassifier()
    stacking = StackingClassifier([lr , rf , gdbt] , meta_classifier = meta_estimator)
    stacking.fit(train_X , train_y)
    get_y = stacking.predict_proba(test_X)
    print(get_y)
    
    pred_y = []
    for i in get_y : 
        pred_y.append(i[1]) 
    return pred_y
    