from sklearn import svm
import numpy as np


def linear_svm(train, trainlabel, tune, tunelabel, test, cc=[0.01, 0.1, 1.0]):
    """
    Notes:
    1. Just use linear svm, don't use other more powerful classifiers;
    2. @ cc, you can define your own cc, pass your own tune list to this function
    """

    print("Performing linear SVM!")
    best_error_tune=1.0
    tune_par_list = cc

    for c in tune_par_list:
        lin_clf=svm.SVC(C=c, kernel="linear")
        
        # train
        svm_x_sample=train
        svm_y_sample=trainlabel
        lin_clf.fit(train, svm_y_sample)

        # dev
        svm_x_sample=tune
        svm_y_sample=tunelabel
        pred=lin_clf.predict(svm_x_sample)
        svm_error_tune=np.mean(pred != svm_y_sample)
        print("c=%f, tune error %f" % (c, svm_error_tune))
        if svm_error_tune < best_error_tune:
            best_error_tune=svm_error_tune
            bestsvm=lin_clf

    # test
    svm_test_sample=test
    pred=bestsvm.predict(svm_test_sample)
    print("tuneerr=%f" % (best_error_tune))

    return best_error_tune, pred
