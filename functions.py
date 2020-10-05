import numpy as np
from sklearn.utils import shuffle
from sklearn.utils.random import sample_without_replacement

def get_eff_faker_vs_feat(feat_interest, X_feats, X, y, fit_clf, thresh=.5):
    
    if feat_interest=='pt':
        idx = X_feats.index('trk_pt')   
        bins = np.logspace(.3,2,12) #2 to 100 GeV in log bins
    if feat_interest=='eta':
        idx = X_feats.index('trk_eta')   
        bins = np.linspace(-2.5,2.5,30)
    # add another feature to study here

    eff = np.zeros(len(bins))
    faker = np.zeros(len(bins))
    err_eff = np.zeros(len(bins))
    err_faker = np.zeros(len(bins))
            
    for ii in range(len(bins)):
        idx_temp = np.digitize(X[:,idx],bins,right=True)==ii
        X_temp = X[idx_temp]
        y_temp = y[idx_temp]
        if len(X_temp)>0:
            try: #works for keras
                y_pred = (fit_clf.predict(X_temp)[:,0]>thresh)*1
            except: #works for sklearn
                y_pred = (fit_clf.predict_proba(X_temp)[:,1]>thresh)*1
            eff[ii], faker[ii], err_eff[ii], err_faker[ii] = get_eff_faker_err(y_pred,y_temp)

    return bins, eff, faker, err_eff, err_faker

def get_eff_faker_err(y_pred, y_true):

    # efficiency = (# reals labeled real)/(# reals) <-this is really TPR
    TP = (y_pred[y_true==1]==1).sum()
    reals = len(y_true[y_true==1])
    eff = TP/reals
    err_eff = np.sqrt(TP*(reals-TP)/reals**3)

    if y_true.all()==1:
        return eff, 0, err_eff, 0
    
    # fake rate = (# fakes labeled real)/(#fakes) <-this is really FPR
    FP = (y_pred[y_true==0]==1).sum()
    fakes = len(y_true[y_true==0])
    faker = FP/fakes
    err_faker = np.sqrt(FP*(fakes-FP)/fakes**3)
    
    return eff, faker, err_eff, err_faker

def train_test_split_by_part(X, y, pdgid, n_mu=2500, n_el=2500, n_had=2500, n_fake=2500):
    
    mu_idx = sample_without_replacement(len(X[abs(pdgid)==13]),n_mu,random_state=23)
    elec_idx = sample_without_replacement(len(X[abs(pdgid)==11]),n_el,random_state=23)
    had_idx = sample_without_replacement(len(X[np.logical_and(abs(pdgid)>37,pdgid!=-999)]),n_had,random_state=23)
    fake_idx = sample_without_replacement(len(X[pdgid==-999]),n_fake,random_state=23)
    
    X_train = np.concatenate((X[abs(pdgid)==13][mu_idx],X[abs(pdgid)==11][elec_idx],\
                              X[np.logical_and(abs(pdgid)>37,pdgid!=-999)][had_idx],X[pdgid==-999][fake_idx]))
    y_train = np.concatenate((y[abs(pdgid)==13][mu_idx],y[abs(pdgid)==11][elec_idx],\
                              y[np.logical_and(abs(pdgid)>37,pdgid!=-999)][had_idx],y[pdgid==-999][fake_idx]))
    pdgid_train = np.concatenate((pdgid[abs(pdgid)==13][mu_idx],pdgid[abs(pdgid)==11][elec_idx],\
                                  pdgid[np.logical_and(abs(pdgid)>37,pdgid!=-999)][had_idx],pdgid[pdgid==-999][fake_idx]))

    X_test = np.concatenate((np.delete(X[abs(pdgid)==13],mu_idx,axis=0),np.delete(X[abs(pdgid)==11],elec_idx,axis=0),\
                             np.delete(X[np.logical_and(abs(pdgid)>37,pdgid!=-999)],had_idx,axis=0),\
                             np.delete(X[pdgid==-999],fake_idx,axis=0)))
    y_test = np.concatenate((np.delete(y[abs(pdgid)==13],mu_idx,axis=0),np.delete(y[abs(pdgid)==11],elec_idx,axis=0),\
                             np.delete(y[np.logical_and(abs(pdgid)>37,pdgid!=-999)],had_idx,axis=0),\
                             np.delete(y[pdgid==-999],fake_idx,axis=0)))
    pdgid_test = np.concatenate((np.delete(pdgid[abs(pdgid)==13],mu_idx,axis=0),np.delete(pdgid[abs(pdgid)==11],elec_idx,axis=0),\
                                 np.delete(pdgid[np.logical_and(abs(pdgid)>37,pdgid!=-999)],had_idx,axis=0),\
                                 np.delete(pdgid[pdgid==-999],fake_idx,axis=0)))

    X_train,y_train,pdgid_train = shuffle(X_train,y_train,pdgid_train,random_state=23)
    X_test,y_test,pdgid_test = shuffle(X_test,y_test,pdgid_test,random_state=23)
    
    return X_train, y_train, pdgid_train, X_test, y_test, pdgid_test
