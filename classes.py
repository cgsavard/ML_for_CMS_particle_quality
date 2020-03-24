import numpy as np

class dataType:
    '''
    Put data into proper format.
    '''
    def __init__(self, data_arrays):
 
        self.init_features = ['trk_pt','trk_phi','trk_eta','trk_z0','trk_chi2pdof',
                              'trk_bendchi2','trk_nstub','trk_stubs_layer','trk_stubs_ps']
        self.trks = trks(data_arrays, self.init_features, False)
        #self.tps = tps(data_arrays)
        
    def summarize(self):
 
        print('L1 tracks info:')
        print('\t',len(self.trks.y),'total tracks')
        print('\t',len(self.trks.y[self.trks.y==0]),'fake tracks (',len(self.trks.y[self.trks.y==0])
              /len(self.trks.y),')')
        print('\t',len(self.trks.y[self.trks.y==1]),'real tracks (',len(self.trks.y[self.trks.y==1])
              /len(self.trks.y),')')
        
        #print('tp matched tracks info:')
        #print('\t',self.tps.num,'total tps')
        #print('\t',len(self.tps.matchtrks.y),'total matched tracks (',len(self.tps.matchtrks.y)/self.tps.num,')')
        #print('\t',len(self.tps.matchtrks.y[self.tps.matchtrks.y==0]),'fake matched tracks (',
        #      len(self.tps.matchtrks.y[self.tps.matchtrks.y==0])/len(self.tps.matchtrks.y),')')
        #print('\t',len(self.tps.matchtrks.y[self.tps.matchtrks.y==1]),'real matched tracks (',
        #      len(self.tps.matchtrks.y[self.tps.matchtrks.y==1])/len(self.tps.matchtrks.y),')') 
        
        return


class trks:
    
    def __init__(self, data_arrays, feats, match_bool):
        
        # data for ML
        self.tp_match = match_bool
        self.X_feats = None
        self.X = self.setX(data_arrays, feats)
        self.y = self.setY(data_arrays)
        self.pdgid = data_arrays['trk_matchtp_pdgid'].flatten()
        self.removeBad()
        
    def setX(self, arrays, feats):
        
        self.X_feats = []        
        tmp_X = [None]*len(feats)
        
        # loop through features and extract them from array
        nstub_idx = -1
        for ii in range(len(feats)):
            feat = feats[ii]
            if 'nstub' in feat:
                nstub_idx = ii
            if 'stubs_layer' in feat:
                tmp_X[ii] = self.setStubsLayer(arrays[feat].flatten(), arrays[feats[nstub_idx]].flatten())
            elif 'stubs_ps' in feat:
                tmp_X[ii] = self.isPs(arrays[feat].flatten(), arrays[feats[nstub_idx]].flatten())
            elif 'stubs_barrel' in feat:
                tmp_X[ii] = self.isBarrel(arrays[feat].flatten(), arrays[feats[nstub_idx]].flatten())
            else:
                tmp_X[ii] = arrays[feat].flatten()
                if self.tp_match:
                    self.X_feats.append(feat[5:])
                else:
                    self.X_feats.append(feat)
            
        # put features in proper format
        X = tmp_X[0]
        for ii in range(1,len(tmp_X)):
            X = np.column_stack((X,tmp_X[ii]))        
        
        return X
    
    def setY(self, arrays):
        
        if self.tp_match:
            y = np.ones(len(self.X[:,0]))
        else:
            y = arrays['trk_fake'].flatten()

        # both hard (1) and soft interactions (2) labeled as 1
        y[y==2] = 1
        
        return y
    
    def setStubsLayer(self, layers, nstubs):
        '''
        Create individual counters for # of stubs in each layer (0-6).
        '''
   
        self.X_feats.extend(('trk_stubs_layer1','trk_stubs_layer2','trk_stubs_layer3',
                            'trk_stubs_layer4','trk_stubs_layer5','trk_stubs_layer6','trk_nlayer_miss'))
        
        n_trks = len(nstubs)   
        stubs_layers = np.zeros((n_trks,7), dtype=int)
        
        kk = 0
        for ii in range(n_trks):
            nstub = nstubs[ii]
            for jj in range(kk,nstub+kk):
                stubs_layers[ii,layers[jj]-1]  = stubs_layers[ii,layers[jj]-1]+1
            kk = kk+nstub
            
            seq_idx = np.where(stubs_layers[ii,0:-1]>0)[0]
            stubs_layers[ii,-1] = np.count_nonzero(stubs_layers[ii,seq_idx[0]:seq_idx[-1]+1]==0)  
    
        return stubs_layers
        
    def isBarrel(self, labels, nstubs):
        '''
        Find majority label for barrel (1)/endcap (0) in all stubs.
        For a tie, label assigned to endcap.
        '''
        
        self.X_feats.append('trk_stubs_barrel')
        
        n_trks = len(nstubs)
        stubs_barrel = np.zeros((n_trks,), dtype=int)
        
        kk = 0
        for ii in range(n_trks):
            nstub = nstubs[ii]
            counts = np.bincount(labels[kk:kk+nstub].astype(int))
            stubs_barrel[ii] = np.argmax(counts)
            kk = kk+nstub
        
        return stubs_barrel
    
    def isPs(self, labels, nstubs):
        '''
        Find majority label for ps module (1)/2s module (0) in all stubs.
        For a tie, label assigned to 2s module.
        '''

        self.X_feats.append('trk_stubs_ps')
        
        n_trks = len(nstubs)
        stubs_ps = np.zeros((n_trks,), dtype=int)
        
        kk = 0
        for ii in range(n_trks):
            nstub = nstubs[ii]
            counts = np.bincount(labels[kk:kk+nstub].astype(int))
            stubs_ps[ii] = np.argmax(counts)
            kk = kk+nstub
        
        return stubs_ps

    def removeBad(self):
        '''
        Take out instances of tracks with nan as a feature and where |z0| is greater than 20 cm.
        '''
        
        bad_idx = np.argwhere(np.isnan(self.X))[0]
        
        while np.isnan(self.X).any():
            bad_i = np.argwhere(np.isnan(self.X))[0][0]
            self.X = np.delete(self.X,bad_i,0)
            self.y = np.delete(self.y,bad_i,0)
            self.pdgid = np.delete(self.pdgid,bad_i,0)
            
        #if self.z0idx != None:
        #    bad_i = np.where(abs(self.X[:,self.z0idx])>20)[0]
        #    if len(bad_i)>0:
        #        self.X = np.delete(self.X,bad_i,0)
        #        self.y = np.delete(self.y,bad_i,0)
        
        return
    
    
class tps:
    
    def __init__(self, data_arrays):
        
        self.init_features = ['matchtrk_pt','matchtrk_phi','matchtrk_eta','matchtrk_z0','matchtrk_chi2pdof',
                            'matchtrk_bendchi2','matchtrk_nstub','matchtrk_stubs_layer','matchtrk_stubs_ps']
        self.matchtrks = trks(self.takeOutFake(data_arrays), self.init_features, True)
        self.num = len(data_arrays['tp_pt'].flatten())
        
    def takeOutFake(self, arrays):
        
        good_idx = (arrays['matchtrk_fake']==1)
    
        fixed_feats = ['matchtrk_pt','matchtrk_phi','matchtrk_eta','matchtrk_z0',
                       'matchtrk_chi2pdof','matchtrk_bendchi2','matchtrk_nstub']
        for feat in fixed_feats:
            arrays[feat] = arrays[feat][good_idx]
            
        return arrays
    