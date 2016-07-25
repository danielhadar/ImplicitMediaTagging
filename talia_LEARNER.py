import numpy as np

from pandas import *
from pandas import DataFrame as DF

import scipy.cluster as cluster
import scipy.stats as sstats

from sklearn import svm
from sklearn import linear_model
from sklearn import cross_validation as cross_val
from sklearn import feature_selection as f_selection
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
import pylab as pl

import os
import pickle
import matplotlib.pyplot as plt
import globalVars


from myUtils import *
from myClasses import *

class newObject(object):
    pass
##

"""--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Learning Class - TODO- Write details HERE
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------""" 
class LearnObject: 

    def __init__(self,FeatureObject,LabelsObject,Details,LabelsObject2='notDefined'):
        self.FeaturesDF=FeatureObject.FeaturesDF
        self.LabelsObject=LabelsObject
        if LabelsObject2=='notDeined':
            self.LabelsObject2=LabelsObject
        else:
            self.LabelsObject2=LabelsObject2
        self.BestFeatures={}
        self.N=LabelsObject.N
        self.model='notDefined'
        self.Details=Details
    class BestFeaturesForLabel(): #class of the best features for certain Labeling method (PatientsVsContols, mentalStatus, PANSS, etc.)
        def __init__(self,FeatureTypeList,LabelingList,n_features):
            self.df=DF(np.zeros([len(FeatureTypeList),n_features]),index=MultiIndex.from_tuples(FeatureTypeList),columns=range(n_features))            
            
        def add(self,bestNfeatures): #adds a feature to best features list (length n_features)   
            BestFeaturesList=[j for j in bestNfeatures]
            FeatureTypeList=self.df.index
            for feature in FeatureTypeList:
                if feature in BestFeaturesList:
                    isFeature=1
                    FeatureLoc=BestFeaturesList.index(feature)
                    self.df.loc[feature][FeatureLoc] +=1 


        #TODO -> add analysis according to facial part (according to excel..)
            #TODO - > add analysis according to learning weights (and not 0.1 : 0.9)
                 
    def run(self,Model='svc',kernel='linear',is_cross_validation=True, cross_validationMethod='LOO', DecompositionMethod='PCA',decompositionLevel='FeatureType',n_components=30, FeatureSelection='TopExplainedVarianceComponents', n_features=10, isPerm=0,isBetweenSubjects=True,isConcatTwoLabels=False,isSaveCsv=None, isSavePickle=None, isSaveFig=None,isSelectSubFeatures=False,SubFeatures='ExpressionLevel',savePath=None):       
        """runs learning with Labels and features from "getFeaturesAndLabels" and returns learning results"""
        # -- TODO :
        # --  # Greedy selection on features + Other feature selection types...
        # --  # remove irelevant data using 'Tracking Success' and consider 'TimeStamps' for feature calculation
        # --  # add f feature analysis by facial part (see excel) 
        # --  # compare svc results with regerssion results (using LOO and different Params for regression  - params for unbalanced data, different kernels, etc.), model evaluation - http://scikit-learn.org/stable/modules/model_evaluation.html) 
        # --  # check how the model weights behave - feature selection analysis
        # --  # calc model error
        # --  # divide data to subparts for training and testing - try within/ between subject, and analyze distribution of features when data is divided
        # --  # add mental status rank scores (0-4)
        # --  # run it over random data (permutation test) 
        

        ## init 
        if isSelectSubFeatures:
            print('Features : ' + SubFeatures)
            f=self.FeaturesDF.copy()
            featureNames=self.FeaturesDF.columns.names
            try:
               f=f[SubFeatures]
               f.columns=MultiIndex.from_product([[SubFeatures],f.columns], names=featureNames)
            except KeyError:
               f.columns=f.columns.swaplevel(0,1)
               f=f[SubFeatures]
               f.columns=MultiIndex.from_product([[SubFeatures],f.columns], names=featureNames)
            self.FeaturesDF=f.copy()
        else:
            SubFeatures='allFeatureTypes'

        FeatureTypeList=[j for j in tuple(self.FeaturesDF.index)]
        self.FullResults=DF()
           
        # set learning params (cross validation method, and model for learning)
        isBoolLabel=self.LabelsObject.isBoolLabel
        isBoolScores=isBoolLabel
        if DecompositionMethod==None and (FeatureSelection == 'TopExplainedVarianceComponents' or FeatureSelection == 'TopNComponents'):
            print("ERROR- feature selection method cannot be '"+ FeatureSelection +"' when X is not decomposed")
            FeatureSelection=raw_input("Choose a different feature selection method ('RFE','f_regression','dPrime','AllFeatures'): ")

        model, isBoolModel= learningUtils.setModel(Model)
        selectFeatures =learningUtils.setFeatureSelection(FeatureSelection,n_features)
        n_components=min(max(n_features,n_components),len(self.FeaturesDF.columns)) #cannot have less components than features, and no more components than full features
        decompositionTitle, decomposeFunction= learningUtils.setDecomposition(DecompositionMethod,n_components,decompositionLevel)
        isDecompose=  decompositionTitle!='noDecomposition'


        # save learning params
        self.Details.update({'Kernel':kernel,'DecompositionMethod':DecompositionMethod,'CrossVal':cross_validationMethod})

        print('\n------------Learning Details------------')
        print(DF.from_dict(self.Details,orient='index'))
        print('\n----' + cross_validationMethod + ' Cross validation Results:----')
        
        #define global variables over modules (to be used in myUtils)

        globalVars.transformMargins=0#lambda x:x         
        globalVars.isBoolLabel=isBoolLabel
        globalVars.isBoolModel=isBoolModel
        global trainLabels_all, testLabels_all, TrueLabels,isAddDroppedSubjects 
        trainLabels_all, testLabels_all, TrueLabels,isAddDroppedSubjects=labelUtils.initTrainTestLabels_all(self.LabelsObject)
        #trainLabels_all2, testLabels_all2, TrueLabels2,isAddDroppedSubjects2=labelUtils.initTrainTestLabels_all(self.LabelsObject2)

        
        LabelingList=trainLabels_all.columns #['N1']#
        self.ResultsDF=DF()
        self.BestFeatures=DF(columns=LabelingList) #dict of BestFeaturesDF according to Labeling methods
        YpredictedOverAllLabels=pandas.Panel(items=range(len(trainLabels_all)),major_axis=LabelingList,minor_axis=TrueLabels.index) #panel: items=cv_ind, major=labels, minor=#TODO 
       
                                              
        ## Create train and test sets according to LabelBy, repeat learning each time on different Labels from LabelingList
        
        isMultivarLabels=False      
        LabelingIndex=enumerate(LabelingList)
        if isMultivarLabels:
            LabelingIndex=enumerate([LabelingList])

        for label_ind, Labeling in LabelingIndex:
            
            #set subjects list according to labels and features
            X,SubjectsList,droppedSubjects,Xdropped=featuresUtils.initX(self.FeaturesDF,trainLabels_all,Labeling)
            #X2,SubjectsList2,droppedSubjects2,Xdropped2=featuresUtils.initX(self.FeaturesDF,trainLabels_all2,Labeling,is2=1)
            
            #init train and test labels
            #TODO- init test labels for photos rating!!
            trainLabels, testLabels, LabelRange = labelUtils.initTrainTestLabels(Labeling,SubjectsList,trainLabels_all, testLabels_all)
            #trainLabels2, testLabels2, LabelRange2 = labelUtils.initTrainTestLabels(Labeling,SubjectsList2,trainLabels_all2, testLabels_all2)
            
            #make sure only labeled subjects are used for classification
            trainLabelsSubjects=list(trainLabels.index)
            X=X.loc[trainLabelsSubjects]
            SubjectIndex=X.index.levels[0]

            #X2=X2.loc[trainLabelsSubjects]
            #X2.index.get_level_values(X2.index.names[0]) 
            #SubjectIndex2=X2.index.levels[0]           
            #init vars
            if isBetweenSubjects:
                cv_param=len(SubjectIndex)
                self.Details['CrossValSubjects']='between'
                isWithinSubjects=False
            else:
                isWithinSubjects=True
                X=X.swaplevel(0,1)
                PieceIndex=list(set(X.index.get_level_values('Piece_ind')))
                cv_param=len(PieceIndex)
                self.Details['CrossValSubjects']='within'       
            
            try:
                print('\n**' + Labeling + '**')
            except TypeError:
                print('\n*******')
                print(Labeling)
            
            cv, crossValScores= initUtils.setCrossValidation(cross_validationMethod,cv_param,trainLabels,isWithinSubjects) 
            
            ## Learning - feature selection for different scoring types, with cross validation - 

            BestFeaturesForLabel=self.BestFeaturesForLabel(FeatureTypeList,LabelingList,n_features) #saves dataframe with best features for each label, for later analysis
            cv_ind=0
            #used for transforming from margins returned from svm to continouse labels (e.g . PANSS)
            trainScores=DF()
            test_index=X.index
            testScores=concat([DF(index=test_index),DF(index=['std_train_err'])])
            testScores2=concat([DF(index=testLabels.index),DF(index=['std_train_err'])]) 
            testProbas=DF(index=X.index)
            testProbas2=DF(index=SubjectIndex)

            #impt=Imputer(missing_values='NaN', strategy='median', axis=0)

            globalVars.LabelRange=LabelRange

            ModelWeights1=DF(columns=range(len(cv)),index=X.columns)
            Components=pandas.Panel(items=range(len(cv)),major_axis=X.columns,minor_axis=range(n_features)) #todo fix this for 1st and second learning
            ExplainedVar=DF(columns=range(len(cv)))
            ModelWeights2=DF(columns=range(len(cv)))
            
            
            #bestNfeaturesPanel=Panel(items=LabelingList,major_axis=range(len(cv)),minor_axis=MultiIndex.from_tuples(('a','b')))
            

            for train, test in cv:

                if not is_cross_validation:
                   train=np.append(train,test)
                   #test=np.append(train,test)
                   self.Details['CrossVal']='NONE'
                   #if cv_ind>0:
                    #    break

                if isBetweenSubjects:
                    #set X and Y
                    train_subjects=trainLabels.iloc[train].index
                    test_subjects=testLabels.iloc[test].index 
                    Xtrain,Xtest, Ytrain, YtrainTrue, Ytest=initUtils.setXYTrainXYTest(X,Labeling,trainLabels,testLabels,TrueLabels,train_subjects,test_subjects)
                    #Xtrain2,Xtest2, Ytrain2, YtrainTrue2, Ytest2=initUtils.setXYTrainXYTest(X2,Labeling,trainLabels2,testLabels2,TrueLabels2,train_subjects,test_subjects)

                    
                    if isConcatTwoLabels: #used when there is more than one doctor
                        Xtrain=concat([Xtrain,Xtrain2])
                        Xtest=concat([Xtest,Xtest2])
                        Ytrain=concat([Ytrain,Ytrain2])
                        YtrainTrue=concat([YtrainTrue,YtrainTrue2])
                        Ytest=concat([Ytest,Ytest2])
                        Xdropped=concat([Xdropped,Xdropped2])
                        SubjectsList=list(set(SubjectsList).intersection(set(SubjectsList2)))
                        droppedSubjects=list(set(droppedSubjects).union(set(droppedSubjects2)).difference(set(SubjectsList)))#diff from SubjectsList to make sure no subjects are both in train and test.
                 

                    #select N best features:
                    Xtrain, Xtest, bestNfeatures, components, explainedVar = learningUtils.decomposeAndSelectBestNfeatures(Xtrain,Xtest,Ytrain,n_features,selectFeatures,decomposeFunction,decompositionLevel)
                    BestFeaturesForLabel.add(bestNfeatures) #todo - delete this??  
                    
                    #train 1 
                    TrainModel=model
                    TrainModel.fit(Xtrain.sort_index(),Ytrain.T.sort_index())
                 
                    if cv_ind==0:
                        ModelWeights1=DF(columns=range(len(cv)),index=range(len(bestNfeatures)))  
                        bestNfeaturesPanel=Panel(items=LabelingList,minor_axis=range(len(cv)),major_axis=range(min(n_features,len(bestNfeatures))))
                    bestNfeaturesPanel[Labeling][cv_ind]=list(bestNfeatures)  
                    ModelWeights1[cv_ind]=TrainModel.coef_.flatten()
                  
                    #get ROC scores without cross validation:
                                           
                    #train 2
                    if isBoolLabel:
                       PiecePrediction_train=DF(TrainModel.predict_proba(Xtrain).T[1],index=Xtrain.index,columns=['prediction'])
                       TrainModel2=svm.SVC(kernel='linear', probability=True,class_weight={0:1,1:1})
                    else:
                       PiecePrediction_train=DF(TrainModel.decision_function(Xtrain),index=Xtrain.index,columns=['prediction'])
                       TrainModel2=linear_model.LinearRegression()

                    Xtrain2, Ytrain2, YtrainTrue2=learningUtils.getX2Y2(Xtrain,Ytrain,YtrainTrue,PiecePrediction_train, isBoolLabel)                 
                    TrainModel2.fit(Xtrain2, Ytrain2)
                    if cv_ind==0:
                        ModelWeights2=DF(columns=range(len(cv)),index= Xtrain2.columns)
                    ModelWeights2[cv_ind]=TrainModel2.coef_.flatten()         

                              
                    #test 1
                    if isAddDroppedSubjects: #take test subjects from cv + subjects that were dropped for labeling used for test
                        if isDecompose:
                            dXdropped=DF(decomposeFunc(Xdropped).values,index=Xdropped.index)
                        XtestDropped=dXdropped[bestNfeatures]
                        YtestDropped=Series(XtestDropped.copy().icol(0))
                        #YTrueDropped=Series(Xdropped.copy().icol(0))
                        for subject in droppedSubjects:
                            YtestDropped[subject]=testLabels_all[Labeling].loc[subject]
                            #YTrueAll.loc[subject]=TrueLabels[Labeling].loc[subject]
                        Ytest=concat([Ytest,YtestDropped]).sort_index()
                        Xtest=concat([Xtest,XtestDropped]).sort_index()


                    if isPerm: #TODO- Check this!!
                        Ytest=y_perms.loc[Ytest.index]
                    Xtest=Xtest.fillna(0.)
                    
                    
                elif isWithinSubjects:
                    #train 1
                    train_pieces=PieceIndex[train]
                    test_pieces=PieceIndex[test] #TODO - make sure that if test/train> piece index, it ignores it and repeate the process
                    
                    XtrainAllFeatures=X.query('Piece_ind == '+ str(list(train_pieces)))
                    Ytrain=Series(index=X.index)
                    Ytest=Series(index=X.index)
                    YtrainTrue=Series(index=X.index)
                    
                    for subject in PieceIndex: 
                        for piece in train_pieces:
                            Ytrain.loc[piece].loc[subject]=trainLabels[subject]
                            YtrainTrue.loc[piece].loc[subject]=TrueLabels[Labeling].loc[subject] 
                            Ytest.loc[piece].loc[subject]=testLabels[subject]   
                    Ytrain=Ytrain.dropna()
                    YtrainTrue=YtrainTrue.dropna() 
                    for subject in test_subjects:
                        Ytest.loc[piece].loc[subject]=testLabels[subject]
                #train scores 1       
                if cv_ind==0:
                    trainScores,YtrainPredicted=learningUtils.getTrainScores(Ytrain,Xtrain,YtrainTrue,TrainModel)
                    plt.figure(1)
                    if len(LabelingList)>1:
                        plt.subplot(round(len(LabelingList)/2),2,label_ind+1)
                    if isBoolLabel:
                        testScores,testProbas=learningUtils.getTestScores(Ytest,Xtest,TrainModel)
                    else:
                        testScores[cv_ind],testProbas=learningUtils.getTestScores(Ytest,Xtest,TrainModel)
                        plt.title(Labeling,fontsize=10)
                else:
                    plt.figure(3)
                    new_trainScores,YtrainPredicted=learningUtils.getTrainScores(Ytrain,Xtrain,YtrainTrue,TrainModel)
                    trainScores=concat([trainScores,new_trainScores],axis=1)
                #test 1   
                testScores[cv_ind],testProbas_new=learningUtils.getTestScores(Ytest,Xtest,TrainModel)
                testProbas=concat([testProbas,testProbas_new])
                
                #train2

                if isBoolLabel:
                    PiecePrediction_test=DF(TrainModel.predict_proba(Xtest).T[1],index=Xtest.index,columns=['prediction'])
                else:
                    PiecePrediction_test=DF(TrainModel.decision_function(Xtest),index=Xtest.index,columns=['prediction'])
                Xtest2, Ytest2 , YtestTrue2 =learningUtils.getX2Y2(Xtest,Ytest,Ytest,PiecePrediction_test,isBoolLabel)
                
                if cv_ind==0:
                    trainScores2,YtrainPredicted2=learningUtils.getTrainScores(Ytrain2,Xtrain2,YtrainTrue2,TrainModel2)
                    YpredictedOverAllLabels[cv_ind].loc[Labeling]=YtrainPredicted2
                    #plt.figure(1)
                    #if len(LabelingList)>1:
                        #plt.subplot(round(len(LabelingList)/2),2,label_ind+1)
                #test2
                    if isBoolLabel:
                        testScores2,testProbas2=learningUtils.getTestScores(Ytest2,Xtest2,TrainModel2)
                    else:
                        testScores2[cv_ind],testProbas2=learningUtils.getTestScores(Ytest2,Xtest2,TrainModel2)
                    #plt.title(Labeling,fontsize=10)
                else:
                    new_trainScores2,YtrainPredicted2=learningUtils.getTrainScores(Ytrain2,Xtrain2,YtrainTrue2,TrainModel2)
                    YpredictedOverAllLabels[cv_ind].loc[Labeling]=YtrainPredicted2
                    trainScores2=concat([trainScores2,new_trainScores2],axis=1)
                    if len(Xtest2)>0: # if there is more than one segment for subject
                        testScores2[cv_ind],testProbas2_new=learningUtils.getTestScores(Ytest2,Xtest2,TrainModel2)     
                        testProbas2=concat([testProbas2,testProbas2_new])
                cv_ind+=1

                #crossValScores=crossValScores.append(CVscoresDF,ignore_index=True) #information about entire train test data. 
            fig2=plt.figure(2)
            if len(LabelingList)>1:
                plt.subplot(round(len(LabelingList)/2),2,label_ind+1)
            #if isAddDroppedSubjects:
               # testLabelsSummary=testLabels_all[Labeling].loc[AllSubjects]
           # else:
               # testLabelsSummary=testLabels
            scoresSummary,rocDF = learningUtils.getScoresSummary(trainScores2,testScores2,testProbas2,TrueLabels[Labeling])

            # reset global vars
            globalVars.fitYscale='notDefined'
            globalVars.beta=DF()

            plt.title(Labeling,fontsize=10)
            plt.xlabel('Ytrue',fontsize=8)
            plt.ylabel('Ypredicted',fontsize=8)
            plt.tick_params(labelsize=6)
            #print(crossValScores.T)    
            scores=scoresSummary.fillna(0.)
            
            #analyze feature weights             
            ModelWeights1=ModelWeights1.dropna(how='all')
            WeightedFeatures1_index0=analysisUtils.getFeaturesWeights(0,bestNfeaturesPanel[Labeling],ModelWeights1) #FeatureAnalysisIndex=0 for featureType, 1= au's (if not decomposed) or component rank (if decomposed)
            WeightedFeatures1_index1=analysisUtils.getFeaturesWeights(1,bestNfeaturesPanel[Labeling],ModelWeights1)
            WeightedFeatures1=concat([DF(index=['-------(A) Index0-------']),WeightedFeatures1_index0,DF(index=['-------(B) Index1 -------']),WeightedFeatures1_index1])
            
            WeightedFeatures2=DF(ModelWeights2.mean(axis=1)).fillna(0)
            #WeightedFeatures2=DF([ModelWeights2.mean(axis=1),ModelWeights2.std(axis=1)],index=['mean','std']).T.fillna(0)
            BestFeatures=concat([DF(index=['------------- Learning 1 -------------']),WeightedFeatures1,DF(index=['------------- Learning 2 -------------']),WeightedFeatures2])
            self.BestFeatures[Labeling]=Series(BestFeatures.values.flatten(),index=BestFeatures.index)

            #analyze decomposition
            if isDecompose:
                Components_mean = Components.mean(axis=0)
                Components_std = Components.std(axis=0)
                normalize=lambda df:DF(StandardScaler().fit_transform(df.T).T,index=df.index,columns=df.columns) 

                        
            #BestFeaturesForLabel.analyze(ByLevel=0) #TODO change to regression coeff
            LabelFullResults=concat([DF(index=[Labeling]),scores]) 
  
            self.FullResults=concat([self.FullResults,LabelFullResults])            
            self.ResultsDF=concat([self.ResultsDF,DF(scores[0],columns=[Labeling])],axis=1)

            #self.BestFeatures[Labeling]=BestFeaturesForLabel.WeightedMean

            #plt.savefig('C:\\Users\\taliat01\\Desktop\\TALIA\\Code-Python\\Results\\'+Labeling+'png')
     #   testScores3=pandas.Panel(items=range(len(X2.index))) #for each cv score...
        FullSubjectsList=YpredictedOverAllLabels[0].columns
        YdroppNans=YpredictedOverAllLabels.dropna(axis=0,how='all')
        YdroppNans=YdroppNans.dropna(axis=1,how='all')
        YpredictedOverAllLabels=YdroppNans.dropna(axis=2,how='all')
        notNans_cv_ind=YpredictedOverAllLabels.items
        notNans_trainSubjects=YpredictedOverAllLabels.minor_axis
        notNans_LabelsList=YpredictedOverAllLabels.major_axis
        notNans_TrueLabels=TrueLabels.T[notNans_trainSubjects].loc[notNans_LabelsList]
        cv_ind=0
        for train, test in cv:
            if cv_ind in notNans_cv_ind:
                print(test)
                train=list(set(FullSubjectsList[train]).intersection(set(notNans_trainSubjects)))
                test=list(set(FullSubjectsList[test]).intersection(set(notNans_trainSubjects)))
                if len(train)>0 and len(test)>0: 
                    AllLabelsYTrainPredicted=YpredictedOverAllLabels[cv_ind][train]
                    AllLabelsYTrainPredicted=AllLabelsYTrainPredicted.fillna(0)
                    AllLabelsYTrainTrue=notNans_TrueLabels[train]
                    AllLabelsYTestPredicted=YpredictedOverAllLabels[cv_ind][test]
                    AllLabelsYTestTrue=notNans_TrueLabels[test]

                    pseudoInverse_AllLabelsYTrainTrue=DF(np.linalg.pinv(AllLabelsYTrainTrue),columns=AllLabelsYTrainTrue.index,index=AllLabelsYTrainTrue.columns)
                    global AllLabelsTransformationMatrix
                    AllLabelsTransformationMatrix=DF(AllLabelsYTrainPredicted.dot(pseudoInverse_AllLabelsYTrainTrue),columns=pseudoInverse_AllLabelsYTrainTrue.columns)#change to real code!!
                TrainModel3=lambda y: y.T.dot(AllLabelsTransformationMatrix)
                #testscores3[cv_ind]=learningUtils.getTestScores(AllLabelsYTrainTrue,AllLabelsYTrainPredicted,TrainModel3)
            cv_ind+=1

        self.BestNFeaturesAll=bestNfeaturesPanel 
        self.ResultsDF=self.ResultsDF.fillna(0.)  
        
        ## Print and save results  
        print('\n')
        print(self.ResultsDF)
        print('\n')
        D=self.Details 
        if 'savePath' not in locals():
            savePath=raw_input('enter save path for learning results: ')
        saveDir='\\'+D['Model']+'_'+D['FeatureMethod']+'_'+D['CutBy']+'_'+str(D['cutParam'])+'_'+FeatureSelection+'_'+D['DecompositionMethod']+'_'+SubFeatures
        saveName=savePath + saveDir + '\\' + str(n_features)+'_features'    
        if isPerm:
            saveName=saveName+'_PERMStest'    
        self.Details['saveDir']=saveName
        dir=os.path.dirname(saveName)
        if not os.path.exists(dir):
            os.makedirs(dir)
        if isSavePickle is None:
            isSavePickle=int(raw_input('Save Results to pickle? '))
        if isSaveCsv is None:
            isSaveCsv= int(raw_input('save Results to csv? '))
        if isSaveFig is None:
            isSaveFig=int(raw_input('save Results to figure? '))

       
        if isSavePickle:        
            self.ResultsDF.to_pickle(saveName+'.pickle')
            self.BestFeatures.to_pickle(saveName+'_bestFeatures.pickle')
                
        if isSaveCsv:
            DetailsDF=DF.from_dict(self.Details,orient='index')
            ResultsCSV=concat([self.ResultsDF,DF(index=['-------Label Details-------']),self.N,DF(index=['-------Learning Details-------']),DetailsDF,DF(index=['-------Selected Features Analysis------']),self.BestFeatures])
            ResultsCSV.to_csv(saveName+'.csv')
            if isBoolLabel:
                ROCfig=learningUtils.save_plotROC(rocDF,isSave=True,saveName=saveName,title=SubFeatures)

        if isSaveCsv or isSavePickle:
            print('successfully saved as:\n' + saveName)
        
        if isSaveFig:
            plt.figure(1)
            plt.savefig(saveName + 'Train.png')
            plt.figure(2)
            plt.savefig(saveName + 'Test.png')
        plt.close()
        plt.close()
"""--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

SET FEATURES AND LABELS FOR LEARNING
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"""    
def SetData(PartName='Interview',isLoadData=True,cutDataBy=None, cutDataParam=None,isSetNewData=False,Variables=[]):
    """This function is sets the data for all parts and processing methods (clustering, quantization and cutting) before learning, and chooses the specific data needed for feature calculations. If features were already calculated, this should not be used.""" 
    os.system('cls')
    Data=DataObject(PartName,Variables)
    if cutDataBy=='stimuli':
        stimuliOperation=cutDataParam
        isCutByStimuli=True
        isCutDataBySegments=False
    elif cutDataBy=='segments':
        PieceLength=cutDataParam
        isCutByStimuli=False
        isCutDataBySegments=True
    elif 'segments' in cutDataBy and 'stimuli' in cutDataBy:
        PieceLength=cutDataParam['segments']
        stimuliOperation=cutDataParam['stimuli']
        isCutByStimuli=True
        isCutDataBySegments=True

    if isSetNewData:
        dataPath=os.path.join(resultsPath,'LearningData',PartName)
        #Variables=raw_input("Choose data variables (deatult:['au17', 'au18', 'au19', 'au1', 'au22', 'au25', 'au26', 'au27', 'au28', 'au29', 'au2', 'au30', 'au31', 'au32', 'au33', 'au34', 'au37', 'au41', 'au43', 'au45', 'au47', 'au48', 'au8']")
        #print('clustering and quantizing all experimental parts...')
        #AllPartsData=pickle.load(open(RawDataPath+'.pickle','rb'))
        #dataUtils.setAndSaveAllDataRawClusteredQuantizedDF(AllPartsData,Variables,dataPath=dataPath)
        if not(isCutDataBySegments):
            isCutDataBySegments=int(raw_input('cut data by segments? '))
        if not(isCutByStimuli):
            isCutDataBySegments=int(raw_input('cut data by stimuli? '))
        if isCutDataBySegments:
            if not(PieceLength):
                PieceLenth=int(raw_input('Piece Length: '))
            print('cutting data to segments size '+str(PieceLength))
            dataUtils.cutDataBySegment([PartName],PieceLength,Variables)
        if isCutByStimuli:
            print('cutting data by stimuli...')
            dataUtils.cutDataByStimuli([PartName],Variables)
        #isLoadData=True

    if isLoadData:
        ## Construct / load DATA object
        Data=dataUtils.loadData(PartName,Variables)
        print(PartName+ ' data successfully loaded!')
        if isCutDataBySegments:
            Data.segmented=dataUtils.loadSegmentedData(PartName,PieceLength)
        if isCutByStimuli:
            if PartName in ['Photos','Videos']:#,'Interview'
                Data.byStimuli=dataUtils.loadDataByStimuli(PartName,stimuliOperation=stimuliOperation)
            else:
                print('Unable to cut data according to stimuli since PartName=='+PartName)
    return Data

def SetFeatures(FeatureMethod,CutDataBy,DataObject=None,isLoadFeatures=True,isGetFeaturesNaNs=0,cutDataParam=None):
    """this function loads gets DataObejct returned from 'setData(), including Data.quantized and Data.clustered, and process it to featuresDF according to 'FeatureMethod'('kMeansClustering','quantization','moments'). 
        Features are saved to library for each experimental part!"""
    PartName=DataObject.details['Part']
    FeaturesPath=os.path.join(resultsPath ,'LearningFeatures',PartName, FeatureMethod + '_Features_'+str(cutDataParam))
    if not(isLoadFeatures):
        # set features params
        Features=newObject()
        Features.details={}
        Features.details['Part']=PartName
        Features.details['CutDataBy']=CutDataBy

        if not(DataObject):
            print('please use SetData to create dataObject, and then continue!')
        if CutDataBy=='stimuli':
            if PartName in ['Photos','Videos']:
                featuresDataObject=DataObject.byStimuli
                cutDataParam=featuresDataObject.details['stimuliOperation']
                Features.details['FeatureMethod']=FeatureMethod
            else:
                print('cannot cut data by stimuli for ' + PartName)
                isBySegments=raw_input('cut data by segments? ')
                if isBySegments:
                    CutDataBy='segments'
        elif CutDataBy=='segments':
            featuresDataObject=DataObject.segmented
            cutDataParam=featuresDataObject.details['PieceLength']
            Features.details['PieceLength']=cutDataParam
        
        #set feature calculation function:
        if FeatureMethod=='Moments':
            Variables=featuresDataObject.quantized.columns
            data=featuresDataObject.raw[Variables]
            n=np.nan
            getFeatures=featuresUtils.calcMomentsFeatures
            columnsNames=['FeatureType','fs-signal']
        if FeatureMethod=='kMeansClustering':
            data=featuresDataObject.clustered
            n=featuresDataObject.details['n_clusters']
            Features.details['n_clusters']=n+1
            getFeatures=featuresUtils.calckMeansClusterFeatures
            columnsNames=['FeatureType','cluster']
        elif FeatureMethod=='Quantization':
            data=featuresDataObject.quantized
            n=featuresDataObject.details['n_quantas']
            Features.details['n_quantas']=n+1
            getFeatures=featuresUtils.calcQuantizationFeatures
            columnsNames=['FeatureType','fs-signal']
        


        DetailsDF=DF.from_dict([Features.details]).T
        print('---------------------------------')
        print(DetailsDF)
        print('-----calculating features...-----')
        
        
        #calc featuresDF
        SubjectsList=featuresDataObject.details['subjectsList']
        AllSegments=list(data.index.levels[1])
        FeaturesDF=DF(index=MultiIndex.from_product([SubjectsList,AllSegments]))
        isFirstIteration=True
        for subject in SubjectsList:
            print(subject)
            subjectData=data.loc[subject]
            segmentsList=subjectData.index.get_duplicates()
            try:
                segmentsList.remove(nan)              
            except ValueError:# name 'segmentsLst' is not defined
                pass
            for segment in segmentsList:
                subjectDataSegment=subjectData.loc[segment]
                if isFirstIteration:
                    FeaturesDF=DF(index=MultiIndex.from_product([SubjectsList,AllSegments]),columns=getFeatures(subjectDataSegment,n).index)
                    isFirstIteration=False
                FeaturesDF.loc[subject,segment]=getFeatures(subjectDataSegment,n)
        FeaturesDF=FeaturesDF.dropna(how='all')
        FeaturesDF=FeaturesDF.fillna(0.)
        FeaturesDF.columns.names=columnsNames
        FeaturesDF.index.names=['sujbect','s'] #s stands for segment or for stimuli
        FeaturesPath=os.path.join(resultsPath ,'LearningFeatures',PartName,FeatureMethod + '_Features_'+str(cutDataParam))

        Features.FeaturesDF=FeaturesDF
        print('Features were succesfully calculated! \n saving to '+ FeaturesPath+'...')
        FeaturesDF.to_csv(FeaturesPath +'.csv')
        FeaturesDF.to_pickle(FeaturesPath+'.pickle')
        DetailsDF.to_pickle(FeaturesPath+'_DETAILS.pickle')
        DetailsDF.to_csv(FeaturesPath+'_DETAILS.csv')
        
        print('Features were succesfully saved!')
      
    if isLoadFeatures:
        Features=newObject()
        print('loading FEATURES from '+ FeaturesPath + '...\n')  
        Features.FeaturesDF=pickle.load(open(FeaturesPath+'.pickle','rb'))
        #Features.FeaturesDF.columns.names=columnsNames
        #Features.FeaturesDF.index.names=['sujbect','s']
        Features.details=pickle.load(open(FeaturesPath+'_DETAILS.pickle','rb')).to_dict()[0]

    return Features
    
    # Set /Load LABELS for Learning
def SetLabels(LabelBy,part=None,isLoadLabels=True, LabelFileName={}):
    if LabelBy=='stimuli':
        resultsPath + '\\LearningLabels\\'+ part +'Rating_Labels'
    else:
        LabelsPath=resultsPath + '\\LearningLabels\\' + LabelBy + '_Labels' #for loading / saving
    LabelsPath2=LabelsPath+'2'
    if isLoadLabels:
        print('loading LABELS from '+LabelsPath+ '...\n')
        try:  
            Labels=pickle.load(open(LabelsPath+".pickle",'rb'))
            print(LabelBy + ' labels successfully loaded!')
            # Labels2=Labels#pickle.load(open(LabelsPath2+".pickle",'rb')) #todo - change this when there is second labeled data (from michael)
        except IOError: #No such file or directory...
            print('label file was not found! calculating labels..')
            isLoadLabels=False
       
    if not(isLoadLabels):
        SubjectsDetailsDF=DF.from_csv('C:\\Users\\taliat01\\Desktop\\TALIA\\Code-Python\\Results\\SubjectsDetailsDF.csv')
        Labels=LabelObject(SubjectsDetailsDF,LabelsPath)
        Labels.getLabels(LabelBy,part=part)
        SubjectsDetailsDF2=DF.from_csv('C:\\Users\\taliat01\\Desktop\\TALIA\\Code-Python\\Results\\SubjectsDetailsDF2-fill with data from michael.csv')
        #Labels2=LabelObject(SubjectsDetailsDF2,LabelsPath2)
        #Labels2.getLabels(LabelBy)
        #Labels.permLabels() #TODO - move this to "not isLoad" or somewhere else. 
        Labels.LabelingMethod= LabelBy
    
    return Labels
 
"""--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

MAIN LEARNING LOOP
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"""       
def main():
    os.system('cls')
    ## init all Learning Params:
    #init loop
    # init Data Params:
    PartNameList=['Interview','Photos','Rest']#'Videos',,,'Clinical',,]#[]#,
    Variables=['au17', 'au18', 'au19', 'au1', 'au22', 'au25', 'au26', 'au27', 'au28', 'au29', 'au2', 'au30', 'au31', 'au32', 'au33', 'au34', 'au37', 'au41', 'au43', 'au45', 'au47', 'au48', 'au8']
    PieceLengthList=[500]
    # init data Params
    isSetNewData=False
    isCutByStimuli=False

    # init Features Params:
    isLoadFeatures=False
    FeatureMethodList=['kMeansClustering','kMeansExplainedVar']#,]#'Moments','Quantization',]#,'All'
    CutDataByList=['stimuli']# 'segments' , 
    cutDataBySegmentsParamList=[str(l) for l in PieceLengthList]
    cutDataByStimuliParamList=['watch']#, 'rate', 'watchAndRate']
    #init Label Params:
    LabelByList= ['PatientsVsControls']#,'PANSS']

    #init Learning Loop Params:
    isLearning=False#False when only want to calculate featuresDF or labelsDF...    
    NFeatureList= [15]#,5]#,10,15]#
    ModelList=['ridge','lasso']
    #TODO - Try different models for multiClass !!
    #from sklearn.naive_bayes import GaussianNB
                    #model = GaussianNB()
                    #sklearn.linear_model.LogisticRegression(multi_class='multinomial')
                    #sklearn.lda.LDA
    DecompositionList=['noDecomposition']#'FeatureType_PCA']#]#]#]#,] #['PCA','noDecomposition','KernelPCA','SparsePCA','ICA']
    FeatureSelectionList=['f_regression']#,'TopNComponents']#,'TopExplainedVarianceComponents']#,#,'f_regression','FirstComponentsAndFregression',]#[,'FirstComponentsAndExplainedVar']  
    is_cross_validation=True
    isSelectSubFeatures=True
    QuantizationSubFeaturesList=['ChangeRatio','ExpressionRatio','ExpressionLength','ExpressionLevel','FastChangeRatio']
    ClusteringSubFeaturesList=['Num','NumOfClusters','ClusterMeanLength','ClusterChangeRatio','Counts','length']
    ExplainedVarSubFeaturesList=['ExplainedVar','ExplainedVar_eachSubject']
  
    #print('cutting data by stimuli...')
    #dataUtils.cutDataByStimuli(PartNameList,Variables) 
    #print('break')

    # Labels Loop:
    for part in PartNameList:
        print('Part = ' +part)
        if part in ['Photos','Videos']:
            CutDataByList_set=CutDataByList
        else:
            CutDataByList_set=['segments']
        for label in LabelByList:
            print('-Label = ' +label)
            Labels=SetLabels(LabelBy=label,part=part)
            labelModelList=initUtils.setModelListAccordingToLabel(ModelList,label)
            if label=='StimuliRating' and part in ['Photos','Videos']:
                CutDataByList_set==['stimuli']
            elif label=='StimuliRating' and not(part in ['Photos','Videos']):
                    print("ERROR- cannot learn 'stimuliRating' labels from "+ part + "data!")
                    continue
            # Data Loop:
            for cutDataBy in CutDataByList_set:
                print('---             - cutBy = ' +cutDataBy)
                if cutDataBy=='stimuli':
                    cutDataParamList=cutDataByStimuliParamList
                    isCutByStimuli=True
                elif cutDataBy=='segments':
                    cutDataParamList=cutDataBySegmentsParamList
                for cutParam in cutDataParamList:
                    print('---             - cutParam = ' +str(cutParam))
                    #print('--DataParams - pieceLength = ' +str(pieceLength))
                    data=SetData(PartName=part,cutDataBy=cutDataBy, cutDataParam=cutParam,Variables=Variables,isSetNewData=isSetNewData)
            
                    # Features Loop:
                    for featureMethod in FeatureMethodList:
                        print('---FeatureParams- featureMethod = ' +featureMethod)
                        if featureMethod=='Quantization' and isSelectSubFeatures:
                            SubFeaturesList=QuantizationSubFeaturesList
                        elif featureMethod=='kMeansClustering' and isSelectSubFeatures:
                            SubFeaturesList=ClusteringSubFeaturesList
                        elif featureMethod=='kMeansExplainedVar' and isSelectSubFeatures:
                            SubFeaturesList=ExplainedVarSubFeaturesList
                        else:
                            SubFeaturesList=['All']
                                               
                        Features=SetFeatures(featureMethod,cutDataBy,cutDataParam=cutParam,DataObject=data,isLoadFeatures=isLoadFeatures)
                        print(Features)
                        if isLearning:
                            print('********Learning********'  )     
                            #save details and make results dir
                            LearningDetails={}
                            for i in ('part', 'label', 'featureMethod','cutDataBy'):
                                LearningDetails[i] = locals()[i]
                            savePath=os.path.join(resultsPath,'LearningResults',part,label)
                            # Learning Loop:
                            [ModelList_set,_,_]=initUtils.setLearningParams(ModelList=ModelList,label=label)                                 
                            for model in ModelList_set:
                                print('-Model = ' +model)
                                for decompositionMethod in DecompositionList:
                                    print('--decompositionMethod = ' + decompositionMethod)
                                    [_,FeatureSelectionList_set,decompositionLevel]=initUtils.setLearningParams(decompositionMethod=decompositionMethod,FeatureSelectionList=FeatureSelectionList)                                        
                                    for fs in FeatureSelectionList_set:
                                        print('---FeatureSelection = ' + fs)
                                        for S in SubFeaturesList:
                                            print('----SubFeature = ' + fs)
                                            for n_features in NFeatureList:
                                                print('-----n_features='+ str(n_features))
                                                #print('Model = ' + model +'\nLabelBy = ' + label + '\nDecomposition = '+ decompositionMethod + 'FeatureSelection = ' + fs + '\nNum Of Features = ' + str(n_features))
                                                Details=Features.details.copy()
                                                LearningDetails={'LabelBy':label,'FeatureMethod':featureMethod,'CutBy':cutDataBy,'cutParam':cutParam,'Model':model, 'DecompositionMethod':decompositionMethod, 'n_components':30, 'FeatureSelection':fs, 'n_features':n_features, 'is_cross_validation':is_cross_validation,'CrossVal':'LOO','is_cross_validation':is_cross_validation,'SubFeatures':S}
                                                Details.update(LearningDetails)
                                                s=LearnObject(Features,Labels,Details)
                                                s.run(Model=model, DecompositionMethod=decompositionMethod,decompositionLevel='FeatureType',n_components=30, FeatureSelection=fs, n_features=n_features, isPerm=0,isBetweenSubjects=True,isConcatTwoLabels=False,isSaveCsv=True, isSavePickle=False, isSaveFig=False,isSelectSubFeatures=isSelectSubFeatures,SubFeatures=S,is_cross_validation=is_cross_validation,savePath=savePath)
                                                LabelNameList=s.ResultsDF.columns #TODO - CHANGE THIS!
                        
"""--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"""
resultsPath='C:\\Users\\taliat01\\Desktop\\TALIA\\Code-Python\\Results'
RawDataPath='C:\\Users\\taliat01\\Desktop\\TALIA\\Code-Python\\AllPartsData'

#Data=dataUtils.loadData(PartName,Variables)
main()

        