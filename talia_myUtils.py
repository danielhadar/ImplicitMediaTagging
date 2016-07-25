#functions for processing faceshift data
import pickle
import numpy as np
from pandas import *
from pandas import DataFrame as DF
from sklearn import svm
from sklearn import linear_model
from sklearn import feature_selection
from sklearn import decomposition
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import scale

from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve, auc
import pylab as pl

import os
import matplotlib.pyplot as plt
import scipy.stats as sstats
from scipy import cluster
class newObject(object):
    pass


# Features functions:
class dataUtils:
    ## Process fs-signal and save results in dataframe   

    @staticmethod
    def setAndSaveAllDataRawClusteredQuantizedDF(AllPartsData,Variables,dataPath='C:\\Users\\taliat01\\Desktop\\TALIA\\Code-Python\\Results\\LearningData\\'):
        Parts=AllPartsData.index.levels[0]
        for part in Parts:
            print('--'+part+'--')
            rawData=AllPartsData.loc[part]
            clusteredDF, clustersCenters, MethodDetails =dataUtils.clusterData(rawData[Variables].T)
            quantizedDF, MethodDetails=dataUtils.quantizeData(rawData[Variables])
            print('saving...')
            clusteredDF.to_pickle(dataPath+part+'\\DATAclustered.pickle')
            quantizedDF.to_pickle(dataPath+part+'\\DATAquantized.pickle')
            clustersCenters.to_pickle(dataPath+part+'\\cluster_centers.pickle')

    @staticmethod
    def loadData(PartName,Variables,dataPath='C:\\Users\\taliat01\\Desktop\\TALIA\\Code-Python\\Results\\LearningData\\'):
        print('loading '+ PartName + ' DATA from '+ dataPath+'...')
        openFile=lambda dataType: open(dataPath + PartName + '\\DATA'+ dataType+'.pickle','rb')
        data=newObject()
        data.raw=pickle.load(openFile('raw'))[Variables]
        data.clustered=pickle.load(openFile('clustered'))
        data.quantized=pickle.load(openFile('quantized'))
        data.details=dataUtils.setDataDetails(data,PartName,dataPath)
        return data

    @staticmethod
    def loadSegmentedData(PartName,PieceLength,dataPath='C:\\Users\\taliat01\\Desktop\\TALIA\\Code-Python\\Results\\LearningData\\'):
        print('loading '+ PartName + ' segmented DATA from '+ dataPath+'...')
        data=newObject()
        openFile=lambda dataType: open(dataPath+PartName+'\\DATA'+dataType+'_'+str(PieceLength)+'.pickle','rb')
        data.raw=pickle.load(openFile(''))
        data.clustered=pickle.load(openFile('clustered'))
        data.quantized=pickle.load(openFile('quantized'))
        data.details=dataUtils.setDataDetails(data,PartName,dataPath,PieceLength=PieceLength)
        return data
                                   
    @staticmethod
    def loadDataByStimuli(PartName,stimuliOperation=None,dataPath='C:\\Users\\taliat01\\Desktop\\TALIA\\Code-Python\\Results\\LearningData\\'):
        data=newObject()
        if PartName in ['Photos','Videos']:
            print('loading '+ PartName + ' byStimuli DATA from '+ dataPath+'...')
            openFile=lambda dataType: open(dataPath +PartName+'\\'+ dataType +'DataBy'+PartName+'_' + stimuliOperation +'.pickle','rb')
            data.raw=pickle.load(openFile('raw'))
            data.clustered=pickle.load(openFile('clustered'))
            data.quantized=pickle.load(openFile('quantized'))
            data.details=dataUtils.setDataDetails(data,PartName,dataPath,stimuliOperation=stimuliOperation)
        return data
                    
    @staticmethod
    def quantizeData(rawData,n_quants=4,QuantizationMethod='random'):
        print('Quantazing data..')
        quantizedDF=DF(index=rawData.index)
        MethodDetails={'NumOfQuants':n_quants,'QuantizationMethod':QuantizationMethod}
        Smoothed=rawData-0.2
        Smoothed[Smoothed<0]=0
        for columns in rawData:
            Quantized=cluster.vq.kmeans2(Smoothed.values ,n_quants, iter=10, thresh=1e-02, minit=QuantizationMethod, missing='warn')
            quantizedDF[columns]=Quantized[1]
        return quantizedDF, MethodDetails
    
    @staticmethod
    def clusterData(rawData, n_clusters=7):
        print('Clustering data...')
        MethodDetails={'NumOfclusters':n_clusters,'ClusteringMethod':'kmeans'}
        kmeans=KMeans(n_clusters=n_clusters)
        #rawData=rawData.drop('time',axis=1)
        kmeans.fit_predict(rawData.T)
        clusteredDF=DF(kmeans.labels_,index=rawData.T.index)
        clustersCenters=DF(kmeans.cluster_centers_,columns=rawData.index)
        return clusteredDF, clustersCenters, MethodDetails     
    
    @staticmethod   
    def cutDataBySegment(Parts,PieceLength,Variables,dataPath='C:\\Users\\taliat01\\Desktop\\TALIA\\Code-Python\\Results\\LearningData\\'):
        """this function cuts the index of raw data to segments in 'PieceLength' defined size"""
        def cut(rawData, PieceLength):
            timeIndex=rawData.index.get_level_values('time').values.flatten()
            subjectIndex=rawData.index.get_level_values('sCode').values.flatten()
            segments=range(0,max(timeIndex),PieceLength)
            sIndex=DF(index=range(len(timeIndex)),columns=segments)
            for s in segments:
                s0=s-1
                s1=s+PieceLength-1
                sIndex[s]=((timeIndex>s0) & (timeIndex<s1))*s
            segmentedIndex=MultiIndex.from_arrays([subjectIndex,sIndex.sum(axis=1)],names=['subject','Piece_ind'])
            segmentedData=rawData.copy()
            segmentedData.index=segmentedIndex
            return segmentedData   

        for part in Parts:
            print('Cutting data...')
            rawData=pickle.load(open(dataPath+part+'\\DATAraw.pickle','rb'))[Variables]
            clusteredData=pickle.load(open(dataPath+part+'\\DATAclustered.pickle','rb'))
            quantizedData=pickle.load(open(dataPath+part+'\\DATAquantized.pickle','rb'))
            segmented_rawData=cut(rawData,PieceLength)
            clusteredData.index=segmented_rawData.index
            quantizedData.index=segmented_rawData.index
            print('saving segmented data...')
            segmented_rawData.to_pickle(dataPath+part+'\\DATA_'+str(PieceLength)+'.pickle')
            clusteredData.to_pickle(dataPath+part+'\\DATAclustered_'+str(PieceLength)+'.pickle')
            quantizedData.to_pickle(dataPath+part+'\\DATAquantized_'+str(PieceLength)+'.pickle') 
    

    @staticmethod
    def cutDataByStimuli(Parts, Variables, dataPath='C:\\Users\\taliat01\\Desktop\\TALIA\\Code-Python\\Results\\LearningData\\',TimeStampsPath='C:\\Users\\taliat01\\Desktop\\TALIA\\Code-Python\\Results\\LearningLabels\\Rating\\Dataframes\\AllSubjectsTimeStamps'):
        print('------cutting data by stimuli------') 
        for part in Parts:
            print('--loading '+ part + ' data...') 
            TimeStampsDF=pickle.load(open(TimeStampsPath+part+'.pickle','rb'))
            TimeStampsDF.index=MultiIndex.from_arrays([TimeStampsDF['subject'],TimeStampsDF['time']])
            TimeStampsDF=TimeStampsDF.drop_duplicates()
            TimeStampsDF.dropna(how='all')
            DataTypes=['raw','clustered','quantized'] #,
            for d in DataTypes: 
                print('--- cutting '+ d + ' data...') 
                data=pickle.load(open(dataPath+part+'\\DATA'+d+'.pickle','rb'))
                data.index.names=['subject','time']
                PartTimeStamps=TimeStampsDF[str.lower(part[:-1])+'Code']
                subjectList=list(set(data.index.levels[0]).intersection(PartTimeStamps.index.levels[0]))
                ErrorSubjectList=[]
                AlignedData=DF()
                AlignedTimeStampsDF=DF()
                for subject in subjectList:
                    print(subject)
                    # align subject Data
                    subjectData=data.loc[subject]
                    subjectTimeRange=subjectData.index
                    subjectIndex=MultiIndex.from_product([subject,subjectTimeRange])
                    subjectData.index=subjectIndex
                    AlignedData=concat([AlignedData,data.loc[subject]])
                    # align subject TimeStamps
                    subjectTimeStamps=TimeStampsDF.loc[subject].iloc[subjectTimeRange]
                    subjectTimeStamps.index=subjectIndex
                    AlignedTimeStampsDF=concat([AlignedTimeStampsDF,subjectTimeStamps])
                   
                #select data by operation in TimeStamps
                TimeStampsIndex=MultiIndex.from_arrays([AlignedTimeStampsDF['subject'],AlignedTimeStampsDF[str.lower(part[:-1])+'Code']])
                AlignedTimeStampsDF.index=TimeStampsIndex
                AlignedData.index=TimeStampsIndex
                watchSegmentedData=AlignedData[AlignedTimeStampsDF['operation']=='watch']
                rateSegmentedData=AlignedData[AlignedTimeStampsDF['operation']=='rate']
                watchAndRateSegmentedData=watchSegmentedData.append(rateSegmentedData).sort_index()
                
                #save
                saveName=dataPath+d+'DataBy'+part+'_'
            
                ##save to pickle
                watchSegmentedData.to_pickle(saveName+'watch.pickle')
                rateSegmentedData.to_pickle(saveName+'rate.pickle')
                watchAndRateSegmentedData.to_pickle(saveName+'watchAndRate.pickle')
                
                
                ##save to csv
                watchSegmentedData.to_csv(saveName+'watch.csv')
                rateSegmentedData.to_csv(saveName+'rate.csv')
                watchAndRateSegmentedData.to_csv(saveName+'watchAndRate.csv')
                
                
               
                print('data successfully saved to '+ saveName+ '!')
            
            #saving timeStamps as labels for learning
            ratingSaveName='C:\\Users\\taliat01\\Desktop\\TALIA\\Code-Python\\Results\\LearningLabels\\'+ part +'Rating_Labels'
            AlignedTimeStampsDF.to_csv(ratingSaveName+'.csv')
            AlignedTimeStampsDF.to_pickle(ratingSaveName+'.pickle')
            print('AlignedTimeStampsDF successfully saved to '+ ratingSaveName+ '!')
            if len(ErrorSubjectList)>0:
                print('##NOTE! the following subjects were not saved:##')
                print(ErrorSubjectList)
    
    @staticmethod
    def setDataDetails(dataObject,PartName,dataPath,stimuliOperation=None,PieceLength=None):
        details={}
        details['Part']=PartName
        details['dataPath']=dataPath
        details['PieceLength']=PieceLength
        details['stimuliOperation']=stimuliOperation
        details['n_clusters']=dataObject.clustered.max().max()
        details['n_quantas']=dataObject.quantized.max().max()
        details['subjectsList']=list(dataObject.raw.index.levels[0])
        return details
        
class featuresUtils:
    @staticmethod
    def setAndSaveAllFeatures(DataObject):
        QuantizationFeatures=calcQuantizationFeatures(DataObject)

    @staticmethod
    def getTransMatrix(SubjectQuantizedData,numOfQuants): 
            """ calcs transition matrix over columns of quantized dataframe  returns a dict with transmatrix for each column of Data """     
            #used for calculating quantized features  - ChangeRatio, FastChangeRatio
            from collections import Counter
            qData=SubjectQuantizedData       
            k=numOfQuants
            TransMatrixDict=dict.fromkeys(qData.columns)
            for col in qData:
                vec=qData[col]
                TransMatrix=np.zeros([k,k])
                for (x,y), c in Counter(zip(vec, vec[1:])).iteritems(): 
                    TransMatrix[x-1,y-1] = c
                TransMatrixDict[col]=TransMatrix
            return TransMatrixDict
            
    @staticmethod
    def countBlocks(SubjectBinaryData):
        """ takes binary DF and returns for each column the number of blocks, their length, etc.."""
        bData=SubjectBinaryData*1 
        BlocksDF=DF(columns=bData.columns,index=['NumOfBlocks','meanBlockLength','stdBlockLength']) #save summary of block details on each column
        BlocksDetails=dict.fromkeys(bData.columns) #save full details about blocks for each column
        DF0=DF([0]*len(bData.columns),index=bData.columns).T
        bData=DF0.append(bData.append(DF0)) #add 0 in the begining and in the end of the vector
        for col in bData:
            vec=np.array(bData[col])            
            diffvec=np.diff(vec)
            blockStart=np.where(diffvec==1)[0]
            blockEnd=np.where(diffvec==-1)[0]
            #fix first and last frames            
            if blockStart.size==0 or blockEnd.size==0: #if there are no blocks
                numOfBlocks=0
                BlockLength=np.array([0])
            else:                    
                if blockStart[0]>blockEnd[0]: #if there's an end of the block before the start ignore it. 
                    blockStart=np.append(0,blockStart)
#                    print('** 0 appended to start')
                if blockStart[-1]>blockEnd[-1]: #if the block starts after the last end, ignore it.        
                    blockEnd=np.append(blockEnd,len(vec))
#                    print('** ' + str(len(vec)) + ' appended to end')
                numOfBlocks=len(blockEnd)
                if len(blockEnd)!=len(blockStart):
                    print(blockEnd)
                    print(blockStart)
                    print(vec)
                    
                BlockLength=blockEnd-blockStart
                #save into dict
                BlocksDetails[col]= DF({'1BlockStartIndex':blockStart,'2BlockEndIndex':blockEnd,'3BlockLength':BlockLength}, index=range(numOfBlocks)).T

            #save summary as DF
            BlocksDF[col].loc['NumOfBlocks']=numOfBlocks
            BlocksDF[col].loc['meanBlockLength']=BlockLength.mean()
            BlocksDF[col].loc['stdBlockLength']=BlockLength.std()
        return BlocksDF.T, BlocksDetails

    @staticmethod
    def calcMomentsFeatures(subjectRawData,n):
            sData=subjectRawData
            cols=sData.columns
            subjectFeaturesDF=DF(columns=['mean','std','skewness','kurtosis'],index=cols)
            subjectFeaturesDF['mean']=sData.mean()
            subjectFeaturesDF['std']=sData.std()
            subjectFeaturesDF['skewness']=sData.skew()
            subjectFeaturesDF['kurtosis']=sData.kurtosis()
            subjectFeaturesSeries=subjectFeaturesDF.unstack()
            return subjectFeaturesSeries


    @staticmethod
    def calcQuantizationFeatures(subjectQuantizedData,n_quants):
            sData=subjectQuantizedData
            cols=sData.columns
            ExpressionInd=sData!=0 
            ExpressionRatio=ExpressionInd.sum()/len(sData)
            ExpressionLevel=sData[ExpressionInd].mean()/3
            [Blocks,_]=featuresUtils.countBlocks(sData>0)
            ExpressionLength=Blocks['meanBlockLength']

            #calc feature using transition matrix:
            TransitionMatrix=featuresUtils.getTransMatrix(sData,n_quants)           
            ChangeRatio=Series(index=cols)
            FastChangeRatio=Series(index=cols)
            for col in cols:
                T=TransitionMatrix[col]
                N=sum(sum(T))
                ChangeFrames=N-(sum(np.diagonal(T)))
                SlowChangeFrames=sum(np.diagonal(T,offset=1,axis1=1,axis2=0)+np.diagonal(T,offset=1))
                FastChangeFrames=ChangeFrames-SlowChangeFrames
                ChangeRatio[col]=ChangeFrames/(N-T[0,0])
                FastChangeRatio[col]=FastChangeFrames/ChangeFrames
            subjectFeaturesSeries=DF([ExpressionRatio,ExpressionLevel,ExpressionLength,ChangeRatio,FastChangeRatio],index=['ExpressionRatio','ExpressionLevel','ExpressionLength','ChangeRatio','FastChangeRatio'])
            subjectFeaturesSeries=subjectFeaturesSeries.unstack().swaplevel(0,1)

            return subjectFeaturesSeries
    
    @staticmethod
    def calckMeansClusterFeatures(subjectClusteredData,n_clusters,isSegmented=1):
        IndexAllClusters=['NumOfClusters','ClusterMeanLength','ClusterChangeRatio','ExplainedVariance']
        IndexEachCluster=['Counts','Length','Num']
        clusterRange=range(n_clusters)
        multIndexEachCluster=MultiIndex.from_product([IndexEachCluster,clusterRange])
        multIndexAllClusters=MultiIndex.from_product([IndexAllClusters,['all']])
        clustersScores=multIndexEachCluster.append(multIndexAllClusters)
        #TODO, make sure it calcs features for all pieces!
        sData=subjectClusteredData
        value_counts=Series(sData[0]).value_counts() #the clusters that actually apeared in subject video and their counts
        value_counts.index=[int(x) for x in value_counts.index]
        
        subjectClusterFeatures=DF(0,columns=IndexEachCluster,index=range(n_clusters+1))

        for c in value_counts.index:
            subjectClusterFeatures['Counts'].loc[c]=value_counts.loc[c]
            [Blocks,_]=featuresUtils.countBlocks(DF(sData==c))
            subjectClusterFeatures['Length'].loc[c]=Blocks['meanBlockLength'][0]
            subjectClusterFeatures['Num'].loc[c]=Blocks['NumOfBlocks'][0]
        subjectClusterFeatures=Series(subjectClusterFeatures.unstack())
            #ClustersTransitionMatrix=featureUtils.getTransMatrix(subjectClusters,n_clusters)   
        subjectClusterFeatures['NumOfClusters']=len(value_counts)
        TransitionMatrix=featuresUtils.getTransMatrix(DF(sData),n_clusters)[0]
        N=sum(sum(TransitionMatrix))
        ChangeFrames=N-(sum(np.diagonal(TransitionMatrix)))
        ChangeRatio=ChangeFrames/(N-TransitionMatrix[0,0])
        subjectClusterFeatures['ClusterMeanLength']=subjectClusterFeatures['Length'].mean()
        subjectClusterFeatures['ClusterChangeRatio']=ChangeRatio  
        
        subjectClusterFeatures=subjectClusterFeatures.fillna(0.)

        return subjectClusterFeatures

    def calckMeansExplainedVar(ClusteredData,RawData,Centroids):
        #calculates explained var by kmeans clustering, and distance from cluster centeroids for each segment for each subject
        from scipy.spatial import distance
        Centroids=DF.from_csv('C:\\Users\\taliat01\\Desktop\\TALIA\\Code-Python\\Results\\Analysis\\clustersCenters.csv')
        Centroids=Centroids.loc[RawData.columns]
        #CentroidData=DF(index=RawData.index,columns=RawData.columns)
        ExplainedVar=DF()
        for subject in RawData.index.levels[0]:
            print(subject)
            subjectData_raw=RawData.loc[subject]
            subjectData_clustered=ClusteredData.loc[subject]
            for segment in subjectData_raw.index.unique():
                subjectData_segment=subjectData_raw.loc[segment].reset_index().drop('Piece_ind',axis=1)
                clusters=subjectData_clustered.loc[segment].reset_index().drop('Piece_ind',axis=1)
                segmentDistances=DF(index=clusters.index,columns=['distance'])
                kmeans=KMeans(n_clusters=1)
                a=kmeans.fit_predict(subjectData_segment)
                segmentVariance=kmeans.inertia_
                for i in subjectData_segment.index:
                    #print((subject,segment,i))
                    segmentDistances.loc[i]=distance.euclidean(Centroids[str(clusters.loc[i][0]+1)],subjectData_segment.loc[i])
            
                SegmentExplainedVar=DF([segmentDistances.mean().values,DF(1-segmentDistances.mean()/segmentVariance).values[0]],columns=[(subject,str(segment))],index=['DistanceFromCentroids','ExplainedVar'])
                ExplainedVar=concat([ExplainedVar,SegmentExplainedVar.T])
                ExplainedVar.index=MultiIndex.from_tuples(ExplainedVar.index)
                return ExplainedVar

    @staticmethod
    def initX(FeaturesDF,trainLabels_all,Labeling,is2=False):
            Xall=FeaturesDF
            if is2:
                pass
                """FeatureSubjectsNames=Xall.index.levels[0]
                FeatureSubjectsNames2=[s+'2' for s in FeatureSubjectsNames]
                Xall.index.levels[0].reindex(FeatureSubjectsNames2) #continue here - change index names!!
                X2=DF(Xall.values,columns=Xall.columns,index=Xall.index.levels[0].reindex(FeatureSubjectsNames2))
                Xall.index.levels[0]=FeatureIndex2
                LabelSubjectsName=trainLabels_all.index
                trainLabels_all.index=[s+'2' for s in LabelSubjectsNames]"""
            FeatureSubjects=Xall.index.levels[0]
            AllSubjects=list(set(FeatureSubjects) & set(trainLabels_all.index)) #intercect label and feature lists 
            LabeledSubjects=trainLabels_all[Labeling].dropna()>-1
            LabeledSubjects=LabeledSubjects[LabeledSubjects].index
            if not(AllSubjects): #if trainLabels_all is a multi index (for labelBy 'stimuliRating')
                AllSubjects=list(set(FeatureSubjects) & set(trainLabels_all.index.levels[0]))
                LabeledSubjects=trainLabels_all[Labeling].dropna()>-1
                LabeledSubjects=LabeledSubjects[LabeledSubjects].index.levels[0]
            SubjectsList=list(set(AllSubjects) & set(LabeledSubjects)) #intercect label and feature lists           
            droppedSubjects=list(set(AllSubjects).difference(set(SubjectsList)))              
            X=Xall.loc[SubjectsList].sort_index() #use only subjects in subjectsList #TODO- fix this for EXPLAINED VARIANCE!
           # X=X.apply(scale)
            #X=Imputer(X)
            Xdropped=Xall.loc[SubjectsList].sort_index()

            return X,SubjectsList,droppedSubjects,Xdropped
    
    @staticmethod
    def getMissingFeatures(Features):
        X=Features.FeaturesDF.T
        FeaturesType=X.columns.levels[0]
        FeaturesAU=X.columns.levels[1]
        nanCount=0
        print('replacing missing values with Nans..')
        for ind in X.index:
            for au in FeaturesAU:
                if X.loc[ind]['ExpressionRatio',au]==0:
                    nanCount+=4
                    X.loc[ind]['ExpressionLength',au]=np.NAN
                    X.loc[ind]['ExpressionLevel',au]=np.NAN
                    X.loc[ind]['ChangeRatio',au]=np.NAN
                    X.loc[ind]['FastChangeRatio',au]=np.NAN
                elif X.loc[ind]['ChangeRatio',au]==0:
                    nanCount+=1
                    X.loc[ind]['FastChangeRatio',au]=np.NAN
        print(str(nanCount) + ' missing values have been found and replaced with NaNs')
        return X.T
    
    @staticmethod
    class DecompositionByLevelClass:
                def __init__(self,n_components,decompositionLevel, decompositionFunc):
                    self.n_components=n_components
                    self.decompositionLevel=decompositionLevel #the level of x on which to decompose
                    self.decompositionFunc= decompositionFunc
                    self.components_=DF()
                    self.explained_variance_=[]
                                

                def fit_transform(self,x,y=None):               
                    ColList=x.columns.get_level_values(self.decompositionLevel)
                    decompositionLevelList=[col for col in ColList.unique() if list(ColList).count(col)>1]
                    NoDecompositionLevelList=[col for col in ColList.unique() if list(ColList).count(col)==1] #only decompose features that has more than one value..  
                    n_components_perType=int(np.round(1.0*self.n_components/len(decompositionLevelList)))-1
                    new_x=DF()
                    dFun=self.decompositionFunc
                    dFun.n_components=n_components_perType
                    #self.dFunDict=dict.fromkeys(decompositionLevelList)
                    """self.fitDict=dict.fromkeys(decompositionLevelList)
                    self.transformDict=dict.fromkeys(decompositionLevelList)
                    self.decompositionLevelList=decompositionLevelList
                    self.NoDecompositionLevelList=NoDecompositionLevelList
                    self.xfMultiIndex=dict.fromkeys(decompositionLevelList)
                    self.explained_variance_=[]
                    for dIndex in decompositionLevelList:
                        xf=x[dIndex].dropna(how='all')
                        self.xfMultiIndex[dIndex]=MultiIndex.from_product([dIndex,range(n_components_perType)])
                        dFun=self.decompositionFunc
                        dFun.fit_transform(xf)
                        new_xf=DF(dFun.fit_transform(xf),index=x.index,columns=self.xfMultiIndex[dIndex])
                        components=DF(dFun.components_,columns=MultiIndex.from_product([dIndex,xf.columns]))
                        self.transformDict[dIndex]=dFun.transform
                        self.fitDict[dIndex]=dFun.fit
                        self.explained_variance_.extend(dFun.explained_variance_)
                        new_x=concat([new_x,new_xf],axis=1)
                        self.components_=concat([self.components_,components],axis=0,ignore_index=True) 
                    if NoDecompositionLevelList: #if it's not empty
                        new_x=concat([new_x,x[NoDecompositionLevelList]],axis=1) #add to featuresDF the features that are not decompoosed
                        components_noDecomposition=DF(np.eye(len(NoDecompositionLevelList)),columns=MultiIndex.from_product([NoDecompositionLevelList,[0]]))
                        self.components_=concat([self.components_,components_noDecomposition],axis=0,ignore_index=True) #add components weights==1 to non decomposed features.
                    self.components_=self.components_.fillna(0)"""
                    return new_x

                def fit(self,x):
                    new_x_fit=DF()
                    for dIndex in self.decompositionLevelList:
                        xf=x[dIndex]
                        new_xf=DF(self.fitDict[dIndex](xf),index=x.index,columns=self.xfMultiIndex[dIndex])
                        new_x_fit=concat( new_x_fit,new_xf)
                    return new_x_fit

                def transform(self,x):
                    new_x_transformed=DF()
                    for dIndex in self.decompositionLevelList:
                        xf=x[dIndex].dropna(how='all')
                        new_xf=DF(self.transformDict[dIndex](xf),index=xf.index,columns=self.xfMultiIndex[dIndex])
                        new_x_transformed=concat([new_x_transformed,new_xf],axis=1)
                    if self.NoDecompositionLevelList:
                        new_x_transformed=concat([new_x_transformed,x[self.NoDecompositionLevelList]],axis=1)

                    return new_x_transformed            
                    
class labelUtils:

        @staticmethod
        def setLabelsForRating(LabelsPath='C:\\Users\\taliat01\\Desktop\\TALIA\\Code-Python\\Results\\LearningLabels\\PhotosRating\\DataFrames\\'):
            """this function is used one time preleminary to learning, in order to generate one dataframe with all information regarding the experiment - frame by frame. this include:
            # stimuli presented (photo\video)
            # rating of the stimuli
            # operation regarding the stimuli (watch/rate)
            # subject name
            # time """
            PhotosRating=DF.from_csv(LabelsPath + 'PhotosRating.csv')
            #VideosRating=DF.from_csv(LabelsPath + 'VideosRating.csv')
            #VideosTiming=DF.from_csv(LabelsPath + 'AllVideoTiming.csv')
            PhotosTiming=DF.from_csv(LabelsPath + 'AllPhotosTiming.csv')
            StartTimes=DF.from_csv(LabelsPath + 'Start-End-Timing.csv')
            print('all DataFrames loaded!')
            AllSubjectsTimeStamps=DF()
            ExcludedSubjects=[]
                        #init stimuli according to video \ photo
            stimuliCode='photoCode' #'photoCode'
            Timing = PhotosTiming
            Rating=PhotosRating
            StartTime=StartTimes['Photos']
            #run 
            SubjectsList=list(set(Timing.index).intersection(set(StartTime.index)))
            #SubjectsList=['c15ovmo']
            for subject in SubjectsList: #continue here, make sure it works with adding SubjectStartingFrame!
                print(subject)
                if subject=='jaslfdja':
                    print('break')
                try:
                    subjectTiming=Timing.loc[subject]-Timing.loc[subject].iloc[0]
                    subjectTiming=subjectTiming[subjectTiming>=0]
                    SubjectStartingFrame=int(StartTime.loc[subject]) #when did we hear the 'bip' in the video, indicating experiment has started (preparation for first stimuli)
                    if SubjectStartingFrame>0:
                        subjectTimeStamps=concat([DF([0],index=['Prepare']),subjectTiming.dropna()+SubjectStartingFrame])
                    else: 
                        subjectTimeStamps=Timing.loc[subject].dropna()+SubjectStartingFrame
                        subjectTimeStamps[subjectTimeStamps<0]=0
                    TimeIndex=subjectTimeStamps.index
                    firstFrame=int(subjectTimeStamps.iloc[0])
                    lastFrame=int(subjectTimeStamps.iloc[-1])
                    stampsRange=range(len(subjectTimeStamps)-1)
                    SubjectTimeStampVector=DF(index=range(lastFrame),columns=['subject','time',stimuliCode,'operation','rating'])
                    SubjectTimeStampVector['subject']=subject
                    SubjectTimeStampVector['time']=range(lastFrame)                    
                    #SubjectTimeStampVector['operation'].loc[range(0,firstFrame)]='Prepare'
                    for i in stampsRange:
                        FullCode=TimeIndex[i]
                        try:
                            Code=FullCode[0:FullCode.index('_')]
                            start_stopCode=FullCode[FullCode.index('_')+1:]
                            subjectRating=Rating[Code].loc[subject]
                            if 'stopped' in start_stopCode:
                                operationCode='rate'
                            elif 'start' in start_stopCode:
                                operationCode='watch'
                        except ValueError:
                            Code=FullCode
                            operationCode=FullCode      
                            subjectRating=np.nan                
                        prepareLocation=range(0,)
                        location=list(np.array(range(int(subjectTimeStamps.iloc[i]),int(subjectTimeStamps.iloc[min(i+1,len(subjectTimeStamps)-1)]))))
                        if location!=0:
                            SubjectTimeStampVector[stimuliCode].loc[location]=Code
                            SubjectTimeStampVector['operation'].loc[location]=operationCode
                            SubjectTimeStampVector['rating'].loc[location]=subjectRating 

                    AllSubjectsTimeStamps=concat([AllSubjectsTimeStamps,SubjectTimeStampVector])
                except ValueError:
                    print('---- excluded!----')
                    ExcludedSubjects=ExcludedSubjects+[subject]
            return AllSubjectsTimeStamps,ExcludedSubjects


        @staticmethod
        def initTrainTestLabels_all(LabelObject):
            import globalVars as glv
            isBoolModel=glv.isBoolModel
            isBoolLabel=glv.isBoolLabel

            while not(isBoolModel) and isBoolLabel: 
                print('ERROR - cannot calculate regression over 0-1 labels!')
                Model=raw_input('Choose model for learning (svc, regression, ridge, lasso): ')
                model, isBoolModel= learningUtils.setModel(Model)

            if not(isBoolModel) and not(isBoolLabel):
                trainLabels_all=LabelObject.contLabelsDF
                testLabels_all=LabelObject.contLabelsDF
                TrueLabels=LabelObject.contLabelsDF
                isAddDroppedSubjects=0
            
            if isBoolModel and isBoolLabel :
                trainLabels_all=LabelObject.boolLabelsDF
                testLabels_all=LabelObject.boolLabelsDF
                TrueLabels=LabelObject.boolLabelsDF
                isAddDroppedSubjects=0

            if isBoolModel and not(isBoolLabel):
                trainLabels_all=LabelObject.boolLabelsDF
                testLabels_all=LabelObject.contLabelsDF
                TrueLabels=LabelObject.contLabelsDF
                isAddDroppedSubjects=1
            glv.transformMargins='notDefined'

            return trainLabels_all, testLabels_all, TrueLabels,isAddDroppedSubjects
        
        @staticmethod
        def initTrainTestLabels(Labeling,SubjectsList,trainLabels_all, testLabels_all):
            trainLabels=trainLabels_all[Labeling].loc[SubjectsList]
            trainLabels=trainLabels[trainLabels>=0] #exclude subjects in Label "-1"
            trainLabels.dropna()
            testLabels=testLabels_all[Labeling].loc[SubjectsList] 
            testLabels=testLabels[testLabels>=0] #exclude subjects in Label "-1"
            testLabels.dropna()
            LabelRange=testLabels.max()-testLabels.min()

            return trainLabels, testLabels, LabelRange   
            
        
## SVM functions: 
class initUtils:
    
        @staticmethod
        def setModelListAccordingToLabel(ModelList,label): #makes sure the bool labeles are learned by bool models!
            
            return ModelList
        
        @staticmethod
        def setLearningParams(ModelList=[],label='skip',decompositionMethod='skip',FeatureSelectionList=[]):
            #set Model list according to Label - makes sure the bool labeles are learned by bool models!
            if label!='skip':
                if label in ['PatientsVsControls', 'boolMentalStatus']: 
                        newModelList=['svc']
                elif label in ['StimuliRating']:
                    newModelList=['multiclass']
                elif label in ['PANSS']:
                    newModelList=ModelList
                    if 'svc' in ModelList:
                        newModelList=ModelList.remove('svc')
                    if 'multiclass' in ModelList:
                        newModelList=ModelList.remove('multiclass')
                print( "ModelList for label="+label+" was reset to:")
                print(newModelList)
            else:
                newModelList=ModelList
            #set Model list according to decomposition method
            if decompositionMethod!='skip':
                if decompositionMethod == 'noDecomposition':
                    newFeatureSelectionList=['f_regression']
                else: 
                    newFeatureSelectionList=FeatureSelectionList
                #set decomposition level according to decomposition method
                if decompositionMethod in ['FeatureType_PCA','fs-signal_PCA']:
                    decompositionLevel=decompositionMethod[:decompositionMethod.index('_')]
                else:
                    decompositionLevel='All'
            else:
                newFeatureSelectionList,decompositionLevel=[None,None]
            
            return newModelList,newFeatureSelectionList,decompositionLevel

        @staticmethod
        def setCrossValidation(cross_validationMethod,cv_param,trainLabels,isWithinSubjects):
                from sklearn import cross_validation
                if cross_validationMethod=='stratifiedKFold': #TODO -> maybe instead of string insert the function cross_val.KFold() as in put to the function run
                    cv=cross_validation.StratifiedKFold(trainLabels)
                    if isWithinSubjects:
                        print('StratifiedKfold does not work with within subject cross validation!!!!!!!!!!!!!')
                elif cross_validationMethod=='KFold':
                    cv=cross_validation.KFold(cv_param)
                elif cross_validationMethod=='LOO': #Very slow.. ):
                    cv=cross_validation.LeaveOneOut(cv_param)
                crossValScores=DF()
                return cv, crossValScores
            
        @staticmethod
        def setXYTrainXYTest(X,Labeling,trainLabels,testLabels,TrueLabels,train_subjects,test_subjects):
                from globalVars import isMultiLabels                 
                #set Y according to train and test subjects
                if not(isMultiLabels): #if labeling is not a list, (isMultiLabel=False)
                    Ytrain=Series(index=X.index)
                    Ytest=Series(index=X.index)
                    YtrainTrue=Series(index=X.index) 
                    for subject in train_subjects:
                        Ytrain.loc[subject]=trainLabels.loc[subject]
                        YtrainTrue.loc[subject]=TrueLabels[Labeling].loc[subject]
                    for subject in test_subjects:
                        Ytest.loc[subject]=testLabels[subject]
  
                if isMultiLabels:  #if we use multivariable labeling
                    Ytrain=DF(index=X.index,columns=Labeling)
                    Ytest=DF(index=X.index,columns=Labeling)
                    YtrainTrue=DF(index=X.index,columns=Labeling)
                    for subject in train_subjects:
                         for l in Labeling:
                            Ytrain[l].loc[subject]=trainLabels[l].loc[subject]
                            YtrainTrue[l].loc[subject]=TrueLabels[l].loc[subject] 
                    for subject in test_subjects:
                        for l in Labeling:
                            Ytest[l].loc[subject]=testLabels[l].loc[subject]

                Ytrain=Ytrain.dropna()
                YtrainTrue=YtrainTrue.dropna()                 
                Ytest=Ytest.dropna() 

                #set X according to train and test subjects,impute non relevant features and 
                XtrainAllFeatures=X.loc[list(train_subjects)]
                XtrainAllFeatures=XtrainAllFeatures.dropna(how='all')
                #XtrainAllFeatures=XtrainAllFeatures.fillna(0.)
                isImpute=0
                if isImpute:
                    XtrainAllFeatures=learningUtils.imputeFeatureMatrix(XtrainAllFeatures,Ytrain) #impute missing X values according to label median

                XtestAllFeatures=X.loc[list(test_subjects)]
                Ytrain=Ytrain.loc[XtrainAllFeatures.index].copy()
                YtrainTrue=YtrainTrue.loc[XtrainAllFeatures.index].copy()
                #select best N features
                
                return XtrainAllFeatures,XtestAllFeatures, Ytrain, YtrainTrue, Ytest 
            
class learningUtils:
        @staticmethod
        def decomposeAndSelectBestNfeatures(Xtrain_allFeatures,Xtest_allFeatures,Ytrain,n_features,selectFeatures,decomposeFunction,decompositionLevel):
            Xtrain_allFeatures=Xtrain_allFeatures.dropna(how='all').fillna(0.)
            Xtest_allFeatures=Xtest_allFeatures.dropna(how='all').fillna(0.)
            
            if decompositionLevel=='All':
                try:
                    Xtrain_allFeatures_decomposed=decomposeFunction.fit_transform(Xtrain_allFeatures) 
                except TypeError: 
                    Xtrain_allFeatures_decomposed=decomposeFunction.fit_transform(Xtrain_allFeatures,y=Ytain)  # use only n features selected for learning
                Xtest_allFeatures_decomposed=decomposeFunction.transform(Xtest_allFeatures)
                components=decomposeFunction.components_
                explainedVar=decomposeFunction.explained_variance_ 
            else: #decompositionByLevel
                
                ColList=Xtrain_allFeatures.columns.get_level_values(decompositionLevel)
                decompositionLevelList=[col for col in ColList.unique() if list(ColList).count(col)>1]
                NoDecompositionLevelList=[col for col in ColList.unique() if list(ColList).count(col)==1] #only decompose features that has more than one value..  
                Xtrain_allFeatures_decomposed=DF()
                Xtest_allFeatures_decomposed=DF()
                components=DF()
                explainedVar=[]
                #n_components=decomposeFunction.n_components
                decomposeFunction_n_components=decomposeFunction.n_components #to reset at the end of function
                try:
                    n_components_perType=int(np.round(1.0*decomposeFunction.n_components/len(decompositionLevelList)))
                    n_components=n_components_perType
                    for dIndex in decompositionLevelList:
                        d_xTrain=Xtrain_allFeatures[dIndex].dropna(how='all')
                        d_xTest=Xtest_allFeatures[dIndex].dropna(how='all')
                        d_n_components=min(n_components,len(d_xTrain.columns))
                        decomposeFunction.n_components=d_n_components
                        #d_n_components=n_components_perType   #TODO- test which works better...
                        dColIndex=MultiIndex.from_product([dIndex,range(d_n_components)])
                        d_xTrain_decomposed=DF(decomposeFunction.fit_transform(d_xTrain),index=d_xTrain.index,columns=dColIndex)
                        d_xTest_decomposed=DF(decomposeFunction.transform(d_xTest),index=d_xTest.index,columns=dColIndex)
                        Xtrain_allFeatures_decomposed=concat([Xtrain_allFeatures_decomposed,d_xTrain_decomposed],axis=1)
                        Xtest_allFeatures_decomposed=concat([Xtest_allFeatures_decomposed,d_xTest_decomposed],axis=1)
                        d_components=DF(decomposeFunction.components_,index=dColIndex,columns=d_xTrain.columns)
                        components=concat([d_components,components],axis=0) 
                except ZeroDivisionError: #if there is no more than one feature, so no decomposition is possible...
                    Xtrain_allFeatures_decomposed=Xtrain_allFeatures
                    Xtest_allFeatures_decomposed=Xtest_allFeatures
                    components=[]
                decomposeFunction.n_components=decomposeFunction_n_components #reset for later iteration
            #select best N features: 
            bestNfeatures,Xtrain_nFeatures =selectFeatures(Xtrain_allFeatures_decomposed,Ytrain,n_features,decomposeFunction) 
            Xtest_nFeatures=Xtest_allFeatures_decomposed[bestNfeatures]
               
            """dFun.fit_transform(xf)
                    new_xf=DF(dFun.fit_transform(xf),index=x.index,columns=self.xfMultiIndex[dIndex])
                    components=DF(dFun.components_,columns=MultiIndex.from_product([dIndex,xf.columns]))
                    self.transformDict[dIndex]=dFun.transform
                    self.fitDict[dIndex]=dFun.fit
                    self.explained_variance_.extend(dFun.explained_variance_)
                    new_x=concat([new_x,new_xf],axis=1)
                    self.components_=concat([self.components_,components],axis=0,ignore_index=True) 
                if NoDecompositionLevelList: #if it's not empty
                    new_x=concat([new_x,x[NoDecompositionLevelList]],axis=1) #add to featuresDF the features that are not decompoosed
                    components_noDecomposition=DF(np.eye(len(NoDecompositionLevelList)),columns=MultiIndex.from_product([NoDecompositionLevelList,[0]]))
                    self.components_=concat([self.components_,components_noDecomposition],axis=0,ignore_index=True) #add components weights==1 to non decomposed features.
                self.components_=self.components_.fillna(0)
                new_x=DF()
                dFun=self.decompositionFunc
                dFun.n_components=n_components_perType
                #self.dFunDict=dict.fromkeys(decompositionLevelList)
                self.fitDict=dict.fromkeys(decompositionLevelList)
                self.transformDict=dict.fromkeys(decompositionLevelList)
                self.decompositionLevelList=decompositionLevelList
                self.NoDecompositionLevelList=NoDecompositionLevelList
                self.xfMultiIndex=dict.fromkeys(decompositionLevelList)
                self.explained_variance_=[]
                
            decompositionLevelList=[col for col in ColList.unique() if list(ColList).count(col)>1]
            NoDecompositionLevelList=[col for col in ColList.unique() if list(ColList).count(col)==1]
            except ValueError:
                 print(Xtest_allFeatures_decomposed)"""
            
            return Xtrain_nFeatures, Xtest_nFeatures, bestNfeatures, components, explainedVar
        
            
        @staticmethod
        def imputeFeatureMatrix(XtrainAllFeatures,Ytrain):
            #this function replace nans in train features with the mean of it's label feaures:
            X=XtrainAllFeatures
            Ximputed=DF(index=X.index,columns=X.columns)
            from globalVars import isMultiLabels
            if not(isMultiLabels):
                LabelValues=list(set(Ytrain.values))
                impt=Imputer(missing_values='NaN', strategy='mean', axis=0)
                for label in LabelValues:
                     X_label=X.loc[Ytrain==label]
                     Ximputed.loc[Ytrain==label]=impt.fit_transform(X_label)
            elif isMultiLabels: #if Y is multiLabeled
                Ximputed=learningUtils.imputeFeatureMatrixMultiLabel(X,Ximputed,Ytrain)
            return Ximputed
        
        @staticmethod    
        def imputeFeatureMatrixMultiLabel(XtrainAllFeatures,Ximputed,Ytrain):
            #this function imputes the feature matrix according regression on labels vector! (:
            #from the non-missing feature values Xi, and the corresponding label vector, we build a model that predicts the value of X in the missing value.
            X=XtrainAllFeatures
            Ximputed=X.copy()
            Missing_Subjects=X[X['ExpressionRatio']==0].index
            ImputeFeaturesIndex=MultiIndex.from_product([['ExpressionLength','ExpressionLevel','ChangeRatio','FastChangeRatio'],X['ExpressionLength'].columns])
            ImputingModel=linear_model.LinearRegression()
            for feature in ImputeFeaturesIndex:
                NonMissing_Data=X[feature].loc[X[feature].apply(np.isnan)==False]
                Missing_Data=X[feature].loc[X[feature].apply(np.isnan)]
                NonMissing_Subjects=NonMissing_Data.index
                Missing_Subjects=Missing_Data.index
                NonMissing_Labels=Ytrain.loc[NonMissing_Subjects]
                Missing_Labels=Ytrain.loc[Missing_Subjects]
                if len(Missing_Subjects)>0:
                    ImputingModel.fit(NonMissing_Labels,NonMissing_Data)
                    Ximputed[feature].loc[Missing_Subjects]=ImputingModel.predict(Missing_Labels)
            return Ximputed
        
        """def imputeLabelMatrix(PredictedLabelsMatrix,TrueLabelsMatrix):
            X=PredictedLabelsMatrix
            Ximputed=DF(index=X.index,columns=X.columns)
            LabelValues=list(set(TrueLabelsMatrix.values.flatten()))
            impt=Imputer(missing_values='NaN', strategy='mean', axis=0)
            for label in LabelValues:
                X_label=X.loc[Ytrain==label]
                Ximputed.loc[Ytrain==label]=impt.fit_transform(X_label)
            return Ximputed"""
        @staticmethod
        class MultipleRegression():
            def __init__(self):
                self.coef_=[]



            def fit(self,Xtrain,Ytrain):
                from numpy.linalg import inv as inverse
                from numpy.linalg import pinv as pseudo_inverse
                x=Xtrain.T
                y=Ytrain.T
                self.A=DF(DF.dot(y.dot(x.T),inverse(x.dot(x.T))),index=y.index) #A=Yx^T(XX^T)^-1
                self.A.columns=x.index
                #self.Apinv=DF(pseudo_inverse(self.A),index=self.A.columns,columns=self.A.index)
                self.coef_=self.A.values

            def predict(self,Xtest):
                Ypredicted=self.A.dot(Xtest.T).T
                return Ypredicted     

            def calcError(self,Ypredicted_AllLabels,Ytrue_AllLabels):
                Labels=Ypredicted_AllLabels.columns
                scoresSummary=DF(columns=Labels)
                trainScoresSummary=DF(['complete!']*3,index=['trainR^2','trainPval','trainError'])
                scores=[]
                scoreNames=[]
                Ypredicted=Ypredicted_AllLabels
                Ytrue=Ytrue_AllLabels
                testErr=((Ytrue-Ypredicted).abs()).mean()/LabelRange
                testErrStd=((Ytrue-Ypredicted).abs()).std()/LabelRange
                testR,testPval=sstats.pearsonr(Ytrue, Ypredicted)   #continue here- fix this
                scores.extend(trainScoresSummary.values.flatten())
                scores.extend([testR,testPval,testErr,testErrStd,LabelRange])
                
                scoreNames.extend(['trainR^2','trainPval','trainError','testR^2','testPval','testError','testErrorStd','LabelRange'])#,'regR','regP'] #decide how to do this + add descriptive of Ytest 
                scoresSummary[label]=DF(scores,index=scoreNames)
                scores


        @staticmethod
        def save_plotROC(rocDF,isSave=False,saveName=[],title='Reciever Operator Curve'):
            rocDF.to_csv(saveName+'DF.csv')
            fpr=rocDF['fpr']
            tpr=rocDF['tpr']
            roc_auc = auc(fpr, tpr)
            #plot ROC curve
            pl.clf()
            pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
            pl.plot([0, 1], [0, 1], 'k--')
            pl.xlim([0.0, 1.0])
            pl.ylim([0.0, 1.0])
            pl.xlabel('False Positive Rate')
            pl.ylabel('True Positive Rate')
            pl.legend(loc="lower right")
            pl.title(title)
            ROCfig=pl
            if isSave:
               pl.savefig(saveName)
            return ROCfig

             
        @staticmethod
        def getX2Y2(X,Y,YTrue,PiecePrediction, isBoolLabel):
            from globalVars import isMultiLabels
            if not(isMultiLabels):
                PiecePrediction=PiecePrediction.unstack()
                if isBoolLabel:
                    #X2=DF(columns=['prediction_mean'], index=PiecePrediction.index)
                    #X2['prediction_mean']=PiecePrediction.mean(axis=1)
                    #X2=DF(columns=['prediction_majority','prediction_mean_proba'], index=PiecePrediction.index)
                    #X2['prediction_majority']=(PiecePrediction.mean(axis=1)>0.5)*1.0
                    #X2['prediction_std']=PiecePrediction.std(axis=1)
                    X2=DF(columns=['m1','m2'], index=PiecePrediction.index)
                    X2['m1']=PiecePrediction.mean(axis=1)
                    X2['m2']=PiecePrediction.std(axis=1)
                    try:
                        Y2=Y.unstack()[0]
                    except KeyError:
                        Y2=Y.unstack()
                        if len(Y2.columns)>1:
                            Y2=Y2.mean(axis=1)

                    YTrue2=Y2
                else:
                    #X2=DF(columns=['m1','m2','skewness','kurtosis'], index=PiecePrediction.index)
                    X2=DF(columns=['m1','m2'], index=PiecePrediction.index)
                    X2['m1']=PiecePrediction.mean(axis=1)
                    X2['m2']=PiecePrediction.std(axis=1)
                    #X2['skewness']=PiecePrediction.skew(axis=1)
                    #X2['kurtosis']=PiecePrediction.kurtosis(axis=1)
                    try:
                        Y2=YTrue.unstack()['0']
                        YTrue2=YTrue.unstack()['0']
                    except KeyError:
                        Y2=Y.unstack()
                        if len(Y2.columns)>1:
                            Y2=Y2.mean(axis=1)
                    YTrue2=Y2

                X2=X2.dropna()
                Y2=Y2.loc[X2.index]
                YTrue2=YTrue2.loc[X2.index]
            elif isMultiLabels: #All Labels together
                labels=PiecePrediction.index
                X=PiecePrediction.copy()
                subjects=list(set(X.columns.get_level_values('subject')))
                X2=DF(columns=MultiIndex.from_product([labels,['m1','m2']]), index=subjects)
                Y2=DF(columns=labels,index=subjects)
                YTrue2=DF(columns=labels,index=subjects)
                for l in labels:
                    X2[(l,'m1')]=X.loc[l].unstack().mean(axis=1)
                    X2[(l,'m2')]=X.loc[l].unstack().std(axis=1)
                    Y2[l]=Y[l].unstack().mean(axis=1)
                    YTrue2[l]=YTrue[l].unstack().mean(axis=1)
            return X2, Y2, YTrue2

        
        @staticmethod
        def getTrainScores(Ytrain,Xtrain,YtrainTrue,TrainModel):    
            #init
            import globalVars as glv
            isBoolLabel=glv.isBoolLabel
            isBoolModel=glv.isBoolModel
            LabelRange=glv.LabelRange 
            isMultiLabels=glv.isMultiLabels  
            Xtrain=Xtrain.sort_index()
            Ytrain=Ytrain.sort_index().loc[Xtrain.index] 
            if not(isMultiLabels):
                if not(isBoolLabel) and not(isBoolModel):
                        YtrainPredicted=Series(TrainModel.decision_function(Xtrain).flatten(),index=Ytrain.index)
                        YtrainPredicted=YtrainPredicted.sort_index()
                        trainR,trainPval=sstats.pearsonr(YtrainPredicted,YtrainTrue)
                        trainError=(YtrainPredicted-Ytrain).abs().mean()/LabelRange
                        plt.figure(1)
                        plt.plot(Ytrain,YtrainPredicted,'.') 
                        plt.xlabel('Ytrue')
                        plt.ylabel('Ypredicted')
                        #plt.tick_params(labelsize=6)
                        trainScoresIndex=['trainR','trainPval','trainError']
                        trainScores=DF([trainR, trainPval, trainError],index=trainScoresIndex)
                        #print('--train--')
                        #print(trainScores)
                elif not(isBoolLabel) and isBoolModel:
                        Ymin=1
                        Ymax=Ymin+LabelRange
                        try: #train2
                            YtrainPredicted=Series(TrainModel.decision_function(Xtrain),index=Ytrain.index)
                        except Exception: #when xtrain is with multiindex (for train1)
                            YtrainPredicted=Series(TrainModel.decision_function(Xtrain).T[0],index=Ytrain.index)
                        YtrainPredicted=YtrainPredicted.sort_index()
                        #YtrainTrue=Series(YtrainTrue.values,index=YtrainTrue.index)
                        YtrainTrue=YtrainTrue.sort_index()
                        Ytrain=Ytrain.sort_index()
                        fitYscale=pandas.stats.ols.OLS(YtrainTrue,YtrainPredicted)
                    
                        YtrainPredictedTransformed=fitYscale.predict(x=YtrainPredicted)
                        plt.figure
                        plt.subplot(1,2,1)
                        plt.plot(YtrainTrue,YtrainPredicted)
                        plt.subplot(1,2,2)
                        plt.plot(YtrainTrue,YtrainPredictedTransformed)
                        glv.fitYscale=fitYscale
                        glv.beta=concat([glv.beta,fitYscale.beta],axis=1,ignore_index=True)
                        YtrainPredicted=YtrainPredictedTransformed 
                        #YtrainPredicted0=YtrainPredicted[Ytrain==0].mean()
                        #YtrainPredicted1=YtrainPredicted[Ytrain==1]
                        #slope, intercept, r_value, p_value, std_err=sstats.linregress(YtrainPredicted,YtrainTrue) #calculate transformation function between margins and actual values based on training data. 
                        """if (Ytrain==0).any(): #make sure it's the first run, if it's already defined, use it the one from previous run saved in glovalVars 
                            print('doubleCheck - pass(:')
                            x1=[YtrainPredicted[Ytrain==0].median(),YtrainTrue[Ytrain==0].median()]
                            x2=[YtrainPredicted[Ytrain==1].median(),YtrainTrue[Ytrain==1].median()]
                            try:
                                slope=(x1[1]-x2[1])/(x1[0]-x2[0])
                            except ZeroDivisionError:
                                slope=0
                            intercept=x2[1]-x2[0]*slope    
                            transformMargins=lambda x: x*slope + intercept
                            glv.transformMargins=transformMargins #use the one calculated on train for train2
                            glv.slope.extend(slope)
                            glv.intercept.extend(intercept)
                        elif(Ytrain>1).any():
                            transformMargins=glv.transformMargins
                        YtrainPredictedTransformed=transformMargins(YtrainPredicted)"""
                        trainR,trainPval=sstats.pearsonr(YtrainPredictedTransformed,YtrainTrue)#if Ypredicted is 0/1 will return nan
                        trainError=(YtrainPredictedTransformed-YtrainTrue).abs().mean()
                        plt.figure(1)
                        plt.plot(YtrainTrue,YtrainPredicted)
                        plt.xlabel('Ytrain - true',fontsize=8)
                        plt.ylabel('Ytrain - margin',fontsize=8)
                        plt.tick_params(labelsize=6)

                        trainScoresIndex=['trainR','trainPval','trainError']
                        trainScores=DF([trainR, trainPval, trainError],index=trainScoresIndex)
                    
                elif isBoolLabel:

                        YtrainPredicted=Series(TrainModel.predict(Xtrain),index=Ytrain.index)    
                        trainScores=DF(index=['trainR','trainPval','trainError'],columns=['0'])
                        trainScores.loc['trainError']=np.mean(YtrainPredicted!=Ytrain)
            elif isMultiLabels: #for multilabel learning
                    labels=Ytrain.columns
                    YtrainPredicted=TrainModel.predict(Xtrain)
                    trainScores=DF(columns=labels,index=['trainR','trainPval','trainError'])
                    for l in labels:
                        trainScores[l].loc['trainR'],trainScores[l].loc['trainPval']=sstats.pearsonr(YtrainPredicted[l],YtrainTrue[l])
                        trainScores[l].loc['trainError']=(YtrainPredicted[l]-Ytrain[l]).abs().mean()/LabelRange[l]
            return trainScores, YtrainPredicted
        
                    #TODO- add plots
        @staticmethod
        def getTestScores(Ytest,Xtest,TrainModel):
            import globalVars as glv
            isBoolLabel=glv.isBoolLabel
            isBoolModel=glv.isBoolModel
            LabelRange=glv.LabelRange 
            isMultiLabels=glv.isMultiLabels
            Ytest=Ytest.sort_index()
            Xtest=Xtest.sort_index()
            if len(Ytest)!=len(Xtest):
                lmin=min(len(Ytest),len(Xtest))
                Ytest=Ytest.iloc[:lmin]
                Xtest=Xtest.iloc[:lmin]
            n=len(Ytest)
            if isBoolLabel:
                Ypredicted=TrainModel.predict(Xtest)  
                tp=np.where((Ytest==1) & (Ypredicted==1)) 
                tp=float(len(tp[0]))
                allp=np.where(Ytest==1)
                allp=float(len(allp[0]))
                tn=np.where((Ytest==0) & (Ypredicted==0))
                tn=float(len(tn[0]))
                alln=np.where(Ytest==0)
                alln=float(len(alln[0]))
                fp=alln-tn
                fn=allp-tp
                testScores=DF([tp,fn,tn,fp,allp,alln],index=['tp','fn','tn','fp','allp','alln'])
                testProbas=DF(TrainModel.predict_proba(Xtest), index=Xtest.index)
                return testScores,testProbas

            if not(isBoolLabel): #(Ytest = continous/discrete variables, Ypredicted - continouse number returned by model)
                if not(isMultiLabels):
                    try:
                        Ypredicted=Series(TrainModel.decision_function(Xtest)[:,0],index=Xtest.index) 
                        Ypredicted=Ypredicted.sort_index()
                    except IndexError:
                        Ypredicted=Series(TrainModel.decision_function(Xtest),index=Xtest.index)
                        Ypredicted=Ypredicted.sort_index()
                    #if not(isBoolModel):
                    if isBoolModel: 
                        Ypredicted=glv.fitYscale.predict(x=Ypredicted) #use transformation function created in train to go from margins to Y true values (1-7 for PANSS)
                        Ypredicted.sort_index()
                        Ytest.sort_index()

                    residuals=(Ypredicted-Ytest).abs()
                    std_err=residuals.mean() /LabelRange
                    std_err_std=residuals.std() /LabelRange
                    try:
                        testResults= DF([Ytest[0],Ypredicted[0],std_err,std_err_std],index=['Ytest','Ypredicted','std_err','std_err_std'])
                        testScores=concat([Ypredicted,DF([std_err],index=['std_train_err'])]) # a matrix of Y predicted for each subject + std train error over all subjects. 
                    except IndexError:
                        print(Ytest)
                        testScores=DF()
                    testProbas=DF()
                elif isMultiLabels:
                    Ypredicted=DF(TrainModel.predict(Xtest),index=Xtest.index)
                    labels=Ypredicted.columns
                    testScores=DF(columns=labels,index=['Ytest','Ypredicted','std_err','std_err_std'])
                    residuals=(Ypredicted-Ytest).abs()
                    testScores.loc['std_err']=residuals.mean() /LabelRange
                    testScores.loc['std_err_std']=residuals.std() /LabelRange
                    testScores.loc['Ytest']=Ytest.iloc[0]#relevant only for train 2 where len(Ytest)=1
                    testScores.loc['Ypredicted']=Ypredicted.iloc[0]# relevant only for train 2
                    testProbas=DF()


                return testScores, testProbas
                #TrainModel. --> calc regression error on train data

        @staticmethod
        def getScoresSummary(trainScores,testScores,testProbas,TrueLabels): #for more scoring  - see http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter   
         # TODO - if the Grouptype is not boolean (e.g. extremPANSS) - calc margin Correlations
            #testScores=testScores.dropna(how='all')
            #trainScores=trainScores.dropna(how='all')            
            import globalVars as glv
            isBoolLabel=glv.isBoolLabel
            isBoolModel=glv.isBoolModel
            LabelRange=glv.LabelRange
            isMultiLabels=glv.isMultiLabels

            if isBoolLabel: #TODO - fix this uusing "testScores" and add results for nonBoolLabels.   
                ssum=testScores.T.sum()
                n=ssum['allp']+ssum['alln']
                if ssum['allp'] !=0: 
                    sensitivity=ssum['tp']/ssum['allp']
                else:
                    sensitivity=float('NaN')
                if ssum['alln'] !=0:
                    specificity=ssum['tn']/ssum['alln']
                else:
                    specificity=float('NaN')
                if ssum['tp']+ssum['fp'] !=0:
                    precision=ssum['tp']/(ssum['tp']+ssum['fp'])
                else: 
                    precision=float('NaN')
                accuracy=(ssum['tp']+ssum['tn'])/n
                recall=sensitivity
                if recall+precision==0:
                    f1=float('NaN')
                else:
                    f1=2*(recall*precision)/(recall+precision)
                ss_mean=(sensitivity+specificity)/2
                #calc AUC
                y_proba=testProbas.copy().sort().values.T[1].flatten()
                y_true=DF(TrueLabels.loc[testProbas.index]).copy().sort().values.flatten()
                fpr, tpr, thresholds = roc_curve(y_true, y_proba)
                roc_auc = auc(fpr, tpr)
                rocDF=DF([thresholds,fpr,tpr],index=['threshold','fpr','tpr']).T
                scores= [ssum['alln'],ssum['allp'],ssum['tn'],ssum['tp'],specificity,sensitivity,precision,f1,accuracy,ss_mean,roc_auc]
                scoreNames=['n0test','n1test','TrueNegatives','TruePositives','specificity','sensitivity','precision','f1','accuracy','ss_mean','roc_auc']
                scoresSummary=DF(scores,index=scoreNames)
                return scoresSummary,rocDF

            if not isBoolLabel:
               if not(isMultiLabels):
                    scoreNames=[]
                    trainScoresSummary=trainScores.mean(axis=1)
                    trainError=testScores.loc['std_train_err'].mean() #TODO- delete this?
                    Ypredicted=testScores[:-1].mean(axis=1)
                    Ypredicted=Ypredicted.sort_index()
                    if isBoolModel:
                        import globalVars as glv
                        beta2=glv.beta[range(1,len(glv.beta.T),2)]#from label range to label range (best fit)
                        beta1=glv.beta[range(0,len(glv.beta.T),2)] #from svm margins to label range values
                        beta= DF(glv.beta.mean(axis=1), index=['b1','b0'])
                        slope2=beta1.loc['x'].mean()
                        intercept2=beta1.loc['intercept'].mean()
                        fitYscale=lambda x: x*slope2 +intercept2
                        Ypredicted=fitYscale(Ypredicted)
                        print(beta1.mean(axis=1))
                    try: #train1
                        Ytrue=Series(index=MultiIndex.from_tuples(Ypredicted.index))
                        Ytrue=Ytrue.sort_index()
                        PredictedSubjects=Ytrue.index.levels[0]
                        for subject in PredictedSubjects:
                            Ytrue[subject]=TrueLabels.loc[subject]
                    except TypeError: #if there is no multiindex (train2)
                        Ytrue=TrueLabels.loc[Ypredicted.index]                

                    testErr=((Ytrue-Ypredicted).abs()).mean()/LabelRange
                    testErrStd=((Ytrue-Ypredicted).abs()).std()/LabelRange
                    Ypredicted=Ypredicted.dropna()
                    Ytrue=Ytrue[Ypredicted.index]
                    testR,testPval=sstats.pearsonr(Ytrue, Ypredicted)
                    #print(DF([Ypredicted,Ytrue]).T)
                    plt.figure(2)
                    plt.plot(Ytrue,Ypredicted,'.') 
                    plt.xlabel('Ytrue',fontsize=8)
                    plt.ylabel('Ypredicted',fontsize=8)
                    plt.tick_params(labelsize=6)

                    valueCounts=Ytrue.value_counts().sort_index() #count the time each number appeared in test data
                    scores=list(valueCounts.values) #count each 
                    scores.extend(trainScoresSummary.values)
                    scores.extend([testR,testPval,testErr,testErrStd,LabelRange])
                
                    for val in valueCounts.index:
                        scoreNames.extend([str(val)+'_counts'])
                    scoreNames.extend(['trainR^2','trainPval','trainError','testR^2','testPval','testError','testErrorStd','LabelRange'])#,'regR','regP'] #decide how to do this + add descriptive of Ytest 
                    scoresSummary=DF(scores,index=scoreNames)
               elif isMultiLabels:
                    YtrueAll=testScores.major_xs('Ytest') #over all subjects
                    YpredictedAll=testScores.major_xs('Ypredicted')
                    PANSS=YtrueAll.index
                    scoresSummary=DF(index=PANSS,columns=['testR^2','pval'])
                    for panss in PANSS:
                        Ytrue=YtrueAll.loc[panss]
                        Ypredicted=YpredictedAll.loc[panss]
                        scoresSummary['testR^2'].loc[panss],scoresSummary['pval'].loc[panss]=sstats.pearsonr(Ytrue, Ypredicted) #check why its minus!!
                    scoresSummary=scoresSummary.T
               rocDF=DF()
               print(scoresSummary)
           
            return scoresSummary, rocDF

        @staticmethod
        def setModel(Model,params='deafult',isBoolModel=None):
            
            #set model params and function:
            if Model=='svc':
                model=svm.SVC(kernel='linear', probability=True,class_weight={0:1,1:1})       
                isBoolModel=1

            if Model=='logist':
                model=linear_model.LogisticRegression()       
                isBoolModel=1

            if Model=='multiclass':
                model=svm.SVC(kernel='linear', probability=True)
                #try also:
                    #from sklearn.naive_bayes import GaussianNB
                    #model = GaussianNB()
                    #sklearn.linear_model.LogisticRegression(multi_class='multinomial')
                    #sklearn.lda.LDA
            if Model=='regression':
                model=linear_model.LinearRegression() #TODO - decide whether I should fit intercept or not. 
                isBoolModel=0
                
            if Model=='ridge':
                model=linear_model.Ridge(alpha=3.)
                isBoolModel=0                

            if Model == 'lasso':
                model=linear_model.Lasso(alpha=0.01)
                isBoolModel=0

            if Model == 'MultipleRegression':
                model=learningUtils.MultipleRegression()
            
            return model, isBoolModel

        @staticmethod
        def setFeatureSelection(FeatureSelectionMethod,n_features): 
            #set feature selection method
            if FeatureSelectionMethod=='AllFeatures': 
                n_features=len(X.columns)
                featureWeight=DF(np.ones(n_features),index=X.columns)   
                def selectFeatures(x,y,n_features,decompositionFunc):
                    FeatureList=x.columns
                    bestNfeatures=X.columns
                    new_x=x
                    return bestNfeatures, new_x
                                  
            elif FeatureSelectionMethod=='dPrime':
                def selectFeatures(x,y,n_features,decompositionFunc):
                    meanDiff=x[y==0].mean()-x[y==1].mean()
                    stdDiff=x[y==0].std() +x[y==1].std()
                    dPrime=meanDiff/(0.5*(stdDiff**0.5))
                    dPrime.index=tuple(dPrime.index)
                    dPrime.sort()
                    bestNfeatures=dPrime[:n_features].index
                    x_new=x(bestNfeature)
                    #featureWeight=DF(range(0,n_features),index=bestNfeatures)
                    return bestNfeatures, new_x

            elif FeatureSelectionMethod=='f_regression':
                featureSelectionFunc = feature_selection.f_regression
                def selectFeatures(x,y,n_features,decompositionFunc):
                    featureList=x.columns
                    f_scores=DF(featureSelectionFunc(x,y)[0],index=featureList)
                    bestNfeatures=f_scores.sort(columns=0,ascending=False)[:n_features].index
                    new_x=x[bestNfeatures]
                    return bestNfeatures, new_x

            elif FeatureSelectionMethod=='RFE':
                featureSelectionFunc = feature_selection.RFE(estimator=model, n_features_to_select=n_features, step=1)
                def selectFeatures(x,y,n_features,decompositionFunc):
                    featureSelectionFunc.fit(x,y)
                    featureList=x.columns
                    ranking=DF(featureSelectionFunc.ranking_,index=featureList) #TODO - CHECK THIS!
                    bestNfeatures=ranking.sort(columns=0,ascending=True)[:n_features].index
                    new_x=x[bestNfeatures]
                    return bestNfeatures,new_x
            
            
            elif FeatureSelectionMethod=='TopNComponents':
                 #returns top N principle components. if decomposition is done by level,  returns N/n_levels top components of each level.
                 def selectFeatures(x,y,n_features,decompositionFunc):
                    try: #if the decomposition is done by level
                        Groups= decompositionFunc.decompositionLevelList.tolist()
                        n_features_perGroup=range(int(round(n_features/len(Groups))))
                        bestNfeatures=MultiIndex.from_product([Groups,n_features_perGroup])
                    except AttributeError: #DecompositionByLevelClass instance has no attribute 'decompositionLevelList'
                        bestNfeatures=x.columns[range(min(n_features,len(x.columns)))] #if the decomposition is not done by level, simply select top components
                    new_x=x[bestNfeatures]
                    return bestNfeatures, new_x


            elif FeatureSelectionMethod=='TopExplainedVarianceComponents':
                def selectFeatures(x,y,n_features,decompositionFunc):
                    sortByExplainedVariance=np.argsort(-np.array(decompositionFunc.explained_variance_))
                    new_x=x[sortByExplainedVariance][range(n_features)]
                    bestNfeatures=new_x.columns
                    return bestNfeatures, new_x

            elif FeatureSelectionMethod=='FirstComponentsAndExplainedVar':
                #create a function that uses FirstComponent from each feature type + best for f-regression/explainedVariance...
                def selectFeatures(x,y,n_features,decompositionFunc):
                    try:
                        Groups= decompositionFunc.decompositionLevelList.tolist()
                        n_features_perGroup=0
                        FirstComponentInLevel=MultiIndex.from_product([Groups,n_features_perGroup]).tolist()
                    except AttributeError:
                        print('no decomposition / decomposition is not done by level!')
                    sortByExplainedVariance=np.argsort(-np.array(decompositionFunc.explained_variance_))
                    bestNfeatures_ByExplainedVar=x[sortByExplainedVariance].columns[range(n_features)]
                    bestNfeatures_ByExplainedVar=list(bestNfeatures_ByExplainedVar)
                    #make sure first components are in besNfeatures
                    bestNfeatures_NotFirstComponents=[f for f in bestNfeatures_ByExplainedVar if f[1]!=0]#not first components
                    bestNfeatures=FirstComponentInLevel
                    bestNfeatures.extend(bestNfeatures_NotFirstComponents[0:(n_features-len(Groups))])
                    new_x=x[bestNfeatures]
                    return bestNfeatures, new_x

            elif FeatureSelectionMethod=='FirstComponentsAndFregression':
                featureSelectionFunc = feature_selection.f_regression
            #create a function that uses FirstComponent from each feature type + best for f-regression/explainedVariance...
                def selectFeatures(x,y,n_features,decompositionFunc):
                    try:
                        Groups= decompositionFunc.decompositionLevelList.tolist()
                        n_features_perGroup=0
                        FirstComponentInLevel=MultiIndex.from_product([Groups,n_features_perGroup]).tolist()
                    except AttributeError:
                        print('no decomposition / decomposition is not done by level!')
                    featureList=x.columns
                    f_scores=DF(featureSelectionFunc(x,y)[0],index=featureList)
                    bestNfeatures_ByFregression=f_scores.sort(columns=0,ascending=False)[:n_features].index
                    bestNfeaturesByFregression=list(bestNfeatures_ByFregression)
                    #make sure first components are in besNfeatures
                    bestNfeatures_NotFirstComponents=[f for f in bestNfeatures_ByFregression if f[1]!=0]#not first components
                    bestNfeatures=FirstComponentInLevel
                    bestNfeatures.extend(bestNfeatures_NotFirstComponents[0:(n_features-len(Groups))])
                    new_x=x[bestNfeatures]
                    return bestNfeatures, new_x



            return selectFeatures #todo- check if I really use featuresSelectionFunc in the main code (maybe I shouldn't return it?)
            
        #set DecompositionMethod
        @staticmethod
        def setDecomposition(DecompositionMethod,n_components,DecompositionLevel='All'):
            decompositionTitle=DecompositionMethod

            if DecompositionMethod=='noDecomposition':
                DecompositionLevel=None
                class doNothingDecomposition:
                    def __init__(self,n_components):
                        self.n_components=n_components
                        self.components_=DF()
                        self.explained_variance_=[]

                    def fit_transform(self,x):
                        return x
                    def fit(self,x):
                        return x
                    def transform(self,x):
                        return x
             
                decompositionTitle='noDecomposition'
                decompositionFunc=doNothingDecomposition(n_components)
            elif DecompositionMethod in ['PCA' ,'FeatureType_PCA','fs-signal_PCA']:
                decompositionFunc=decomposition.PCA(n_components=n_components, copy=True, whiten=True)
            elif DecompositionMethod=='ICA':
                decompositionFunc=decomposition.FastICA(n_components=n_components, whiten=True)
            elif DecompositionMethod=='KernelPCA':
                decompositionFunc=decomposition.KernelPCA(n_components=n_components,  kernel='linear')
            elif DecompositionMethod=='SparsePCA':
                decompositionFunc=decomposition.SparsePCA(n_components=n_components)
               
            return  decompositionTitle, decompositionFunc      

class analysisUtils:
    @staticmethod
    def combinePermScores(permfileDir):
        dirlist=os.listdir(permfileDir)
        permutationFilesList=[f for f in dirlist if 'permutation' in f]
        CombinedPerms=DF()
        numOfFiles=len(permutationFilesList)
        fileNum=1
        for pFile in permutationFilesList:
            pFilePath=os.path.join(permfileDir,pFile)
            print('appending '+str(fileNum)+'/'+str(numOfFiles)+' files...')
            newFile=DF.from_csv(pFilePath).dropna(axis=1, how='all')
            newCols=newFile.columns
            if fileNum==1:
                CombinedPerms=newFile
            else: 
                CombinedPerms[newCols]=newFile #TODO- make sure it doesn't overwrite old columns , handle inf and nans          
            fileNum+=1

        isSave=int(raw_input('save combined files? '))
        if isSave:
            filename=raw_input('enter combined file name (.csv): ')
            CombinedPerms.to_csv(permfileDir+'\\'+filename) #not working - test this!
            print('combinded permutation file was saved!')
        isDel=int(raw_input('delete source files? '))
        if isDel:
            for pFile in permutationFilesList:
                pPath=os.path.join(permfileDir,pFile)
                os.remove(pPath)
            print('all files succesfully removed!')

        
    @staticmethod
    def getPermScores(permfilePath, permfileRange,resultfilePath,numOfFeatures,resultfileRange,savefilePath):
        Perms=DF.from_csv(permfilePath)
        Results=DF.from_csv(resultfilePath)
        Results=Results[str(numOfFeatures)]
        nPerms=len(Perms.columns)
        ScoresIndex=['R^2','pval','Error']
        ResultsIndex=list(set(Results.dropna().index))
        ResultsIndex.sort()
        PermProbability=DF(index=ResultsIndex,columns=ScoresIndex)
        SumResults=DF(index=ResultsIndex,columns=ScoresIndex)
        for ind in Results.index:
            iResults=DF(Results.loc[ind].values, index=ScoresIndex)
            iPerms=DF(Perms.loc[ind].values, ScoresIndex)
            SumResults.loc[ind]=iResults.T.values
            for score in ScoresIndex:
                Prob=sum(iPerms.loc[score].values<iResults.loc[score].values)/nPerms #what is the chance of getting a smaller value in perms?
                PermProbability.loc[ind][score]=Prob  

        #PermutationResults=concat(DF(index='------Results------'),SumResults, DF(index='------Permutation test results (%(x<Xperms)------')
        #PermutationResults.to_csv(savefilePath)   
        
    @staticmethod
    def getFeaturesWeights(FeatureAnalysisIndex,bestNfeaturesForLabel,ModelWeightsForLabel):
        global ii
        ii=FeatureAnalysisIndex
        try:
            LabelFeatures=bestNfeaturesForLabel.applymap(lambda a: a[ii])    
        except IndexError: #if there is no cross validation
            LabelFeatures=DF(bestNfeaturesForLabel[0]).applymap(lambda a: a[ii])      
        FeaturesList=list(set(LabelFeatures.values.flatten()))
        LabelingFeatureWeights=Series(index=FeaturesList)

        for f in FeaturesList:
            try:
                isFeature=LabelFeatures==f
            except TypeError: #:Could not compare ['all'] with block values - run an exhaustive search
                isFeature=DF(index=LabelFeatures.index,columns=LabelFeatures.columns)
                for col in isFeature.columns:
                    for ind in isFeature.index:
                        if LabelFeatures[col][ind]==f:
                            isFeature[col][ind]=True
                        else:
                            isFeature[col][ind]=False
            LabelingFeatureWeights.loc[f]=DF(isFeature*ModelWeightsForLabel).sum().mean()

 
        return LabelingFeatureWeights     
        



