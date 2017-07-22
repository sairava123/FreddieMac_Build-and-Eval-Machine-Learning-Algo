
# coding: utf-8

# In[1]:

import mechanicalsoup as ms
import requests
from zipfile import ZipFile
import os
from io import BytesIO
from os.path import basename
from requests import get  # to make GET request
from pathlib import Path
import pandas as pd
from sklearn import linear_model
import numpy as np
from sklearn.metrics import r2_score
from sklearn.feature_selection import f_regression
from sklearn.metrics import *
import math
lr=linear_model.LinearRegression()
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFE
from sklearn.neural_network import MLPRegressor
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets.supervised import SupervisedDataSet 
from pybrain.supervised.trainers import BackpropTrainer

def downloadhistoricaldata(trainQ, testQ, t,s, flag):
    for l in t:
        if(trainQ in l['href'] or testQ in l['href']):
            c = 'https://freddiemac.embs.com/FLoan/Data/' + l['href']
            r = s.get(c)
            z = ZipFile(BytesIO(r.content)) 
            z.extractall(os.getcwd())
            flag = 1
    return flag

def login(login, password, trainQ, testQ):
    flag = 0
    s = requests.Session()
    url = "https://freddiemac.embs.com/FLoan/secure/auth.php"
    url2 = "https://freddiemac.embs.com/FLoan/Data/download.php"
    browser = ms.Browser(session = s)
    print("Logging in....")
    login_page = browser.get(url)
    login_form = login_page.soup.find("form",{"class":"form"})
    login_form.find("input", {"name":"username"})["value"] = login
    login_form.find("input", {"name":"password"})["value"] = password
    response = browser.submit(login_form, login_page.url)
    login_page2 = browser.get(url2)
    print("To the continue page...")

    next_form = login_page2.soup.find("form",{"class":"fmform"})
    a= next_form.find("input",{"name": "accept"}).attrs
    a['checked']=True

    response2 = browser.submit(next_form, login_page2.url)
    print("Start Downloading from..."+ response2.url)
    table = response2.soup.find("table",{"class":"table1"})

    t = table.find_all('a')
    flag = downloadhistoricaldata(trainQ, testQ, t,s, flag) 

    if flag == 1:
        print("Data downloaded successfully!!")
    else:
        print("Error in downloading data")

    
def generatecsv(trainQ, testQ):
    trainfile = "historical_data1_"+ trainQ + ".txt"
    testfile = "historical_data1_"+ testQ + ".txt"
    f1 = "train_" + trainQ + ".csv"
    f2 = "test_" + testQ + ".csv"
    with open(f1, 'w',encoding='utf-8') as file: 
        df = pd.read_csv(trainfile ,delimiter ="|", names=['credit_score','first_payment_date','fthb_flag','matr_date','msa',"mortage_insurance_pct",'no_of_units','occupancy_status','cltv','dti_ratio','original_upb','original_ltv','original_int_rt','channel','ppm_flag','product_type','property_state', 'prop_type','zipcode','loan_seq_number','loan_purpose', 'original_loan_term','number_of_borrowers','sellers_name','servicer_name','super_conforming_flag'],skipinitialspace=True)   
        df.to_csv(file, header=True,index=False, mode='a')
        print("%s csv generated!"%file )
        
    with open(f2, 'w',encoding='utf-8') as file: 
        df = pd.read_csv(testfile ,delimiter ="|", names=['credit_score','first_payment_date','fthb_flag','matr_date','msa',"mortage_insurance_pct",'no_of_units','occupancy_status','cltv','dti_ratio','original_upb','original_ltv','original_int_rt','channel','ppm_flag','product_type','property_state', 'prop_type','zipcode','loan_seq_number','loan_purpose', 'original_loan_term','number_of_borrowers','sellers_name','servicer_name','super_conforming_flag'],skipinitialspace=True)   
        df.to_csv(file, header=True,index=False, mode='a')
        print("%s csv generated!"%file )
        
def cleanData(data):
    data = data.drop(data['credit_score'].loc[(data['credit_score'] < 301) | (data['credit_score'] > 850)].index)
    data = data.dropna(subset=['credit_score'])
    data['fthb_flag'] = data['fthb_flag'].fillna("NA") 
    data = data.dropna(subset=['msa'])
    data['mortage_insurance_pct'] = data['mortage_insurance_pct'].fillna(0)
    data['no_of_units'] = data['no_of_units'].fillna(0)
    data['cltv'] = data['cltv'].fillna(0)
    data['dti_ratio'] = data['dti_ratio'].fillna(0)
    data['original_ltv'] = data['original_ltv'].fillna(0)
    data['ppm_flag'] = data['ppm_flag'].fillna("U")
    data['prop_type']=data['prop_type'].fillna('NA') 
    data['loan_purpose']=data['loan_purpose'].fillna('NA')
    data = data.dropna(subset=['zipcode'])
    data['number_of_borrowers'] = data['number_of_borrowers'].fillna(1)
    data['super_conforming_flag'] = data['super_conforming_flag'].fillna("N")
    return data

def convertNumer(data):
    data['fthb_flag'] = data['fthb_flag'].replace(['Y','N','NA'],[1,2,3])
    data['occupancy_status'] = data['occupancy_status'].replace(['I','O','S'],[1,2,3])
    data['channel'] = data['channel'].replace(['B','C','R','T'],[1,2,3,4])
    data['ppm_flag'] = data['ppm_flag'].replace(['Y','N','U'],[1,2,3])
    data['prop_type'] = data['prop_type'].replace(['CO','LH','PU','MH','SF','CP','NA'],[1,2,3,4,5,6,7])
    data['loan_purpose'] = data['loan_purpose'].replace(['P','C','N','NA'],[1,2,3,4])
    data['super_conforming_flag'] = data['super_conforming_flag'].replace(['Y','N'],[0,1])
    return data

def changedatatype(data):
    data[['credit_score','msa','no_of_units','mortage_insurance_pct','cltv','dti_ratio','original_ltv','zipcode','number_of_borrowers']]=data[['credit_score','msa','no_of_units','cltv','mortage_insurance_pct','dti_ratio','original_ltv','zipcode','number_of_borrowers']].astype('int64')
    data[['fthb_flag','occupancy_status','channel']] = data[['fthb_flag','occupancy_status','channel']].astype('int64')
    data[['ppm_flag','prop_type','loan_purpose','super_conforming_flag']]= data[['ppm_flag','prop_type','loan_purpose','super_conforming_flag']].astype('int64')
    data[['product_type','property_state']] = data[['product_type','property_state']].astype('str')
    data[['loan_seq_number','sellers_name','servicer_name']] = data[['loan_seq_number','sellers_name','servicer_name']].astype('str')
    return data

def perform_linear_regression(lr, xaxis, yaxis):
    print ("Start of linear regression")
    #Fit the linear model
    lr.fit(xaxis,yaxis)
    print ("Intercept is ",lr.intercept_)
    #Calculate variance score
    var_score =lr.score(xaxis,yaxis) 
    #Print coefficients for x-axis only once
    print("Coefficient is ",len(lr.coef_))
    #print(pd.DataFrame(list(zip(xaxis.columns,lr.coef_ )), columns=['Features','Estimated Coefficients']))
    #To calculate difference between estimated and actual y-axis values
    diff =r2_score(yaxis,lr.predict(xaxis))
    print("Linear Regression Score is: ",diff)
    print ("End of linear regression")
    return lr

def perform_selectKBest(lr, xaxis, yaxis):
    feat=SelectKBest(f_regression,k=10) 
    feat = feat.fit(xaxis, yaxis)
    #To find out score, we need to reduce xaxis to selected features
    X = feat.fit_transform(xaxis,yaxis)
    fit = lr.fit(X,yaxis)
    var_score = lr.score(X, yaxis)
    diff=r2_score(yaxis,lr.predict(X))
    #idxs_selected = feat.get_support(indices=True)
    features=pd.DataFrame(list(zip(xaxis,sorted(feat.scores_, reverse = True))),columns=["features","scores"])
    print("K best r2 Score is: ",diff)
    print(features)
    
def perform_lassoLinear(lr,xaxis, yaxis):
    lasso = linear_model.Lasso(alpha=0.1) 
    lasso.fit(xaxis,yaxis) 
    predict=lasso.predict(xaxis)
    score=r2_score(yaxis,predict)
    features=pd.DataFrame(list(zip(xaxis,sorted(lasso.coef_, reverse = True))),columns=["features","coefficient"])
    print("Lasso Regression r2 score:", score)
    print(features)

    
def perform_recursiveFE(lr,xaxis,yaxis):
    selector = RFE(lr,10)
    feat = selector.fit(xaxis, yaxis)
    prediction=feat.predict(xaxis)
    score=r2_score(yaxis,prediction)
    print("RFE r2 score: ",score)
    rankfeatures=pd.DataFrame(list(zip(xaxis.columns,sorted(feat.ranking_))),columns=["features","ranking"])
    print(rankfeatures)

   
    
def calculatestat(lr,xaxis,yaxis):
    y_pred=lr.predict(xaxis)
    MAE=mean_absolute_error(yaxis,y_pred)
    print("Mean Absolute Error: ", MAE) 
    RMSE=math.sqrt(mean_squared_error(yaxis,y_pred))
    print("Root of Mean Squared Deviation: ",RMSE)
    MAPE=np.mean(np.abs((yaxis - y_pred) / yaxis)) * 100
    print("Mean Absolute Percentage Error: ",MAPE)
    
def keepReqColumns(df):
    df=df.drop('zipcode',axis=1)
    df=df.drop('number_of_borrowers',axis=1)
    df=df.drop('original_loan_term',axis=1)
    df=df.drop('original_ltv',axis=1)
    df=df.drop('channel',axis=1)
    df=df.drop('ppm_flag',axis=1)
    df=df.drop('prop_type',axis=1)
    df=df.drop('loan_purpose',axis=1)
    df=df.drop('super_conforming_flag',axis=1)      
    return df

def getTrainTest(trainQ, testQ):
    f1 = "train_" + trainQ + ".csv"
    f2 = "test_" + testQ + ".csv"
    trf = pd.read_csv(f1)
    tsf = pd.read_csv(f2)
    return (trf, tsf)
    
def LinearRegressionAnalysis(xtrain,ytrain,xtest,ytest):
    trainreg  = lr.fit(xtrain, ytrain)
    print("---------------Linear Regression---------------")
    print("Train Data:")
    print(calculatestat(trainreg, xtrain, ytrain ))
    print("Test Data:")
    print(calculatestat(trainreg, xtest, ytest))
          
def Random_Forest(xtrain,ytrain,xtest,ytest):
    rand_forest = RandomForestRegressor(n_estimators=15,max_depth=10)
    rand_forest = rand_forest.fit(xtrain,ytrain)
    print("---------------Random Forest---------------")
    print("Train Data:")
    print(calculatestat(rand_forest, xtrain, ytrain ))
    print("Test Data:")
    print(calculatestat(rand_forest,xtest,ytest))

def Neural_Network(xtrain,ytrain,xtest,ytest):
    #Hidden nodes
    hidden_net = 2
    #Epoch is a single pass through the entire training set, followed by testing of the verification set.
    epoch = 2
    ytrain = ytrain.reshape(-1,1)
    input_cnt = xtrain.shape[1]
    target_cnt = ytrain.shape[1]
    dataset = SupervisedDataSet(input_cnt, target_cnt)
    dataset.setField( 'input', xtrain )
    dataset.setField( 'target', ytrain )
    network = buildNetwork( input_cnt, hidden_net, target_cnt, bias = True )
    #Trainer that trains the parameters of a module according to a supervised dataset (potentially sequential) by backpropagating the errors (through time).
    trainer = BackpropTrainer( network,dataset )
    print("---------------Neural Network---------------")
    print("Train Data")
    for e in range(epoch):
        mse = trainer.train()
        rmse = math.sqrt(mse)   
    print("MSE, epoch {}: {}".format(e + 1, mse))
    print("RMSE, epoch {}: {}".format(e + 1, rmse))
    
    ytest=ytest.reshape(-1,1)
    input_size = xtest.shape[1]
    target_size = ytest.shape[1]
    dataset = SupervisedDataSet( input_size, target_size )
    dataset.setField( 'input', xtest)
    dataset.setField( 'target', ytest)
    model = network.activateOnDataset(dataset)

    mse = mean_squared_error(ytest, model )
    rmse =math.sqrt(mse)
    print("Test Data:")
    print("MSE: ", mse)
    print("RMSE: ", rmse)
   
    
    
def select_features(trainQ,testQ):
    print("Starting feature selection")
    df = getTrainTest(trainQ, testQ)
    df = cleanData(df[0])
    df = convertNumer(df)
    trainframe = changedatatype(df)
    yaxis=trainframe.original_int_rt
    xaxis= trainframe.drop('original_int_rt',axis=1)._get_numeric_data()
    perform_linear_regression(lr, xaxis, yaxis)
    perform_selectKBest(lr, xaxis, yaxis)
    perform_recursiveFE(lr,xaxis,yaxis)
    perform_lassoLinear(lr,xaxis, yaxis)
    

    
def perform_predication(trainQ, testQ):
    data = getTrainTest(trainQ, testQ)
    df = cleanData(data[0])
    df = convertNumer(df)
    trainframe = changedatatype(df)
    trainf=keepReqColumns(trainframe)
    df1 = cleanData(data[1])
    df1 = convertNumer(df1)
    testframe = changedatatype(df1)
    testf=keepReqColumns(testframe)
    ytrain = trainf.original_int_rt
    xtrain = trainf.drop('original_int_rt',axis=1)._get_numeric_data()
    ytest = testf.original_int_rt
    xtest = testf.drop('original_int_rt',axis=1)._get_numeric_data()
    LinearRegressionAnalysis(xtrain,ytrain,xtest,ytest)
    Random_Forest(xtrain,ytrain,xtest,ytest)
    Neural_Network(xtrain,ytrain,xtest,ytest)

def financial_crisis_economic(trainQ, testQ):
    from sklearn.ensemble import RandomForestRegressor 
    data1 = getTrainTest(trainQ, testQ)
    df2 = cleanData(data1[0])
    df2 = convertNumer(df2)
    trainframe = changedatatype(df2)
    trainf=keepReqColumns(trainframe)
    df3 = cleanData(data1[1])
    df3 = convertNumer(df3)
    testframe = changedatatype(df2)
    testf=keepReqColumns(testframe)      
    ytrain = trainf.original_int_rt
    xtrain = trainf.drop('original_int_rt',axis=1)._get_numeric_data()
    #n-jobs:No ristriction on use of processors
    #The out-of-bag (OOB) error is the average error for each z_i calculated using predictions from the trees that do not contain z_i in their respective bootstrap sample.
    rand_forest = RandomForestRegressor(n_jobs=-1,  oob_score = True,max_depth=10)
    rand_forest.fit(xtrain,ytrain)
    print("---------------Financial Crisis---------------")
    print("Train Data: ",trainQ)
    print(calculatestat(rand_forest, xtrain, ytrain ))
    ytest=testf.original_int_rt
    xtest = testf.drop('original_int_rt',axis=1)._get_numeric_data()
    print("Test Data: ",testQ)
    print(calculatestat(rand_forest,xtest,ytest))
    
def main():
    username =  joshi.sn@husky.neu.edu
    passw = gpvc`w]~
    trainQ= 'Q12007'
    testQ= 'Q22007'
    print("Username:", username)
    print("Password:",passw)
    print("Train Quarter:", trainQ)
    print("Test Quarter", testQ)
    login(username,passw,trainQ,testQ)
    generatecsv(trainQ, testQ)
    select_features(trainQ,testQ)
    perform_predication(trainQ, testQ)
    financial_crisis_economic(trainQ, testQ) 

if __name__ == '__main__':
    main()
