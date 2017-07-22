
# coding: utf-8

# In[10]:

import mechanicalsoup as ms
import requests
from zipfile import ZipFile
import urllib
import os
from io import BytesIO
from urllib import request
from requests import get  # to make GET request
import glob
import pandas as pd
import sys
from pathlib import Path


def login(login,passw):
    print("Pass:"+str(passw))
    url = "https://freddiemac.embs.com/FLoan/secure/auth.php"
    url2 = "https://freddiemac.embs.com/FLoan/Data/download.php"
    s = requests.Session()
    browser = ms.Browser(session = s)
    print("Logging in....")
    login_page = browser.get(url)
    login_form = login_page.soup.find("form",{"class":"form"})
    login_form.find("input", {"name":"username"})["value"] = login
    login_form.find("input", {"name":"password"})["value"] = passw
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
    for x in range(76,88):
        c = 'https://freddiemac.embs.com/FLoan/Data/' + t[x]['href']
        r = s.get(c)
        z = ZipFile(BytesIO(r.content)) 
        z.extractall(os.getcwd())  
    print("Downloaded all sample successfully!")
    
#function to change data type of columns
def changedatatype(dataframe):
    dataframe['repurchase_flag'] = dataframe['repurchase_flag'].astype('str')
    dataframe['modification_flag'] = dataframe['modification_flag'].astype('str')
    dataframe['zero_bal_date'] = dataframe['zero_bal_date'].astype('str')
    dataframe['ddlpi'] = dataframe['ddlpi'].astype('str')
    dataframe['net_sale_proceeds'] = dataframe['net_sale_proceeds'].astype('str')
    dataframe['delq_status'] = dataframe['delq_status'].astype('int64')
    dataframe['loan_age'] = dataframe['loan_age'].astype('int64')
    dataframe['rem_months'] = dataframe['rem_months'].astype('int64')
    dataframe['zero_balance_code'] = dataframe['zero_balance_code'].astype('int64')
    dataframe['current_def_upb'] = dataframe['current_def_upb'].astype('int64')
    dataframe['actual_loss_calc'] = dataframe['actual_loss_calc'].astype('int64')
    return dataframe

#function to fill nan values
def fillnulls(dataframe):
    dataframe['delq_status']=dataframe['delq_status'].fillna(0)
    dataframe['loan_age']=dataframe['loan_age'].fillna(0)
    dataframe['rem_months']=dataframe['rem_months'].fillna(0)
    dataframe['repurchase_flag']=dataframe['repurchase_flag'].fillna('NA')
    dataframe['modification_flag']=dataframe['modification_flag'].fillna('Not Modified')
    dataframe['zero_balance_code']=dataframe['zero_balance_code'].fillna(00)
    dataframe['zero_bal_date']=dataframe['zero_bal_date'].fillna('NA')
    dataframe['current_def_upb']=dataframe['current_def_upb'].fillna(0)
    dataframe['ddlpi']=dataframe['ddlpi'].fillna('NA')
    dataframe['mi_recoveries']=dataframe['mi_recoveries'].fillna(0)
    dataframe['net_sale_proceeds']=dataframe['net_sale_proceeds'].fillna('U')
    dataframe['non_mi_recoveries']=dataframe['non_mi_recoveries'].fillna(0)
    dataframe['expenses']=dataframe['expenses'].fillna(0)
    dataframe['legal_costs']=dataframe['legal_costs'].fillna(0)
    dataframe['maint_pres_costs']=dataframe['maint_pres_costs'].fillna(0)
    dataframe['taxes_ins']=dataframe['taxes_ins'].fillna(0)
    dataframe['misc_expenses']=dataframe['misc_expenses'].fillna(0)
    dataframe['actual_loss_calc']=dataframe['actual_loss_calc'].fillna(0)
    dataframe['modification_cost']=dataframe['modification_cost'].fillna(0)
    return dataframe
                                                                                                                                                                                                                                                   
#function to summarize performance data
def get_month(group):
    return {'month': group.max()}
def get_current_actual_upb(group):
    return {'max_current_actual_upb': group.max(), 'min_current_actual_upb': group.min()}
def get_delq_status(group):                                                                                                                
    return {'delq_status': group.max()} 
def get_loan_age(group):
    return {'loan_age': group.max()}
def get_rem_months(group):
    return {'rem_months': group.min()}
def get_repurchase_flag(group):
    return {'repurchase_flag': group.max()}
def get_modification_flag(group):
    return {'modification_flag': group.max()}
def get_zero_bal_code(group):
    return {'zero_balance_code': group.max()}
def get_zero_bal_date(group):
    return {'zero_bal_date': group.max()}
def get_current_int_rate(group):
    return {'current_int_rate': group.max()}
def get_current_def_upb(group):
    return {'current_def_upb': group.max()}
def get_ddlpi(group):
    return {'ddlpi': group.max()}
def get_mi_recoveries(group):
    return {'mi_recoveries': group.max()}
def get_net_sale_proceeds(group):
    return {'net_sale_proceeds': group.max()}
def get_non_mi_recoveries(group):                                                                                                                                                                                                           
    return {'non_mi_recoveries': group.min()} 
def get_expenses(group):
    return {'expenses': group.min()}
def get_legal_costs(group):
    return {'legal_costs': group.min()}
def get_maint_pres_costs(group):
    return {'maint_pres_costs': group.max()}  
def get_taxes_ins(group):
    return {'taxes_ins': group.min()} 
def get_misc_expenses(group):
    return {'misc_expenses': group.max()} 
def get_actual_loss_calc(group):
    return {'actual_loss_calc': group.max()} 
def get_modification_cost(group):
    return {'modification_cost': group.max()} 
 
def cleanorigin(data):
    data = data.drop(data['credit_score'].loc[(data['credit_score'] < 301) | (data['credit_score'] > 850)].index)
    data = data.dropna(subset=['credit_score'])
    data['fthb_flag'] = data['fthb_flag'].fillna("NA")
    data = data.dropna(subset=['msa'])
    data['mortage_insurance_pct'] = data['mortage_insurance_pct'].fillna("NA")
    data['cltv'] = data['cltv'].fillna(0)
    data['dti_ratio'] = data['dti_ratio'].fillna(0)
    data['original_ltv'] = data['original_ltv'].fillna(0)
    data['ppm_flag'] = data['ppm_flag'].fillna("U")
    data = data.dropna(subset=['zipcode'])
    data['number_of_borrowers'] = data['number_of_borrowers'].fillna(1)
    data['super_conforming_flag'] = data['super_conforming_flag'].fillna("N")
    return data
    
def clean_merge_generate_origin_csv():
    print("In a directory: " + os.getcwd())
    OrigFiles=str(os.getcwd())+"/sample_orig_*.txt"
    heading = 0
    filename= "sample_orig_combined.csv"
    path= Path(filename)
    files = glob.glob(OrigFiles)
    if len(files) != 0:
        print("Total %d sample original files" %len(files) )
        if path.is_file():
            print("'sample_orig_combined.csv' already exits!")
        else:
            with open(filename, 'w',encoding='utf-8') as file:
                for f in files: 
                    df = pd.read_csv(f ,delimiter ="|", names=['credit_score','first_payment_date','fthb_flag','matr_date','msa',"mortage_insurance_pct",'no_of_units','occupancy_status','cltv','dti_ratio','original_upb','original_ltv','original_int_rt','channel','ppm_flag','product_type','property_state', 'prop_type','zipcode','loan_seq_number','loan_purpose', 'original_loan_term','number_of_borrowers','sellers_name','servicer_name','super_conforming_flag'],skipinitialspace=True,low_memory=False) 
                    if heading == 0:
                        df = cleanorigin(df)
                        df.to_csv(file, header=True,index=False, mode='a')
                        heading = 1
                    else:
                        df = cleanorigin(df)
                        df.to_csv(file, header=False, index=False, mode='a')
                print("'sample_orig_combined.csv' generated!" )
    else:
        print("Origination file list is empty!!")
        
def clean_merge_generate_perform_csv():
    PerfFiles=str(os.getcwd())+"/sample_svcg_*.txt"
    fileName = "Summarized_performance_data.csv"
    filepath = Path(fileName)
    performance_files = glob.glob(PerfFiles)   
    flag = 0                                                                                                                                                                                                                                      
    if(performance_files) != 0:
        print("Total %d sample performance files" %len(performance_files))
        if filepath.is_file():
            print("'Summarized_performance_data.csv' already exits!")
        else:
            with open(fileName, 'w',encoding='utf-8', newline="") as file:
                for f in performance_files:                                                                                                                    
                    print("Processing " + f)
                    performance_df = pd.read_csv(f ,sep="|", names=['loan_seq_number','month','current_actual_upb','delq_status','loan_age','rem_months', 'repurchase_flag','modification_flag','zero_balance_code', 'zero_bal_date','current_int_rate','current_def_upb','ddlpi','mi_recoveries', 'net_sale_proceeds','non_mi_recoveries','expenses', 'legal_costs', 'maint_pres_costs','taxes_ins','misc_expenses','actual_loss_calc', 'modification_cost'],skipinitialspace=True) 
                    performance_df['delq_status'] = [ 999 if x=='R' else x for x in (performance_df['delq_status'].apply(lambda x: x))]
                    performance_df['delq_status'] = [ 0 if x=='XX' else x for x in (performance_df['delq_status'].apply(lambda x: x))]
                    performance_df = fillnulls(performance_df)
                    performance_df = changedatatype(performance_df)
                    summary_df = pd.DataFrame()
                    summary_df['loan_seq_number']= performance_df['loan_seq_number'].drop_duplicates()
                    summary_df=summary_df.join((performance_df['month'].groupby(performance_df['loan_seq_number']).apply(get_month).unstack()), on='loan_seq_number')
                    summary_df=summary_df.join((performance_df['current_actual_upb'].groupby(performance_df['loan_seq_number']).apply(get_current_actual_upb).unstack()),on='loan_seq_number')
                    summary_df=summary_df.join((performance_df['delq_status'].groupby(performance_df['loan_seq_number']).apply(get_delq_status).unstack()),on='loan_seq_number')
                    summary_df=summary_df.join((performance_df['loan_age'].groupby(performance_df['loan_seq_number']).apply(get_loan_age).unstack()),on='loan_seq_number')
                    summary_df=summary_df.join((performance_df['rem_months'].groupby(performance_df['loan_seq_number']).apply(get_rem_months).unstack()),on='loan_seq_number')
                    summary_df=summary_df.join((performance_df['repurchase_flag'].groupby(performance_df['loan_seq_number']).apply(get_repurchase_flag).unstack()),on='loan_seq_number')
                    summary_df=summary_df.join((performance_df['modification_flag'].groupby(performance_df['loan_seq_number']).apply(get_modification_flag).unstack()),on='loan_seq_number')
                    summary_df=summary_df.join((performance_df['zero_balance_code'].groupby(performance_df['loan_seq_number']).apply(get_zero_bal_code).unstack()),on='loan_seq_number')
                    summary_df=summary_df.join((performance_df['zero_bal_date'].groupby(performance_df['loan_seq_number']).apply(get_zero_bal_date).unstack()),on='loan_seq_number')
                    summary_df=summary_df.join((performance_df['current_int_rate'].groupby(performance_df['loan_seq_number']).apply(get_current_int_rate).unstack()),on='loan_seq_number')
                    summary_df=summary_df.join((performance_df['current_def_upb'].groupby(performance_df['loan_seq_number']).apply(get_current_def_upb).unstack()),on='loan_seq_number')
                    summary_df=summary_df.join((performance_df['ddlpi'].groupby(performance_df['loan_seq_number']).apply(get_ddlpi).unstack()),on='loan_seq_number')
                    summary_df=summary_df.join((performance_df['mi_recoveries'].groupby(performance_df['loan_seq_number']).apply(get_mi_recoveries).unstack()),on='loan_seq_number')
                    summary_df=summary_df.join((performance_df['net_sale_proceeds'].groupby(performance_df['loan_seq_number']).apply(get_net_sale_proceeds).unstack()),on='loan_seq_number')
                    summary_df=summary_df.join((performance_df['non_mi_recoveries'].groupby(performance_df['loan_seq_number']).apply(get_non_mi_recoveries).unstack()),on='loan_seq_number')
                    summary_df=summary_df.join((performance_df['expenses'].groupby(performance_df['loan_seq_number']).apply(get_expenses).unstack()),on='loan_seq_number')
                    summary_df=summary_df.join((performance_df['legal_costs'].groupby(performance_df['loan_seq_number']).apply(get_legal_costs).unstack()),on='loan_seq_number')
                    summary_df=summary_df.join((performance_df['maint_pres_costs'].groupby(performance_df['loan_seq_number']).apply(get_maint_pres_costs).unstack()),on='loan_seq_number')
                    summary_df=summary_df.join((performance_df['taxes_ins'].groupby(performance_df['loan_seq_number']).apply(get_taxes_ins).unstack()),on='loan_seq_number')
                    summary_df=summary_df.join((performance_df['misc_expenses'].groupby(performance_df['loan_seq_number']).apply(get_misc_expenses).unstack()),on='loan_seq_number')
                    summary_df=summary_df.join((performance_df['actual_loss_calc'].groupby(performance_df['loan_seq_number']).apply(get_actual_loss_calc).unstack()),on='loan_seq_number')
                    summary_df=summary_df.join((performance_df['modification_cost'].groupby(performance_df['loan_seq_number']).apply(get_modification_cost).unstack()),on='loan_seq_number')
                    if flag == 0:    
                        summary_df.to_csv(file, mode='a', header=True,index=False)
                        flag = 1
                    else:
                        summary_df.to_csv(file, mode='a', header=False,index=False)
                        
                        
def main():
    args = sys.argv[1:]
    counter = 0
    if len(args) == 0:
        print("Please provide login and password")
        exit(0)
    for arg in args:
        if counter == 0:
            username = str(arg)
        elif counter == 1:
            passw = str(arg)
        counter += 1
    print("Login:", username)
    print("Pass:", passw)
    login(username,passw)
    clean_merge_generate_origin_csv()
    clean_merge_generate_perform_csv()
        
        
if __name__ == '__main__':
    main()


# In[ ]:



