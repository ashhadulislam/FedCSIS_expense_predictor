import streamlit as st
import pandas as pd
import geopy.distance
from pickle import load
import time
import copy

import torch

import os
import shutil


from PIL import Image
def load_image(image_file):
    print("Loading ",image_file)
    img = Image.open(image_file)
    img.save(os.path.join("data",image_file.name))
    return img


def calc_distance(df):
    first_load_lats=list(df.first_load_lat)
    first_load_lons=list(df.first_load_lon)


    last_unload_lats=list(df.last_unload_lat)
    last_unload_lons=list(df.last_unload_lon)

    loading_distances=[]

    for i in range(len(first_load_lats)):
        if i%30000==0:
            print("{}/{}".format(i,len(first_load_lats)))
        coords_1 = (first_load_lats[i], first_load_lons[i])
        coords_2 = (last_unload_lats[i], last_unload_lons[i]) # Jamesie
        dist=geopy.distance.geodesic(coords_1, coords_2).km
        loading_distances.append(dist)
        
        

    df["loading_distance"]=loading_distances




    route_start_lats=list(df.route_start_lat)
    route_start_lons=list(df.route_start_lon)


    route_end_lats=list(df.route_end_lat)
    route_end_lons=list(df.route_end_lon)

    route_distances=[]

    for i in range(len(route_start_lats)):
        if i%30000==0:
            print("{}/{}".format(i,len(route_start_lats)))
        coords_1 = (route_start_lats[i], route_start_lons[i])
        coords_2 = (route_end_lats[i], route_end_lons[i]) # Jamesie
        dist=geopy.distance.geodesic(coords_1, coords_2).km
        route_distances.append(dist)
        
        

    df["route_distance"]=route_distances

    return df    

def process_dataframe(df):
    drop_cols=["temperature","first_load_country","last_unload_country",
          "route_start_country","route_end_country","prim_train_line",
          "prim_ferry_line"]
    df=df.drop(columns=drop_cols)
    df['route_end_datetime'] = pd.to_datetime(df['route_end_datetime'], errors='coerce')
    df['route_start_datetime'] = pd.to_datetime(df['route_start_datetime'], errors='coerce')

    df['handling_time'] = (df['route_end_datetime'] - df['route_start_datetime']).dt.total_seconds() /60 

    # get the fuel part
    df_fuel=pd.read_csv("data/fuel_prices.csv",sep=";")
    df_fuel['date'] = pd.to_datetime(df_fuel['date'], errors='coerce')    

    disel_type1_price_list=[]
    disel_type2_price_list=[]
    disel_type3_price_list=[]
    for each_id in list(df.id_contract):
        df_id=df[df.id_contract==each_id]
        start_date=list(df_id.route_start_datetime)[0]
        end_date=list(df_id.route_end_datetime)[0]    
        df_fuel_dated=df_fuel[(df_fuel.date>=start_date) & (df_fuel.date<end_date)]
        df_fuel_dated=df_fuel_dated.drop(columns=["date"])
        df_fuel_dated_sum=df_fuel_dated.sum()
        disel_type1_price_list.append(df_fuel_dated_sum["disel_type1_price"])
        disel_type2_price_list.append(df_fuel_dated_sum["disel_type2_price"])
        disel_type3_price_list.append(df_fuel_dated_sum["disel_type3_price"])    
    df["disel_type1_price"]=disel_type1_price_list
    df["disel_type2_price"]=disel_type2_price_list
    df["disel_type3_price"]=disel_type3_price_list

    # end of fuel part

    # replace lat long with distance
    df=calc_distance(df)
    lat_lon_cols=["route_start_lat",
              "route_start_lon","route_end_lat","route_end_lon"]
    df=df.drop(columns=lat_lon_cols)
    if "expenses" in df.columns:
        df=df.drop(columns=["expenses"])
    # more_columns_dropped=["km_total"]
    # df=df.drop(columns=more_columns_dropped)

    # now to calculate steps and haversine distance
    df_routes=pd.read_csv("data/css_routes_test.csv",sep=";")
    res={}
    res["id_contract"]=[]
    res["km_haversine"]=[]
    res["num_steps"]=[]
    count=0
    for id_contract in list(df.id_contract):
        res["id_contract"].append(id_contract)
        res["km_haversine"].append(df_routes[df_routes.id_contract==id_contract].km_haversine.sum())
        res["num_steps"].append(df_routes[df_routes.id_contract==id_contract].shape[0])

    df_routes=pd.DataFrame(res)

    df = pd.merge(df, df_routes, how='inner')

    return df


def final_pre_process(df):
    print(df.head())
    print(df.columns)
    final_selected_features=['direction', 'km_nonempty', 'km_total', 'last_unload_lat',
       'ferry_duration', 'train_km', 'contract_type', 'last_unload_lon',
       'max_weight', 'first_load_lat', 'first_load_lon', 'id_currency',
       'load_size_type', 'km_haversine', 'num_steps']
    df=df[final_selected_features]
    print("Final")
    print(df.columns)
    
    categ_cols=["direction","contract_type","id_currency","load_size_type"]


    for col in categ_cols:
        uniq_values=list(df[col].unique())
        uniq_labels=[i for i in range(len(uniq_values))]
        
        df[col].replace(uniq_values, uniq_labels, inplace=True)    

    Xscaler = load(open('data/Xscaler.pkl', 'rb'))
    X_test=df.values
    X_test=Xscaler.transform(X_test)
    return X_test
    

def get_predictions(df_original,X_test):
    models=os.listdir("models")
    if ".DS_Store" in models:
        models.remove(".DS_Store")

    for model in models:
        model_name=model.split(".")[0]
        print("Name = ",model_name)
        reg=load(open('models/'+model, 'rb'))
        predictions=reg.predict(X_test)
        # print(model_name,predictions.shape)
        df_original[model_name+"_expenses"]=list(predictions)
        df_original.to_csv("outputs/result"+model_name+".csv",index=False,sep=";")
        with st.container():
            st.info(model_name+" predictions are as follows (scroll right to see the predictions)")

            with open("outputs/result"+model_name+".csv") as f:
                st.download_button('Download Predictions by '+model_name, f, 'text/csv')  # Defaults to 'text/plain'
            st.dataframe(df_original)
            st.markdown("""---""")

        df_original=df_original.drop(columns=[model_name+"_expenses"])

    

def app():
    cur_dir=os.getcwd()
    header=st.container()
    result_all = st.container()
    with header:
        st.subheader("Check expenses for contracts")
        data_file = st.file_uploader("Upload csv file")

        if data_file is not None:
            # To See details
            file_details = {"filename":data_file.name, "filetype":data_file.type,
                          "filesize":data_file.size}
            st.write(file_details)
            df = pd.read_csv(data_file,sep=";")

            # image_file=os.path.join("data",fname)
        else:
            with open("data/css_main_test.csv") as f:
                st.download_button('Download Sample File', f, 'text/csv')  # Defaults to 'text/plain'
            fname="css_main_test.csv"
            proxy_csv_file="data/"+fname
            df=pd.read_csv(proxy_csv_file,sep=";")
        df_original=copy.deepcopy(df)
        st.dataframe(df)
        st.write("Starting pre process . . . ")


        df=process_dataframe(df)
        st.write("After Pre processing and feature extraction [Scroll Right to see new features]")
        st.dataframe(df)
        st.markdown("""---""")
        X_test=final_pre_process(df)
        st.subheader("Check predictions by different models below")
        st.write("You can download predicitons from each model by clicking on the Download buttons")
        get_predictions(df_original,X_test)
        # df_predictions.to_csv("outputs/result.csv",index=False,sep=";")
        
        
        
            # predict on this


    # with result_all:     
    #     weight_path=os.path.join(cur_dir,"yolov5/runs/train/RoadTrainModel/weights/best.pt")
    #     shutil.rmtree('yolov5/runs/detect/')

    #     detect.run(weights=weight_path,name="RoadTestModel", source=image_file)  
    #     image_file_output="yolov5/runs/detect/RoadTestModel/"+fname
    #     img = Image.open(image_file_output)
    #     st.subheader("Road Defect Detections")    
    #     st.image(img,width=250)            
        

    #     