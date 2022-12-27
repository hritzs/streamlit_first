import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
diamonds_data = pd.read_csv("diamondpriceprediction\diamonds.csv")
new_diamonds_data = pd.read_csv("diamondpriceprediction/new_diamonds.csv")
new_diamonds_data = pd.get_dummies(new_diamonds_data,columns=['cut','color','clarity'])
model_training = st.container()
with model_training:
    st.header("Model Training")
    st.title("In this we are going to train our model")
    selection_col, display_col = st.columns(2)
    selection_col.text('List Of features: ')
    selection_col.write(diamonds_data.columns)
    input_feature = selection_col.text_input('Give an input feature to be feeded to the model','carat')
    max_depth = st.slider("What should be the max depth of model", min_value=10,max_value=100,value=20,step=10)
    number_of_trees = st.selectbox("Count of trees",options = [100,200,300],index=0)
    model = RandomForestRegressor(max_depth=max_depth,n_estimators=number_of_trees)
    X=diamonds_data[[input_feature]]
    y= diamonds_data[['price']]
    model.fit(X,y)
    prediction = model.predict(new_diamonds_data[[input_feature]])
    new_diamonds_data['price']=prediction
    st.write(new_diamonds_data[['carat','price']].head(5))
