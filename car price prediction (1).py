#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Data Understanding

# In[2]:


df = pd.read_csv('car data.csv')
df.head()


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


df.duplicated


# In[6]:


df.drop_duplicates()


# In[7]:


df.isnull().sum()


# # Data Preprocessing

# In[8]:


df['Year'].value_counts()


# In[9]:


df['Age_of_car'] = 2022 - df['Year']


# In[10]:


df.drop('Year',axis = 1,inplace = True)


# In[11]:


df.head()


# In[12]:


df.rename(columns = {'Selling_Price':'Selling_Price(lacs)','Present_Price':'Present_Price(lacs)','Owner':'Past_Owners'},inplace = True)


# In[13]:


df.columns


# In[14]:


num_cols = ['Selling_Price(lacs)','Present_Price(lacs)','Kms_Driven','Age_of_car']
cat_cols = ['Fuel_Type','Seller_Type','Transmission','Past_Owners','Car_Name']


# # EDA

# In[15]:


####----Distribution of Selling Price
plt.figure(figsize=(10, 6))
sns.histplot(df['Selling_Price(lacs)'], bins=30, kde=True)
plt.title('Distribution of Selling Price')
plt.xlabel('Selling Price (lacs)')
plt.ylabel('Frequency')
plt.show()


# In[16]:


####----Distribution of Present Price
plt.figure(figsize=(10, 6))
sns.histplot(df['Present_Price(lacs)'], bins=30, kde=True)
plt.title('Distribution of Present Price')
plt.xlabel('Present Price (lacs)')
plt.ylabel('Frequency')
plt.show()


# In[17]:


####---- Distribution of Kms Driven
plt.figure(figsize=(10, 6))
sns.histplot(df['Kms_Driven'], bins=30, kde=True)
plt.title('Distribution of Kms Driven')
plt.xlabel('Kms Driven')
plt.ylabel('Frequency')
plt.show()


# In[18]:


#####---- Distribution of Age of Car
plt.figure(figsize=(10, 6))
sns.histplot(df['Age_of_car'], bins=30, kde=True)
plt.title('Distribution of Age of Car')
plt.xlabel('Age of Car')
plt.ylabel('Frequency')
plt.show()


# In[19]:


####----- Scatter plot of Present Price vs Selling Price
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Present_Price(lacs)', y='Selling_Price(lacs)', data=df)
plt.title('Present Price vs Selling Price')
plt.xlabel('Present Price (lacs)')
plt.ylabel('Selling Price (lacs)')
plt.show()


# In[20]:


####---Scatter plot of Kms Driven vs. Selling Price
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Kms_Driven', y='Selling_Price(lacs)', data=df)
plt.title('Kms Driven vs Selling Price')
plt.xlabel('Kms Driven')
plt.ylabel('Selling Price (lacs)')
plt.show()


# In[21]:


####----Scatter plot of Age of Car vs. Selling Price
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age_of_car', y='Selling_Price(lacs)', data=df)
plt.title('Age of Car vs Selling Price')
plt.xlabel('Age of Car')
plt.ylabel('Selling Price (lacs)')
plt.show()


# In[22]:


#####----- Correlation matrix
plt.figure(figsize=(10, 8))
correlation_matrix = df[num_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Numerical Features')
plt.show()


# In[23]:


df.corr()['Selling_Price(lacs)']


# categorical values plots

# In[24]:


####---Distribution of Fuel Type
plt.figure(figsize=(10, 6))
sns.countplot(x='Fuel_Type', data=df)
plt.title('Distribution of Fuel Type')
plt.xlabel('Fuel Type')
plt.ylabel('Count')
plt.show()


# In[25]:


####---Distribution of Seller Type
plt.figure(figsize=(10, 6))
sns.countplot(x='Seller_Type', data=df)
plt.title('Distribution of Seller Type')
plt.xlabel('Seller Type')
plt.ylabel('Count')
plt.show()


# In[26]:


####---Distribution of Transmission
plt.figure(figsize=(10, 6))
sns.countplot(x='Transmission', data=df)
plt.title('Distribution of Transmission')
plt.xlabel('Transmission')
plt.ylabel('Count')
plt.show()


# In[27]:


####---Distribution of Past Owners
plt.figure(figsize=(10, 6))
sns.countplot(x='Past_Owners', data=df)
plt.title('Distribution of Past Owners')
plt.xlabel('Past Owners')
plt.ylabel('Count')
plt.show()


# In[28]:


####--Distribution of Car Name
plt.figure(figsize=(12, 8))
top_car_names = df['Car_Name'].value_counts().head(10)  # Top 10 car names
sns.barplot(x=top_car_names.index, y=top_car_names.values)
plt.title('Top 10 Car Names')
plt.xlabel('Car Name')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()


# In[29]:


####---Boxplot of Selling Price by Fuel Type
plt.figure(figsize=(10, 6))
sns.boxplot(x='Fuel_Type', y='Selling_Price(lacs)', data=df)
plt.title('Selling Price by Fuel Type')
plt.xlabel('Fuel Type')
plt.ylabel('Selling Price (lacs)')
plt.show()


# In[30]:


####---Boxplot of Selling Price by Seller Type
plt.figure(figsize=(10, 6))
sns.boxplot(x='Seller_Type', y='Selling_Price(lacs)', data=df)
plt.title('Selling Price by Seller Type')
plt.xlabel('Seller Type')
plt.ylabel('Selling Price (lacs)')
plt.show()


# In[31]:


###---Boxplot of Selling Price by Transmission
plt.figure(figsize=(10, 6))
sns.boxplot(x='Transmission', y='Selling_Price(lacs)', data=df)
plt.title('Selling Price by Transmission')
plt.xlabel('Transmission')
plt.ylabel('Selling Price (lacs)')
plt.show()


# In[32]:


df.drop('Car_Name',axis = 1,inplace= True)


# In[33]:


df.drop('Past_Owners',axis = 1,inplace= True)


# In[34]:


df.head()


# In[35]:


df = pd.get_dummies(df,drop_first = True)
df.head()


# # Train-Test-split

# In[36]:


X = df[num_cols].drop('Selling_Price(lacs)',axis = 1)
y = df['Selling_Price(lacs)']


# In[37]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)


# In[38]:


from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score


# In[39]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[40]:


CV = []
R2_train = []
R2_test = []

def car_pred_model(model,model_name):
    # Training model
    model.fit(X_train,y_train)
            
    # R2 score of train set
    y_pred_train = model.predict(X_train)
    R2_train_model = r2_score(y_train,y_pred_train)
    R2_train.append(round(R2_train_model,2))
    
    # R2 score of test set
    y_pred_test = model.predict(X_test)
    R2_test_model = r2_score(y_test,y_pred_test)
    R2_test.append(round(R2_test_model,2))
    
     # R2 mean of train set using Cross validation
    cross_val = cross_val_score(model ,X_train ,y_train ,cv=5)
    cv_mean = cross_val.mean()
    CV.append(round(cv_mean,2))
    
    # Printing results
    print("Train R2-score :",round(R2_train_model,2))
    print("Test R2-score :",round(R2_test_model,2))
    print("Train CV scores :",cross_val)
    print("Train CV mean :",round(cv_mean,2))


# # Linear Regression

# In[41]:


from sklearn.linear_model import LinearRegression

lr = LinearRegression()
car_pred_model(lr,"Linear_regressor.pkl")


# # Lasso Regression

# In[42]:


from sklearn.linear_model import Lasso
lasso_model = Lasso()
car_pred_model(lasso_model,"Lasso_regressor.pkl")


# # Ridge Regression

# In[43]:


from sklearn.linear_model import Ridge
ridge_model = Ridge()
car_pred_model(ridge_model,"Ridge_regressor.pkl")


# # Random Forest

# In[44]:


from sklearn.ensemble import RandomForestRegressor
rmodel = RandomForestRegressor()
car_pred_model(rmodel,"Random_Forest.pkl")


# # Gradiant boost

# In[45]:


from sklearn.ensemble import GradientBoostingRegressor
gb_model = GradientBoostingRegressor()
car_pred_model(gb_model,"Gradient boost.pkl")


# # from the all above models that we considered of there results the best model and perfect model is Linear Regression

# # Price finding

# In[46]:


import joblib
joblib.dump(lr, 'car_price_model.pkl')
joblib.dump(scaler, 'scaler.pkl')


# In[47]:


model = joblib.load('car_price_model.pkl')
scaler = joblib.load('scaler.pkl')


# In[48]:


new_data = pd.DataFrame({
    'Present_Price(lacs)': [5.0, 4.5],
    'Kms_Driven': [25000, 30000],
    'Age_of_car': [2, 3]
})


# In[49]:


new_data_scaled = scaler.transform(new_data)


# In[50]:


predictions = model.predict(new_data_scaled)


# In[51]:


for i, price in enumerate(predictions):
    print(f"Predicted Selling Price for new car {i + 1}: {price:.2f} lacs")


# In[52]:


df.columns


# # Streamlit App

# In[53]:


import streamlit as st
import pandas as pd
import joblib 


# In[54]:


# Load the trained model and label encoder
model = joblib.load('car_price_model.pkl')


# In[55]:


def load_model():
    # Load your pre-trained model from a file
    model = joblib.load('car_price_model.pkl')
    return model


# In[56]:


def main():
    st.title('Car Price Prediction')

    # Load the model
    model = load_model()

    # User input for prediction
    present_price = st.number_input('Present Price (in lacs)', min_value=0.0)
    kms_driven = st.number_input('Kilometers Driven', min_value=0)
    age_of_car = st.number_input('Age of Car (in years)', min_value=0)
    
    # Create a DataFrame for prediction
    input_data = pd.DataFrame({
        'Present_Price(lacs)': [present_price],
        'Kms_Driven': [kms_driven],
        'Age_of_car': [age_of_car],
    })
    
     # Predict
    if st.button('Predict'):
        prediction = model.predict(input_data)[0]
        st.write(f'Estimated Car Price: {prediction:.2f} lacs')

if __name__ == "__main__":
    main()


# In[ ]:




