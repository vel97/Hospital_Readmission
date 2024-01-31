import pandas as pd
import streamlit as st
import base64
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# Title
st.set_page_config(page_title="Readmission Analytics", layout="wide")

# Heading
st.header('Prediction on Hospital Readmission')

# Define background image with base64 format
# def get_img(file):
# 	with open(file, "rb") as f:
# 		img = f.read()
# 	return base64.b64encode(img).decode()

# bg_img = get_img("C:/Users/SriramvelM/Desktop/sklearn_vs/Pharma1.png")

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
bg_img = add_bg_from_local('Pharma2.png')

# styles = f"""
# 			<style>
#                 [data-testid="stApp"] {{
# 				    background-image: url('data:image/png;base64,{bg_img}');
# 				    background-size: cover;
#                 }}
#             </style>
# 		"""
   
# Inject CSS into Streamlit app
# st.markdown(styles, unsafe_allow_html=True)

import pandas as pd
from snowflake.snowpark import Session
session = Session.builder.configs({'user': 'svel',
                                   'password': 'October1897',
                                   'account': 'ywjkphp-xi98015',
                                   'warehouse': 'COMPUTE_WH',
                                   'database': 'hospital',
                                   'schema': 'readmission',
                                   'role':'ACCOUNTADMIN'}).create()

# Uploaded Data
uploaded_file = st.file_uploader("Choose a file", type=["csv", "txt"])

if uploaded_file is not None:
    # button = st.button("Select Mode:", ["Single", "Batch"])
    upd_df = pd.read_csv(uploaded_file)
    # Change data types of multiple columns
    dtype_dict_upd = {
        'MEDICAL_SPECIALTY': 'int64',
        'DIAG_1': float,
        'DIAG_2': float
    }

    # Change data types of multiple columns
    upd_df = upd_df.astype(dtype_dict_upd) 

# Data
result = session.sql("SELECT * FROM DATA")
list = result.collect()
df =  pd.DataFrame(list)

# Define dictionary for mapping column names to desired data types
dtype_dict = {
    'ENCOUNTER_ID': 'int64',
    'PATIENT_NBR': 'int64',
    'RACE': 'int64',
    'GENDER': 'int64',
    'AGE': 'int64',
    'ADMISSION_TYPE_ID': 'int64',
    'DISCHARGE_DISPOSITION_ID': 'int64',
    'ADMISSION_SOURCE_ID': 'int64',
    'TIME_IN_HOSPITAL': 'int64',
    'DIAG_1': float,
    'DIAG_2': float,
    'MAX_GLU_SERUM': 'int64',
    'A1CRESULT': 'int64',
    'INSULIN': 'int64',
    'CHANGE': 'int64',
    'DIABETESMED': 'int64',
    'READMITTED': 'int64'
}

# Change data types of multiple columns
df = df.astype(dtype_dict) 

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()

col = ['MEDICAL_SPECIALTY']
for i in col:
    df[i] = label_encoder.fit_transform(df[i])

df['READMITTED'].value_counts()

# #Correlation heatmap
# import matplotlib.pyplot as plt
# import seaborn as sns
# corr_m = df.corr()
# sns.heatmap(corr_m)

# sns.countplot(data = df, x='READMITTED')
# plt.show()


# col = df.select_dtypes(include='number')
# for i in col:
#     fig, ax = plt.subplots(1,1)
#     sns.kdeplot(df[df["READMITTED"]==2][i], fill=True, color="red", label="No", ax=ax)
#     sns.kdeplot(df[df["READMITTED"]==1][i], fill=True, color="blue", label="<30", ax=ax)
#     sns.kdeplot(df[df["READMITTED"]==0][i], fill=True, color="green", label=">30", ax=ax)
#     ax.set_xlabel(i)
#     ax.set_ylabel("readmitted")
#     fig.suptitle(i +" "+"vs readmitted")
#     ax.legend()
#     # fig.show()

#     st.pyplot(fig)

#Splitting the dataset as train and test
x = df.drop(['READMITTED'], axis='columns')
y = df['READMITTED']

#Train test split
import numpy as np
from sklearn .model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.75, random_state=0)


# Assign data to x_test from uploaded dataset
if uploaded_file is not None:
    x_test = upd_df

########################## random forest model #########################
from sklearn.metrics import accuracy_score, roc_auc_score
# from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(bootstrap=False, max_depth=8, min_samples_leaf=2, min_samples_split=4, n_estimators=70)

# Create One-vs-Rest Classifier
from sklearn.multiclass import OneVsRestClassifier
ovr_classifier = OneVsRestClassifier(rfc)

# Train the model
ovr_classifier.fit(x_train, y_train)

############################################################################ Make predictions on the test set ################################################
y_pred = ovr_classifier.predict(x_test)

if uploaded_file is not None:
    # button = st.button("Single", "Batch")
    if st.button('Single'):
        y_pred = pd.DataFrame(y_pred)
        y_pred.rename(columns={0: 'READMITTED'}, inplace=True)
        st.subheader('Predicted Data')
        # st.write(y_pred)
        
        s1 = upd_df['PATIENT_NBR'][0]
        # Description of predicted result
        if y_pred['READMITTED'].iloc[0] == 0:
            # string = "The patient" + upd_df['PATIENT_NBR'][0] + "is not likely to be readmitted."
            st.write(f'**The patient {s1} is not likely to be readmitted.**')
        elif y_pred['READMITTED'].iloc[0] == 1:
            st.write(f'**The patient {s1} is likely to be readmitted within 30days.**')
        elif y_pred['READMITTED'].iloc[0] == 2:
            st.write(f'**The patient {s1} is likely to be readmitted after 30days.**')
            
        # y_test = y_test.head(99)
        
    if st.button('Batch') :
        y_pred = pd.DataFrame(y_pred)
        y_pred.rename(columns={0: 'READMITTED'}, inplace=True)
        a = pd.concat([upd_df['PATIENT_NBR'], y_pred], axis=1)
        # Save the concatenated DataFrame to BytesIO
        csv_file = BytesIO()
        a.to_csv(csv_file, index=False)

        # Create a download link
        st.download_button(
        label='Download Predictions CSV',
        data=csv_file.getvalue(),
        file_name='predictions.csv',
        key='download_button')

        st.write('The Predictions are downloaded')
        # st.write("Predictions are completed")
        # if st.button('Download as CSV'):



# Evaluate the performance
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy:.2f}")

# y_pred1 = ovr_classifier.predict_proba(x_test)
# roc_rf = roc_auc_score(y_test, y_pred1, multi_class='ovo')

# Display classification report
# from sklearn.metrics import accuracy_score, classification_report
# print("Classification Report:")
# print(classification_report(y_test, y_pred))

st.markdown("<h1 style='color: black;'>Readmission Analysis</h1>", unsafe_allow_html=True)
# st.header("Readmission Analysis", unsafe_allow_html=True)

# Page Layout
c1, c2 = st.columns(2)

with c1:
    # Calculate the percentage of patients readmitted within 30 days by age group
    readmitted_within_30_days = df[df['READMITTED'] == 1]
    readmitted_counts_by_age = readmitted_within_30_days.groupby('AGE').size()
    total_counts_by_age = df.groupby('AGE').size()
    readmission_rates_by_age = ((readmitted_counts_by_age / total_counts_by_age) * 100).round(2)

    # Create a Barchart plotting for percentage of patients readmitted within 30 days by age group
    readmission_rates_age = pd.DataFrame({'Age Group': readmission_rates_by_age.index,
                                        'Readmission Rate (%)': readmission_rates_by_age.values})

    # Sort the DataFrame by age group
    readmission_rates_age = readmission_rates_age.sort_values(by='Age Group')

    # Plot the bar chart
    fig = px.bar(readmission_rates_age, x='Age Group', y='Readmission Rate (%)',
                title='Percentage of Patients Readmitted within 30 Days by Age Group',
                labels={'Readmission Rate (%)': 'Readmission Rate (%)'},
                text='Readmission Rate (%)',
                color='Age Group', width=600)
    
    # Set x-axis range
    fig.update_xaxes(range=[10, 100]) 
    
    # Set transparent background
    fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')

    # show chart
    st.plotly_chart(fig)

with c2:
    # Calculate the percentage of patients readmitted later than 30 days
    readmitted_later_than_30_days = df[df['READMITTED'] == 2]

    # Create a Stacked Pie chart plotting for percentage of patients readmitted later than 30 days by gender
    readmitted_counts_by_gender = readmitted_later_than_30_days.groupby('GENDER').size()
    total_counts_by_gender = df.groupby('GENDER').size()
    readmission_rates_by_gender = ((readmitted_counts_by_gender / total_counts_by_gender) * 100).round(2)

    # Create a DataFrame for plotting
    readmission_rates_gen = pd.DataFrame({'Gender': readmission_rates_by_gender.index,
                                        'Readmission Rate (%)': readmission_rates_by_gender.values})

    # Plot the Pie chart
    fig = px.pie(readmission_rates_gen, values='Readmission Rate (%)', names='Gender',
                title='Percentage of Patients Readmitted Later than 30 Days by Gender',
                labels={'Readmission Rate (%)': 'Readmission Rate (%)'},
                color= 'Gender', width=600)

    # Set transparent background
    fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')

    # Show the plot
    st.plotly_chart(fig)
    

# Page Layout
c1, c2 = st.columns(2)

with c1:
    # Calculate readmission rate by number of diagnoses from both early and later readmission
    readmitted_counts = df[df['READMITTED'].isin([1,2])].groupby('NUMBER_DIAGNOSES').size()
    total_counts_by_diag = df.groupby('NUMBER_DIAGNOSES').size()
    readmission_rate_by_diag = ((readmitted_counts / total_counts_by_diag) * 100).round(2)

    readmission_diag = pd.DataFrame({'Number of Diagnoses': readmission_rate_by_diag.index, 'Readmission Rate (%)': readmission_rate_by_diag.values})

    # Sort the DataFrame by number of diagnoses
    readmission_diag.sort_values(by='Number of Diagnoses', inplace=True)

    # Create a line chart
    fig = px.line(readmission_diag, x='Number of Diagnoses', y='Readmission Rate (%)', 
                title='Readmission Rate by Number of Diagnoses',
                labels={'Number of Diagnoses': 'Number of Diagnoses', 'Readmission Rate (%)': 'Readmission Rate (%)'},
                text='Readmission Rate (%)', width=600)
    # Set transparent background
    fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')

    st.plotly_chart(fig)
    

with c2:
    # Histogram for Time in Hospital Stay By Readmission
    fig = px.histogram(df, x='TIME_IN_HOSPITAL', color='READMITTED',
                                        title='Distribution of Hospital Stay Duration by Readmission',
                                        labels={'TIME_IN_HOSPITAL': 'Time in Hospital (days)'},
                                        histnorm='percent')
    # Set transparent background
    fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
   
    st.plotly_chart(fig)
 
   
# Page Layout
c1, c2 = st.columns(2)
 
with c1:
    # Count the occurrences of readmission status for each combination of Diag1 and Diag2
    diag_readmission_count = df.groupby(['DIAG_1', 'DIAG_2', 'READMITTED']).size().reset_index(name='count')
 
    # Create a radio button
    options_mapping = ['DIAG_1', 'DIAG_2']
    selected_ds = st.radio("***Diagnosis***", options_mapping)
 
    if selected_ds == "DIAG_1":
        # Create a bar chart
        fig = px.bar(diag_readmission_count, x=selected_ds, y='count', color='READMITTED',
                        category_orders={'DIAG_1': sorted(df['DIAG_1'].unique())},
                        labels={'count': 'Count of Readmission by Diagnosis 1', 'DIAG_1': 'Diagnosis 1'},
                        title='Readmission Trends Across Diagnoses')
 
        # Set transparent background
        fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
   
   
        # Render the chart using Streamlit
        st.plotly_chart(fig)
       
    elif selected_ds == "DIAG_2":
        # Create a bar chart
        fig = px.bar(diag_readmission_count, x=selected_ds, y='count', color='READMITTED',
                        category_orders={'DIAG_2': sorted(df['DIAG_2'].unique())},
                        labels={'count': 'Count of Readmission by Diagnosis 2', 'DIAG_2': 'Diagnosis 2'},
                        title='Readmission Trends Across Diagnoses')
 
        # Set transparent background
        fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
   
   
        # Render the chart using Streamlit
        st.plotly_chart(fig)
