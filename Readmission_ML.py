import pandas as pd
import streamlit as st
import base64
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# Title
st.set_page_config(page_title="Hospital Re-Admission Analysis", layout="wide")

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
    
    [data-testid="stFileUploader"] {{
		width: 500px;
    }}
    
    [data-testid="stSelectbox"] {{
		width: 200px;
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

st.markdown("<h1 style='color: #3d9df3;text-align: center'>Hospital Re-Admission Analysis</h1>", unsafe_allow_html=True)
st.markdown('<br><br>',unsafe_allow_html=True)

# Page Layout
c1, c2 = st.columns(2)

# Function to rename readmitted subcategories
def readm_code_to_name(code):
    if code == 0:
        return 'Not Admitted'
    elif code == 1:
        return '<30days'
    elif code == 2:
        return '>30days'
    else:
         return code
   
   
chart_df = df.copy()
   
# Renameing df column TIME_IN_HOSPITAL, DIAG_1, DIAG_2
chart_df = chart_df.rename(columns={'TIME_IN_HOSPITAL': 'LENGTH_OF_STAY', 'DIAG_1': 'Primary', 'DIAG_2': 'Secondary'})

# Apply the function to the Readminssion column
chart_df['READMITTED'] = chart_df['READMITTED'].apply(readm_code_to_name)

# Function to rename gender category
def gen_code_to_name(code):
    if code == 0:
        return 'Female'
    elif code == 1:
        return 'Male'
    else:
        return code
    
# Apply the function to the Gender column
chart_df['GENDER'] = chart_df['GENDER'].apply(gen_code_to_name)

# Function to rename DiabetMed category
def dbt_code_to_name(code):
    if code == 0:
        return 'Non-diabetic'
    elif code == 1:
        return 'Diabetic'
    else:
        return code
    
# Apply the function to the DiabetMed column
chart_df['DIABETESMED'] = chart_df['DIABETESMED'].apply(dbt_code_to_name)

# Function to rename Inpatient category
def inpat_code_to_name(code):
    if code < 1:
        return 0
    elif code >= 1:
        return 1
    else:
        return code
   
# Apply the function to the Inpatient column
chart_df['NUMBER_INPATIENT'] = chart_df['NUMBER_INPATIENT'].apply(inpat_code_to_name)
 
# Function to rename Emergency category
def emrg_code_to_name(code):
    if code < 1:
        return 0
    elif code >= 1:
        return 1
    else:
        return code
   
# Apply the function to the Emergency column
chart_df['NUMBER_EMERGENCY'] = chart_df['NUMBER_EMERGENCY'].apply(emrg_code_to_name)

with c1:    
    
    # Heading
    st.markdown("<h5 style='color: #0068c9;'>Rate of Patients Readmitted within 30 Days by Age Group</h5>", unsafe_allow_html=True)

    # Calculate the percentage of patients readmitted within 30 days by age group
    readmitted_within_30_days = chart_df[chart_df['READMITTED'] == '<30days']
    readmitted_counts_by_age = readmitted_within_30_days.groupby('AGE').size()
    total_counts_by_age = chart_df.groupby('AGE').size()
    readmission_rates_by_age = ((readmitted_counts_by_age / total_counts_by_age) * 100).round(2)

    # Create a Barchart plotting for percentage of patients readmitted within 30 days by age group
    readmission_rates_age = pd.DataFrame({'Age Group': readmission_rates_by_age.index,
                                        'Readmission Rate (%)': readmission_rates_by_age.values})

    # Sort the DataFrame by age group
    readmission_rates_age = readmission_rates_age.sort_values(by='Age Group')

    # Plot the bar chart
    fig = px.bar(readmission_rates_age, x='Age Group', y='Readmission Rate (%)',
                labels={'Readmission Rate (%)': 'Readmission Rate (%)'},
                text='Readmission Rate (%)',
                color='Age Group', width=600)
    
    # Set x-axis range
    fig.update_xaxes(range=[10, 100]) 
    
    # Set transparent background
    fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
    
    # Customize tick and label colors for x-axis and y-axis
    fig.update_xaxes(tickfont=dict(color='#5CA8F1'),  # Change x-axis tick color to blue
                    titlefont=dict(color='#EF7B45'))  # Change x-axis label color to blue
    fig.update_yaxes(tickfont=dict(color='#5CA8F1'),  # Change y-axis tick color to green
                    titlefont=dict(color='#EF7B45'))  # Change y-axis label color to blue

    # show chart
    st.plotly_chart(fig)

with c2:
    
    # Heading
    st.markdown("<h5 style='color: #0068c9;'>Rate of Patients Readmitted Later than 30 Days by Gender</h5>", unsafe_allow_html=True)
    
    # Calculate the percentage of patients readmitted later than 30 days
    readmitted_later_than_30_days = chart_df[chart_df['READMITTED'] == '>30days']

    # Create a Stacked Pie chart plotting for percentage of patients readmitted later than 30 days by gender
    readmitted_counts_by_gender = readmitted_later_than_30_days.groupby('GENDER').size()
    total_counts_by_gender = chart_df.groupby('GENDER').size()
    readmission_rates_by_gender = ((readmitted_counts_by_gender / total_counts_by_gender) * 100).round(2)

    # Create a DataFrame for plotting
    readmission_rates_gen = pd.DataFrame({'Gender': readmission_rates_by_gender.index,
                                        'Readmission Rate (%)': readmission_rates_by_gender.values})

    # Plot the Pie chart
    fig = px.pie(readmission_rates_gen, values='Readmission Rate (%)', names='Gender',
                labels={'Readmission Rate (%)': 'Readmission Rate (%)'},
                color= 'Gender', width=600)

    # Set transparent background
    fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')

    # Show the plot
    st.plotly_chart(fig)
    

# Page Layout
c1, c2 = st.columns(2)

with c1:
    
    # Heading
    st.markdown("<h5 style='color: #0068c9;'>Rate of Readmission by Number of Diagnoses</h5>", unsafe_allow_html=True)

    # Calculate readmission rate by number of diagnoses from both early and later readmission
    readmitted_counts = chart_df[chart_df['READMITTED'].isin(['<30days','>30days'])].groupby('NUMBER_DIAGNOSES').size()
    total_counts_by_diag = chart_df.groupby('NUMBER_DIAGNOSES').size()
    readmission_rate_by_diag = ((readmitted_counts / total_counts_by_diag) * 100).round(2)

    readmission_diag = pd.DataFrame({'Number of Diagnoses': readmission_rate_by_diag.index, 'Readmission Rate (%)': readmission_rate_by_diag.values})

    # Sort the DataFrame by number of diagnoses
    readmission_diag.sort_values(by='Number of Diagnoses', inplace=True)

    # Create a line chart
    fig = px.line(readmission_diag, x='Number of Diagnoses', y='Readmission Rate (%)', 
                labels={'Number of Diagnoses': 'Number of Diagnoses', 'Readmission Rate (%)': 'Readmission Rate (%)'},
                text='Readmission Rate (%)', width=600)
    # Set transparent background
    fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
    
    # Customize tick and label colors for x-axis and y-axis
    fig.update_xaxes(tickfont=dict(color='#5CA8F1'),  # Change x-axis tick color to blue
                    titlefont=dict(color='#EF7B45'))  # Change x-axis label color to blue
    fig.update_yaxes(tickfont=dict(color='#5CA8F1'),  # Change y-axis tick color to green
                    titlefont=dict(color='#EF7B45'))  # Change y-axis label color to blue
    
    fig.update_traces(textposition='top center',  # Move text to the top center of each point
                  textfont=dict(color='red'),  # Customize text font
                  texttemplate='%{text:.1f}',  # Format text
                  hoverinfo='skip')  # Hide hover info to only display text

    st.plotly_chart(fig)
    

with c2:
    
    # Heading
    st.markdown("<h5 style='color: #0068c9;'>Average Length of Stay (LOS) by Readmission</h5>", unsafe_allow_html=True)
    
    # Histogram for Time in Hospital Stay By Readmission
    # fig = px.histogram(chart_df, x='LENGTH_OF_STAY', color='READMITTED',
    #                                     labels={'LENGTH_OF_STAY': 'Length of stay in Hospital (days)'},
    #                                     histnorm='percent')
    
    # Group data based on readmitted status and calculate the average time_in_hospital
    grouped_df = (chart_df.groupby('READMITTED')['LENGTH_OF_STAY'].mean().round(2)).reset_index()

    # Create a grouped bar chart
    fig = px.pie(grouped_df, values='LENGTH_OF_STAY', names='READMITTED',hole=0.5,
                labels={'READMITTED': 'Readmission Type', 'LENGTH_OF_STAY': 'Avg Length of stay in Hospital (days)'}, width=600)

    # fig = px.bar(grouped_df, x='READMITTED', y='LENGTH_OF_STAY', color='READMITTED',
    #                         # category_orders={'NUMBER_INPATIENT': sorted(inpat_not_visited['NUMBER_INPATIENT'].unique())},
    #                         labels={'READMITTED': 'Readmission Type', 'LENGTH_OF_STAY': 'Length of stay in Hospital (days)'}, width=600,
    #                         text='LENGTH_OF_STAY', orientation='h')

    # Set transparent background
    fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
    # fig.update_traces(text=chart_df['LENGTH_OF_STAY'], textposition='auto')
    
    # Customize tick and label colors for x-axis and y-axis
    # fig.update_xaxes(tickfont=dict(color='#5CA8F1'),  # Change x-axis tick color to blue
    #                 titlefont=dict(color='#EF7B45'))  # Change x-axis label color to blue
    # fig.update_yaxes(tickfont=dict(color='#5CA8F1'),  # Change y-axis tick color to green
    #                 titlefont=dict(color='#EF7B45'))  # Change y-axis label color to blue
    
    st.plotly_chart(fig)
 
   
# Page Layout
c1, c2 = st.columns(2)

with c1:
    
    # Heading
    st.markdown("<h5 style='color: #0068c9;'>Analyzing Readmission Trends in Diabetic Cases</h5>", unsafe_allow_html=True)
    
    # Count the occurrences of Readmission for each Diabetes type
    filter_option = ('All', 'Diabetic', 'Non-diabetic')

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        selected_dbm = st.selectbox("***Diabetic Status***", filter_option)
    
    # Filter the DataFrame based on the selected option
    if selected_dbm == 'All':
        filtered_df = chart_df[chart_df['DIABETESMED'].isin(['Diabetic', 'Non-diabetic'])]
    elif selected_dbm == 'Non-diabetic':
        filtered_df = chart_df[chart_df['DIABETESMED'].isin(['Non-diabetic'])]
    elif selected_dbm == 'Diabetic':
        filtered_df = chart_df[chart_df['DIABETESMED'].isin(['Diabetic'])]

    diabetes_readmission = filtered_df.groupby(['DIABETESMED', 'READMITTED']).size().reset_index(name='count')
    
    # Create a horizontal bar chart
    fig = px.bar(diabetes_readmission, x='count', y='READMITTED', color='DIABETESMED', orientation='h',
                labels={'count': 'Count', 'DIABETESMED': 'Diabetic Status', 'READMITTED': 'Readmission Type'}, width=600,
                text='count')

    # Set transparent background
    fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
    
    # Customize tick and label colors for x-axis and y-axis
    fig.update_xaxes(tickfont=dict(color='#5CA8F1'),  # Change x-axis tick color to blue
                    titlefont=dict(color='#EF7B45'))  # Change x-axis label color to blue
    fig.update_yaxes(tickfont=dict(color='#5CA8F1'),  # Change y-axis tick color to green
                        titlefont=dict(color='#EF7B45'))  # Change y-axis label color to blue
    # Render the chart using Streamlit
    st.plotly_chart(fig)
    
with c2:
    
    # Heading
    st.markdown("<h5 style='color: #0068c9;'>Readmission Trends Across Diagnoses</h5>", unsafe_allow_html=True)
    # Count the occurrences of readmission status for each combination of Diag1 and Diag2
    diag_readmission_count = chart_df.groupby(['Primary', 'Secondary', 'READMITTED']).size().reset_index(name='count')
 
    # Create a radio button
    options_mapping = ['Primary', 'Secondary']
    selected_ds = st.radio("***Diagnosis***", options_mapping)
 
    if selected_ds == "Primary":
        # Create a bar chart
        fig = px.bar(diag_readmission_count, x=selected_ds, y='count', color='READMITTED',
                        category_orders={'Primary': sorted(chart_df['Primary'].unique())},
                        labels={'count': 'Count of Readmission by Primary', 'Primary': 'Primary Diagnosis'},width=600)
        
        # Customize tick and label colors for x-axis and y-axis
        fig.update_xaxes(tickfont=dict(color='#5CA8F1'),  # Change x-axis tick color to blue
                        titlefont=dict(color='#EF7B45'))  # Change x-axis label color to blue
        fig.update_yaxes(tickfont=dict(color='#5CA8F1'),  # Change y-axis tick color to green
                        titlefont=dict(color='#EF7B45'))  # Change y-axis label color to blue
 
        # Set transparent background
        fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
   
   
        # Render the chart using Streamlit
        st.plotly_chart(fig)
       
    elif selected_ds == "Secondary":
        # Create a bar chart
        fig = px.bar(diag_readmission_count, x=selected_ds, y='count', color='READMITTED',
                        category_orders={'Secondary': sorted(chart_df['Secondary'].unique())},
                        labels={'count': 'Count of Readmission by Secondary', 'Secondary': 'Secondary Diagnosis'},
                        title='Readmission Trends Across Diagnoses')
 
        # Set transparent background
        fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
        
        # Customize tick and label colors for x-axis and y-axis
        fig.update_xaxes(tickfont=dict(color='#5CA8F1'),  # Change x-axis tick color to blue
                        titlefont=dict(color='#EF7B45'))  # Change x-axis label color to blue
        fig.update_yaxes(tickfont=dict(color='#5CA8F1'),  # Change y-axis tick color to green
                        titlefont=dict(color='#EF7B45'))  # Change y-axis label color to blue
   
   
        # Render the chart using Streamlit
        st.plotly_chart(fig)


# Page Layout
c1, c2 = st.columns(2)

with c1:
    
    # Heading
    st.markdown("<h5 style='color: #0068c9;'>Analysis of Readmission Across Inpatient Visits For the Preceding Year</h5>", unsafe_allow_html=True)
        
    # Count the occurrences of readmission status for Inpatient Visits
    inpat_status = st.radio("**Inpatient Status:**", ['***Visited***', '***Not Visited***'])
        
    if inpat_status == '***Not Visited***':
    
        # Heading
        st.markdown("<h6 style='color: #0068c9;'>Understanding Readmission Patterns for Patients without Inpatient History</h6>", unsafe_allow_html=True)
        
        inpat_not_visited = chart_df[chart_df['NUMBER_INPATIENT'] < 1]
        inpat_readmission_count = inpat_not_visited.groupby(['NUMBER_INPATIENT', 'READMITTED']).size().reset_index(name='count')
        
        # fig = px.bar(chart_df, x='NUMBER_INPATIENT', y='READMITTED', title='Number of Inpatient Visits by Patient Number', color='NUMBER_INPATIENT'
        #         #  labels={'NUMBER_INPATIENT': 'Number of Inpatient Visits', 'GENDER': 'Patient Number'}
        #          )

        # fig = px.bar(inpat_readmission_count, x='NUMBER_INPATIENT', y='count', color='READMITTED',
        #                     category_orders={'NUMBER_INPATIENT': sorted(inpat_not_visited['NUMBER_INPATIENT'].unique())},
        #                     labels={'count': 'Count of Inpatient Visit on Current Years', 'NUMBER_INPATIENT': 'No. of Inpatient Visits'},
        #                      text='count')
        
        fig = px.pie(inpat_readmission_count, values='count', names='READMITTED',hole=0.5,
                labels={'READMITTED': 'Readmission Type', 'count': 'Count of Inpatient Visit on Current Years'}, width=600)

    
        # Set transparent background
        fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
        
        # Customize tick and label colors for x-axis and y-axis
        fig.update_xaxes(tickfont=dict(color='#5CA8F1'),  # Change x-axis tick color to blue
                        titlefont=dict(color='#EF7B45'))  # Change x-axis label color to blue
        fig.update_yaxes(tickfont=dict(color='#5CA8F1'),  # Change y-axis tick color to green
                        titlefont=dict(color='#EF7B45'))  # Change y-axis label color to blue

        # Set x-axis range
        fig.update_xaxes(range=[0, 0]) 
        
        # Render the chart using Streamlit
        st.plotly_chart(fig)
    
    elif inpat_status == '***Visited***':
        
        # Heading
        st.markdown("<h6 style='color: #0068c9;'>Analyzing Readmission Trends For Previous Year's Inpatient Admission</h6>", unsafe_allow_html=True)

        inpat_not_visited = chart_df[chart_df['NUMBER_INPATIENT'] > 0]
        inpat_readmission_count = inpat_not_visited.groupby(['NUMBER_INPATIENT', 'READMITTED']).size().reset_index(name='count')
        
        # fig = px.bar(chart_df, x='NUMBER_INPATIENT', y='READMITTED', title='Number of Inpatient Visits by Patient Number', color='NUMBER_INPATIENT'
        #         #  labels={'NUMBER_INPATIENT': 'Number of Inpatient Visits', 'GENDER': 'Patient Number'}
        #          )

        # fig = px.bar(inpat_readmission_count, x='count', y='NUMBER_INPATIENT', color='READMITTED',
        #                     category_orders={'NUMBER_INPATIENT': sorted(inpat_not_visited['NUMBER_INPATIENT'].unique())},
        #                     labels={'count': 'Count of Inpatient Visit on Previous Years', 'NUMBER_INPATIENT': 'No. of Inpatient Visits'},
        #                     text='count', orientation='h')

        fig = px.pie(inpat_readmission_count, values='count', names='READMITTED',hole=0.5,
                labels={'READMITTED': 'Readmission Type', 'count': 'Count of Inpatient Visit on Previous Years'}, width=600)


        # Set transparent background
        fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
                
        # Customize tick and label colors for x-axis and y-axis
        fig.update_xaxes(tickfont=dict(color='#5CA8F1'),  # Change x-axis tick color to blue
                        titlefont=dict(color='#EF7B45'))  # Change x-axis label color to blue
        fig.update_yaxes(tickfont=dict(color='#5CA8F1'),  # Change y-axis tick color to green
                        titlefont=dict(color='#EF7B45'))  # Change y-axis label color to blue

        # Set axis range
        fig.update_yaxes(range=[1, 21]) 
        fig.update_xaxes(range=[0, 9000]) 
        
        # Render the chart using Streamlit
        st.plotly_chart(fig)
        
# with c2:
#   # Heading
#     st.markdown("<h5 style='color: #0068c9;'>Analysis of Readmission Trends Across Over Emergency Visits</h5>", unsafe_allow_html=True)
        
#     # Count the occurrences of readmission status for Inpatient Visits
#     emg_status = st.radio("**Emergency Status**:", ['***Visited***', '***Not Visited***'])
        
#     if emg_status == '***Not Visited***':
    
#         # Heading
#         st.markdown("<h6 style='color: #0068c9;'>Understanding Readmission Rates among Patients without Prior Emergency Visits</h6>", unsafe_allow_html=True)
        
#         emg_not_visited = chart_df[chart_df['NUMBER_EMERGENCY'] < 1]
#         # st.write(emg_not_visited)
#         emg_readmission_count = emg_not_visited.groupby(['NUMBER_EMERGENCY', 'READMITTED']).size().reset_index(name='count')
#         # st.write(emg_readmission_sum)
        
#         # fig = px.bar(chart_df, x='NUMBER_INPATIENT', y='READMITTED', title='Number of Inpatient Visits by Patient Number', color='NUMBER_INPATIENT'
#         #         #  labels={'NUMBER_INPATIENT': 'Number of Inpatient Visits', 'GENDER': 'Patient Number'}
#         #          )

#         # fig = px.bar(emg_readmission_sum, x='NUMBER_EMERGENCY', y='avg', color='READMITTED',
#         #                     category_orders={'NUMBER_EMERGENCY': sorted(emg_not_visited['NUMBER_EMERGENCY'].unique())},
#         #                     labels={'avg': 'Sum of Emergency Visit on the Current Year', 'NUMBER_EMERGENCY': 'No. of Emergency Visits'},
#         #                      text='avg')
        
#         fig = px.pie(emg_readmission_count, values='count', names='READMITTED',hole=0.5,
#                 labels={'READMITTED': 'Readmission Type', 'count': 'Count of Inpatient Visit on Current Years'}, width=600)

    
#         # Set transparent background
#         fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
        
#         # Customize tick and label colors for x-axis and y-axis
#         fig.update_xaxes(tickfont=dict(color='#5CA8F1'),  # Change x-axis tick color to blue
#                         titlefont=dict(color='#EF7B45'))  # Change x-axis label color to blue
#         fig.update_yaxes(tickfont=dict(color='#5CA8F1'),  # Change y-axis tick color to green
#                         titlefont=dict(color='#EF7B45'))  # Change y-axis label color to blue

#         # Set x-axis range
#         # fig.update_xaxes(range=[0, 0]) 
        
#         # Render the chart using Streamlit
#         st.plotly_chart(fig)
    
#     elif emg_status == '***Visited***':
        
#         # Heading
#         st.markdown("<h6 style='color: #0068c9;'>Analyzing Readmission Trends For Previous Year's Emergency Admissions</h6>", unsafe_allow_html=True)

#         emg_not_visited = chart_df[chart_df['NUMBER_EMERGENCY'] >= 0]
#         emg_readmission_count = emg_not_visited.groupby(['NUMBER_EMERGENCY', 'READMITTED']).size().reset_index(name='count')
        
#         # fig = px.bar(chart_df, x='NUMBER_INPATIENT', y='READMITTED', title='Number of Inpatient Visits by Patient Number', color='NUMBER_INPATIENT'
#         #         #  labels={'NUMBER_INPATIENT': 'Number of Inpatient Visits', 'GENDER': 'Patient Number'}
#         #          )

#         # fig = px.bar(inpat_readmission_count, x='count', y='NUMBER_INPATIENT', color='READMITTED',
#         #                     category_orders={'NUMBER_INPATIENT': sorted(inpat_not_visited['NUMBER_INPATIENT'].unique())},
#         #                     labels={'count': 'Count of Inpatient Visit on Previous Years', 'NUMBER_INPATIENT': 'No. of Inpatient Visits'},
#         #                     text='count', orientation='h')

#         fig = px.pie(emg_readmission_count, values='count', names='READMITTED',hole=0.5,
#                 labels={'READMITTED': 'Readmission Type', 'count': 'Count of Emergency Visits on Previous Years'}, width=600)


#         # Set transparent background
#         fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
                
#         # Customize tick and label colors for x-axis and y-axis
#         fig.update_xaxes(tickfont=dict(color='#5CA8F1'),  # Change x-axis tick color to blue
#                         titlefont=dict(color='#EF7B45'))  # Change x-axis label color to blue
#         fig.update_yaxes(tickfont=dict(color='#5CA8F1'),  # Change y-axis tick color to green
#                         titlefont=dict(color='#EF7B45'))  # Change y-axis label color to blue

#         # Set axis range
#         fig.update_yaxes(range=[1, 21]) 
#         fig.update_xaxes(range=[0, 9000]) 
        
#         # Render the chart using Streamlit
#         st.plotly_chart(fig)  
    
with c2:
  # Heading
    st.markdown("<h5 style='color: #0068c9;'>Analysis of Readmission Across Emergency Visits For the Preceding Year</h5>", unsafe_allow_html=True)
        
    # Count the occurrences of readmission status for Inpatient Visits
    emg_status = st.radio("**Emergency Status**:", ['***Visited***', '***Not Visited***'])
        
    if emg_status == '***Not Visited***':
    
        # Heading
        st.markdown("<h6 style='color: #0068c9;'>Understanding Readmission Rates among Patients without Prior Emergency Visits</h6>", unsafe_allow_html=True)
        
        emg_not_visited = chart_df[chart_df['NUMBER_EMERGENCY'] < 1]
        # st.write(emg_not_visited)
        emg_readmission_count = emg_not_visited.groupby(['NUMBER_EMERGENCY', 'READMITTED']).size().reset_index(name='count')        

        fig = px.treemap(emg_readmission_count, 
                 path=['READMITTED', 'count'], 
                 values='count',
                #  custom_data=['customdata'],
                 hover_data={'count': True},
                 color='READMITTED', 
                 color_continuous_scale='Viridis', width=600)

        # Update layout
        fig.update_layout(title='Sample Treemap Chart',
                        treemapcolorway=['gold', 'mediumturquoise', 'darkorange', 'lightgreen'],
                        margin=dict(t=50, l=25, r=25, b=25))
    
    
    
        # Set transparent background
        fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
        
        # Customize tick and label colors for x-axis and y-axis
        fig.update_xaxes(tickfont=dict(color='#5CA8F1'),  # Change x-axis tick color to blue
                        titlefont=dict(color='#EF7B45'))  # Change x-axis label color to blue
        fig.update_yaxes(tickfont=dict(color='#5CA8F1'),  # Change y-axis tick color to green
                        titlefont=dict(color='#EF7B45'))  # Change y-axis label color to blue

        # Set x-axis range
        # fig.update_xaxes(range=[0, 0]) 
        
        # Render the chart using Streamlit
        st.plotly_chart(fig)
    
    elif emg_status == '***Visited***':
        
        # Heading
        st.markdown("<h6 style='color: #0068c9;'>Analyzing Readmission Trends For Previous Year's Emergency Admissions</h6>", unsafe_allow_html=True)

        emg_not_visited = chart_df[chart_df['NUMBER_EMERGENCY'] > 0]
        emg_readmission_count = emg_not_visited.groupby(['NUMBER_EMERGENCY', 'READMITTED']).size().reset_index(name='count')

        fig = px.treemap(emg_readmission_count, 
                 path=['READMITTED', 'count'], 
                 values='count',
                #  custom_data=['customdata'],
                 hover_data={'count': True},
                 color='READMITTED', 
                 color_continuous_scale='Viridis', width=600)

        # Update layout
        fig.update_layout(
            # title='Sample Treemap Chart',
                        treemapcolorway=['gold', 'mediumturquoise', 'darkorange', 'lightgreen'],
                        margin=dict(t=50, l=25, r=25, b=25))

        # Set transparent background
        fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
                
        # Customize tick and label colors for x-axis and y-axis
        fig.update_xaxes(tickfont=dict(color='#5CA8F1'),  # Change x-axis tick color to blue
                        titlefont=dict(color='#EF7B45'))  # Change x-axis label color to blue
        fig.update_yaxes(tickfont=dict(color='#5CA8F1'),  # Change y-axis tick color to green
                        titlefont=dict(color='#EF7B45'))  # Change y-axis label color to blue

        # Set axis range
        fig.update_yaxes(range=[1, 21]) 
        fig.update_xaxes(range=[0, 9000]) 
        
        # Render the chart using Streamlit
        st.plotly_chart(fig)  
    
    
# st.header('Prediction on Hospital Readmission')
st.markdown("<h2 style='color: #3d9df3;'>Prediction</h2>", unsafe_allow_html=True)

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
    tab_titles = ['Single', 'Batch']
    tablet = st.tabs(tab_titles)
    
    # Custom CSS for tabs
    custom_tabs_css = """
    <style>
    .stTabs [data-baseweb="tab-list"] {
        display: flex;
        justify-content: space-between;
        width: 200px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 35px;
        width: 130px;
        white-space: pre-wrap;
        background-color: #5ca8f1;
        border-radius: 10px 10px 10px 10px;
        color: white;
        gap: 50px;
        padding-top: 10px;
        padding-bottom: 10px;
        text-align: center;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white;
        border-radius: 10px 10px 0px 0px;
        color: #5ca8f1;
    }
    </style>
    """
    
    # Inject custom CSS
    st.markdown(custom_tabs_css, unsafe_allow_html=True)
    
    with tablet[0]:
        y_pred = pd.DataFrame(y_pred)
        y_pred.rename(columns={0: 'READMITTED'}, inplace=True)
        st.markdown("<h4 style='color: #3d9df3;'>Predicted Data</h4>", unsafe_allow_html=True)
        # st.write(y_pred)
        
        s1 = upd_df['PATIENT_NBR'][0]
        # Description of predicted result
        if y_pred['READMITTED'].iloc[0] == 0:
            # string = "The patient" + upd_df['PATIENT_NBR'][0] + "is not likely to be readmitted."
            # st.write(f'**The patient {s1} is not likely to be readmitted.**')
            st.markdown(f"<h6>The patient <span style='color: #EF7B45;'>{s1}</span> is not likely to be readmitted</h6>", unsafe_allow_html=True)
        elif y_pred['READMITTED'].iloc[0] == 1:
            # st.write(f'**The patient {s1} is likely to be readmitted within 30days.**')
            st.markdown(f"<h6>The patient <span style='color: #EF7B45;'>{s1}</span> is likely to be readmitted within 30days.</h6>", unsafe_allow_html=True)
        elif y_pred['READMITTED'].iloc[0] == 2:
            # st.write(f'**The patient {s1} is likely to be readmitted after 30days.**')
            st.markdown(f"<h6>The patient <span style='color: #EF7B45;'>{s1}</span> is likely to be readmitted after 30days.</h6>", unsafe_allow_html=True)
                
    with tablet[1]:
        y_pred = pd.DataFrame(y_pred)
        y_pred.rename(columns={0: 'READMITTED'}, inplace=True)
        
        # Function to rename gender category
        def pred_readm_code_to_desc(code):
            if code == 0:
                return 'Not likely to be readmitted'
            elif code == 1:
                return 'Likely to be readmitted within 30 days'
            elif code == 2:
                return 'Likely to be readmitted after 30 days'
            else:
                return code
            
        # Apply the function to the Gender column
        y_pred['READMITTED'] = y_pred['READMITTED'].apply(pred_readm_code_to_desc)

        a = pd.concat([upd_df['PATIENT_NBR'], y_pred], axis=1)
        # Save the concatenated DataFrame to BytesIO
        csv_file = BytesIO()
        a.to_csv(csv_file, index=False)

        
        # Create a download link
        if st.download_button(label='Download Predictions CSV',data=csv_file.getvalue(),file_name='predictions.csv',key='download_button'):
            # st.success("The predictions are downloaded!")

            st.markdown("<I><h6 style='color: #03c04a;'>The predictions are downloaded!</h6></I>", unsafe_allow_html=True)
    
    
    # button = st.button("Single", "Batch")
    # if st.button('Single'):
    #     y_pred = pd.DataFrame(y_pred)
    #     y_pred.rename(columns={0: 'READMITTED'}, inplace=True)
    #     st.subheader('Predicted Data')
    #     # st.write(y_pred)
        
    #     s1 = upd_df['PATIENT_NBR'][0]
    #     # Description of predicted result
    #     if y_pred['READMITTED'].iloc[0] == 0:
    #         # string = "The patient" + upd_df['PATIENT_NBR'][0] + "is not likely to be readmitted."
    #         st.write(f'**The patient {s1} is not likely to be readmitted.**')
    #     elif y_pred['READMITTED'].iloc[0] == 1:
    #         st.write(f'**The patient {s1} is likely to be readmitted within 30days.**')
    #     elif y_pred['READMITTED'].iloc[0] == 2:
    #         st.write(f'**The patient {s1} is likely to be readmitted after 30days.**')
            
    #     # y_test = y_test.head(99)
        
    # if st.button('Batch') :
    #     y_pred = pd.DataFrame(y_pred)
    #     y_pred.rename(columns={0: 'READMITTED'}, inplace=True)
    #     a = pd.concat([upd_df['PATIENT_NBR'], y_pred], axis=1)
    #     # Save the concatenated DataFrame to BytesIO
    #     csv_file = BytesIO()
    #     a.to_csv(csv_file, index=False)

    #     # Create a download link
    #     st.download_button(
    #     label='Download Predictions CSV',
    #     data=csv_file.getvalue(),
    #     file_name='predictions.csv',
    #     key='download_button')

    #     st.write('The Predictions are downloaded')
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
