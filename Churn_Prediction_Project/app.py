import streamlit as st
import pickle
import pandas as pd
from PIL import Image

html_temp = """
<div style="background-color:tomato;padding:10px">
<h2 style="border:2px solid white;color:white;text-align:center;">Employee Churn Prediction Model</h2>
</div>"""

st.markdown(html_temp,unsafe_allow_html=True)

im = Image.open("hr.jpg")
st.image(im, width=697)#, caption="Employee Churn Prediction")

model = pickle.load(open("smothe_model", "rb"))

st.sidebar.info('Employee Characteristics')

satisfaction_level=st.sidebar.slider('Satisfaction Level',0.0,1.0)
last_evaluation=st.sidebar.slider("Last Performance", 0.0,1.0)
number_project=st.sidebar.slider("Number of project", 0,10,step=1)
average_montly_hours=st.sidebar.slider("Average Monthly Hours" , 70,350)
time_spend_company=st.sidebar.slider("Time Spend Company(year)",0,10)
Work_accident=st.sidebar.selectbox("Work Accident", ('Yes','No'))
promotion_last_5years=st.sidebar.selectbox('Promotion Last Five Years',('Yes','No'))
departments=st.sidebar.selectbox("Departments",('IT','RandD','accounting','hr','management','marketing','product_mng','sales','support','technical'))
salary=st.sidebar.selectbox("Salary",('low','medium','high'))

my_dict = {
	'satisfaction_level':satisfaction_level,
	'last_evaluation':last_evaluation,
	'number_project':number_project,
	'average_montly_hours':average_montly_hours,
	'time_spend_company':time_spend_company,
	'Work_accident':Work_accident,
	'promotion_last_5years':promotion_last_5years,
	'departments':departments,
	'salary':salary
}

df = pd.DataFrame.from_dict([my_dict])

df['salary'] = df['salary'].map({'low' : 0, 'medium' : 1, 'high' : 2})

df['Work_accident'] = df['Work_accident'].map({'No' : 0, 'Yes' : 1})

df['promotion_last_5years'] = df['promotion_last_5years'].map({'No' : 0, 'Yes' : 1})

columns = ['satisfaction_level','last_evaluation','number_project',
		   'average_montly_hours','time_spend_company','Work_accident',
		   'promotion_last_5years','salary','IT','RandD','accounting','hr',
		   'management','marketing','product_mng','sales','support','technical']



df2 = pd.get_dummies(df).reindex(columns=columns, fill_value=0)
st.table(df)
prediction = model.predict(df2)

#st.table(df)

if st.button('Predict'):
    if prediction == 1.0:
    	st.warning('Yes, the employee will left.')
    else:
        st.success("No, the employee won't left.")