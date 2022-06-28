import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from  PIL import Image
import base64

data = pd.read_csv('diabetes.csv')
# data.head()
# print(data)
columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction",
           "Age"]
data[columns] = data[columns].replace({'0': np.nan, 0: np.nan})

#imputing using median
imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
imp_median.fit(data.values)
data_median = imp_median.transform(data.values)
data_median = pd.DataFrame(data_median)
data_median.columns =['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                      'DiabetesPedigreeFunction', 'Age', 'Outcome']

# Scaling data
newData = data_median[['Pregnancies', 'Glucose', 'BMI', 'DiabetesPedigreeFunction']]
newData.columns = ['Pregnancies', 'Glucose', 'BMI', 'DiabetesPedigreeFunction']
minmaxScale = MinMaxScaler()
X = minmaxScale.fit_transform(newData.values)
transformedDF = minmaxScale.transform(X)
data_transformedDF = pd.DataFrame(X)
data_transformedDF.columns = ['Pregnancies', 'Glucose', 'BMI', 'DiabetesPedigreeFunction']
data_transformedDF['Outcome'] = data_median['Outcome']
# Show top 5 rows
# data_transformedDF.head()

# Splitting the dataset
features = data_transformedDF.drop(["Outcome"], axis=1)
labels = data_transformedDF["Outcome"]
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.20, random_state=7)

X = x_train.append(x_test)
y = y_train.append(y_test)

logreg = LogisticRegression()
logreg = logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)

model = logreg.fit(x_train, y_train)

pickle_out = open("logisticRegr.pkl", "wb")
pickle.dump(logreg, pickle_out)
pickle_out.close()

pickle_in = open('logisticRegr.pkl', 'rb')
classifier = pickle.load(pickle_in)

with st.sidebar:
    st.set_page_config(initial_sidebar_state='expanded')

    choose_option = ['Home', 'About DiaPred', 'Diabetes', 'Calculate']
    choose = option_menu("DiaPred ", ["Home", "DiaPred", "Diabetes", "Calculate"],
                             icons=['house', 'info-circle', 'card-text', 'calculator'],
                             menu_icon="app-indicator", default_index=0,
                             #orientation="horizontal",
                             styles={
                                "container": {"padding": "5!important", "background-color": "#fafafa"},
                                "icon": {"color": "orange", "font-size": "25px"},
                                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#FEC9C9"},
                                "nav-link-selected": {"background-color": "#F9665E"},
                             }
                         )

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
        <style>
            .stApp {
            background-image: url("data:image/png;base64,%s");
            background-size: contain;
        }
        </style>
        ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

if choose == "DiaPred":
    set_background('blank.png')
    calc = Image.open('calculator2.png')
    st.image(calc)

    st.markdown(""" <style> .font {
       font-size:18x ; font-family: 'roboto light'; color: #000000;} 
       </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">DiaPred is a prediction calculator that aims to determine the risk of having '
                'type 2 diabetes based on the data provided by the user. The calculator was modelled using PIMA Indian Diabetes'
                ' dataset and logistic regression algorithm for classification with an application of median imputation'
                ' to handle the missing values from the dataset. The model reached an accuracy rate of 78.1 percent.</p>', unsafe_allow_html=True)


elif choose == "Home":
    set_background('blank.png')
    logo = Image.open('logo (3).png')
    st.image(logo)

    st.markdown(""" <style> .font {
        font-size:50px ; font-family: 'Roboto'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
    _, _, _, col, _, _, _ = st.columns([1] * 6 + [1.18])
    clicked = col.button('Tutorial')

    if clicked:
        col1, col2, col3 = st.columns(3)
        st.markdown(""" <style> .font {
                            font-size:30px ; font-family: 'roboto'; color: #fafafa;} 
                            </style> """, unsafe_allow_html=True)
        st.markdown(""" <style> .font2 {
                                        font-size:18px ; font-family: 'roboto light'; color: #000000;} 
                                        </style> """, unsafe_allow_html=True)
        with col1:

            st.markdown('<p class="font">Step 1</p>', unsafe_allow_html=True)
            st.markdown('<p class="font2">Navigate to Calculate Page.</p>', unsafe_allow_html=True)
            step1 = Image.open('step1.png')
            st.image(step1)

        with col2:
            st.markdown('<p class="font">Step 2</p>', unsafe_allow_html=True)
            st.markdown('<p class="font2">Input the necessary data.</p>', unsafe_allow_html=True)
            step2 = Image.open('step 2.png')
            st.image(step2)

        with col3:
            st.markdown('<p class="font">Step 3</p>', unsafe_allow_html=True)
            st.markdown('<p class="font2">Download the result.</p>', unsafe_allow_html=True)
            step3 = Image.open('step 3.png')
            st.image(step3)

elif choose == "Diabetes":
    #set_background('blank.png')
    diabetes = Image.open('diabetes (1).png')
    st.image(diabetes)
    st.markdown(""" <style> .font {
    font-size:18x ; font-family: 'roboto light'; color: #2E2E2E;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Diabetes is a disease that occurs when your blood glucose,'
                'also called blood sugar, is too high. Blood glucose is your main source of '
                'energy and comes from the food you eat. Insulin, a hormone made by the pancreas, '
                'helps glucose from food get into your cells to be used for energy.  '
                'Sometimes your body doesn’t make enough—or any—insulin or doesn’t use insulin well.'
                'Glucose then stays in your blood and doesn’t reach your cells.</p>', unsafe_allow_html=True)
    st.markdown('<p </p>', unsafe_allow_html=True)

    types_diabetes = Image.open('types.png')
    st.image(types_diabetes)
    st.markdown('<p class="font">The most common types of diabetes are type 1, type 2, and gestational diabetes.</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        types_1 = Image.open('type1.png')
        st.image(types_1)
        st.markdown('<p class="font"> Type 1 diabetes is a condition in which your immune system destroys insulin-making cells in your pancreas. '
                    'These are called beta cells. The condition is usually diagnosed in children and young people, so it used to be called juvenile diabetes.'
                    '</p>', unsafe_allow_html=True)

    with col2:
        types_1 = Image.open('type2.png')
        st.image(types_1)
        st.markdown('<p class="font"> Type 2 diabetes is an impairment in the way '
                    'the body regulates and uses sugar (glucose) as a fuel. '
                    'This long-term (chronic) condition results in too much sugar '
                    'circulating in the bloodstream. Eventually, high blood sugar '
                    'levels can lead to disorders of the circulatory, nervous and immune '
                    'systems </p>', unsafe_allow_html=True)

    gestational = Image.open('gestational.png')
    st.image(gestational)
    st.markdown(' <p class ="font"> Gestational diabetes develops in some women when they are pregnant. '
                'Most of the time, this type of diabetes goes away after the baby is born. '
                'However, if you’ve had gestational diabetes, you have a greater chance of developing type '
                '2 diabetes later in life. Sometimes diabetes diagnosed during pregnancy is actually type 2 diabetes.'
                '</p', unsafe_allow_html=True)

    type2_big = Image.open('type 2 big.png')
    st.image(type2_big)

    st.markdown('<p class="font"> Type 2 diabetes used to be known as adult-onset diabetes, '
                'but both type 1 and type 2 diabetes can begin during childhood and adulthood. '
                'Type 2 is more common in older adults, but the increase in the number of '
                'children with obesity has led to more cases of type 2 diabetes in younger '
                'people. </p>', unsafe_allow_html=True)

    st.markdown('<p class="font"> There is no cure for type 2 diabetes, but losing weight, eating well '
                'and exercising can help you manage the disease. If diet and exercise are not '
                'enough to manage your blood sugar, you may also need diabetes medications or '
                'insulin therapy. </p>', unsafe_allow_html=True)

    col1_diabetes1, col2_diabetes2 = st.columns(2)

    with col1_diabetes1:
        causes = Image.open('Causes.png')
        st.image(causes)

        st.markdown('<p class="font">'
                    '<ul> <li class="font">Cells in muscle, fat and the liver become resistant to insulin.</li> '
                    '<li class="font">The pancreas is unable to produce enough insulin to manage blood sugar levels.</li>'
                    '</ul> </p>', unsafe_allow_html=True)

    with col2_diabetes2:
        symptoms = Image.open('Symptoms.png')
        st.image(symptoms)

        st.markdown('<p class="font">'
                    '<ul> <li class="font">Increased thirst</li> '
                    '<li class="font">Frequent urination</li> <li class="font">Increased hunger</li> '
                    '<li class="font">Unintended weight loss</li>'
                    '<li class="font">Blurred vision</li>'
                    '<li class="font">Fatigue</li>'
                    '<li class="font">Slow-healing sores</li>'
                    '<li class="font">Frequent infections</li>'
                    '<li class="font">Numbness or tingling in the hands or feet</li>'
                    '</ul class="font"> </p>', unsafe_allow_html=True)

    complication = Image.open('complication.png')
    st.image(complication)

    st.markdown('<p class="font"> Type 2 diabetes affects many major organs, '
                'including your heart, blood vessels, nerves, eyes and kidneys. '
                'The complications of diabetes include:'
                '</p>', unsafe_allow_html=True)
    st.markdown('<p class="font">'
                '<ul> <li class="font">Heart and blood vessel disease</li> '
                '<li class="font">Kidney disease</li> <li class="font">Eye damage</li> '
                '<li class="font">Sleep apnea</li>'
                '<li class="font">Nerve damage (neuropathy) in limbs</li>'
                '</ul> </p>', unsafe_allow_html=True)

    risk = Image.open('risk factors.png')
    st.image(risk)

    st.markdown('<p class="font"> The factors of having diabetes include: . </p>', unsafe_allow_html=True)
    st.markdown('<p class="font">'
                '<ul> <li class="font">Weight</li> '
                '<li class="font">Fat distribution</li> <li class="font">Family history</li> '
                '<li class="font">Blood lipid levels</li>'
                '<li class="font">Age</li>'
                '<li class="font">Pregnancy-related risks</li>'
                '<li class="font">Prediabetes</li>'
                '</ul> </p>', unsafe_allow_html=True)

    prevention = Image.open('prevention.png')
    st.image(prevention)

    st.markdown('<p class="font"> Healthy lifestyle choices can help prevent type 2 diabetes. It includes: . </p>', unsafe_allow_html=True)
    st.markdown('<p class="font">'
                '<ul> <li class="font">Eating healthy foods</li> '
                '<li class="font">Getting active</li> <li class="font">Losing weight</li> '
                '</ul> </p>', unsafe_allow_html=True)


    with st.expander('Show Sources'):
        st.write("""
                 https://www.webmd.com/diabetes/guide/understanding-diabetes-symptoms
                 https://www.mayoclinic.org/diseases-conditions/type-2-diabetes/symptoms-causes/syc-20351193
                 https://www.niddk.nih.gov/health-information/diabetes/overview/what-is-diabetes
                 https://www.mayoclinic.org/diseases-conditions/type-2-diabetes/symptoms-causes/syc-20351193
             """)




elif choose == "Calculate":
    set_background('blank.png')
    calc = Image.open('calculator2.png')
    st.image(calc)

    text_contents = ''''''
    pregnancy_contents = 'No. of pregnancy: '
    glucose_contents = 'Plasma Glucose Concentration: '
    bmi_contents = 'Body Mass Index (BMI): '
    dpf_contents = 'Diabetes Pedigree Function: '
    classification_contents = 'Classification: '

    with st.form(key='columns_in_form2',clear_on_submit=True): #set clear_on_submit=True so that the form will be reset/cleared once it's submitted
        #st.write('Please help us improve!')
        pregnancy = st.number_input("No. of pregnancy:", min_value=0, max_value=27, value=0)
        glucose = st.number_input("Plasma Glucose Concentration :",  value=0)
        bmi = st.number_input("Body Mass Index (BMI) :", value=0.000)
        dpf = st.number_input("Diabetes Pedigree Function:", value=0.000)
        submitted = st.form_submit_button('Predict')


        if submitted:
             if pregnancy < 0 or pregnancy > 27:
                  st.write('Incorrect input for pregnancy.')
             elif glucose < 1 or glucose > 2656:
                  st.write('Incorrect input for glucose.')
             elif bmi < 7.5 or bmi > 185:
                  st.write('Incorrect input for BMI.')
             elif dpf < 0 or dpf > 2:
                  st.write('Incorrect input for diabetes pedigree function.')
             else:
                arr = np.array([[pregnancy, glucose, bmi, dpf]])
                X_scaled = arr.reshape(1,-1)
                X_minmax_scaled = minmaxScale.transform(X_scaled)
                prediction = classifier.predict(X_minmax_scaled)
                probability = classifier.predict_proba(X_minmax_scaled)
                #probability_0 = round(probability[0][0] * 100, 2)
                probability_1 = round(probability[0][1] * 100, 2)
                if prediction == 1:
                    classification_contents = classification_contents + str(probability_1) + '% risk of Diabetes.'
                    st.write('You have ', str(probability_1), '% risk of having Diabetes.')
                    st.write('High risk.')

                else:
                    classification_contents = classification_contents + str(probability_1) + '% risk of Diabetes.'
                    st.write('You have ', str(probability_1), '% risk of having Diabetes.')
                    st.write('Low risk.')

                pregnancy_contents = pregnancy_contents + str(pregnancy)
                glucose_contents = glucose_contents + str(glucose)
                bmi_contents = bmi_contents + str(bmi)
                dpf_contents = dpf_contents + str(dpf)

    text_contents = pregnancy_contents + '\n' + glucose_contents + '\n' + bmi_contents + '\n' + dpf_contents + '\n' +  classification_contents
    st.download_button('Download result',  text_contents)



footer_style ='''
<style>
.main
</style>
'''