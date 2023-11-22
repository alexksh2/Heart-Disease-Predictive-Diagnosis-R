# Heart_Disease_Predictive_Diagnosis
According to the World Health Organisation, cardiovascular disease are the most significant cause of mortality annually as it results in the deaths of __approximately  17.9 million individuals every year__ (World Health Organization, 2019). While __the healthcare industry generates substantial data__ on patients, diseases and diagnosis, it is found that __these data are not properly analysed__.  (Hassan, 2022) <br>

The key root cause for heart disease often in the context of __sex, smoking, age, family history, poor diet, cholesterol, physical inactivity, high blood pressure, overweightness, and alcohol use.__ <br>

Therefore, the objective of this study is to effectively predicting the presence of coronary heart disease using machine learning classifiers such as __Logistic Regression, K-Nearest Neighbours, Classification and Regression Tree and Random Forest__.



<br>
<br>


![UDMI_Cardiovascular-Disease](https://github.com/alexksh2/Heart_Disease_Predictive_Diagnosis/assets/138288828/575e295e-984a-4fc9-9acd-7eb3f4c85f71)

# The Dataset

__Content__ <br>
This data set dates from 1988 and consists of four databases: Cleveland, Hungary, Switzerland, and Long Beach V. It contains 76 attributes, including the predicted attribute, but all published experiments refer to using a subset of 14 of them. The "target" field refers to the presence of heart disease in the patient. It is integer valued 0 = no disease and 1 = disease. <br>

https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

__Context__ <br>
1. age <br>
2. sex <br>
3. chest pain type (4 values) <br>
4. resting blood pressure <br>
5. serum cholestoral in mg/dl <br>
6. fasting blood sugar > 120 mg/dl <br>
7. resting electrocardiographic results (values 0,1,2) <br>
8. maximum heart rate achieved <br>
9. exercise induced angina <br>
10. oldpeak = ST depression induced by exercise relative to rest <br>
11. the slope of the peak exercise ST segment <br>
12. number of major vessels (0-3) colored by flourosopy <br>
13. thal: 0 = normal; 1 = fixed defect; 2 = reversable defect <br>
14. The names and social security numbers of the patients were recently removed from the database, replaced with dummy values. <br>


# Data Treatment
__Undersampling__ have been done to reduce class imbalance such that the ML model outputs are not biased towards the majority class.  (Yenigün, O. 2023).



# Model Results
__1. Logistic Regression__ <br>
   Overall Accuracy Rate: 208/250 = __83.2%__ <br>
   F1 score: 0.8384615 <br>
   F2 score: 0.8582677 <br>

__2. K-Nearest Neighbours__ <br>
   Overall Accuracy Rate: 211/250 = __84.4%__ <br>
   F1 score: 0.8368201 <br>
   F2 score: 0.8143322 <br>

__3. Classification and Regression Tree (Classification)__ <br>
   Overall Accuracy Rate: 234/250 = __93.6%__ <br>
   F1 score: 0.936 <br>
   F2 score: 0.936 <br>

__4. Random Forest__ <br>
   Overall Accuracy Rate: 244/250 = __97.6%__ <br>
   F1 score: 0.9758065 <br>
   F2 score: 0.9711075 <br>
   

__Intrepretation of Model Results:__ <br>
The analytical results found that __Random Forest Model has the highest predictive accuracy on heart disease predictive diagnosis__ as it has the highest overall accuracy rate (97%), F1 score and F2 score as compared to __logistic regression model, K-Nearest Neighbors model and Classification and Regression Tree Model__.

<br>
<br>

__Variable Importance Bar Chart of CART Model:__ <br>
![Variable Importance Bar Chart (Optimal CART Model)](https://github.com/alexksh2/Heart_Disease_Predictive_Diagnosis/assets/138288828/d2e662ee-4a1c-4985-ab5d-43f2863e43f6)

<br>
<br>

__Variable Importance Plot of Random Forest:__ <br>
![Variable Importance Plot of Random Forest](https://github.com/alexksh2/Heart_Disease_Predictive_Diagnosis/assets/138288828/4036fe18-2a2b-42c4-be40-daf8ef6fe2a2)

<br>
<br>



Conclusion: __Both tree models (CART and Random Forest) results indicated that Thal is the most significant variable in heart disease predictive diagnosis__.

# Citation
World Health Organization: WHO. (2019, June 11). Cardiovascular diseases. Who.int; World Health Organization: WHO. https://www.who.int/health-topics/cardiovascular-diseases/#tab=tab_1 <br>

Hassan, Ch. A. ul, Iqbal, J., Irfan, R., Hussain, S., Algarni, A. D., Bukhari, S. S. H., Alturki, N., & Ullah, S. S. (2022). Effectively Predicting the Presence of Coronary Heart Disease Using Machine Learning Classifiers. Sensors, 22(19), 7227. https://doi.org/10.3390/s22197227 <br>

Yenigün, O. (2023, March 29). Handling Class Imbalance in Machine Learning. MLearning.ai. https://medium.com/mlearning-ai/handling-class-imbalance-in-machine-learning-cb1473e825ce
