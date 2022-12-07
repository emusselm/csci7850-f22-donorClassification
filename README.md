# Alumni Donor Classification and Categorization
### CSCI 7850 - Fall 2022

This project seeks to (1) classify MTSU alumni as donors and non-donors using basic demographic and socioeconomic information. It also attempts to (2) classify donors into five categories of average giving, as described in the table below. 

<div align=center>

 **Class 0**: Average Gift <= $10 <br> 
 **Class 1**: $10 < Average Gift < $21   
 **Class 2**:  $21 <= Average Gift < $50   
 **Class 3**:  $50 <= Average Gift <= $100   
 **Class 4**:  Average Gift >= 100
 
 </div>
 

We present two deep residual neural network architectures for alumni donor classification and categorization. These architectures can be found in the python scripts `donor_classification.py` and `donor_categorization.py`. These models are trained using compiled survey data from the American Community Survey and the US Census Data. The survey data is available for download [here](https://csci7850-f22-semesterproject.nyc3.digitaloceanspaces.com/survey_data.csv). <br>

Due to the confidentiality and sensitivity of information, the alumni data for this project is unavailable for download. However, a demo of the trained donor classification and categorization models is available.  

To access the demo, clone the repository and navigate to the demo folder. The demo is implemented using a python notebook, and therefore, an IDE that can run this file is essential to the demo. 

Once downloaded, the donor classification demo (`demo-donorClassification.ipynb`) and the donor categorization demo (`demo-donorCategorization.ipynb`) should be immediately ready for testing. The files contain functions for downloading the survey data and trained models, an interface for entering in alumni data, and steps for preprocessing the data to prepare for model testing. 

When run end-to-end, the first demo should provide a non-donor or donor classification for the given alumni information, and the second should provide a category for average donor giving for an alumni.
