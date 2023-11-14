# Repository: Berkley-ML-AI-CapstoneProject

## This repository contains a project completed for Berkley AI ML course
#####                                                  Author - Lalitya Sawant

### **Overview**
#### Sales Forecasting for Walmart dataset.

In this application, we will explore a dataset from Kaggle which holds walmart sales information.

### **Problem Statement:**
Sales forecasting is a common problem for lot of organizations. This leads to loss of revenue generation and profits. Knowing the trend for the sales can help organizations to order for required quantity of good fordifferent departments and locations.

**Solution notebook link:** [apstoneProject_Berkley.ipynb: ](https://github.com/LalityaSawant/Berkley-ML-AI-CapstoneProject/blob/master/CapstoneProject_Berkley.ipynb)

To achieve this, a comprehensive analysis of the dataset is imperative. It includes below steps:

**Data Cleaning:** A meticulous review of the dataset to identify and rectify any inconsistencies or inaccuracies.

**Outlier Detection:** Identifying potential outliers and evaluating whether they should be excluded from the analysis.

**Bias Assessment:** Scrutinizing the dataset for any biases and implementing appropriate measures to address them.

**Data Transformation:** Converting textual/bollean data into a format understandable by the predictive model.

Once these preprocessing steps are accomplished, the subsequent task is to distribute the data in the training and testing set and then apply different algorithms to reach to an accurate prediction of the forecasts. 

### **Some insights on Data:**
Original shape of data: (421570, 16)

Int64Index: 421570 entries, 0 to 421569
Data columns (total 16 columns):
| #   | Column        | Non-Null Count | Dtype    |
| --- | ------------- | -------------- | -------- |
| 0   | Store         | 421570         | int64    |
| 1   | Dept          | 421570         | int64    |
| 2   | Date          | 421570         | object   |
| 3   | Weekly_Sales  | 421570         | float64  |
| 4   | IsHoliday     | 421570         | bool     |
| 5   | Temperature   | 421570         | float64  |
| 6   | Fuel_Price    | 421570         | float64  |
| 7   | MarkDown1     | 150681         | float64  |
| 8   | MarkDown2     | 111248         | float64  |
| 9   | MarkDown3     | 137091         | float64  |
| 10  | MarkDown4     | 134967         | float64  |
| 11  | MarkDown5     | 151432         | float64  |
| 12  | CPI           | 421570         | float64  |
| 13  | Unemployment  | 421570         | float64  |
| 14  | Type          | 421570         | object   |
| 15  | Size          | 421570         | int64    |
dtypes: bool(1), float64(10), int64(3), object(2)
memory usage: 51.9+ MB

Shape of the data after data processing/cleanup: (420212, 11)


## **Findings from the EDA:**                         
# Key Findings from Data Exploration

1. **Data Compilation:**
   - The data was initially provided in 4 separate CSV files.
   - We merged the store, features, and train CSVs to create a comprehensive dataset.

2. **Data Quality Enhancement:**
   - Identified and addressed null values in markdown columns by removing those columns.
   - Ensured better data quality for subsequent analysis.

3. **Sales Data Anomalies:**
   - Detected and addressed rows with negative sales values, likely data anomalies.
   - Removed such instances, maintaining the integrity of the dataset.

4. **Key Attributes Impacting Sales:**
   - Explored attributes like holidays, fuel price, unemployment, and temperature.

5. **Holiday Analysis:**
   - Categorized holidays into four types: Labor Day, Super Bowl, Thanksgiving, and Christmas.
   - Thanksgiving showed a strong positive impact on sales, while Super Bowl had a moderate impact.
   - Labor Day and Christmas did not exhibit a significant positive impact on sales.

6. **Other Sales Influencers:**
   - Explored factors beyond holidays, finding no clear positive or negative impact on sales.

7. **Yearly Sales Trend:**
   - Observed a consistent pattern of increased sales at the end of each year.

These insights provide a foundational understanding for our further analysis and decision-making processes.


### **Further Data Processing Steps:**

1. **Feature Transformation:**
   - Addressed remaining categorical and ordinal fields requiring transformation for modeling.

2. **Correlation Analysis:**
   - Explored the correlation of features with weekly sales.


### **Feture Selction:**
Upon executing Ridge Regression for feature selection, we obtained the following correlation coefficient data:
| #   | Features              | Coefs           |
| --- | --------------------- | --------------- |
| 3   | Size                  | 6111.455355     |
| 1   | Dept                  | 3272.028832     |
| 9   | Type_C                | 1379.949679     |
| 5   | Month                 | 1168.625192     |
| 2   | Fuel_Price            | 701.886808      |
| 17  | Thanksgiving_True     | 341.183575      |
| 18  | Christmas_False       | 206.217420      |
| 14  | Labor_Day_False       | 94.529108       |
| 13  | Super_Bowl_True        | 80.195029       |
| 11  | IsHoliday_True         | 62.867514       |
| 10  | IsHoliday_False        | -62.867514      |
| 12  | Super_Bowl_False       | -80.195029      |
| 15  | Labor_Day_True         | -94.529108      |
| 19  | Christmas_True         | -206.217420     |
| 16  | Thanksgiving_False     | -341.183575     |
| 7   | Type_A                | -410.515532     |
| 8   | Type_B                | -427.978958     |
| 4   | Week                 | -430.756689     |
| 6   | Year                 | -663.120183     |
| 0   | Store                | -1681.637899    |


Below is the output from the permutation_importance:
| Feature           | Mean           | Standard Deviation |
| ------------------ | -------------- | ------------------ |
| Size              | 73663404.289   | 675128.606         |
| Dept              | 23966202.892   | 296850.476         |
| Store             | 5440509.582    | 179566.791         |
| Type_C            | 4518730.235    | 145167.318         |
| Month             | 884192.429     | 110500.216         |
| Week              | 677551.526     | 40811.852          |
| Type_B            | 478886.728     | 44116.877          |
| Type_A            | 334070.412     | 46556.820          |
| Fuel_Price        | 235328.534     | 51785.721          |
| Super_Bowl_True   | 26413.466      | 10746.130          |
| Super_Bowl_False  | 26413.466      | 10746.130          |
| IsHoliday_False   | 14263.435      | 4812.539           |
| IsHoliday_True    | 14263.435      | 4812.539           |

### Time Series Analysis and Modeling

After performing time series decomposition and the augmented Dickey-Fuller test, we concluded that the data is nonstationary. Subsequent decomposition at weekly and monthly intervals revealed a repetitive pattern in the data.

To address nonstationarity, we applied difference, shift, and log algorithms. The differential data emerged as the most effective in achieving stationarity.

For the final time series model, we utilized the auto_arima algorithm, identifying the following as the optimal model for predictions:

**Best model:** ARIMA(3,0,2)(0,0,0)[1] intercept

**Total fit time:** 10.236 seconds



### **Conclusion:**
#### The predictions from the above model exhibit a slightly lower trend than the test data. Further tuning or exploring alternative algorithms may help achieve a closer alignment between the predictions and the test data.
