## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
       ```
       import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
![image](https://github.com/user-attachments/assets/56af7505-8adb-4644-9370-a2cd72069c93)

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/2e4e7373-8f5f-497d-976f-e48640c42e7d)

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/user-attachments/assets/9134c10c-27ae-42a6-95be-d869ba61d6a1)
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/73bafa07-daac-4929-a4dc-bdd5ccaec9ab)

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
```
```
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/67c4882c-7dd4-4ca5-a28f-0c9247e6cbf0)

```
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/user-attachments/assets/9ec42b14-f9e5-4cdd-8cbb-38c30d68b15d)

```
pip install --upgrade category_encoders
````
```
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
```
```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```
```
dfb=pd.concat([df,nd],axis=1)
dfb
```
![image](https://github.com/user-attachments/assets/9a083768-1f47-4fae-ada7-907bc842c62e)

```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
````

![image](https://github.com/user-attachments/assets/4057e8ba-b0e9-454d-98b7-379fb37125bd)

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```

![image](https://github.com/user-attachments/assets/f8da2b8d-5cf9-477b-a08b-b3a2bb7ee7b4)

```
df.skew()
````

![image](https://github.com/user-attachments/assets/3d774b98-186a-4e76-8724-2b1bf81a2ac8)

```
np.log(df["Highly Positive Skew"])
````

![image](https://github.com/user-attachments/assets/14811e81-c7b4-4581-8980-bfbc0de49a56)

````
np.reciprocal(df["Moderate Positive Skew"])
````

![image](https://github.com/user-attachments/assets/d02b1982-e1d3-4a3a-88ae-c5ba93ae4ac1)

```
np.sqrt(df["Highly Positive Skew"])
```
```
np.square(df["Highly Positive Skew"])
```
````
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```

![image](https://github.com/user-attachments/assets/3c2b78b4-07b4-4762-9c80-9a3eb01a0dc1)

```
df.skew()
```

![image](https://github.com/user-attachments/assets/0724a800-79a0-4f6a-89f8-6d27af8e9e68)

```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```

![image](https://github.com/user-attachments/assets/7916609f-9538-4b20-a4cc-e0a7926ba819)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```

![image](https://github.com/user-attachments/assets/8fc153de-ae67-4570-b08c-bcc13481fd9c)

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/778497b6-78ca-4426-b95f-3723e7f14092)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/3959c2d3-bd64-4ecf-b737-230e24cf39fc)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
````

![image](https://github.com/user-attachments/assets/15c3a214-a5aa-4b71-b91f-03c9dab9aa25)

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```


![image](https://github.com/user-attachments/assets/0d7563e8-59e8-4325-b6ce-63817ed858d2)


```
dt=pd.read_csv("titanic_dataset.csv")
dt
````
````
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45') 
plt.show()
```

![image](https://github.com/user-attachments/assets/c572a2fc-4005-4079-accf-de896f1520c7)

```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```


![image](https://github.com/user-attachments/assets/4b17f763-9f2a-4fd5-9eb6-200ff946f654)


# RESULT:
      Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.

       
