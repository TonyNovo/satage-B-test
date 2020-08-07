# satage-B-test
shows error 'Found input variables with inconsistent numbers of samples: [13814, 5921]'
# stage B test
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#url= https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv

energy_data=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv')
#energy_data.head(3)

column_names = { 'date': 'date_time','Appliances': 'appliances_(Wh)', 'lights': 'lights_(Wh)',
                'T1': 'kitchen_temperature_C', 'RH_1': 'kitchen_humidity_%', 
                'T2': 'living_room_temperature_C', 'RH_2': 'living_room_humidity_%', 
                'T3': 'laundry-room_temperature_C', 'RH_3': 'laundry_room_humidity_%', 
                'T4': 'office-room_temperature_C', 'RH_4': 'office_room_humidity_%', 
                'T5': 'bathroom_temperature_C', 'RH_5': 'bathroom_humidity_%', 
                'T6': 'northside_outdoor_temperature_C', 'RH_6': 'northside_outdoor_humidity_%', 
                'T7': 'ironing_room_temperature_C', 'RH_7': 'ironing_room_humidity_%', 
                'T8': 'teenager_room_temperature_C', 'RH_8': 'teenager_room_humidity_%', 
                'T9': 'parents_room_temperature_C', 'RH_9': 'parents_room_humidity_%', 
                'T_out': 'outside_temperature_C', 'RH_out': 'outside_humidity_%', 
                'Press_mm_hg': 'pressure_mm_hg', 'Windspeed ': 'windspeed_m/s', 
                'Visibility': 'visibility_km', 'Tdewpoint': 't-dew-point_A^o_C','rv1':'random_variable_1',
                'rv2':'random_variable_2'
               }
energy_data=energy_data.rename(columns=column_names)
energy_data.head(3)

#df=df.drop('date_time',axis=1)
energy_data.drop(['date_time'], axis=1, inplace=True)
energy_data.drop(['lights_(Wh)'], axis=1, inplace=True)
#analysing my model
from sklearn.preprocessing import MinMaxScaler
Scaler=MinMaxScaler()
normalised_data=pd.DataFrame(Scaler.fit_transform(energy_data), columns=energy_data.columns)
normalised_data.head(3)

# predictors
features = normalized_features.drop(['Appliances'], axis=1)

# target variable
target = normalized_features['Appliances']

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


features = normalised_data.drop(['appliances_(Wh)'], axis=1)

# target variable
target = normalised_data['appliances_(Wh)']

x_train,y_train,x_test,y_test= train_test_split(features, target,train_size=0.7, test_size=0.3, random_state=42)

linear_Reg= LinearRegression()
linear_Reg.fit(x_train,y_train)
values=linear_Reg.predict(x_test)
print(values)
