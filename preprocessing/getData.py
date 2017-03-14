import pandas as pd

time = []
speed = []
data = pd.read_csv('D:\Study\Project\Python\wind-speed-forecasting_v1\data\wind speed4.csv') 
# print(data.iloc[0, 2])
for index in range(len(data['rqsj'])):
	if data['rqsj'][index].split(':')[1] == '05' or data['rqsj'][index].split(':')[1] == '35':
		time.append(data['rqsj'][index])
		speed.append(data.iloc[index, 2])
data2 = pd.DataFrame({'time':time,
                      'speed':speed
	})
data2.to_csv('D:\Study\Project\Python\wind-speed-forecasting_v1\data\G4_speed.csv')
