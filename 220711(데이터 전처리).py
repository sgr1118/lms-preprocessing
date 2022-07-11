#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csv_file_path = os.getenv('HOME')+'/aiffel/data_preprocess/data/trade.csv'
trade = pd.read_csv(csv_file_path) 
trade = trade
trade.head()


# # 10-2. 결측치

# In[3]:


print('전체 데이터 건수:', len(trade))


# In[13]:


#print('컬럼별 결측치 개수')
len(trade) - trade.count()
# print(trade.info())
#print(trade.isna().count())
# 기타 사항은 전부 결측치


# In[14]:


# 기타사항 삭제
trade = trade.drop(columns = ['기타사항'])
trade.info()


# In[17]:


trade[trade.isnull().any(axis=1)]


# In[18]:


trade.dropna(how='all', subset = ['수출건수', '수출금액', '수입건수', '수입금액', '무역수지'], inplace = True)
trade[trade.isnull().any(axis=1)]


# In[19]:


trade.loc[[188, 191, 194]]


# In[23]:


# 평균으로 결측치 채우기 (수출금액)
trade.loc[191, '수출금액'] = (trade.loc[188, '수출금액'] + trade.loc[194, '수출금액']) / 2
trade.loc[[191]]


# In[24]:


# 계산으로 결측치 채우기
trade.loc[191, '무역수지'] = (trade.loc[191, '수출금액'] - trade.loc[191, '수입금액'])
trade.loc[[191]]


# In[28]:


# 컬럼의 평균값을 구해서 결측치 채우기
print(round(trade['수출금액'].mean(),1))
# 6580880.8
trade.loc[191, '수출금액'] = 6580880.8
trade.loc[[191]]


# # 10-3. 중복된 데이터

# In[32]:


# 중복 확인
# trade[trade.duplicated()] index 186, 187 중복
trade[(trade['기간'] == '2020년 03월')&(trade['국가명'] == '중국')]


# In[34]:


# 중복 데이터 제거
trade.drop_duplicates(inplace=True)


# In[36]:


# DataFrame.drop_duplicates 자세히 알아보기
df = pd.DataFrame({'id':['001', '002', '003', '004', '002'], 
                   'name':['Park Yun', 'Kim Sung', 'Park Jin', 'Lee Han', 'Kim Min']})
df


# In[39]:


df.drop_duplicates(subset=['id'], keep='last')
df['id'].sort_values(ascending=True)


# # 10-4. 이상치 

# In[40]:


# z-score method
def outlier(df, col, z):
    return df[abs(df[col] - np.mean(df[col]))/np.std(df[col])>z].index


# In[45]:


# z-score method 계산
trade.loc[outlier(trade, '무역수지', 3)].count()


# In[47]:


# 이상치 아닌 값만 출력하기
def not_outlier(df, col, z):
    return df[abs(df[col] - np.mean(df[col]))/np.std(df[col]) <= z].index


# In[48]:


trade.loc[not_outlier(trade, '무역수지', 1.5)]


# In[49]:


# IQR method (사분위 범위수 활용)

np.random.seed(2020)
data = np.random.randn(100)  # 평균 0, 표준편차 1의 분포에서 100개의 숫자를 샘플링한 데이터 생성
data = np.concatenate((data, np.array([8, 10, -3, -5])))      # [8, 10, -3, -5])를 데이터 뒤에 추가함
data


# In[50]:


fig, ax = plt.subplots()
ax.boxplot(data)
plt.show()


# In[54]:


# IQR method  계산식
# IQR = Q3 - Q1 (3분위수 - 1분위수)
# Q3, Q1 = np.percentile(data, [75, 25])
#print(Q3, Q1) # 0.5166477538712722 -0.6478448291078243
IQR = Q3 - Q1
IQR # 1.1644925829790964


# In[55]:


data[(Q1-1.5*IQR > data)|(Q3+1.5*IQR < data)]


# In[57]:


# 이상치 실습
def outlier2(df, col):
    return df[col].quantile(0.75) - df[col].quantile(0.25)

outlier2(trade, '무역수지')


# # 10-5. 정규화

# In[67]:


# 정규분포를 따라 랜덤하게 데이터 x를 생성합니다. 
x = pd.DataFrame({'A': np.random.randn(100)*4+4,
                 'B': np.random.randn(100)-1})
x


# In[69]:


# 데이터 x를 Standardization 기법으로 정규화합니다. 
x_stand = (x - x.mean())/x.std()
x_stand


# In[70]:


# 데이터 x를 min-max scaling 기법으로 정규화합니다.
x_min_max = (x-x.min()) / (x.max() - x.min())
x_min_max


# In[72]:


# 스케일링 후 데이터 분호 확인
fig, axs = plt.subplots(1,2, figsize=(12, 4),
                        gridspec_kw={'width_ratios': [2, 1]})

axs[0].scatter(x['A'], x['B'])
axs[0].set_xlim(-5, 15)
axs[0].set_ylim(-5, 5)
axs[0].axvline(c='grey', lw=1)
axs[0].axhline(c='grey', lw=1)
axs[0].set_title('Original Data')

axs[1].scatter(x_stand['A'], x_stand['B'])
axs[1].set_xlim(-5, 5)
axs[1].set_ylim(-5, 5)
axs[1].axvline(c='grey', lw=1)
axs[1].axhline(c='grey', lw=1)
axs[1].set_title('Data after standardization')

plt.show()


# In[73]:


fig, axs = plt.subplots(1,2, figsize=(12, 4),
                        gridspec_kw={'width_ratios': [2, 1]})

axs[0].scatter(x['A'], x['B'])
axs[0].set_xlim(-5, 15)
axs[0].set_ylim(-5, 5)
axs[0].axvline(c='grey', lw=1)
axs[0].axhline(c='grey', lw=1)
axs[0].set_title('Original Data')

axs[1].scatter(x_min_max['A'], x_min_max['B'])
axs[1].set_xlim(-5, 5)
axs[1].set_ylim(-5, 5)
axs[1].axvline(c='grey', lw=1)
axs[1].axhline(c='grey', lw=1)
axs[1].set_title('Data after min-max scaling')

plt.show()


# In[75]:


# Standardization
# trade 데이터를 Standardization 기법으로 정규화합니다. 
# x_stand = (x - x.mean())/x.std()
cols = ['수출건수', '수출금액', '수입건수', '수입금액', '무역수지']
trade_Standard = (trade[cols]-trade[cols].mean()) / trade[cols].std()
# trade_Standard.head()
trade_Standard.describe()


# In[91]:


# trade 데이터를 min-max scaling 기법으로 정규화합니다. 
# x_min_max = (x-x.min()) / (x.max() - x.min())
trade[cols] = (trade[cols] - trade[cols].min()) / (trade[cols].max() - trade[cols].min())
trade.head()


# In[79]:


# sklearn을 활용한 Scaler
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train = [[10, -10], [30, 10], [50, 0]]
test = [[0, 1]]


# In[80]:


scaler.fit_transform(train)


# In[81]:


scaler.fit_transform(test)


# # 10-6 원-핫 인코딩 (One-Hot Encoding)

# In[92]:


# #trade 데이터의 국가명 컬럼 원본
#print(trade['국가명'].head())
# get_dummies를 통해 국가명 원-핫 인코딩
country = pd.get_dummies(trade['국가명'])
country.head()


# In[93]:


trade = pd.concat([trade, country], axis = 1)
trade.head()


# In[ ]:


# 필요없


# In[97]:


trade2 = pd.read_csv(csv_file_path)
trade2.info()


# In[101]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
trade2['국가명'] = encoder.fit_transform(trade2['국가명'])
trade2.head()


# In[ ]:





# # 10-7. 구간화

# In[118]:


salary = pd.Series([4300, 8370, 1750, 3830, 1840, 4220, 3020, 2290, 4740, 4600, 
                    2860, 3400, 4800, 4470, 2440, 4530, 4850, 4850, 4760, 4500, 
                    4640, 3000, 1880, 4880, 2240, 4750, 2750, 2810, 3100, 4290, 
                    1540, 2870, 1780, 4670, 4150, 2010, 3580, 1610, 2930, 4300, 
                    2740, 1680, 3490, 4350, 1680, 6420, 8740, 8980, 9080, 3990, 
                    4960, 3700, 9600, 9330, 5600, 4100, 1770, 8280, 3120, 1950, 
                    4210, 2020, 3820, 3170, 6330, 2570, 6940, 8610, 5060, 6370,
                    9080, 3760, 8060, 2500, 4660, 1770, 9220, 3380, 2490, 3450, 
                    1960, 7210, 5810, 9450, 8910, 3470, 7350, 8410, 7520, 9610, 
                    5150, 2630, 5610, 2750, 7050, 3350, 9450, 7140, 4170, 3090])
print("👽 Almost there..")


# In[103]:


# 데이터 구간 확인
salary.hist()


# In[104]:


# bins = [0, 2000, 4000, 6000, 8000, 10000]


# In[ ]:





# In[119]:


ctg = pd.cut(salary, bins=[0, 2000, 4000, 6000, 8000, 10000])
ctg


# In[108]:


print('salary[5]:', salary[5])
print('salary[5]가 속한 카테고리:', ctg[0])


# In[109]:


ctg.value_counts().sort_index()


# In[110]:


ctg = pd.cut(salary, bins=6)
ctg


# In[111]:


ctg.value_counts().sort_index()


# In[112]:


ctg = pd.qcut(salary, q=5)
ctg


# In[113]:


print(ctg.value_counts().sort_index())


# # 10-7. 데이터 전처리 총 복습

# In[122]:


csv_file_path = os.getenv('HOME')+'/aiffel/data_preprocess/data/vgsales.csv'
vgsales = pd.read_csv(csv_file_path) 
vgsales_copy = vgsales.copy()
vgsales.head()


# In[123]:


vgsales_copy.info()


# In[124]:


# 결측치 확인
vgsales_copy.isnull().sum() # year, Publisher


# In[127]:


vgsales_copy.dropna(how='any', subset=['Year', 'Publisher'], inplace = True)
vgsales_copy.isnull().sum()


# In[128]:


vgsales_copy.info()


# In[130]:


# 중복된 데이터 확인
vgsales_copy[vgsales_copy.duplicated()]
# 중복 데이터 없음


# In[141]:


# 이상치 확인 (IQR사용)
def outlier2(df, col):
    return df[col].quantile(0.75) - df[col].quantile(0.25)

#outlier2(vgsales_copy, 'NA_Sales') 0.24
#outlier2(vgsales_copy, 'EU_Sales') 0.11
#outlier2(vgsales_copy, 'JP_Sales') 0.04
#outlier2(vgsales_copy, 'Other_Sales') 0.04
Global_Sales_IQR = outlier2(vgsales_copy, 'Global_Sales') #0.42


# In[143]:


# 정규화
# 수치형 컬럼의 범위가 그렇게 크지않기 때문에 정규화가 필요없어보이지만 일단 한다
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
vgsales_copy = scaler.fit_transform


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




