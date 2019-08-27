#!/usr/bin/env python
# coding: utf-8

# ## 練習時間
# #### 請寫一個函式用來計算 Mean Square Error
# $ MSE = \frac{1}{n}\sum_{i=1}^{n}{(Y_i - \hat{Y}_i)^2} $
# 
# ### Hint: [如何取平方](https://googoodesign.gitbooks.io/-ezpython/unit-1.html)

# # [作業目標]
# - 仿造範例的MAE函數, 自己寫一個MSE函數(參考上面公式)

# # [作業重點]
# - 注意程式的縮排
# - 是否能將數學公式, 轉換為 Python 的函式組合? (In[2], Out[2])

# In[ ]:


# 載入基礎套件與代稱
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


def mean_absolute_error(y, yp):
    """
    計算 MAE
    Args:
        - y: 實際值
        - yp: 預測值
    Return:
        - mae: MAE
    """
    mae = MAE = sum(abs(y - yp)) / len(y)
    return mae

# 定義 mean_squared_error 這個函數, 計算並傳回 MSE
def mean_squared_error(yp , y):
    """
    請完成這個 Function 後往下執行
    """
    return sum(pow(y - yp , 2))


# In[ ]:


# 與範例相同, 不另外解說
w = 3
b = 0.5
x_lin = np.linspace(0, 100, 101)
y = (x_lin + np.random.randn(101) * 5) * w + b

plt.plot(x_lin, y, 'b.', label = 'data points')
plt.title("Assume we have data points")
plt.legend(loc = 2)
plt.show()


# In[ ]:


# 與範例相同, 不另外解說
y_hat = x_lin * w + b
plt.plot(x_lin, y, 'b.', label = 'data')
plt.plot(x_lin, y_hat, 'r-', label = 'prediction')
plt.title("Assume we have data points (And the prediction)")
plt.legend(loc = 2)
plt.show()


# In[ ]:


# 執行 Function, 確認有沒有正常執行
MSE = mean_squared_error(y, y_hat)
MAE = mean_absolute_error(y, y_hat)
print("The Mean squared error is %.3f" % (MSE))
print("The Mean absolute error is %.3f" % (MAE))


# # [作業2]
# 
# 請上 Kaggle, 在 Competitions 或 Dataset 中找一組競賽或資料並寫下：
# 
# 1. 你選的這組資料為何重要
# 
# 2. 資料從何而來 (tips: 譬如提供者是誰、以什麼方式蒐集)
# 
# 3. 蒐集而來的資料型態為何
# 
# 4. 這組資料想解決的問題如何評估
#
# Data set : Medical Cost Personal Datasets
#
# A1 : 因為這組資料刻劃了美國人的醫療保險費用，並給了我們諸如病患的性別、bmi、是否吸菸、所屬地區等
#      與醫療費用息息相關的資料。重要的議題加上不會乾淨、完整的資料，我認為這筆資料很適合拿來練習分析。
#      
# A2 : 資料的提供者 : Miri Choi，資料來源未知
#
# A3 : 資料的型態為 .csv，裡面的變數部分是連續型的(歲數、bmi)，部分是離散型的(性別、是否吸菸)
#
# A4 : 我覺得可以用 MAE、RMSE來評估，因為這組 data set 的作者想要看看能不能正確的預測醫療保險的金額，
#      也就是說，這應該是個回歸問題，而回歸問題最常見的評估方法應該是用MAE、RMSE，所以我覺得可以先嘗試
#      這兩個評估方法
#
# # [作業3]
# 
# 想像你經營一個自由載客車隊，你希望能透過數據分析以提升業績，請你思考並描述你如何規劃整體的分析/解決方案：
# 
# 1. 核心問題為何 (tips：如何定義 「提升業績 & 你的假設」)
# 
# 2. 資料從何而來 (tips：哪些資料可能會對你想問的問題產生影響 & 資料如何蒐集)
# 
# 3. 蒐集而來的資料型態為何
# 
# 4. 你要回答的問題，其如何評估 (tips：你的假設如何驗證)
#
# A1 : 如何提升載客率、如何提升客戶黏著度、如何讓正在使用其他載客車隊的客戶轉向我們、如何開發不常使用載客服務的客群
#      如何利用數據進行定價、如何基於數據來管理整個車隊
#
# A2 : 資料從內部過去的載客資料蒐集(比如說 : 如果有app的話，蒐集各個帳號的年齡、性別、工作、常出現的地方等資料)，
#      問卷調查、買資料、其他資料(視個別的case去爬)
#
# A3 : 看標籤，部分是連續的資料，部分是離散的資料，我們也可以視情況把連續的資料作離散化
#
# A4 : 看上述的核心問題有沒有被解決、公司的業績是否有有效提升、知名度是否有提升、註冊人數是否變多
