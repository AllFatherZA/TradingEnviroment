import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import curve_fit
from scipy import signal
from  scipy.optimize import OptimizeWarning
import warnings
from finta import TA
import os
from utils import TradingGraph, Write_to_file
from datetime import date, datetime, timedelta
import math
from collections import deque
import neat
import pickle as pickle
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import random 
from sklearn.preprocessing import PolynomialFeatures

class CustomEnv:
     def __init__(self,df,Drawdown_P,Crystalball,Stoploss,LotSize,AccountSize,Decimals,Spread,Multiple,df_normalized,Convertion=1,initial_balance=5000,lookback_window_size=100,Render_range=100,show_reward=False,Show_indicators=False):
         self.df=df.reset_index()
         df.total_steps=len(self.df)-1
         self.initial_balance=initial_balance
         self.lookback_window_size=lookback_window_size
         self.Render_range=Render_range
         self.show_reward=show_reward
         self.Show_indicators=Show_indicators
         self.df_total_steps = len(self.df)-1        
         self.orders_history = deque(maxlen=self.lookback_window_size)
         self.Accurate=0
         self.df_normalized=df_normalized
         self.market_history = deque(maxlen=self.lookback_window_size)
         self.PNL=0
         self.TradesAnalyst=[0]
         self.Decimals=Decimals
         self.LotSize=LotSize
         self.AccountSize=AccountSize
         self.Spread=Spread
         self.Multiple=Multiple
         self.Convertion=Convertion
         self.Stoploss=Stoploss
         self.Drawdown_P=Drawdown_P
         self.Crystalball=Crystalball
         
         self.columns = list(self.df.columns[5:])
     
     def reset(self,env_steps_size=4320):
         self.visualization = TradingGraph(Render_range=self.Render_range, Show_reward=self.show_reward, Show_indicators=self.Show_indicators) # init visualization
         self.trades = deque(maxlen=self.Render_range)
         self.balance = self.initial_balance
         self.net_worth = self.initial_balance
         self.prev_episode_orders = 0
         self.prev_net_worth = self.initial_balance
         self.positions_held = 0
         self.Sells = 0
         self.Buys = 0
         self.Holding=False
         self.HoldValue=0
         self.OrderBuy=False
         self.OrderSell=False


         self.episode_orders = 0 # track episode orders count
         self.prev_episode_orders = 0 # track previous episode orders count
         self.env_steps_size = env_steps_size
         self.punish_value = 0
         if env_steps_size > 0: # used for training dataset
             self.start_step =self.df_total_steps - env_steps_size
             self.end_step = self.start_step + env_steps_size
         else: # used for testing dataset
             self.start_step = self.lookback_window_size
             self.end_step = self.df_total_steps
            
         self.current_step = self.start_step
         
         for i in reversed(range(self.lookback_window_size)):
            current_step = self.current_step - i
            self.orders_history.append([self.balance,
                                        self.net_worth,
                                        self.Buys,
                                        self.Sells,
                                        self.positions_held
                                        ])

            # one line for loop to fill market history withing reset call
            self.market_history.append([self.df.loc[current_step, column] for column in self.columns])
            
         state = np.concatenate((self.orders_history, self.market_history), axis=1)

         return self.df[self.current_step-self.Crystalball:self.current_step]
         
     def next_observation(self):
         #self.market_history.append([self.df.loc[self.current_step, column] for column in self.columns])
         #print(self.df.loc[self.current_step:,'Volume':'CloseDiff'])
         obs = np.concatenate((self.orders_history, self.market_history), axis=1)
         print('Observation',self.current_step-50)
         return self.df.loc[(self.current_step-self.Crystalball):self.current_step,'Open':'CloseDiff']
      
     def step(self, action):
            # Set the current price to a random price between open and close
            #current_price = random.uniform(
            #    self.df.loc[self.current_step, 'Open'],
            #    self.df.loc[self.current_step, 'Close'])
            current_price = self.df.loc[self.current_step, 'Close']
            Date = self.df.loc[self.current_step, 'Date'] # for visualization
            High = self.df.loc[self.current_step, 'High'] # for visualization
            Low = self.df.loc[self.current_step, 'Low'] # for visualization
            if self.OrderBuy==True:
                 Pnl =self.BuyPNLIndex(current_price)
                 if Pnl<-1*self.Stoploss:
                     self.balance+=Pnl
                     print("Stoploss Hit")
                     self.PNL+=Pnl
                     self.OrderBuy=False
                     self.trades.append({'Date' : Date, 'High' : High, 'Low' : Low, 'total': self.Buys, 'type': "close", 'current_price': current_price})
                     self.Holding=False
                     self.TradesAnalyst.append(Pnl)
                     
            if self.OrderSell==True:
                 Pnl =self.SellPNLIndex(current_price)
                 if Pnl<-1*self.Stoploss:
                     self.balance+=Pnl
                     print("Stoploss Hit")
                     self.PNL+=Pnl
                     self.OrderSell=False
                     self.trades.append({'Date' : Date, 'High' : High, 'Low' : Low, 'total': self.Buys, 'type': "close", 'current_price': current_price})
                     self.Holding=False
                     self.TradesAnalyst.append(Pnl)
                     
             
            if self.OrderBuy==True and action==-1:
             Pnl = self.BuyPNLIndex(current_price)
             self.balance+=Pnl
             print('Buy Close ...PNL:',Pnl)
             if Pnl>0:
                 self.Accurate+=1
             self.PNL+=Pnl
             self.OrderBuy=False
             self.trades.append({'Date' : Date, 'High' : High, 'Low' : Low, 'total': self.Buys, 'type': "close", 'current_price': current_price})
             self.Holding=False
             self.TradesAnalyst.append(Pnl)

            if self.OrderSell==True and action==1:
             Pnl = self.SellPNLIndex(current_price)
             self.trades.append({'Date' : Date, 'High' : High, 'Low' : Low, 'total': self.Buys, 'type': "close", 'current_price': current_price})
             self.balance+=Pnl
             self.PNL+=Pnl
             print('Sell Close ....PNL:',Pnl)
             if Pnl>0:
                 self.Accurate+=1
             self.OrderSell=False
             self.Holding=False
             self.TradesAnalyst.append(Pnl)

            if action == 1 and self.balance > self.initial_balance*self.Drawdown_P:
                # Buy with 0,1
                if self.OrderSell==False and self.Holding==False:
                    self.Buys +=1
                    self.trades.append({'Date' : Date, 'High' : High, 'Low' : Low, 'total': self.Buys, 'type': "buy", 'current_price': current_price})
                    self.episode_orders += 1
                    self.Holding=True
                    self.HoldValue=current_price
                    self.OrderBuy=True
                    print("Buy Opened")


            if action == -1 and self.balance > self.initial_balance*self.Drawdown_P:
                # Sell 0,1       
                if self.OrderBuy==False and self.Holding==False:
                    self.trades.append({'Date' : Date, 'High' : High, 'Low' : Low, 'total': self.Sells, 'type': "sell", 'current_price': current_price})
                    self.episode_orders += 1
                    self.Sells+=1
                    self.HoldValue=current_price
                    self.OrderSell=True
                    print("Sell Opened")

                
                
            


            if self.OrderBuy==True:
             Pnl = self.BuyPNLIndex(current_price)
             self.prev_net_worth = self.net_worth
             self.net_worth=self.balance+Pnl
             print("updating system...Currently Buy")
            if self.OrderSell==True:
             Pnl = self.SellPNLIndex(current_price)
             self.net_worth=self.balance+Pnl
             print("updating system...Currently Sell")

           
       
            
            self.orders_history.append([self.balance,
                                            self.net_worth,
                                            self.Buys,
                                            self.Sells,
                                            self.positions_held
                                            ])

            # Receive calculated reward

            self.current_step += 1
            if self.net_worth <= self.initial_balance*self.Drawdown_P:
                done = True
            else:
                done = False

            obs = self.next_observation()
            
            return obs, done
            

            
         # render environment
     def render(self, visualize = False):
        #print(f'Step: {self.current_step}, Net Worth: {self.net_worth}')
         if visualize:
            # Render the environment to the screen
                img = self.visualization.render(self.df.loc[self.current_step], self.net_worth, self.trades)
                return img
                
     def BuyPNL(self,current_price):
         return self.Convertion*round((math.trunc(current_price*self.Multiple)-math.trunc(self.HoldValue*self.Multiple)-self.Spread)*(self.LotSize*(self.Decimals*self.AccountSize)/current_price),2)
             
             
     def SellPNL(self,current_price):
         return self.Convertion*round((math.trunc(self.HoldValue*self.Multiple)-math.trunc(current_price*self.Multiple)-self.Spread)*(self.LotSize*(self.Decimals*self.AccountSize)/current_price),2)             
     
     def SellPNLIndex(self,current_price):
         return round(self.Convertion*(self.HoldValue-current_price-self.Spread)*self.LotSize*self.Multiple,2)
         
     def BuyPNLIndex(self,current_price):
         return round(self.Convertion*(current_price-self.HoldValue-self.Spread)*self.LotSize*self.Multiple,2)


def Random_games(env, visualize, test_episodes = 1, comment=""):
    Days=0
    average_net_worth = 0
    average_orders = 0
    no_profit_episodes = 0
    
    step=10/1440
    for episode in range(test_episodes):
        state = env.reset()
        while True:
            env.render(visualize)
            action = getaction(state)
            print('Position',action)
            state, done = env.step(action)
            os.system('cls')
            Days=Days+step
            print("Accuracy %: {}, \nnet_worth: {},  \naverage_profit_per_trade: {},  \norders: {},\nmax_profit: {},\nmax_loss: {},\nTotalAccurateTrades {},\nDays:{},Profit% {} :".format(env.Accurate/env.episode_orders, env.balance, env.PNL/env.episode_orders,
                                                                                                                                     env.episode_orders,max(env.TradesAnalyst),min(env.TradesAnalyst),env.Accurate,Days,(env.PNL/env.initial_balance)*100))
            if done ==True:
                 break

            if env.current_step == env.end_step:
                average_net_worth += env.net_worth
                average_orders += env.episode_orders
                if env.net_worth < env.initial_balance: no_profit_episodes += 1 # calculate episode count where we had negative profit through episode
                print("Accuracy %: {}, \nnet_worth: {}, average_profit_per_trade: {}, orders: {},max_profit: {},max_loss: {},TotalAccurateTrades {},Days {},Profit% {} :".format(env.Accurate/env.episode_orders, env.balance, env.PNL/env.episode_orders,
                                                                                                                                     env.episode_orders,max(env.TradesAnalyst),min(env.TradesAnalyst),env.Accurate,Days,(env.PNL/env.initial_balance)*100))
     
                prompt ="\nAccuracy %: {}, \nnet_worth: {}, \naverage_profit_per_trade: {}, \norders: {},\nmax_profit: {},\nmax_loss: {},\nTotalAccurateTrades {},\nDays {},\nProfit% {},\nInitial_balance :,\nLotSize : {},\nSpread,".format(env.Accurate/env.episode_orders, env.balance, env.PNL/env.episode_orders,
                                                                                                                                     env.episode_orders,max(env.TradesAnalyst),min(env.TradesAnalyst),env.Accurate,Days,(env.PNL/env.initial_balance)*100,env.initial_balance,env.LotSize,env.Spread)
                with open('Analysis.txt','w') as f:
                     f.writelines(prompt)
                
                break

 
def getaction(df):
    future_days =1
    df[str(future_days)+'_Day_Price_Forecast']=df[['CloseDiff']].shift(-future_days)
    df=df[['CloseDiff',str(future_days)+'_Day_Price_Forecast']]
    #print(df.tail())
    X=np.array(df[['CloseDiff']])
    X=X[:df.shape[0]-future_days]
    #print('Inputs',len(X))
    y=np.array(df[str(future_days)+'_Day_Price_Forecast'])
    y=y[:-future_days]
    
    
    #x=x.reshape(-1,1)
    poly=PolynomialFeatures(degree=4)
    
    X_poly=poly.fit_transform(X)
    print(X_poly)
    poly.fit(X_poly,y)
    
    LinReg=LinearRegression()
    LinReg.fit(X_poly,y)
    
    
    
    
    
    
    
    
    
    
    
    #print('Outputs',len(y))
    #ytest=np.array(df_tst[['Close']])
    #print('====================================')
    #svr_rbf=SVR(kernel='rbf',C=1e3,gamma=0.00001)
    #svr_rbf=LinearRegression()
    
    #svr_rbf.fit(X,y)

    svr_rbf_confidence=LinReg.score(X_poly,y)
    print("SVR ACCURACY",svr_rbf_confidence)
    y=np.array(df['CloseDiff'].iloc[-1])
    #print("Current value:",y)
    
    y2=0#LinReg.predict(X_poly[-1].reshape(-1,1))
    #print("Prediction:",y2)
    #df=df.drop(df.index[0])
    #print('end before',df.tail(1))
    #print('====================================')
    #df.loc[i,'Close']=df_tst.loc[i,'Close']
    #print('End after',df.tail(1))
    prediction=0
    if 0>y:
         prediction=1
         print("Buy prediction",y)
    elif 0<y:
         prediction=-1
         print("Sell prediction",y2)
         
    return prediction
    

df = pd.read_csv('../US30(10).csv')
df.columns=['Date','Open','High','Low','Close','Volume']
df['CloseDiff']=df['Close'].diff()
#r=random.randint(1,15000)
#r2=random.randint(1,20000)
df=df[3000:]
env=CustomEnv(df=df,Crystalball=1000,Drawdown_P=0.95,df_normalized=df,LotSize=0.02,AccountSize=1,Decimals=1,Spread=6,Multiple=1,Convertion=17.99,Stoploss=60)
Days=0
Random_games(env,visualize=True);

 
 