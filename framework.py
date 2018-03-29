from yahoo_finance import Share
from pprint import pprint
from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import datetime as dt
from scipy.fftpack import idct, dct
import numpy as np
import itertools
from matplotlib.dates import DateFormatter, WeekdayLocator,DayLocator, MONDAY

class Stocks():
    def __init__(self):
        #self.startTime = 1
        self.comp_tick = {}
        self.close = {}
        self.cos_trans = {}
        self.mov_avg = {}
        self.grad = {}
        self.grad2 = {}
        self.tickers = []
        #most recent
        self.first_date = '2017-08-04'
        self.last_date =  '2016-08-04'
        self.my_stocks = []
    def fill(self):
        self.close = pickle.load( open('sp500.pickle', "rb"))
        self.cos_trans = pickle.load( open('cos_trans.pickle', "rb"))
        self.mov_avg = pickle.load( open('moving_averages.pickle', "rb"))
        self.comp_tick = pickle.load( open('tick_and_comp.pickle', "rb"))
        self.grad = pickle.load( open('grads.pickle', "rb"))
        self.grad2 = pickle.load( open('2grads.pickle', "rb"))
        self.tickers = pickle.load( open('selfticks.pickle', "rb"))

    def save(self):
        pickle.dump(self.close, open('sp500.pickle', "wb"))
        pickle.dump(self.cos_trans, open('cos_trans.pickle', "wb"))
        pickle.dump(self.mov_avg, open('moving_averages.pickle', "wb"))
        pickle.dump(self.comp_tick, open('tick_and_comp.pickle', "wb"))
        pickle.dump(self.grad, open('grads.pickle', "wb"))
        pickle.dump(self.grad2, open('2grads.pickle', "wb"))
        pickle.dump(self.tickers, open('selfticks.pickle', "wb"))
        
    def set_my_stocks(self, list_of_stock):
        self.my_stocks = list_of_stock
        
    def get_my_stocks(self):
        return self.my_stocks
    
    def add_preference(self, stock):
        self.my_stocks.append(stock)

    def new_company_list(self, comps, tickers):
        insert = list(zip(comps, tickers))
        for i in insert:
            self.comp_tick[i[0]] = i[1]
            self.tickers.append(i[1])
        pickle.dump(self.comp_tick, open('tick_and_comp.pickle', "wb") )
            
    def populate(self):
        self.datacollection(self.tickers)
        self.moving_averages(self.tickers)
        self.dct(self.tickers)
        self.gradients(self.tickers)

    def datacollection(self, tickers): 
        #strings in the form '2016-7-30'
        data_source = 'google'
        # We would like all available data from 01/01/2000 until 12/31/2016.
        # User pandas_reader.data.DataReader to load the desired data. As simple as that.
        panel_data = data.DataReader(tickers, data_source, self.last_date, self.first_date)
        # Getting just the adjusted closing prices. This will return a Pandas DataFrame
        # The index in this DataFrame is the major index of the panel_data.
        close = panel_data['Close']
        all_weekdays = pd.date_range(start=self.last_date, end=self.first_date, freq='B')
        original_close = close.reindex(all_weekdays)
        original_close.fillna(method = 'bfill', inplace=True)
        for tick in tickers:
            self.close[tick] = original_close[tick]
        pickle.dump(self.close, open('sp500.pickle', "wb") )
        
    def moving_averages(self, tickers):
        for i in tickers:
            l = []
            l.append(np.round(self.close[i].rolling(window = 5, center = False).mean(), 2))
            l.append(np.round(self.close[i].rolling(window = 10, center = False).mean(), 2))
            l.append(np.round(self.close[i].rolling(window = 20, center = False).mean(), 2))
            l.append(np.round(self.close[i].rolling(window = 50, center = False).mean(), 2))
            l.append(np.round(self.close[i].rolling(window = 100, center = False).mean(), 2))
            self.mov_avg[i] = l
        pickle.dump(self.mov_avg, open('moving_averages.pickle', "wb") )
        
    def dct(self, tickers):
        for i in tickers:
            data1 = self.close[i]
            l = []
            ckD =  dct(data1, norm = 'ortho')
            ckD[round(.2 * len(ckD)):] = 0
            smoothF = idct(ckD, norm = 'ortho')
            l.append(smoothF) 
            ckD =  dct(data1, norm = 'ortho')
            ckD[round(.1 * len(ckD)):] = 0
            smoothF = idct(ckD, norm = 'ortho')
            l.append(smoothF) 
            ckD =  dct(data1, norm = 'ortho')
            ckD[round(.05 * len(ckD)):] = 0
            smoothF = idct(ckD, norm = 'ortho')
            l.append(smoothF) 
            ckD =  dct(data1, norm = 'ortho')
            ckD[round(.02 * len(ckD)):] = 0
            smoothF = idct(ckD, norm = 'ortho')
            l.append(smoothF) 
            ckD =  dct(data1, norm = 'ortho')
            ckD[round(.01 * len(ckD)):] = 0
            smoothF = idct(ckD, norm = 'ortho')
            l.append(smoothF) 
            self.cos_trans[i] = l
        pickle.dump(self.cos_trans, open('cos_trans.pickle', "wb") )
        
    def gradients(self, tickers):
        for i in tickers:
            l = []
            m = []
            stock = self.
            trans[i]
            for cos in range(len(stock)):
                grad = np.gradient(np.asarray(stock[cos]))
                l.append(grad)
                grad2 = np.gradient(grad)
                m.append(grad2)
            self.grad[i] = l
            self.grad2[i] = m
        pickle.dump(self.grad, open('grads.pickle', "wb") )
        pickle.dump(self.grad2, open('2grads.pickle', "wb") )

    def search(self, stock, date = None, end = None, start = None):
        stock = stock.lower()
        #start is the most recent date as a string
        if end is not None:
            if len(end) != 10:
                end = dt.datetime.strptime(end + '-0', "%Y-W%W-%w")
            #if (datetime.strptime(self.first_date, '%Y-%m-%d') - datetime.strptime(end, '%Y-%m-%d')).days() < 0:
             #   end = 
        if stock not in self.my_stocks:
            self.my_stocks.append(stock)
        
        symbol = self.comp_tick[stock]

        ##################################### GIVES PRICE ON DATE OR OF LAST CLOSE ############################################
        if date is not None:
            print1 = self.close[symbol][date]
        else:
            print1 = self.close[symbol][-1]

        #################################  THIS AREA GIVES MAX CLOSE OVER RANGE OF TIME ###################################
        if start is not None and end is not None:
            start_datetime = dt.datetime.strptime(start, '%Y-%m-%d')
            end_datetime = dt.datetime.strptime(end, '%Y-%m-%d')
            number_of_days = (start_datetime - end_datetime).days
            days_from_start = (dt.datetime.strptime(self.first_date, '%Y-%m-%d') - start_datetime).days
            designated_closes = np.asarray(self.close[symbol][-days_from_start - number_of_days: -days_from_start])
            print2 = designated_closes.max()
            print2days = number_of_days
        elif end is not None:
            end_datetime = dt.datetime.strptime(end, '%Y-%m-%d')
            number_of_days = (dt.datetime.strptime(self.first_date, '%Y-%m-%d') - end_datetime).days
            designated_closes = np.asarray(self.close[symbol][-number_of_days:])
            print2 = designated_closes.max()
            print2days = number_of_days
        else:
            print(self.first_date)
            print2 = self.close[symbol].max()
            #the last this many days
            s = dt.datetime.strptime(self.first_date, '%Y-%m-%d')
            l = dt.datetime.strptime(self.last_date, '%Y-%m-%d')
            print2days = (s - l).days


        print3 = self.mov_avg[symbol]
        if self.mov_avg[symbol][-2][0] - self.mov_avg[symbol][-2][130] <= 0.0 and self.mov_avg[symbol][-2][130] - self.mov_avg[symbol][-2][-1] <= 0.0:
            print5 = 'trending in the last year'
        elif self.mov_avg[symbol][-2][0] - self.mov_avg[symbol][-2][130] >= 0.0 and self.mov_avg[symbol][-2][130] - self.mov_avg[symbol][-2][-1] >= 0.0:
            print5 = 'declining in popularity in the last year'
        else:
            print5 = 'varying in popularity. Maybe check out news buddy to find out why!'
        points = self.grad[symbol][3]
        slope = (points[-8], points[-4], points[-1])
        if slope[2] - slope[1] >= 0.0 and slope[1] >= slope[0]:
            print4 = 'bullish'
        elif slope[0] >= slope[1] and slope[1] >= slope[2]:
            print4 = 'bearish'
        else:
            print4 = 'volatile'
        ##################################### 
        return (print1,print2, print2days, print3, print4, print5)
    
    def my_companies(self):
        all_stocks = []
        for stock in self.my_stocks:
            all_stocks.append(self.search(stock))
        return all_stocks