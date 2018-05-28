#!/bin/env python
# -*- coding: UTF-8 -*-

import requests
import pandas as pd
import datetime as dt
import os
from project.freshair.config.global_config import GlobalConfig

'''
API detail:
https://biendata.com/competition/airquality/{city}/{start_time}/{end_time}/2k0d1d8
https://biendata.com/competition/meteorology/{city}/{start_time}/{end_time}/2k0d1d8
https://biendata.com/competition/meteorology/{city}_grid/{start_time}/{end_time}/2k0d1d8
Example:
For Beijing:

Air Quality data:
https://biendata.com/competition/airquality/bj/2018-04-01-0/2018-04-01-23/2k0d1d8
Observed Meteorology:
https://biendata.com/competition/meteorology/bj/2018-04-01-0/2018-04-01-23/2k0d1d8
Meteorology Grid Data:
https://biendata.com/competition/meteorology/bj_grid/2018-04-01-0/2018-04-01-23/2k0d1d8
For London:

Air Quality data:
https://biendata.com/competition/airquality/ld/2018-04-01-0/2018-04-01-23/2k0d1d8
Meteorology Grid Data:
https://biendata.com/competition/meteorology/ld_grid/2018-04-01-0/2018-04-01-23/2k0d1d8
Here is a piece of code in Python 3 to retrieve data:
'''


def download_aq_meo(city='bj', type_data='aq', now_datetime=dt.datetime.utcnow(),
            api_dir=GlobalConfig.API_DATA_DIR, total_dir=GlobalConfig.API_TOTAL_DIR):
    '''
    包含两个功能，1. 把历史整天AQ保存，并合并到总表。2. 下载最近一天数据，并输出。
    :param city: bj or ld
    :param type: aq, meo, meo_grid
    :param now_datetime: 当前时刻。
    :param api_dir: 历史AQ文件加
    :param total_dir: 总表文件
    :return:
    '''
    if not os.path.exists(api_dir):  # 判断当前路径是否存在，没有则创建new文件夹
        os.makedirs(api_dir)
    if not os.path.exists(total_dir):
        os.mkdir(total_dir)

    path = api_dir + city + '_' + type_data + '/'
    total_file = total_dir + city + '_' + type_data + '_2018-4-1_now.csv'

    def download_save_read(stat_time, end_time, path, save_name=None):
        if save_name is None:
            save_name = time.strftime('%Y-%m-%d') + ".csv"
        if type_data == 'aq':
            url = 'https://biendata.com/competition/airquality/' + city + '/' + stat_time.strftime('%Y-%m-%d-%H') \
                  + '/' + end_time.strftime('%Y-%m-%d-%H') + '/2k0d1d8'
        elif type_data == 'meo':
            url = 'https://biendata.com/competition/meteorology/' + city + '/' + stat_time.strftime('%Y-%m-%d-%H') \
                  + '/' + end_time.strftime('%Y-%m-%d-%H') + '/2k0d1d8'
        elif type_data == 'meo_grid':
            url = 'https://biendata.com/competition/meteorology/' + city + '_grid/' + stat_time.strftime('%Y-%m-%d-%H') \
                  + '/' + end_time.strftime('%Y-%m-%d-%H') + '/2k0d1d8'
        else:
            raise 'type_data is error!'
        response = requests.get(url)
        with open(path + save_name, 'w') as f:
            f.write(response.text)
        # 数据标准化
        res = pd.read_csv(path + save_name, parse_dates=[2]).iloc[:, 1:]
        if type_data == 'aq':
            res.rename(columns={'PM25_Concentration': 'PM2.5', 'PM10_Concentration': 'PM10',
                                'NO2_Concentration': 'NO2', 'CO_Concentration': 'CO', 'O3_Concentration': 'O3',
                                'SO2_Concentration': 'SO2', 'time': 'utc_time'}, inplace=True)
        elif type_data == 'meo' or type_data == 'meo_grid':
            res.rename(columns={'time': 'utc_time'}, inplace=True)

        res = res.sort_values(['utc_time', 'station_id']).reset_index(drop=True)
        if type_data == 'aq' and city == 'bj':
            res = res[['station_id', 'utc_time', 'PM2.5', 'PM10', 'O3', 'NO2', 'CO', 'SO2']]
        if type_data == 'aq' and city == 'ld':
            res = res[['station_id', 'utc_time', 'PM2.5', 'PM10', 'NO2', 'CO', 'SO2']]
        res.to_csv(path + save_name, index=False)
        return res

    if not os.path.exists(path):  # 判断当前路径是否存在，没有则创建new文件夹
        os.makedirs(path)
    total_exist = True
    if not os.path.exists(total_file):
        total_exist = False
    total_opend = False
    for time in pd.date_range(dt.datetime(2018, 4, 1), now_datetime.date() - dt.timedelta(days=1)):
        if os.path.exists(path + time.strftime('%Y-%m-%d') + ".csv"):
            continue
        else:
            print('now we are getting', time, city, type_data)
            res = download_save_read(stat_time=time, end_time=time + dt.timedelta(hours=23), path=path)
            if total_exist:
                if not total_opend:
                    total_res = pd.read_csv(total_file, parse_dates=[1])
                    total_opend = True
                total_res = total_res.append(res)
                total_res.reset_index(drop=True, inplace=True)
            else:
                total_res = res.copy()
                total_opend = True
                total_exist = True
        total_res.to_csv(total_file, index=False)
    return download_save_read(stat_time=now_datetime.date(), end_time=now_datetime, path=path, save_name='temp.csv')


def download_meo_forecast(city='bj', now_datetime=dt.datetime.utcnow(),
                          api_dir=GlobalConfig.API_DATA_DIR, total_dir=GlobalConfig.API_TOTAL_DIR):
    path = api_dir + city + '_meo_fore/'
    total_file = total_dir + city + '_meo_fore_2018-4-11_now.csv'

    def download_save_read(time, path=path, city=city, save_name=None):
        if save_name is None:
            save_name = time.strftime('%Y-%m-%d') + ".csv"
        url = 'http://kdd.caiyunapp.com/competition/forecast/' + city + '/' \
              + (time - dt.timedelta(hours=1)).strftime('%Y-%m-%d') + '-23/2k0d1d8'
        response = requests.get(url)
        with open(path + save_name, 'w') as f:
            f.write(response.text)
        res = pd.read_csv(path + save_name, parse_dates=[2]).iloc[:, 1:]
        res.rename(columns={'forecast_time': 'utc_time'}, inplace=True)
        res = res.sort_values(['utc_time', 'station_id']).reset_index(drop=True)
        res = res[res.utc_time.dt.date == time.date()]
        if (res.wind_speed > 100).any():
            wind_d = res['wind_speed'].copy()
            res['wind_speed'] = res['wind_direction']
            res['wind_direction'] = wind_d
        res.to_csv(path + save_name, index=False)
        return res

    if not os.path.exists(path):  # 判断当前路径是否存在，没有则创建new文件夹
        os.makedirs(path)
    total_exist = True
    if not os.path.exists(total_file):
        total_exist = False
    total_opend = False
    if city == 'bj':
        start_time = dt.datetime(2018, 4, 11, 0)
    else:
        start_time = dt.datetime(2018, 4, 12, 0)
    for time in pd.date_range(start_time, now_datetime.date() - dt.timedelta(days=1)):

        if os.path.exists(path + time.strftime('%Y-%m-%d') + ".csv"):
            continue
        else:
            print('now we are getting', time.strftime('%Y-%m-%d'), city, 'moe forecast.')
            res = download_save_read(time)
            if total_exist:
                if not total_opend:
                    total_res = pd.read_csv(total_file, parse_dates=[1])
                    total_opend = True
                if total_res.utc_time.iloc[-1].date() < res.utc_time.iloc[0].date():
                    total_res = total_res.append(res)
                    total_res.reset_index(drop=True, inplace=True)
            else:
                total_res = res.copy()
                total_opend = True
                total_exist = True
        total_res.to_csv(total_file, index=False)
    return

def download_new_meo_forecast(city='bj', now_datetime=dt.datetime.utcnow(), api_dir=GlobalConfig.API_DATA_DIR):
    path = api_dir + city + '_meo_fore/'
    now_datetime = now_datetime - dt.timedelta(minutes=60)
    url = 'http://kdd.caiyunapp.com/competition/forecast/' + city + '/' + now_datetime.strftime('%Y-%m-%d-%H') + '/2k0d1d8'
    response = requests.get(url)
    with open(path + 'temp.csv', 'w') as f:
        f.write(response.text)
    res = pd.read_csv(path + 'temp.csv', parse_dates=[2]).iloc[:, 1:]
    res.rename(columns={'forecast_time': 'utc_time'}, inplace=True)
    res = res.sort_values(['utc_time', 'station_id']).reset_index(drop=True)
    res.to_csv(path + 'temp.csv', index=False)
    return res

def download_today_meo_forecast(city='bj', now_datetime=dt.datetime, path=GlobalConfig.API_TOTAL_DIR):
    path = path + city + '_meo_fore/'
    now_datetime = now_datetime.date()
    url = 'http://kdd.caiyunapp.com/competition/forecast/' + city + '/' \
          + (now_datetime - dt.timedelta(days=1)).strftime('%Y-%m-%d') + '-23/2k0d1d8'
    response = requests.get(url)
    with open(path + 'temp.csv', 'w') as f:
        f.write(response.text)
    res = pd.read_csv(path + 'temp.csv', parse_dates=[2]).iloc[:, 1:]
    res.rename(columns={'forecast_time': 'utc_time'}, inplace=True)
    res = res.sort_values(['utc_time', 'station_id']).reset_index(drop=True)
    res = res[res.utc_time.dt.date == now_datetime]
    res.to_csv(path + 'temp.csv', index=False)
    return res


if __name__ == '__main__':
    now_datetime = dt.datetime.utcnow()
    download_aq_meo(city='bj', type_data='aq', now_datetime=now_datetime)
    download_aq_meo(city='bj', type_data='meo', now_datetime=now_datetime)
    download_aq_meo(city='bj', type_data='meo_grid', now_datetime=now_datetime)
    download_meo_forecast(city='bj', now_datetime=now_datetime)
    # download_meo_forecast(now_datetime=now_datetime, city='ld')