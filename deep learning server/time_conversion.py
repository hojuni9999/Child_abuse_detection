import time
import datetime

def convert_datetime(unixtime):
    """Convert unixtime to datetime"""
    date = datetime.datetime.fromtimestamp(unixtime).strftime('%Y-%m-%d %H_%M_%S')
    return date # format : str

def convert_unixtime(date_time):
    """Convert datetime to unixtime"""
    unixtime = datetime.datetime.strptime(date_time,'%Y-%m-%d %H_%M_%S').timestamp()
    return unixtime
