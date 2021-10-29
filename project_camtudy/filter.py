# def format_datetime(value, fmt='%Y년 %m월 %d일 %H:%M'):
#     return value.strftime(fmt)


import locale
import datetime

locale.setlocale(locale.LC_ALL, '')

now = datetime.datetime.now()

def format_datetime(value, fmt='%Y년 %m월 %d일 %H:%M'):
    return now.strftime(fmt)