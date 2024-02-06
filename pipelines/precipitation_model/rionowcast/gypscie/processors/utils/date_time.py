from datetime import datetime, date
from dateutil import parser
from dateutil.relativedelta import relativedelta


def is_valid_date(str_date: str):
    try:
        parser.parse(str_date).date()
        return True
    except:
        return False


def day_of_year(date_obj: datetime or date):
    assert isinstance(date_obj, (datetime, date))
    return date_obj.timetuple().tm_yday


def str_day_of_year(date_obj: datetime or date,
                    num_digits=3):
    assert isinstance(date_obj, (datetime, date))
    str_day = str(day_of_year(date_obj))
    return str_day.zfill(num_digits)


def parse_to_date(str_date: str):
    try:
        return parser.parse(str_date).date()
    except Exception as e:
        print(e)
        
    
def parse_to_datetime(str_date: str):
    try:
        return parser.parse(str_date)
    except Exception as e:
        print(e)
  

def split_date(dt: date):
    return [dt.year, dt.month, dt.day]


def today():
    return datetime.today().date()
  
  
def next_day(dt: date):
    return dt + relativedelta(days=+1)
