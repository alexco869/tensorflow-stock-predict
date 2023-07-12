from  stock_ai import stock_ai
from datetime import datetime


obj = stock_ai('GOOGL', datetime(2023, 7, 10), 3650)

obj.update()

obj.fill_deltas()

obj.train()

obj.test()