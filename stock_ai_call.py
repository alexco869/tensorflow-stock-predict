from  stock_ai import stock_ai
from datetime import datetime


obj = stock_ai('V', datetime(2023, 7, 12), 3650)
print(" GO TO UPDATE")
obj.update()

obj.fill_deltas()

obj.train(3000)

obj.test()

obj.evaluate()