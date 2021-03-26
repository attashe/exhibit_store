import schedule
import time
import subprocess

from loguru import logger


def job():
    logger.info('Job was run')
    time.sleep(5)

# =============
# test 1
# schedule.every(1).seconds.do(job)

# while 1:
#     schedule.run_pending()
#     time.sleep(0.1)

for i in range(5):
    subprocess.run(['python3', 'sleep.py'])
