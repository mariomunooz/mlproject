# Logger is for the purpose that any execution that probably happens we should be able 
# to log it into files. This is a good practice to keep track of the execution of the code.
import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), 'logs', LOG_FILE)
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)


#Format argument defines the format of each log entry. The placeholders in the string are automatically replaced by the logging system:
#
#    %(asctime)s: The timestamp of when the log entry was created.
#    %(lineno)d: The line number in the source code where the log was generated.
#    %(name)s: The name of the logger (usually the module or file name).
#    %(levelname)s: The severity level of the log message (e.g., INFO, DEBUG, ERROR).
#    %(message)s: The actual log message that was passed to the logger (via methods like logging.info() or logging.error()).

logging.basicConfig(
    filename=LOG_FILE_PATH, 
    format='[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    )

#if __name__ == '__main__':
#    logging.info("Logging is working fine")
   
    

