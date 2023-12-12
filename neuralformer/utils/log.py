import logging


def setup_logger(filename):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(filename)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


# from datetime import datetime
# from time import time


# class mylog(object):
#     def __init__(
#         self, logFile="log/logFile", fileOutput=True, screenOutput=True, reset=False
#     ):
#         self.__st = time()
#         self.__logFile = logFile
#         self.__fileOutput = fileOutput
#         self.__screenOutput = screenOutput
#         if reset:
#             f = open(logFile, "w")

#     def get_start(self):
#         return self.__st

#     def set_start(self, st):
#         self.__st = st

#     def reset(self):
#         self.__st = time()

#     def get_time(self):
#         return time() - self.__st

#     def set_fileOutput(self, fileOutput):
#         self.__fileOutput = fileOutput

#     def set_screenOutput(self, screenOutput):
#         self.__screenOutput = screenOutput

#     def set_logFile(self, logFile):
#         self.__logFile = logFile

#     def log(self, msg, fileOutput=None, screenOutput=None):
#         if fileOutput == None:
#             fileOutput = self.__fileOutput
#         if screenOutput == None:
#             screenOutput = self.__screenOutput

#         currentTime = self.get_time()

#         if fileOutput == True:
#             with open(self.__logFile, "a+", encoding="utf-8") as f:
#                 now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#                 f.write(now + " : " + msg + "\n")

#         if screenOutput == True:
#             now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             print(now, ":", msg)
