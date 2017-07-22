FROM python:3

RUN pip install jupyter
RUN pip install pandas 
RUN pip install  seaborn 
RUN pip install  matplotlib
RUN pip install   numpy 
RUN pip install mechanicalsoup
RUN pip install pathlib


RUN mkdir /assignment3

WORKDIR /assignment3

EXPOSE 8888

ADD part1_EDA.ipynb /assignment3/
ADD part1_Script.py /assignment3/
ADD part1_run.sh /assignment3/
RUN chmod 755 /assignment3/*
	
CMD ["sh" , "/assignment3/part1_run.sh"]
