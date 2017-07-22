FROM python:3

RUN pip install jupyter
RUN pip install pandas 
RUN pip install  seaborn 
RUN pip install  matplotlib
RUN pip install   numpy 
RUN pip install mechanicalsoup
RUN pip install pathlib
RUN pip install requests
RUN pip install sklearn
RUN pip install https://github.com/pybrain/pybrain/archive/0.3.3.zip


RUN mkdir /assignment3pt2

WORKDIR /assignment3pt2

EXPOSE 8888

ADD Script_part2.py /assignment3pt2/
ADD Script_pt2.ipynb /assignment3pt2/
RUN chmod 755 /assignment3pt2/*

ENV PASSWD='sha1:98b767162d34:8da1bc3c75a0f29145769edc977375a373407824'
	
CMD /bin/bash -c 'jupyter notebook --no-browser --ip=* --NotebookApp.password="$PASSWD" "$@"'
