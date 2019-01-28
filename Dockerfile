FROM tensorflow/tensorflow:latest-gpu 


RUN mkdir -p /subspace/
RUN mkdir -p /subspace/Data/
RUN mkdir -p /subspace/models/
RUN mkdir -p /subspace/models_DSC/
RUN mkdir - p /subspace/logs/
RUN pip install munkres 

COPY ./Data /subspace/Data/
COPY ./models /subspace/models/
COPY *.py /subspace/

ENV MAT EYB_fc
ENV MODEL EYBfc_af

WORKDIR /subspace/

CMD ["sh", "-c", "python affinity_fusion.py --mat ${MAT} --model ${MODEL} && /bin/bash"]
