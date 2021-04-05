FROM pytorch/torchserve
COPY model-store/vectors.txt /home/model-server/model-store/vectors.txt
COPY model-store/biobert_batch.mar /home/model-server/model-store/biobert_batch.mar

RUN pip install numpy

EXPOSE 8080/tcp
EXPOSE 8080/udp
EXPOSE 8081/tcp
EXPOSE 8081/udp

WORKDIR "/home/model-server/"
CMD ["torchserve", "--start", "--ncs", "--model-store", "model-store", "--models", "biobert_batch.mar"]
