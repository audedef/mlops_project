
# src/webapp/Dockerfile
FROM python:3.9

WORKDIR /app

COPY app.py .

RUN pip install streamlit torchvision pillow \
    torch==2.7.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu


EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=127.0.0.1"]

