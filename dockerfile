FROM python:3.11-slim
   
##Environmental variables 
ENV PYTHONDONTWRITEBYTECODE=1 \
PYTHONUNBUFFERED=1       

#Working directory
WORKDIR /app

# Installing system dependencies 
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        # for Aws cli 
        awscli \
    && rm -rf /var/lib/apt/lists/*


# Project Files copying 

COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt


# Expose Streamlit port
EXPOSE 8501



# Addding src to Python path
ENV PYTHONPATH="${PYTHONPATH}:/app/src"


#  MAIN APP the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]



