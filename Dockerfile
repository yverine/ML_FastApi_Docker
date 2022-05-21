# 
FROM python:3.9

# 
WORKDIR /thecode

# 
COPY ./requirements.txt /thecode/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade -r /thecode/requirements.txt

# 
COPY . /thecode/app

#
EXPOSE 80

# 
CMD ["uvicorn", "deploy:app", "--host", "0.0.0.0", "--port", "80"]