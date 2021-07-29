FROM eosineye1/gpt2-prideandprejudice:1

WORKDIR /app

RUN pip install flask transformers torch

COPY . .

CMD ["python3", "app.py"]
