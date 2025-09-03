FROM python:3.11-slim 

WORKDIR /api

COPY src/api .

RUN pip install -r requirements.txt

RUN mkdir -p /api/models/trained

COPY models/trained/*.pkl /api/models/trained/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
