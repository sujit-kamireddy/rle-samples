FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8080

# Base URL of the separately deployed harbor-server (../harbor-server). This
# has no default because a shim with no Harbor server to call cannot do
# anything useful -- fail at startup, not on the first rollout.
ENV HARBOR_SERVER_URL=""

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
