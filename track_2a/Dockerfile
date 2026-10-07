FROM python:3.11.14-slim-bookworm@sha256:65a93d69fa75478d554f4ad27c85c1e69fa184956261b4301ebaf6dbb0a3543d
WORKDIR /app
COPY requirements-pdf.lock ./
RUN --mount=type=secret,id=cloud_ca \
    if [ -f /run/secrets/cloud_ca ]; then export PIP_CERT=/run/secrets/cloud_ca; fi; \
    pip install --no-cache-dir --no-deps --require-hashes -r requirements-pdf.lock
COPY pyproject.toml ./
COPY src ./src
COPY deployment ./deployment
COPY tests ./tests
COPY data/example-booklet.json ./data/example-booklet.json
RUN chmod -R a+rX /app
ENV PYTHONPATH=/app/src PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
USER 65534:65534
EXPOSE 8000
ENTRYPOINT ["python", "-m", "ost_nli"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
