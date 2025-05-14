FROM ghcr.io/astral-sh/uv:alpine
WORKDIR /work
COPY ./3rd-party/repo/util /work
RUN uv pip install -r requirements.txt
ENTRYPOINT [ "uv", "run", "python", "genpubs.py" ]