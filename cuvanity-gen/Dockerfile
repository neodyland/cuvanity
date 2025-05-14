FROM ghcr.io/astral-sh/uv:alpine
WORKDIR /work
COPY ./3rd-party/repo/util /work
RUN uv python install $(cat .python-version)
RUN uv pip install -r requirements.txt --system -U
ENTRYPOINT [ "uv", "run", "python", "genpubs.py" ]