FROM ghcr.io/astral-sh/uv:debian-slim
WORKDIR /work
COPY ./3rd-party/repo/util /work
RUN uv python install $(cat .python-version)
RUN uv venv && uv pip install -r requirements.txt -U
ENV PATH="/work/.venv/bin:$PATH"
ENTRYPOINT [ "uv", "run", "python", "genpubs.py" ]