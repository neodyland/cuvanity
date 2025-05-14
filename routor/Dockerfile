FROM alpine:latest
RUN apk add --no-cache tor
WORKDIR /work
COPY ./entrypoint.sh /work/entrypoint.sh
RUN chmod +x ./entrypoint.sh
ENTRYPOINT ["/work/entrypoint.sh"]