FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS build
COPY ./3rd-party/repo /work
WORKDIR /work
RUN apt update && apt install cmake make -y && rm -rf /var/lib/apt/lists/*
RUN cmake -B build -DCMAKE_CXX_COMPILER=g++
RUN cmake --build build -j $(nproc)
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS runtime
WORKDIR /work
COPY --from=build /work/build/vanity_torv3_cuda /work/vanity
ENTRYPOINT ["./vanity"]