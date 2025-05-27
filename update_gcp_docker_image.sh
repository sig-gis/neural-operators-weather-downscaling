docker build -t downscaling .
docker tag downscaling:latest us-west1-docker.pkg.dev/eofm-benchmark/climate-downscaling/downscaling:latest
docker push us-west1-docker.pkg.dev/eofm-benchmark/climate-downscaling/downscaling:latest