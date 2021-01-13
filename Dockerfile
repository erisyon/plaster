FROM buildpack-deps:focal AS flann-build
RUN apt-get -qq update && DEBIAN_FRONTEND=noninteractive apt-get -qq -y install \
    cmake \
    libgtest-dev \
    libhdf5-dev \
    liblz4-dev

# We do not need the entire git history for this project; thus --depth 1
RUN cd / && git clone --depth 1 https://github.com/mariusmuja/flann.git
RUN touch /flann/src/cpp/empty.cpp
RUN cd /flann/src/cpp && sed -i 's/SHARED ""/SHARED "empty.cpp"/g' CMakeLists.txt
RUN echo "#pragma GCC diagnostic ignored \"-Wmisleading-indentation\"" | cat - /flann/src/cpp/flann/util/any.h > /tmp/out && mv /tmp/out /flann/src/cpp/flann/util/any.h
RUN cd /flann && cmake . && make flann

# STAGE 0.2: Setup the lmfits library
# --------------------------------------------------------------------------------------------------------
FROM buildpack-deps:focal AS lmfits-build
RUN apt-get -qq update && DEBIAN_FRONTEND=noninteractive apt-get -qq -y install \
    libblas-dev \
    liblapack-dev \
    f2c \
    libgfortran5

RUN cd / && git clone --depth 1 https://github.com/zsimpson/zbs.lmfits.git && cd /zbs.lmfits && git checkout 8c4c4131d888049c1725f0fc62ae975212459979
RUN cd /zbs.lmfits/levmar-2.6 && ENV_OPTS="-fPIC" make
RUN cd /zbs.lmfits && LEVMAR_FOLDER=/zbs.lmfits/levmar-2.6 DST_FOLDER=/zbs.lmfits ./build.sh

# STAGE 1: Install apt packages; base for images that want to run plaster jobs
# --------------------------------------------------------------------------------------------------------
FROM ubuntu:20.04 AS plaster-base

ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV USER=root
USER root
WORKDIR /root

COPY ./scripts/apt_packages.txt /erisyon/plaster/scripts/apt_packages.txt

RUN apt-get -qq update \
    && { \
    cat /erisyon/plaster/scripts/apt_packages.txt \
    | grep -o '^[^#]*' | xargs apt-get -qq -y install --no-install-recommends ; \
    } \
    && update-alternatives --quiet --install /usr/bin/pip pip /usr/bin/pip3 1 \
    && update-alternatives --quiet --install /usr/bin/python python /usr/bin/python3 1 \
    && pip install pipenv \
    && rm -rf /var/lib/apt/lists/* \
    && locale-gen "en_US.UTF-8"

# Copy over the FLANN files; once erisyon.plaster doesn't trigger compilation during
# installation we can remove these
COPY --from=flann-build /flann/lib /flann/lib
COPY --from=flann-build /flann/src/cpp/flann /flann/src/cpp/flann/
COPY --from=lmfits-build /zbs.lmfits /zbs.lmfits
ENV LD_LIBRARY_PATH="/flann/lib:/zbs.lmfits:${LD_LIBRARY_PATH}"


# STAGE 2: Install pip packages with build tools into venv; pluck the venv from this image
# if you need any element of our python environment
# --------------------------------------------------------------------------------------------------------
FROM plaster-base AS plaster-with-pips

# Add build tools so that source distros can build
RUN apt-get -qq update && apt-get -qq -y install --no-install-recommends build-essential

# Have pipenv build the venv into /.venv
# so that we can pluck it out in later stage
# and eliminate the build tools in this stage
WORKDIR /venv
COPY ./Pipfile ./Pipfile
COPY ./Pipfile.lock ./Pipfile.lock
RUN PIPENV_VENV_IN_PROJECT=1 pipenv sync --dev --python /usr/bin/python


# Stage 3: Copy over the venv, source, and infra tools into the final developer image
# --------------------------------------------------------------------------------------------------------
FROM plaster-base AS final

# See the comment near _MOUNT_SUDOERS in p
# The container must grant sudoers to all by default
RUN echo "root ALL=(ALL:ALL) ALL" > /etc/sudoers
RUN echo "%docker ALL=(ALL:ALL) NOPASSWD:ALL" >> /etc/sudoers

ENV VIRTUAL_ENV=/venv/.venv

COPY --from=plaster-with-pips $VIRTUAL_ENV $VIRTUAL_ENV

ENV ERISYON_DOCKER_ENV="1"
ENV ERISYON_TMP="/tmp"
ENV ERISYON_ROOT="/erisyon"
ENV PLASTER_ROOT="/erisyon/plaster"
ENV PYTHONPATH="${PLASTER_ROOT}:${PLASTER_ROOT}/plaster/vendor:${VIRTUAL_ENV}/lib/python3.8/site-packages:${VIRTUAL_ENV}/lib/python3.8/site-packages/plaster/vendor"
ENV PATH="${VIRTUAL_ENV}/bin:${ERISYON_ROOT}/plaster:${PATH}"

# Using a copy here instead of a touch to prevent busting cache
# Note, plaster_root is being copied to erisyon_root (just an empty file)
COPY ./plaster_root /erisyon/erisyon_root

# Copy Erisyon related files
WORKDIR /erisyon

COPY ./plaster_root /erisyon/plaster/plaster_root
COPY ./plas /erisyon/plaster/plas
COPY ./scripts/docker_entrypoint.sh ./plaster/scripts/docker_entrypoint.sh
# TODO: need a plaster autocomp
# COPY ./internal/.autocomp ./internal/.autocomp

## COPY only the C source files and have Cython to build
## this is to prevent having to run the compile after every change of ANY plaster file.
## COPY ./plaster/run/nn_v2/c ./plaster/plaster/run/nn_v2/c
#COPY ./plaster/run/sim_v2/fast ./plaster/plaster/run/sim_v2/fast
#COPY ./plaster/run/survey_v2/fast ./plaster/plaster/run/survey_v2/fast
#COPY ./plaster/tools/c_common ./plaster/plaster/tools/c_common
#COPY ./README.md ./plaster/README.md
#COPY ./plaster/version.py ./plaster/plaster/version.py
#COPY ./setup.py ./plaster/setup.py
#RUN cd /erisyon/plaster && /erisyon/plaster/scripts/docker_entrypoint.sh
## At this point the .so files are built and are in:
##    ./plaster/plaster/run/sim_v2/fast/sim_v2_fast.cpython-38-x86_64-linux-gnu.so
##    ./plaster/plaster/run/survey_v2/fast/survey_v2_fast.cpython-38-x86_64-linux-gnu.so
## But they are about to be overwritten by the following COPY line so
## we stash them into an alternative folder and then copy them back
#RUN cp \
#    ./plaster/plaster/run/sim_v2/fast/sim_v2_fast.cpython-38-x86_64-linux-gnu.so /tmp \
#    && cp ./plaster/plaster/run/survey_v2/fast/survey_v2_fast.cpython-38-x86_64-linux-gnu.so /tmp

# COPY the LATEST local copy of plaster thus over-riding the
# version of plaster that was actually published to PyPi
COPY ./plaster ./plaster/plaster

# COPY the correct version of the .so files into place.
# This COPY prevents having to re-build the Cython files when ANY un-related plaster file changes.
RUN cp \
    /tmp/sim_v2_fast.cpython-38-x86_64-linux-gnu.so ./plaster/plaster/run/sim_v2/fast \
    && cp /tmp/survey_v2_fast.cpython-38-x86_64-linux-gnu.so ./plaster/plaster/run/survey_v2/fast

COPY ./scripts ./plaster/scripts

# The gid bit (2XXX) on /root and /home so that all files created in there
# will be owned by the group root. This is so that when the container is run
# as the local user we can propagate the userid to the host file system on new files
# Also, all subdirectories must be read-writeable by any user that comes
# into the container since they all share the same home directory
# Note that this must be in one line so that we don't duplicate all the
# /root and /home files in a duplicate layer when chmods are run.
RUN chmod -R 2777 /root && chmod -R 2777 /home

WORKDIR /erisyon/plaster
# ENTRYPOINT [ "/erisyon/plaster/scripts/docker_entrypoint.sh" ]
