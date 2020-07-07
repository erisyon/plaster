# Stage 1
# ------------------------------------------------------
FROM ubuntu:20.04 AS base-image

ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV USER=root
USER root

WORKDIR /erisyon/plaster

COPY ./scripts/apt_packages.txt ./scripts/apt_packages.txt

RUN apt-get -qq update \
    && { cat ./scripts/apt_packages.txt | grep -o '^[^#]*' | xargs apt-get -qq -y install --no-install-recommends ; } \
    && update-alternatives --quiet --install /usr/bin/pip pip /usr/bin/pip3 1 \
    && update-alternatives --quiet --install /usr/bin/python python /usr/bin/python3 1 \
    && pip install pipenv \
    && rm -rf /var/lib/apt/lists/* \
    && locale-gen "en_US.UTF-8"


# Stage 2
# ------------------------------------------------------
FROM base-image AS pip-image

# Add build tools so that source distros can build
RUN apt-get -qq update && apt-get -qq -y install --no-install-recommends build-essential

# Have pipenv put the venv into /venv/.venv
# so that we can pluck it out in later stage
# and eliminate the build tools in this stage
WORKDIR /venv
COPY ./Pipfile ./Pipfile
COPY ./Pipfile.lock ./Pipfile.lock
RUN PIPENV_VENV_IN_PROJECT=1 pipenv sync --python /usr/bin/python


# Stage 3
# ------------------------------------------------------

FROM base-image AS final-image

WORKDIR /erisyon
RUN echo "This file is a sentinel to mark that this is the erisyon root source folder" > ./erisyon_root

WORKDIR /erisyon/plaster
ENV VIRTUAL_ENV=/venv/.venv
COPY --from=pip-image $VIRTUAL_ENV $VIRTUAL_ENV

ENV ERISYON_TMP="/tmp"
ENV ERISYON_ROOT="/erisyon"
ENV PLASTER_ROOT="/erisyon/plaster"
# if PYTHONPATH changes please call it out explicitly in PR, we need to keep it
# in sync with our internal codebase
ENV PYTHONPATH="${PLASTER_ROOT}:${PLASTER_ROOT}/plaster/vendor:${VIRTUAL_ENV}/lib/python3.8/site-packages"
ENV PATH="${VIRTUAL_ENV}/bin:${ERISYON_ROOT}:${ERISYON_ROOT}/plaster:${PATH}"

COPY . .

# The gid bit (2XXX) on /root and /home so that all files created in there
# will be owned by the group root. This is so that when the container is run
# as the local user we can propagate the userid to the host file system on new files
# Also, all subdirectories must be read-writeable by any user that comes
# into the container since they all share the same home directory
# Note that this must be in one line so that we don't duplicate all the
# /root and /home files in a duplicate layer when chmods are run.
RUN chmod -R 2777 /root && chmod -R 2777 /home
