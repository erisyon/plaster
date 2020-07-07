# Plaster: Erisyon's Fluorosequencing Informatic Pipeline

This is the repository for Erisyon's analysis software for analyzing and simulating
runs on our Fluoro-Sequecning platform.

It consists of the following two parts:

1. gen: A CLI tool that generates instructions for plaster.
1. run: The tool that runs the analysis. Run is comprised of the following parts:
    * Sigproc: The signal processor that reads raw images from the instrument
       and coverts that into a "radmat" (rad-iometry mat-rix).
    * Simulator aka Virtual Fluoro-Sequencing (VFS): Uses an error model of the chemistry
       and instrument to produce simulated reads by Monte Carlo sampling.
       Used to train and evaluate the classifier.
    * Classifier: Trained against simulated reads, produces peptide or protein calls from reads,
       either real or simulated from VFS.

# Example Usages

1. Show plaster "gen" instructions
    ```bash
    $ plas gen --help    # abbreviated
    $ plas gen --readme  # detailed
    ```

1. Show plaster "run" help
    ```bash
    $ plas run --help
    ```

1. Virtual Fluoro-Sequencing (VFS) on 10 random human proteins.
    ```bash
    # Step 1: Generate a job...
    $ plas gen classify \
      --job=/jobs_folder/human_random_2 \
      --sample='human proteome 2 random proteins' \
      --protein_csv=./sample_data/human_proteome_random_proteins_2.csv \
      --label_set='DE,Y,C' \
      --n_edmans=10 \
      --n_pres=1

    # Run the job...    
    $ plas run /jobs_folder/human_random_2
    # Report in: /jobs_folder/human_random_2/report.html
    ```

## Requirements

Plaster is only tested on Linux. It is typically run as a docker container
and can therefore run as a container under Linux, OSX, or Windows.

If you are not using docker, a list of Debian packages is listed in:
`./scripts/apt_packages.txt`

## Build & test the docker container

Assuming you have docker installed and have pulled plaster into $PLASTER_ROOT

```bash
$ cd $PLASTER_ROOT
$ docker build -t plaster:latest .
$ docker run -it plaster:latest /bin/bash
$ plas test
# All the tests should pass 
```

Typically you will use data from outside the container and save your jobs
to a permanent folder outside of the container. For example, assuming you've
built plaster:latest as per above.

    ```bash
    # Step 1: Build the cotainer
    $ docker build -t plaster:latest .
    
    # Step 2: Start the container with volume mount(s)
    $ docker run -it --volume ${HOME}/jobs_folder:/jobs_folder:rw plaster:latest /bin/bash

    # Step 3, generate a job
    $ plas gen classify \
      --job=/jobs_folder/human_random_2 \
      --sample='human proteome 2 random proteins' \
      --protein_csv=./sample_data/human_proteome_random_proteins_2.csv \
      --label_set='DE,Y,C' \
      --n_edmans=10 \
      --n_pres=1

    # Step 4: Run the job...    
    $ plas run /jobs_folder/human_random_2
    ```

## Running Jupyter from a Docker container

Jupyter notebooks beed to open ports so you need to add this
to the Docker command line. 

    ```bash
    $ docker run -it --volume ${HOME}/jobs_folder:/jobs_folder:rw -p 8080:8080 plaster:e2e plas jupyter
    ```
