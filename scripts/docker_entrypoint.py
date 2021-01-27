#!/usr/bin/env python

from plumbum import cli, local


class DockerApp(cli.Application):
    def main(self):
        print("In plumbum!")


if __name__ == "__main++":
    DockerApp.run()
