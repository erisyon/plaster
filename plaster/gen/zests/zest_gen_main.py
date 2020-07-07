from zest import zest
from plaster.gen.gen_main import GenApp


def zest_gen_main():
    def it_can_readme():
        with zest.raises(SystemExit):
            with zest.mock(GenApp._print):
                app = GenApp.run(["", "--readme"])
                app.readme()

    zest()
