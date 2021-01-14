Reports are notebooks which are not templated
and are executed by the indexer.

During generation they are emitted into _reports
under the job folder. This in contrast to nb_templates
which are constructed from templates into the root
of the generated job folder.

Reports are not templated. They should execute stand-alone
and use the `job = Job.from_context(dev_default="")`
pattern to load from context or dev.

No important information should be conveyed in their
code blocks. Use Markdown for all information.
The code blocks will be stripped from the final display.

It is also not expected/permitted that these reports
ever be modified in place. Rather, it is expected that
an old version found in any job folder can be over-written
by a newer version. Thus, it is not allowed to make changes
as they can be unexpectedly blown away. Again, this is in
contrast to templated reports which are exepcted to be
modified by the analyst.

Reports are of two flavors:

some_stable_report.ipynb
_some_unsable_report.ipynb

Those that are stable are expected to be integration
tested and should have cooresponding zest_*.py files
in the zests/ sub-folder

Those that are unstable are expected to start with an
underscore and are not required to have tests.


