import numpy as np
import re
from contextlib import contextmanager
import binascii
import os


def dropdown(df, description, value):
    import ipywidgets as widgets  # Defer slow import

    return widgets.Dropdown(
        options=sorted(df.columns), description=description, value=value
    )


def tooltips(df):
    tooltips = [(key, f"@{key}") for key in sorted(df.columns)]
    tooltips += [
        ("peak", "$index"),
        (
            "(x,y)",
            "($x, $y)<style>.bk-tooltip>div:not(:first-child) {display:none;}</style>",
        ),
    ]
    return tooltips


def restart_kernel():
    from IPython.display import display_html  # Defer slow imports

    display_html("<script>Jupyter.notebook.kernel.restart()</script>", raw=True)


def _css_for_collapsible():
    from IPython.core.display import display, HTML  # Defer slow imports

    display(
        HTML(
            """
            <style>
                .wrap-collabsible {
                  margin-bottom: 0.2rem 0;
                }
    
                input[type='checkbox'] {
                  display: none;
                }
    
                .lbl-toggle {
                  display: block;
                  font-weight: bold;
                  font-size: 130%;
                  //text-transform: uppercase;
                  cursor: pointer;
                  border-radius: 7px;
                  transition: all 0.25s ease-out;
                }
    
                .lbl-toggle::before {
                  content: ' ';
                  display: inline-block;
                  border-top: 5px solid transparent;
                  border-bottom: 5px solid transparent;
                  border-left: 5px solid currentColor;
                  vertical-align: middle;
                  margin-right: .7rem;
                  transform: translateY(-2px);
                  transition: transform .2s ease-out;
                }
    
                .toggle:checked + .lbl-toggle::before {
                  transform: rotate(90deg) translateX(-3px);
                }
    
                .collapsible-content {
                  max-height: 0px;
                  overflow: hidden; 
                  transition: max-height .25s ease-in-out;
                }
    
                .toggle:checked + .lbl-toggle + .collapsible-content {
                  max-height: 10000000350px;
                }
    
                .toggle:checked + .lbl-toggle {
                  border-bottom-right-radius: 0;
                  border-bottom-left-radius: 0;
                }
    
                .collapsible-content .content-inner {
                  border: 2px solid rgba(0,0,0,0.2);
                  border-radius: 6px;
                  padding: .5rem 1rem;
                }
            </style>
        """
        )
    )


@contextmanager
def wrap_in_collapsible(header):
    from IPython.core.display import display, HTML  # Defer slow imports
    from IPython.utils.capture import capture_output  # Defer slow imports

    def _id():
        return binascii.b2a_hex(os.urandom(16)).decode("ascii")

    with capture_output(stdout=True, stderr=True, display=True) as captured:
        yield captured

    # captured now has the content which will be rendered into .output
    # divs in the display() call below. Before and after that special
    # sentinel divs are added so that the divs between those can be pulled
    # into a collapsible.
    _css_for_collapsible()

    top_id = _id()
    display(HTML(f"<div id='{top_id}'></div>"))

    for o in captured.outputs:
        o.display()

    bot_id = _id()
    display(HTML(f"<div id='{bot_id}'></div>"))

    # Run some jQuery kung-fu to pull the .output_subareas into the collapsible
    display(
        HTML(
            """
        <script>
            var top0 = $('#"""
            + top_id
            + """').closest('.output_area');
            var bot0 = $('#"""
            + bot_id
            + """').closest('.output_area');
            var foundOutputAreas = $(top0).nextUntil($(bot0));
            var topSubArea = $(top0).find('.output_subarea');
            $(topSubArea).empty();
            $(topSubArea).html(
                "<div class='wrap-collabsible'>" +
                    "<input id='collapsible-"""
            + top_id
            + """' class='toggle' type='checkbox'>" +
                    "<label for='collapsible-"""
            + top_id
            + """' class='lbl-toggle'>"""
            + header
            + """</label>" +
                    "<div class='collapsible-content'>" +
                        "<div class='content-inner'>" +
                        "<p></p>" +
                        "</div>" +
                    "</div>" +
                "</div>"
            );
            var insideOfCollapsable = $(top0).find("p");
            var foundOutputSubAreas = $(foundOutputAreas).find('.output_subarea');
            $(foundOutputSubAreas).detach().appendTo(insideOfCollapsable);
        </script>
    """
        )
    )


"""
Example usage of hd, h

hd("h1", "Some title")

hd("div",
    h("p.some_class.another_class", "paragraph 1"),
    h("p#the_id", "paragraph 2"),
    h(".a_class_on_a_div", "A div"),
)

hd("div",
    h("div", 
        h("p", "paragraph 1"),
        h("p", "paragraph 2"),
    ),
    h("div",
        h("p", "paragraph 3")
    )
)
"""


def _h_fmt(_tag):
    id = ""
    classes = ""
    tag = "div"
    for part in re.split(r"([.#][^.#]+)", _tag):
        if part.startswith("#"):
            id = f" {part[1:]}"
        elif part.startswith("."):
            classes += f" {part[1:]}"
        elif part != "":
            tag = part

    return tag, id, classes


def h(tag, *strings):
    tag, id, classes = _h_fmt(tag)
    return f"<{tag} id='{id}' class='{classes}'>{' '.join([str(s) for s in strings])}</{tag}>"


def hd(tag, *strings):
    from IPython.core.display import display, HTML  # Defer slow imports

    display(HTML(h(tag, *strings)))


def md(string):
    from IPython.core.display import display, Markdown  # Defer slow imports

    display(Markdown(string))


def v(vec, prec=2):
    """format a vector"""
    if isinstance(vec, list):
        vec = np.array(vec)
    return ", ".join([f"{i:2.{prec}f}" for i in vec.squeeze()])


def m(mat, prec=2, indent=""):
    """format a matrix"""
    assert mat.ndim == 2
    return "\n".join([indent + v(row, prec) for row in mat])


def pv(vec, prec=2):
    """print a vector"""
    print(v(vec, prec))


def pm(mat, prec=2):
    """print a matrix"""
    print(m(mat, prec))


def title(title):
    md(f"# {title}")


def subtitle(title):
    md(f"### {title}")


def fix_auto_scroll():
    from IPython.core.display import display, HTML  # Defer slow imports

    display(
        HTML(
            """
                <script>
                    var curOutput = $(".jupyter-widgets-output-area", Jupyter.notebook.get_selected_cell().element.get(0));
                    var curOutputChildren = $(curOutput).children(".output");
                    var mut = new MutationObserver(function(mutations) {
                        mutations.forEach(function (mutation) {
                            $(mutation.target).removeClass("output_scroll");
                          });
                    });
                    mut.observe( $(curOutputChildren)[0], {'attributes': true} );
                </script>
            """
        )
    )


def qgrid_mono():
    from IPython.core.display import display, HTML  # Defer slow imports

    display(
        HTML(
            "<style>.slick-cell { font-family: monospace, monospace !important; }</style>"
        )
    )


def css_for_markdown():
    """
    Not yet tested. The idea is to limit the width of markdown.
    """

    from IPython.core.display import display, HTML  # Defer slow imports

    display(
        HTML(
            """
            <style>
                .text_cell {
                    max-width: 300;
                }
            </style>
        """
        )
    )
