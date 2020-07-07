from plaster.tools.zplots.zplots import ZPlots
from plaster.run.plots import plots

# This is taken directly from the same-named fn in plots_dev_mhc.
# The only difference here is titles and labeling for proteins and not peptides.
#
def plot_best_runs_pr(best_pr, all_pr, run_info, filters, **kwargs):
    df = best_pr.sort_values(by=["prec", "recall"], ascending=[False, False])
    z = kwargs.get("_zplots_context", None) or ZPlots()
    title = f"PR curves, protein identification ({len(df.pro_i.unique())} proteins), best runs ({filters.classifier})."
    with z(
        f_title=title,
        _merge=True,
        _legend="bottom_right",
        f_y_axis_label="precision",
        f_x_axis_label="read recall",
    ):
        color_by_run = len(run_info.run_iz) > 1
        groups = df.groupby("run_i")
        for run_i, run_label in zip(run_info.run_iz, run_info.run_labels):
            group = groups.get_group(run_i)
            if color_by_run:
                color = z.next()
            for i, row in group.iterrows():
                if not color_by_run:
                    color = z.next()
                pep_i = row.pep_i
                pro_id = row.pro_id
                legend_label = f"{run_label} {pro_id}"
                line_label = f"{pro_id} pep{row.pep_i:03d} {row.seqstr} {row.flustr}"
                prdf = all_pr[(all_pr.run_i == run_i) & (all_pr.pep_i == pep_i)]
                prsa = (prdf.prec.values, prdf.recall.values, prdf.score.values, None)
                plots._plot_pr_curve(
                    prsa,
                    color=color,
                    legend_label=legend_label,
                    _label=line_label,
                    _zplots_context=z,
                    **kwargs,
                )
