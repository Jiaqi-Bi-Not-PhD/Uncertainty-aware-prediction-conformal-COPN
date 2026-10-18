"""Format Supplement S7 Table S11 from authorized private aggregate inputs."""
import argparse
import csv
from pathlib import Path
from copn_analysis.private_io import private_input, private_output
POLICIES = tuple((f'A{i}' for i in range(6)))
CLASSES = ('PD', 'AP', 'HC')
VARIATION_GROUPS = ('Lowest variation', 'Lower variation', 'Middle variation', 'Higher variation', 'Highest variation')

def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))

def percent(value):
    return f'{100 * float(value):.2f}'

def ssi_panel(rows, budget, panel):
    text = []
    if panel != 'A':
        text.append('\\clearpage')
    text.extend(['\\begin{center}\\small\\textbf{Table S11. Whole-cohort robustness summaries' + (' (continued)' if panel != 'A' else '') + '.}\\end{center}', '\\begingroup\\small', '\\setlength{\\tabcolsep}{4pt}', '\\renewcommand{\\arraystretch}{1.08}', '\\setlength{\\TableWidth}{\\dimexpr\\linewidth-10\\tabcolsep\\relax}', f'\\noindent\\textbf{{{panel}. ' + ('Primary assessment budget J3 (levels 1--3)' if budget == '3' else 'Expanded assessment budget J4 (levels 1--4)') + '}\\par\\smallskip', '\\noindent\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{0.08\\TableWidth}>{\\raggedleft\\arraybackslash}p{0.14\\TableWidth}>{\\raggedleft\\arraybackslash}p{0.21\\TableWidth}>{\\raggedleft\\arraybackslash}p{0.19\\TableWidth}>{\\raggedleft\\arraybackslash}p{0.19\\TableWidth}>{\\raggedleft\\arraybackslash}p{0.19\\TableWidth}@{}}', '\\toprule', '\\textbf{Policy} & \\textbf{Average labels} & \\textbf{All three labels, count (\\%)} & \\textbf{PD coverage (\\%)} & \\textbf{AP coverage (\\%)} & \\textbf{HC coverage (\\%)} \\\\', '\\midrule'])
    for group in range(1, 6):
        selected = [r for r in rows if r['budget'] == budget and r['display_group'] == VARIATION_GROUPS[group - 1]]
        by_policy = {r['method_id']: r for r in selected}
        first = selected[0]
        denominators = '; '.join((f"{c}: {int(first[f'n_{c}']):,}" for c in CLASSES))
        label = VARIATION_GROUPS[group - 1]
        text.append(f'\\multicolumn{{6}}{{@{{}}l@{{}}}}{{\\textbf{{{label}: ' + f"{int(first['prediction_records']):,} records ({denominators})" + '}} \\\\')
        for policy in POLICIES:
            r = by_policy[policy]
            coverages = [percent(r[f'core_coverage_{c}']) if int(r[f'n_{c}']) else 'NA' for c in CLASSES]
            cells = [policy, f"{float(r['mean_set_size']):.3f}", f"{int(r['full_set_count']):,} ({percent(r['full_set_rate'])})", *coverages]
            text.append(' & '.join(cells) + ' \\\\')
        if group < 5:
            text.append('\\midrule')
    text.extend(['\\bottomrule', '\\end{tabular}\\par', '\\endgroup', '\\medskip'])
    if panel == 'A':
        text.append("\\noindent Group headings give record counts for Parkinson's disease (PD), the atypical Parkinsonian category (AP) and healthy controls (HC), shared by all policies. Coverage is the percentage of records in a class whose returned set includes that class. An all-three-label output is a full set; its count and percentage use all records in the group. The accompanying data file retains covered counts and unrounded estimates.")
    else:
        text.append('\\noindent Budget and policy summaries use the supplied private prediction records across cross-validation repeats. Groups run from lowest to highest variation in the possible diagnoses returned across scenarios; lower, middle and higher are the three intermediate groups in increasing order. Equal scores stay together, so group sizes can differ. Groups are formed separately at each budget and shared by all policies. They differ from Q1--Q4, the four groups ordered by the final information-deficit score. Policy key: A0, one label; A1, ensemble-only set; A2, union of scenario sets; A3, combined-score set; A4, reliability enlargement; A5, cost-based enlargement. These are descriptive estimates without uncertainty intervals or a separate coverage guarantee in each group.')
    return '\n'.join(text)

def m1_panel(rows):
    text = ['\\clearpage', '\\begin{center}\\small\\textbf{Table S11. Whole-cohort robustness summaries (continued).}\\end{center}', '\\noindent\\textbf{C. Highest M1-based information-deficit group (D4)}\\par', '\\medskip', 'Highest M1-based deficit group: summaries of fixed prediction policies.\\par', '\\medskip', '\\begingroup\\small', '\\setlength{\\tabcolsep}{4pt}', '\\renewcommand{\\arraystretch}{1.2}', '\\setlength{\\TableWidth}{\\dimexpr\\linewidth-8\\tabcolsep\\relax}', '\\noindent\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{0.12\\TableWidth}>{\\raggedleft\\arraybackslash}p{0.15\\TableWidth}>{\\raggedleft\\arraybackslash}p{0.21\\TableWidth}>{\\raggedleft\\arraybackslash}p{0.28\\TableWidth}>{\\raggedleft\\arraybackslash}p{0.24\\TableWidth}@{}}', '\\toprule', '\\textbf{Budget} & \\textbf{Prediction records} & \\textbf{A2 mean labels} & \\textbf{A5 HC covered / HC records} & \\textbf{A5 HC coverage (\\%)} \\\\', '\\midrule']
    for r in rows:
        text.append(' & '.join([f"J{r['budget']}", r['prediction_records'], f"{float(r['a2_mean_set_size']):.3f}", f"{r['a5_hc_covered']}/{r['a5_hc_records']}", percent(r['a5_hc_coverage'])]) + ' \\\\')
    text.extend(['\\bottomrule', '\\end{tabular}\\par', '\\endgroup', '\\medskip', '\\noindent Earlier M1 groups D1--D4 run from the lowest (D1) to the highest (D4) information deficit, with D2 and D3 intermediate. They were defined by empirical quartiles of held-out scores pooled across repeats within each budget. Their construction and role in the evaluated reliability score are documented in Supplement S4.4. These groups differ from the training-derived, split-specific cutoffs for final M5-based Q. Figure~\\ref{fig:s7-m1-deficit} shows all four earlier M1 groups at J3. HC denotes healthy controls. These are descriptive summaries; the separate M1 data file retains the unrounded values.'])
    return '\n'.join(text)

def build(ssi: Path, m1: Path, output: Path):
    rows = read_csv(private_input(ssi))
    m1_rows = read_csv(private_input(m1))
    content = [ssi_panel(rows, "3", "A"), ssi_panel(rows, "4", "B"), m1_panel(m1_rows)]
    private_output(output).write_text("\n\n".join(content) + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssi", type=Path, required=True)
    parser.add_argument("--m1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.ssi, args.m1, args.output)
