# ------------------------------------------------------------------
# 0 · Imports  (install once:  pip install biopython pyahocorasick )
# ------------------------------------------------------------------
from Bio import SeqIO
import ahocorasick
from collections import defaultdict
import re, pathlib, pprint

# ------------------------------------------------------------------
# 1 · FASTA  →  {prefix : [protein strings]}
# ------------------------------------------------------------------
HEADER_RE = re.compile(r'^([^_]+)_')          # >Pfl_…  →  "Pfl"

def fasta_by_prefix(fasta_path, ignore_numeric=False):
    org2seqs = defaultdict(list)

    for rec in SeqIO.parse(fasta_path, "fasta"):
        m = HEADER_RE.match(rec.id)
        if not m:
            continue
        prefix = m.group(1)
        if ignore_numeric and prefix.isdigit():
            continue
        org2seqs[prefix].append(str(rec.seq))

    return dict(org2seqs)

# ------------------------------------------------------------------
# 2 · Build one Aho–Corasick automaton from all peptides
# ------------------------------------------------------------------
def build_ac_automaton(peptides):
    A = ahocorasick.Automaton()
    for idx, pep in enumerate(peptides):
        A.add_word(pep, (idx, pep))
    A.make_automaton()
    return A

# ------------------------------------------------------------------
# 3 · Map peptides → taxa  (global scan)
# ------------------------------------------------------------------
def map_peptides_ac(peptides, org2seqs):
    idx2pep   = dict(enumerate(peptides))
    pep2tax   = [set() for _ in peptides]      # list-of-sets is fast
    A         = build_ac_automaton(peptides)

    for taxon, protein_list in org2seqs.items():
        for prot in protein_list:
            for _, (idx, _pep) in A.iter(prot):
                pep2tax[idx].add(taxon)

    # convert back to dict[str,list[str]]
    return {idx2pep[i]: sorted(taxa) for i, taxa in enumerate(pep2tax) if taxa}

def canon_pep2orgs(mapping):
    """Return {peptide: set[str]} with trimmed labels; drop Nones/empties."""
    out = {}
    for pep, orgs in mapping.items():
        if orgs is None:
            continue
        if isinstance(orgs, (set, list, tuple)):
            items = orgs
        else:
            items = [orgs]
        keep = {str(o).strip() for o in items if o is not None and str(o).strip()}
        if keep:
            out[pep] = keep
    return out

def invert_peptide_map(pep2orgs, all_orgs=None, keep_empty=True):
    """
    Given {peptide: [org1, org2, ...]} return {org: [peptide1, ...]}.
    If keep_empty=True and all_orgs is provided, include orgs with [].
    """
    org_to_peps = defaultdict(list)
    if keep_empty and all_orgs is not None:
        for org in all_orgs:
            org_to_peps[org]  # pre-create empty list

    for pep, orgs in pep2orgs.items():
        for org in orgs:
            org_to_peps[org].append(pep)

    return dict(org_to_peps)