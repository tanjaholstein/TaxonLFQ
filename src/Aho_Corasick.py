# ------------------------------------------------------------------
# 0 · Imports  (install once:  pip install biopython pyahocorasick )
# ------------------------------------------------------------------
from Bio import SeqIO
import ahocorasick
from collections import defaultdict
import re, pathlib, pprint, warnings

# ------------------------------------------------------------------
# 1 · FASTA  →  {taxon label : [protein strings]}
#
# Taxon labels are extracted via a cascade of recognised header formats:
#   P1  UniProt OX=  —  OX=9606          (NCBI taxonomy integer)
#   P2  UniProt OS=  —  OS=Homo sapiens   (organism name string)
#   P3  NCBI/RefSeq  —  [Homo sapiens]    (bracket at end of header)
#   P4  custom DB    —  TaxonPrefix_…     (prefix before first '_')
# If no header in the file matches any pattern a ValueError is raised.
# ------------------------------------------------------------------
_OX_RE     = re.compile(r'\bOX=(\d+)')
_OS_RE     = re.compile(r'\bOS=(.+?)(?=\s+[A-Z]{2}=|\s*$)')
_NCBI_RE   = re.compile(r'\[(.+?)\]\s*$')
_PREFIX_RE = re.compile(r'^([^_]+)_')


def _extract_taxon_label(description: str):
    """Return the taxon label from a FASTA description line, or None."""
    m = _OX_RE.search(description)
    if m:
        return m.group(1)
    m = _OS_RE.search(description)
    if m:
        return m.group(1).strip()
    m = _NCBI_RE.search(description)
    if m:
        return m.group(1).strip()
    first_word = description.split()[0] if description.strip() else ""
    m = _PREFIX_RE.match(first_word)
    if m:
        return m.group(1)
    return None


def fasta_by_prefix(fasta_path, ignore_numeric=False):
    org2seqs = defaultdict(list)
    n_total = 0
    n_unmatched = 0

    for rec in SeqIO.parse(fasta_path, "fasta"):
        n_total += 1
        label = _extract_taxon_label(rec.description)
        if label is None:
            n_unmatched += 1
            continue
        if ignore_numeric and label.isdigit():
            continue
        org2seqs[label].append(str(rec.seq))

    if not org2seqs:
        raise ValueError(
            f"Could not infer taxon labels from any of the {n_total} FASTA headers. "
            "None of the supported formats were recognised:\n"
            "  • UniProt:    OX=<taxID>  or  OS=<organism name>\n"
            "  • NCBI/RefSeq: [Organism name] at end of header\n"
            "  • Custom DB:  <TaxonPrefix>_<ProteinID>\n\n"
            "Please provide your own peptide→taxon mapping file instead of a FASTA."
        )

    if n_unmatched > 0:
        warnings.warn(
            f"{n_unmatched}/{n_total} FASTA records had no recognisable taxon pattern "
            "and were skipped. If this is unexpected, provide your own "
            "peptide→taxon mapping file instead of a FASTA.",
            stacklevel=2,
        )

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