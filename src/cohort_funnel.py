"""Reconstruct and verify, from the data themselves, how each cohort was selected: compare the samples retained in the
.npz files with the original metadata of each release, and report a selection funnel (counts at every step), the
inclusion rules actually applied, and whether subsampling kept whole patients.

Usage: python cohort_funnel.py [data_root] [datasets_dir]
Writes cohort_funnel.md and prints the same content.
"""
from config import CAMELYON_METADATA, DATA_ROOT, DATASETS_DIR  # noqa: E402,F401
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = sys.argv[1] if len(sys.argv) > 1 else DATA_ROOT
DS = sys.argv[2] if len(sys.argv) > 2 else DATASETS_DIR
OUT = []


def say(s=""):
    print(s)
    OUT.append(s)


def find(name_patterns, max_hits=5):
    """Locate metadata files under ROOT (skips folders of images)."""
    hits = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        if len(filenames) > 2000:                 # image folders: do not descend further
            dirnames[:] = []
        for f in filenames:
            if any(re.fullmatch(p, f, flags=re.I) for p in name_patterns):
                hits.append(os.path.join(dirpath, f))
        if len(hits) >= max_hits:
            break
    return sorted(hits, key=len)


def npz(name):
    z = np.load(os.path.join(DS, f"{name}.npz"), allow_pickle=True)
    return {k: z[k] for k in z.files}


def section(title):
    say(f"\n## {title}\n")


def subsample_check(meta_ids_by_patient, kept_ids, eligible_ids):
    """Did subsampling keep whole patients? Fraction of eligible images of selected patients that were kept."""
    kept, elig = set(kept_ids), set(eligible_ids)
    sel_pat = [p for p, ids in meta_ids_by_patient.items() if kept & set(ids)]
    tot = sum(len(elig & set(meta_ids_by_patient[p])) for p in sel_pat)
    got = sum(len(kept & set(meta_ids_by_patient[p])) for p in sel_pat)
    return len(sel_pat), got, tot


# ------------------------------------------------------------------ NIH ChestX-ray14
def nih():
    section("NIH ChestX-ray14 (effusion vs no finding)")
    z = npz("nih_cxr")
    kept = [os.path.basename(str(p)) for p in z["paths"]]
    y = z["y"].astype(int)
    f = find([r"Data_Entry_2017(_v2020)?\.csv"])
    if not f:
        say("- metadata file Data_Entry_2017.csv not found under the root; funnel not computed")
        return
    m = pd.read_csv(f[0])
    say(f"- metadata: `{f[0]}`")
    m["eff"] = m["Finding Labels"].str.split("|").apply(lambda l: "Effusion" in l)
    m["nofind"] = m["Finding Labels"] == "No Finding"
    pos, neg = m[m.eff], m[m.nofind]
    say(f"- release: {len(m):,} images, {m['Patient ID'].nunique():,} patients")
    say(f"- eligible: effusion (label list contains 'Effusion') {len(pos):,} images; 'No Finding' {len(neg):,} images")
    say(f"  - of the effusion images, with no other finding: {int((pos['Finding Labels'] == 'Effusion').sum()):,}; "
        f"with co-occurring findings: {int((pos['Finding Labels'] != 'Effusion').sum()):,}")
    km = m.set_index("Image Index").loc[[k for k in kept if k in set(m["Image Index"])]]
    say(f"- retained: {len(kept):,} images ({int((y == 1).sum()):,} label 1, {int((y == 0).sum()):,} label 0); "
        f"found in metadata: {len(km):,}; patients {km['Patient ID'].nunique():,}")
    if len(km) == len(kept):
        say(f"- rule check: label-1 images whose labels contain 'Effusion': {100 * km.eff.values[y == 1].mean():.1f}%; "
            f"label-0 images that are exactly 'No Finding': {100 * km.nofind.values[y == 0].mean():.1f}%")
    else:
        say(f"- rule check skipped: {len(kept) - len(km)} retained images not found in the metadata")
    say(f"- retained effusion images with co-occurring findings: {int((km.eff & (km['Finding Labels'] != 'Effusion')).sum()):,}")
    say(f"- view position, retained: {km['View Position'].value_counts().to_dict()}; eligible: "
        f"{m[m.eff | m.nofind]['View Position'].value_counts().to_dict()}")
    by_pat = m[m.eff | m.nofind].groupby("Patient ID")["Image Index"].apply(list).to_dict()
    n_pat, got, tot = subsample_check(by_pat, km.index, m[m.eff | m.nofind]["Image Index"])
    say(f"- subsampling: {n_pat:,} of {len(by_pat):,} eligible patients selected; {got:,} of the {tot:,} eligible images of "
        f"the selected patients retained ({100 * got / max(tot, 1):.1f}%; 100% = whole patients kept)")


# ------------------------------------------------------------------ CheXpert
def chexpert():
    section("CheXpert (pleural effusion vs no finding)")
    z = npz("chexpert")
    key = lambda p: re.sub(r"^.*?(train|valid)/", r"\1/", str(p).replace("\\", "/"))  # noqa: E731
    kept = [key(p) for p in z["paths"]]
    y = z["y"].astype(int)
    f = [x for x in find([r"train\.csv"], max_hits=10) if "chexpert" in x.lower()]
    if not f:
        say("- metadata file train.csv (CheXpert) not found; funnel not computed")
        return
    parts = [pd.read_csv(f[0])]
    v = os.path.join(os.path.dirname(f[0]), "valid.csv")
    if os.path.exists(v):
        parts.append(pd.read_csv(v))
    m = pd.concat(parts, ignore_index=True)
    say(f"- metadata: `{f[0]}`" + (" + valid.csv" if len(parts) > 1 else ""))
    m["key"] = m["Path"].map(key)
    m["patient"] = m["Path"].str.extract(r"(patient\d+)")[0]
    say(f"- release (train and valid): {len(m):,} images, {m['patient'].nunique():,} patients")
    fr = m[m["Frontal/Lateral"] == "Frontal"]
    say(f"- frontal views: {len(fr):,}")
    pe, nf = fr["Pleural Effusion"], fr["No Finding"]
    say(f"- pleural effusion among frontal: positive {int((pe == 1).sum()):,}, uncertain {int((pe == -1).sum()):,}, "
        f"negative {int((pe == 0).sum()):,}, blank {int(pe.isna().sum()):,}; 'No Finding' = 1: {int((nf == 1).sum()):,}")
    elig = fr[(pe == 1) | ((nf == 1) & ~pe.isin([1, -1]))]          # inclusion rule: uncertain effusion excluded
    say(f"- eligible (inclusion rule): effusion = 1: {int((pe == 1).sum()):,}; no finding = 1 with effusion neither 1 nor -1: "
        f"{int(((nf == 1) & ~pe.isin([1, -1])).sum()):,}; excluded because effusion uncertain: {int((pe == -1).sum()):,}")
    km = m.set_index("key").loc[[k for k in kept if k in set(m["key"])]]
    say(f"- retained: {len(kept):,} images ({int((y == 1).sum()):,} label 1, {int((y == 0).sum()):,} label 0); "
        f"found in metadata: {len(km):,}; patients {km['patient'].nunique():,}")
    say(f"- rule check: retained images that are frontal {100 * (km['Frontal/Lateral'] == 'Frontal').mean():.1f}%")
    if len(km) == len(kept):
        pe_k, nf_k = km["Pleural Effusion"].values, km["No Finding"].values
        say(f"  - label 1: Pleural Effusion = 1 in {100 * np.mean(pe_k[y == 1] == 1):.1f}%; label 0: No Finding = 1 in "
            f"{100 * np.mean(nf_k[y == 0] == 1):.1f}%, Pleural Effusion uncertain (-1) in {int(np.sum(pe_k[y == 0] == -1)):,}, "
            f"blank in {int(np.sum(pd.isna(pe_k[y == 0]))):,}")
    else:
        say(f"  - per-label check skipped: {len(kept) - len(km)} retained images not found in the metadata")
    if "AP/PA" in km:
        say(f"- AP/PA, retained: {km['AP/PA'].value_counts().to_dict()}")
    by_pat = elig.groupby("patient")["key"].apply(list).to_dict()
    n_pat, got, tot = subsample_check(by_pat, km.index, elig["key"])
    say(f"- subsampling: {n_pat:,} of {len(by_pat):,} eligible patients selected; {got:,} of {tot:,} eligible images of the "
        f"selected patients retained ({100 * got / max(tot, 1):.1f}%)")


# ------------------------------------------------------------------ HAM10000
def ham():
    section("HAM10000 (melanoma vs nevus)")
    z = npz("ham10000")
    ids = [re.search(r"(ISIC_\d+)", str(s)).group(1) for s in (z["sample_id"] if "sample_id" in z else z["paths"])
           if re.search(r"(ISIC_\d+)", str(s))]
    f = find([r"HAM10000_metadata(\.csv)?"])
    if not f:
        say("- HAM10000_metadata.csv not found; counts from the .npz only")
    else:
        m = pd.read_csv(f[0])
        say(f"- metadata: `{f[0]}`")
        say(f"- release: {len(m):,} images, {m['lesion_id'].nunique():,} lesions; diagnoses {m['dx'].value_counts().to_dict()}")
        e = m[m.dx.isin(["mel", "nv"])]
        say(f"- eligible (mel or nv): {len(e):,} images ({int((e.dx == 'mel').sum()):,} mel, {int((e.dx == 'nv').sum()):,} nv), "
            f"{e['lesion_id'].nunique():,} lesions")
        km = m[m.image_id.isin(ids)]
        say(f"- retained: {len(ids):,} images matched to metadata: {len(km):,}; by source: "
            f"{km.groupby(['dataset', 'dx']).size().to_dict()}")
    say(f"- .npz: {len(z['y']):,} samples, label 1 = {int((z['y'] == 1).sum()):,}; sites {pd.Series(z['site']).value_counts().to_dict()}")


# ------------------------------------------------------------------ PTB-XL
def ptbxl():
    section("PTB-XL (myocardial infarction vs normal)")
    z = npz("ptbxl")
    f = find([r"ptbxl_database\.csv"])
    g = find([r"scp_statements\.csv"])
    say(f"- .npz: {len(z['y']):,} records, label 1 = {int((z['y'] == 1).sum()):,}, "
        f"{len(np.unique(z['group'])):,} patients, {len(np.unique(z['site'])):,} sites")
    if not f or not g:
        say("- ptbxl_database.csv or scp_statements.csv not found; funnel not computed")
        return
    import ast
    m = pd.read_csv(f[0], index_col="ecg_id")
    s = pd.read_csv(g[0], index_col=0)
    say(f"- metadata: `{f[0]}`")
    diag = s[s.diagnostic == 1]["diagnostic_class"].to_dict()
    sup = m.scp_codes.map(lambda c: {diag[k] for k in ast.literal_eval(c) if k in diag})
    say(f"- release: {len(m):,} records, {m.patient_id.nunique():,} patients")
    has_mi, is_norm_only = sup.map(lambda x: "MI" in x), sup.map(lambda x: x == {"NORM"})
    is_mi_only = sup.map(lambda x: x == {"MI"})
    say(f"- eligible (inclusion rule, single superclass): MI only {int(is_mi_only.sum()):,}, NORM only {int(is_norm_only.sum()):,}; "
        f"excluded: MI with another superclass {int((has_mi & ~is_mi_only).sum()):,}, other or no superclass "
        f"{int((~has_mi & ~is_norm_only).sum()):,}")
    say(f"- diagnostic superclasses: MI present {int(has_mi.sum()):,}; NORM only {int(is_norm_only.sum()):,}; "
        f"MI and NORM {int(sup.map(lambda x: {'MI', 'NORM'} <= x).sum()):,}; no diagnostic statement {int(sup.map(len).eq(0).sum()):,}")
    ids = []
    for v in (z["sample_id"] if "sample_id" in z else z["paths"]):
        mm = re.findall(r"\d+", str(v))
        ids.append(int(mm[-1]) if mm else -1)
    ids = np.array(ids)
    ok = np.isin(ids, m.index)
    if ok.mean() < 0.9:
        say(f"- could not match retained records to ecg_id reliably ({100 * ok.mean():.0f}% matched); rule check skipped")
        return
    y = z["y"].astype(int)[ok]
    ks = sup.loc[ids[ok]]
    say(f"- rule check (retained): label 1 with MI {100 * ks[y == 1].map(lambda x: 'MI' in x).mean():.1f}%, of which also NORM "
        f"{int(ks[y == 1].map(lambda x: 'NORM' in x).sum()):,}; label 0 NORM only {100 * ks[y == 0].map(lambda x: x == {'NORM'}).mean():.1f}%; "
        f"label 1 with other superclasses besides MI {int(ks[y == 1].map(lambda x: len(x - {'MI'}) > 0).sum()):,}")


# ------------------------------------------------------------------ ABIDE
def abide():
    section("ABIDE I and II (autism vs control)")
    z = npz("abide")
    say(f"- .npz: {len(z['y']):,} slices, label 1 = {int((z['y'] == 1).sum()):,}, {len(np.unique(z['group'])):,} identifiers, "
        f"{len(np.unique(z['site'])):,} sites")
    say(f"- sites: {pd.Series(z['site']).value_counts().to_dict()}")
    f = find([r".*Phenotypic.*\.csv"], max_hits=5)
    say(f"- phenotypic files found: {[os.path.basename(x) for x in f]}" if f else "- phenotypic files not found under the root")


if __name__ == "__main__":
    say("# Cohort selection funnel, reconstructed from the retained samples and the original metadata")
    say(f"root: `{ROOT}`; datasets: `{DS}`")
    for fn in (nih, chexpert, ham, ptbxl, abide):
        try:
            fn()
        except Exception as e:                       # report and continue with the next dataset
            say(f"- ERROR in {fn.__name__}: {type(e).__name__}: {e}")
    open("cohort_funnel.md", "w").write("\n".join(OUT) + "\n")
    print("\nwritten: cohort_funnel.md")
