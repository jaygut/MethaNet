"""Sequence/coordinate checks and explicit caller crosswalks for selected loci."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re
import time
import warnings

from Bio.Seq import Seq
from Bio.Data import CodonTable
from Bio import BiopythonWarning

from .model import digest
from .molecular_audit import local_path, read_tsv, write_json, write_tsv
from .molecular_stage import MUCC, add_source, stable_id


def fasta(path):
    """Stream FASTA retaining full source header and one-based record locator."""
    opener = gzip.open if str(path).endswith(".gz") else open
    header, chunks, ordinal = None, [], 0
    with opener(path, "rt") as stream:
        for line in stream:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    yield ordinal, header, "".join(chunks).upper()
                ordinal += 1
                header, chunks = line[1:], []
            elif line:
                if header is None:
                    raise ValueError("Sequence before FASTA header")
                chunks.append(line)
        if header is not None:
            yield ordinal, header, "".join(chunks).upper()


def sequence_digest(seq):
    # Normalization is explicit and separate from the source file byte digest.
    return hashlib.sha256(seq.rstrip("*").upper().encode("ascii")).hexdigest()


def fasta_index(path):
    result = {}
    for ordinal, header, seq in fasta(path):
        key = header.split()[0]
        if key in result:
            raise ValueError(f"Duplicate FASTA record ID: {key}")
        result[key] = (ordinal, header, seq)
    return result


def prodigal_coordinates(header):
    parts = header.split(" # ")
    if len(parts) < 5:
        return None
    try:
        start, end, strand = int(parts[1]), int(parts[2]), int(parts[3])
    except ValueError:
        return None
    gene_id = parts[0].split()[0]
    contig, sep, suffix = gene_id.rpartition("_")
    if not sep or not suffix.isdigit():
        return None
    attributes = dict(x.split("=", 1) for x in parts[4].split(";") if "=" in x)
    return {"contig_id": contig, "start": start, "end": end, "strand": strand,
            "partial": attributes.get("partial", "unrecorded"),
            "start_type": attributes.get("start_type", "unrecorded"),
            "caller": "Prodigal_style_source_header", "caller_version": "unrecorded",
            "coordinate_system": "1_based_inclusive", "genetic_code": 11,
            "genetic_code_evidence": "validation_hypothesis_standard_bacterial_archaeal_table_11_not_historical_run_proof"}


def check_translation(dna, protein, start, end, strand, table=11):
    if strand not in (-1, 1) or start < 1 or end < start or end > len(dna):
        return {"status": "invalid_coordinates_or_strand", "admit_encodes": False}
    nt = dna[start-1:end]
    if strand == -1:
        nt = str(Seq(nt).reverse_complement())
    if len(nt) % 3:
        return {"status": "non_triplet_interval", "admit_encodes": False}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BiopythonWarning)
        translated = str(Seq(nt).translate(table=table)).rstrip("*")
    target = protein.rstrip("*")
    match = translated == target
    alternative_start = False
    if not match and translated and target and translated[1:] == target[1:] and target[0] == "M":
        alternative_start = nt[:3] in CodonTable.unambiguous_dna_by_id[table].start_codons
        match = alternative_start
    return {"status": "exact_with_allowed_initiator_methionine" if match and alternative_start else "exact_translation" if match else "translation_mismatch",
            "admit_encodes": match, "genetic_code_tested": table,
            "nucleotide_interval_sha256": hashlib.sha256(nt.encode()).hexdigest(),
            "translated_sequence_sha256": sequence_digest(translated),
            "protein_sequence_sha256": sequence_digest(target),
            "translation_normalization": f"upper_case;terminal_stop_stripped;initiator_M_only_if_table{table}_start_codon"}


def source_coordinates(event):
    r = event["raw"]
    try:
        return {"contig_id": r["scaffold"], "start": int(r["start_position"]), "end": int(r["end_position"]),
            "strand": int(r["strandedness"]), "caller": "source_DRAM_feature", "caller_version": "unrecorded",
            "coordinate_system": "source_fields_tested_as_1_based_inclusive", "genetic_code": 11,
            "genetic_code_evidence": "validation_hypothesis_table_11_not_source_version_proof"}
    except (ValueError, KeyError, TypeError):
        return None


def extract(repo, stagedir, out):
    started = time.monotonic()
    executing_code_sha256 = digest(Path(__file__))
    out.mkdir(parents=True, exist_ok=True)
    events = json.loads((stagedir / "selected-annotation-events.json").read_text())
    magcontext = json.loads((stagedir / "selected-mag-context.json").read_text())
    contexts = {(m["lane_id"], m["proteome_id"]): m for m in magcontext}
    groups = defaultdict(list)
    for e in events:
        groups[(e["lane_id"], e["proteome_id"], e["run_id"], e["gene_id"])].append(e)
    bymag = defaultdict(list)
    for key in groups:
        bymag[key[:2]].append(key)
    sources, loci, crosswalks, issues = {}, [], [], []
    for file in (stagedir / "selected-annotation-events.json", stagedir / "selected-mag-context.json"):
        add_source(repo, sources, file, "selected_event_input")
    previous = {r["source_gene_id"]: r for r in read_tsv(repo / "docs/white-paper/v17/analysis/curation_audit/marker_label_review.tsv")}
    for (lid, pid), keys in sorted(bymag.items()):
        context = contexts[(lid, pid)]
        m = context["warehouse"]
        assembly_path = local_path(repo, m.get("mag_fasta") or m["local_fna_path"])
        protein_path = local_path(repo, m["proteome_faa"]) if m.get("proteome_faa") else None
        dna_source = add_source(repo, sources, assembly_path, "selected_assembly")
        faa_source = add_source(repo, sources, protein_path, "selected_proteome") if protein_path and protein_path.is_file() else None
        dna = fasta_index(assembly_path)
        proteins = fasta_index(protein_path) if faa_source else {}
        # One selected run may have a raw Bakta JSON with explicit orig_id map.
        # It is not inferred by matching contig_1 to the first input contig.
        bakta, bakta_source, record_source = None, None, None
        qc_translation_code, qc_code_source = None, None
        run = context.get("selected_run", {})
        input_hash_status = "not_recorded"
        if run.get("run_dir"):
            record_path = local_path(repo, run["run_dir"]) / "curated/run_record.json"
            if record_path.is_file():
                record_source = add_source(repo, sources, record_path, "selected_curated_run_record")
                record = json.loads(record_path.read_text())
                if record.get("proteome_id") != pid or record.get("run_id") != m["run_id"]:
                    raise ValueError("Curated run identity mismatch")
                expected = record.get("inputs", {}).get("input_sha256")
                input_hash_status = "byte_digest_matches_recorded_input" if expected == dna_source["sha256"] else "recorded_input_digest_not_same_as_current_file_bytes" if expected else "not_recorded"
                # A gzipped container can differ from the recorded plain FASTA.
                if expected and input_hash_status != "byte_digest_matches_recorded_input" and assembly_path.suffix == ".gz":
                    h = hashlib.sha256()
                    with gzip.open(assembly_path,"rb") as stream:
                        for chunk in iter(lambda:stream.read(1024*1024), b""):
                            h.update(chunk)
                    if h.hexdigest() == expected:
                        input_hash_status = "decompressed_bytes_match_recorded_input"
                quality = record.get("outputs", {}).get("checkm2_quality", {}).get("path")
                if quality and local_path(repo, quality).is_file():
                    qp = local_path(repo,quality)
                    qr = read_tsv(qp)
                    if len(qr)==1 and qr[0].get("Translation_Table_Used"):
                        qc_translation_code = int(qr[0]["Translation_Table_Used"])
                        qc_code_source = add_source(repo,sources,qp,"CheckM2_reported_translation_table_context")
                bk = record.get("outputs", {}).get("bakta_json", {}).get("path")
                if bk:
                    bp = local_path(repo,bk)
                    if not bp.is_file() and Path(str(bp)+".gz").is_file():
                        bp = Path(str(bp)+".gz")
                    if bp.is_file():
                        bakta_source = add_source(repo,sources,bp,"Bakta_feature_and_original_contig_crosswalk")
                        opener = gzip.open if bp.suffix == ".gz" else open
                        with opener(bp,"rt") as stream:
                            bakta = json.load(stream)
        bakta_candidates = defaultdict(list)
        if bakta:
            contig_map = {}
            for s in bakta.get("sequences", []):
                original = s.get("orig_id")
                if original in dna and s.get("nt", "").upper() == dna[original][2]:
                    contig_map[s["id"]] = original
                else:
                    issues.append({"lane_id":lid,"proteome_id":pid,"issue":"Bakta_original_contig_sequence_crosswalk_failed","source_contig":s["id"]})
            for f in bakta.get("features", []):
                if f.get("type") == "cds" and f.get("sequence") in contig_map:
                    key = (contig_map[f["sequence"]], int(f["start"]), int(f["stop"]), 1 if f["strand"] == "+" else -1)
                    bakta_candidates[key].append(f)
        for key in sorted(keys):
            _,_,run_id,gene_id = key
            evs = groups[key]
            locus_id = stable_id("locus", lid,pid,run_id,gene_id)
            protein_id = stable_id("protein",lid,pid,run_id,gene_id)
            source_protein = proteins.get(gene_id)
            coords = source_coordinates(evs[0]) if lid == MUCC else prodigal_coordinates(source_protein[1]) if source_protein else None
            # Source records disagreeing on a locus cannot be flattened.
            if lid == MUCC and any(source_coordinates(e) != coords for e in evs):
                coords = None
                issues.append({"lane_id":lid,"proteome_id":pid,"gene_id":gene_id,"issue":"conflicting_coordinate_rows"})
            result = {"status":"missing_sequence_or_coordinate", "admit_encodes":False}
            if coords and source_protein:
                if coords["contig_id"] in dna:
                    result = check_translation(dna[coords["contig_id"]][2],source_protein[2],coords["start"],coords["end"],coords["strand"])
                    if not result["admit_encodes"] and qc_translation_code and qc_translation_code != 11:
                        alternative = check_translation(dna[coords["contig_id"]][2],source_protein[2],coords["start"],coords["end"],coords["strand"],table=qc_translation_code)
                        result["alternative_translation_test"] = alternative
                        result["alternative_code_evidence"] = qc_code_source
                        if alternative["admit_encodes"]:
                            result = {**alternative,"standard_table11_status":"translation_mismatch",
                                "alternative_code_evidence":qc_code_source,
                                "code_interpretation":"Source sequence exactly translates under the independently recorded CheckM2 code; original protein caller code remains unrecorded"}
                            coords["genetic_code"] = qc_translation_code
                            coords["genetic_code_evidence"] = "CheckM2_reported_code_and_direct_sequence_consistency_not_original_caller_proof"
                else:
                    result = {"status":"unresolved_source_contig_identity", "admit_encodes":False}
            row = {"locus_id":locus_id,"protein_id":protein_id,"lane_id":lid,"proteome_id":pid,"run_id":run_id,"gene_id":gene_id,
                "annotation_event_ids":[e["event_id"] for e in evs],"source_annotation_sha256":evs[0]["source_sha256"],
                "assembly_source":dna_source,"protein_source":faa_source,"curated_run_source":record_source,
                "recorded_input_integrity":input_hash_status,"protein_record_ordinal":source_protein[0] if source_protein else None,
                "protein_header":source_protein[1] if source_protein else None,"protein_length":len(source_protein[2].rstrip("*")) if source_protein else None,
                "protein_sequence_sha256":sequence_digest(source_protein[2]) if source_protein else None,
                "protein_digest_normalization":"uppercase_ASCII_terminal_stop_stripped_v1",
                "sequence_available":bool(source_protein),"coordinates":coords,"translation_check":result,
                "sequence_identity_status":"source_header_and_coordinate_translation_verified" if result["admit_encodes"] else "sequence_only_no_locus_translation" if source_protein else "source_locator_sequence_unavailable",
                "caller_namespace":"source_DRAM_original_or_expression_derived" if lid==MUCC else "Prodigal_style_query_namespace",
                "caller_version":"unrecorded_in_selected_protein_header","review_status":"pending_independent_review"}
            if qc_code_source:
                row["checkm_translation_code"] = qc_translation_code
                row["checkm_code_source"] = qc_code_source
            if bakta:
                row["bakta_translation_code"] = bakta.get("genome",{}).get("translation_table")
                row["gene_calling_code_agreement"] = "disagreement_between_source_methods" if qc_translation_code and qc_translation_code != row["bakta_translation_code"] else "no_reported_code_disagreement"
            if lid == MUCC and gene_id in previous and source_protein:
                expected = previous[gene_id]["sequence_sha256_stop_stripped"]
                row["historical_sequence_digest_status"] = "matches" if expected==row["protein_sequence_sha256"] else "drift"
                if expected and expected != row["protein_sequence_sha256"]:
                    raise ValueError(f"Sequence drift against V17 source audit: {gene_id}")
            if coords and bakta and source_protein:
                bkkey = (coords["contig_id"],coords["start"],coords["end"],coords["strand"])
                exact = [f for f in bakta_candidates.get(bkkey,[]) if sequence_digest(f.get("aa", ""))==row["protein_sequence_sha256"]]
                for f in exact:
                    crosswalks.append({"lane_id":lid,"proteome_id":pid,"locus_id":locus_id,"query_gene_id":gene_id,
                        "bakta_locus_tag":f.get("locus"),"bakta_feature_id":f.get("id"),"bakta_version":bakta.get("version",{}),
                        "source":bakta_source,"identity_rule":"explicit_orig_id_and_equal_contig_sequence_and_equal_coordinates_strand_and_protein_SHA256",
                        "status":"exact_locus_translation_crosswalk" if len(exact)==1 else "ambiguous_duplicate_Bakta_features",
                        "product":f.get("product"),"gene_symbols":f.get("genes",[]),"db_xrefs":f.get("db_xrefs",[]),
                        "independent_function_validation":False})
                row["bakta_exact_crosswalk_count"] = len(exact)
            loci.append(row)
        if len(loci) % 100 < len(keys):
            print(f"SEQUENCES {len(loci)} selected source-scoped loci checked",flush=True)
    # Sequence-identical proteins remain distinct loci/assemblies.
    shared = defaultdict(list)
    for row in loci:
        if row["protein_sequence_sha256"]:
            shared[row["protein_sequence_sha256"]].append(row["locus_id"])
    identities = [{"sequence_sha256":h,"locus_ids":sorted(ids),"relation":"same_normalized_protein_sequence_not_same_locus_or_organism"} for h,ids in sorted(shared.items()) if len(ids)>1]
    write_json(out/"loci.json",loci)
    write_json(out/"bakta-crosswalks.json",crosswalks)
    write_json(out/"shared-protein-digests.json",identities)
    write_json(out/"sequence-issues.json",issues)
    write_json(out/"sources.json",list(sources.values()))
    write_tsv(out/"locus-audit.tsv",[{k:v for k,v in row.items() if k not in ("protein_header",)} for row in loci])
    receipt={"status":"sequence_audit_complete_with_explicit_gaps","selected_loci":len(loci),
        "sequence_available":sum(r["sequence_available"] for r in loci),
        "translation_verified":sum(r["translation_check"]["admit_encodes"] for r in loci),
        "translation_statuses":dict(Counter(r["translation_check"]["status"] for r in loci)),
        "source_locus_IDs_are_not_deduplicated_by_sequence":True,
        "bakta_exact_crosswalk_rows":sum(r["status"]=="exact_locus_translation_crosswalk" for r in crosswalks),
        "shared_sequence_digest_groups":len(identities),"issues":len(issues),
        "elapsed_seconds":round(time.monotonic()-started,3),"code_sha256":executing_code_sha256,
        "outputs":{p.name:digest(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='sequence-receipt.json'}}
    write_json(out/"sequence-receipt.json",receipt)
    return receipt


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",type=Path,default=Path.cwd())
    parser.add_argument("--staging",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(extract(args.repo.resolve(),args.staging.resolve(),args.out.resolve()),indent=2))
