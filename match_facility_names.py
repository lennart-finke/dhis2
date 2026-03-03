from pathlib import Path
import pandas as pd
import re
import Levenshtein
from tqdm import tqdm

def norm(s):
    if pd.isna(s): return ""
    s = str(s).lower().strip()
    s = re.sub(r"\s*\(.*?\)", "", s)
    s = re.sub(r"_phcu", "", s)
    s = re.sub(r"\s+phcu", "", s)
    s = re.sub(r"\s+health\s+(center|centre|post|station)$", "", s)
    s = re.sub(r"\s+hospital$", "", s)
    s = re.sub(r"\s+clinic$", "", s)
    s = re.sub(r"\s+hp$", "", s)
    s = re.sub(r"\s+hc$", "", s)
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()

def norm_region(s):
    if pd.isna(s): return ""
    s = str(s).lower().strip()
    s = re.sub(r"\s+regional state$", "", s)
    s = re.sub(r"\s+regional health bureau$", "", s)
    s = re.sub(r"\s+region$", "", s)
    s = re.sub(r"\s+city administration$", "", s)
    s = re.sub(r"-", " ", s)
    s = re.sub(r"[^a-z\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()

dhis_files = list(Path("data/dhis2/25_12_12").glob("*.csv"))
if not dhis_files: print("No DHIS2 files")
dhis = pd.concat([pd.read_csv(f, usecols=lambda c: c.startswith("orgunit") or c.startswith("organisation"), low_memory=False) for f in dhis_files], ignore_index=True)

dhis_facilities = dhis[["orgunitlevel2", "orgunitlevel3", "orgunitlevel4", "organisationunitname", "organisationunitcode", "organisationunitid"]].drop_duplicates().dropna(subset=["organisationunitname"])
dhis_facilities = dhis_facilities.rename(columns={"orgunitlevel2": "dhis_region", "orgunitlevel3": "dhis_zone", "orgunitlevel4": "dhis_woreda", "organisationunitname": "dhis_facility", "organisationunitcode": "dhis_facility_code", "organisationunitid": "dhis_facility_id"})
dhis_facilities["dhis_facility_norm"] = dhis_facilities["dhis_facility"].apply(norm)
dhis_facilities["dhis_facility_code"] = dhis_facilities["dhis_facility_code"].astype(str)
dhis_facilities["dhis_facility_id"] = dhis_facilities["dhis_facility_id"].astype(str)

region_map = {"Central Ethiopian region": "SNNP Region", "South Ethiopia": "SNNP Region", "South Ethiopia Regional State": "SNNP Region", "South Ethiopia Region": "SNNP Region", "South West Ethiopia Region": "SNNP Region", "Southern West Ethiopia Region": "SNNP Region", "Sidama Region": "SNNP Region"}
dhis_facilities["dhis_region_mapped"] = dhis_facilities["dhis_region"].replace(region_map)
dhis_facilities["dhis_region_norm"] = dhis_facilities["dhis_region_mapped"].apply(norm_region)

regional_sheets = ["SWE", "SLI", "SNNP", "SD", "OR", "GM", "BG", "AMH", "AFA", "AA", "HAR", "DD"]
mfr_list = [pd.read_excel("data/mfr/MFR List All.xlsx", sheet_name=s) for s in regional_sheets]
main_sheet = pd.read_excel("data/mfr/MFR List All.xlsx", sheet_name="MFR List All - Feb 23")
mfr_list.append(main_sheet[main_sheet["region"].astype(str).str.contains("Tigray", case=False, na=False)].copy())

mfr = pd.concat(mfr_list, ignore_index=True)
mfr_facilities = mfr[["region", "zone", "woreda", "name", "id", "dhis2_id", "latitude", "longitude"]].drop_duplicates().dropna(subset=["name"])
mfr_facilities["mfr_facility_norm"] = mfr_facilities["name"].apply(norm)
mfr_facilities["mfr_region_norm"] = mfr_facilities["region"].apply(norm_region)
mfr_facilities = mfr_facilities.rename(columns={"region": "mfr_region", "zone": "mfr_zone", "woreda": "mfr_woreda", "name": "mfr_facility", "id": "mfr_id", "dhis2_id": "mfr_dhis2_id", "latitude": "mfr_latitude", "longitude": "mfr_longitude"})
mfr_facilities["mfr_id"] = mfr_facilities["mfr_id"].astype(str)
mfr_facilities["mfr_dhis2_id"] = mfr_facilities["mfr_dhis2_id"].astype(str)

mapping_by_code = dhis_facilities.merge(mfr_facilities, left_on="dhis_facility_code", right_on="mfr_id", how="left")
matched_by_code = mapping_by_code[mapping_by_code["mfr_facility"].notna()]
print("Matched by code:", len(matched_by_code))

unmatched_after_code = dhis_facilities.drop(matched_by_code.index, errors="ignore")
mapping_by_id = unmatched_after_code.merge(mfr_facilities, left_on="dhis_facility_id", right_on="mfr_dhis2_id", how="left")
matched_by_id = mapping_by_id[mapping_by_id["mfr_facility"].notna()]
print("Matched by id:", len(matched_by_id))

unmatched_after_id = unmatched_after_code.drop(matched_by_id.index, errors="ignore")
mapping_by_name = unmatched_after_id.merge(mfr_facilities, left_on=["dhis_facility_norm", "dhis_region_norm"], right_on=["mfr_facility_norm", "mfr_region_norm"], how="left")
matched_by_name = mapping_by_name[mapping_by_name["mfr_facility"].notna()]
print("Matched by name:", len(matched_by_name))

unmatched_after_name = unmatched_after_id.drop(matched_by_name.index, errors="ignore")
matched_mfr_ids = pd.concat([matched_by_code["mfr_id"].dropna(), matched_by_id["mfr_dhis2_id"].dropna(), matched_by_name["mfr_id"].dropna()])
mfr_unmatched = mfr_facilities[~mfr_facilities["mfr_id"].isin(matched_mfr_ids)]

# Fuzzy matching
print("Starting fuzzy matching...")
mfr_grouped = {r: grp.to_dict('records') for r, grp in mfr_unmatched.groupby("mfr_region_norm")}
dhis_records = unmatched_after_name.to_dict('records')

fuzzy_matches = []
for dhis_row in tqdm(dhis_records, desc="Fuzzy"):
    d_name = dhis_row["dhis_facility_norm"]
    d_region = dhis_row["dhis_region_norm"]
    if d_region in mfr_grouped:
        for mfr_row in mfr_grouped[d_region]:
            if Levenshtein.distance(d_name, mfr_row["mfr_facility_norm"], score_cutoff=1) <= 1:
                fuzzy_matches.append({**dhis_row, **mfr_row})
                break

mapping_by_fuzzy = pd.DataFrame(fuzzy_matches) if fuzzy_matches else pd.DataFrame()
matched_by_fuzzy = mapping_by_fuzzy[mapping_by_fuzzy["mfr_facility"].notna()] if fuzzy_matches else pd.DataFrame()

mapping = pd.concat([matched_by_code, matched_by_id, matched_by_name, mapping_by_fuzzy], ignore_index=True)
mapping = mapping.rename(columns={"dhis_region": "region", "dhis_zone": "zone", "dhis_woreda": "woreda", "dhis_facility": "facility_dhis2", "dhis_facility_code": "facility_dhis2_code", "dhis_facility_id": "facility_dhis2_id", "mfr_facility": "facility_mfr", "mfr_id": "facility_mfr_id", "mfr_dhis2_id": "facility_mfr_dhis2_id", "mfr_latitude": "lat", "mfr_longitude": "lon"})
output_cols = ["region", "zone", "woreda", "facility_dhis2", "facility_dhis2_code", "facility_dhis2_id", "facility_mfr", "facility_mfr_id", "facility_mfr_dhis2_id", "lat", "lon"]
mapping = mapping[[c for c in output_cols if c in mapping.columns]].drop_duplicates()
mapping = mapping[mapping["facility_dhis2"].notna()]

total = dhis_facilities["dhis_facility"].nunique()
matched = mapping[mapping["facility_mfr"].notna()]["facility_dhis2"].nunique()
print("TOTAL MATCHED:", matched, " TOTAL:", total)

with_coords = mapping[mapping["lat"].notna() & mapping["lon"].notna() & (mapping["lat"] != 0) & (mapping["lon"] != 0)]["facility_dhis2"].nunique()
print("WITH COORDS:", with_coords, " OUT OF MATCHED:", matched)
mapping.to_csv("estimates/facility_mapping.csv", index=False)
