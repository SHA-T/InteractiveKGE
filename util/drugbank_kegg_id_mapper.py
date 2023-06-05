import numpy as np
import pandas as pd
import json
from SPARQLWrapper import SPARQLWrapper, JSON


def set_default(obj):
    """
    Returns a list version of obj, if obj is a set.
    """
    if isinstance(obj, set):
        return list(obj)
    raise TypeError


def get_mapper():
    """
    Creates a KeggID-DrugBankID mapper for drugs from querying wikidata.org and merging it with a mapper from
    https://github.com/iit-Demokritos/drug_id_mapping/blob/main/drug-mappings.tsv.
    :return:
        A dictionary of the format:
        {
            'KeggID_1': 'DrugBankID_1',
            'KeggID_2': 'DrugBankID_2',
            ...
        }
    """
    # SPARQL-querying wikidata.org
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.setQuery("""
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>

    SELECT ?item ?drugbank_id ?kegg_id
    WHERE
    {
      ?item wdt:P31 wd:Q11173 .             # ?item  Property:instance_of  Entity:ChemicalCompound  (Item is instance of ChemicalCompound)
      ?item wdt:P715 ?drugbank_id .         # ?item  Property:DrugBank_ID  ?drugbank_id             (Item has DrugBank ID)
      ?item wdt:P665 ?kegg_id               # ?item  Property:KEGG_ID      ?kegg_id                 (Item has KEGG ID)
    }
    """)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    # Load first mapper from query response
    df_1 = pd.json_normalize(results['results']['bindings'])[['drugbank_id.value', 'kegg_id.value']]
    df_1 = df_1.rename(columns={'drugbank_id.value': 'drugbank_id',
                                       'kegg_id.value': 'kegg_id'})
    df_1['drugbank_id'] = 'DB' + df_1['drugbank_id'].astype(str)    # Add 'DB' before drugbank_ids
    df_1 = df_1[~df_1.kegg_id.str.contains("C")]    # Remove entries, where kegg_id starts with 'C' identifier
    df_1 = df_1.drop_duplicates()                                   # Drop duplicate rows
    df_1 = df_1.drop_duplicates(subset=['kegg_id'], keep='first')   # Typically, doesn't occur in this data
    df_1 = df_1.drop_duplicates(subset=['drugbank_id'], keep='first')  # Some drugbank_ids have 2 kegg_ids, drop second
    df_1 = df_1.reset_index()
    print('\ndf_1 (mapper from query response):\n', df_1)

    # Load second mapper from Github
    tsv_url = 'https://raw.githubusercontent.com/iit-Demokritos/drug_id_mapping/main/drug-mappings.tsv'
    df_2 = pd.read_csv(tsv_url, sep='\t', on_bad_lines='skip')[['drugbankId', 'kegg_id']]
    df_2 = df_2.rename(columns={'drugbankId': 'drugbank_id'})
    df_2 = df_2.dropna()
    df_2 = df_2.drop_duplicates()                                   # Drop duplicate rows
    df_2 = df_2.drop_duplicates(subset=['kegg_id'], keep='first')   # Some kegg_ids have 2 drugbank_ids, drop second
    df_2 = df_2.drop_duplicates(subset=['drugbank_id'], keep='first')  # Typically, doesn't occur in this data
    df_2 = df_2.reset_index()
    print('\ndf_2 (mapper from github.com/iit-Demokritos/drug_id_mapping):\n', df_2)

    # Merge both mappers
    df = pd.concat([df_2, df_1], ignore_index=True, sort=False)
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset=['kegg_id'], keep='first')
    df = df.drop_duplicates(subset=['drugbank_id'], keep='first')
    df = df.reset_index()
    print('\ndf (merged mapper):\n', df)

    # Convert mapper to dictionary
    mapper = dict(df[['kegg_id', 'drugbank_id']].values)
    print('\ndictionary:\n', mapper, '\n')

    return mapper


if __name__ == '__main__':
    mapper_dict = get_mapper()
    dump = json.dumps(mapper_dict, indent=4, default=set_default)
    f = open(f"drugbank_kegg_id_map.json", "w+")
    f.write(dump)
    f.close()

    # Load unsplitted original datasets
    drug_e = pd.read_csv('../data/Yamanishi/Unsplitted_Data/KeggID_Original/bind_orfhsa_drug_e.csv')
    drug_gpcr = pd.read_csv('../data/Yamanishi/Unsplitted_Data/KeggID_Original/bind_orfhsa_drug_gpcr.csv')
    drug_ic = pd.read_csv('../data/Yamanishi/Unsplitted_Data/KeggID_Original/bind_orfhsa_drug_ic.csv')
    drug_nr = pd.read_csv('../data/Yamanishi/Unsplitted_Data/KeggID_Original/bind_orfhsa_drug_nr.csv')
    drug_targetsAll = pd.read_csv('../data/Yamanishi/Unsplitted_Data/KeggID_Original/bind_orfhsa_drug_targetsAll.csv')

    # Map DrugBank-IDs
    drug_e['drugbank_id'] = drug_e.drug.map(mapper_dict)
    drug_gpcr['drugbank_id'] = drug_gpcr.drug.map(mapper_dict)
    drug_ic['drugbank_id'] = drug_ic.drug.map(mapper_dict)
    drug_nr['drugbank_id'] = drug_nr.drug.map(mapper_dict)
    drug_targetsAll['drugbank_id'] = drug_targetsAll.drug.map(mapper_dict)
    print(drug_e)
    print(drug_gpcr)
    print(drug_ic)
    print(drug_nr)
    print(drug_targetsAll)

    # Count 'NaN'-entries
    print('\nAmount of NaN-entries:')
    print(drug_e['drugbank_id'].isna().sum())
    print(drug_gpcr['drugbank_id'].isna().sum())
    print(drug_ic['drugbank_id'].isna().sum())
    print(drug_nr['drugbank_id'].isna().sum())
    print('all:', drug_targetsAll['drugbank_id'].isna().sum())

    # Replace Kegg-IDs with DrugBank-IDs if not NaN, else keep Kegg-ID
    drug_e['drug'] = np.where(~drug_e['drugbank_id'].isnull(), drug_e['drugbank_id'], drug_e['drug'])
    drug_gpcr['drug'] = np.where(~drug_gpcr['drugbank_id'].isnull(), drug_gpcr['drugbank_id'], drug_gpcr['drug'])
    drug_ic['drug'] = np.where(~drug_ic['drugbank_id'].isnull(), drug_ic['drugbank_id'], drug_ic['drug'])
    drug_nr['drug'] = np.where(~drug_nr['drugbank_id'].isnull(), drug_nr['drugbank_id'], drug_nr['drug'])
    drug_targetsAll['drug'] = np.where(~drug_targetsAll['drugbank_id'].isnull(), drug_targetsAll['drugbank_id'], drug_targetsAll['drug'])

    # Save datasets with mapped IDs
    drug_e.drop(columns='drugbank_id').to_csv(
        '../data/Yamanishi/Unsplitted_Data/DrugBankID_Mapped/bind_orfhsa_drug_e.csv', index=False)
    drug_gpcr.drop(columns='drugbank_id').to_csv(
        '../data/Yamanishi/Unsplitted_Data/DrugBankID_Mapped/bind_orfhsa_drug_gpcr.csv', index=False)
    drug_ic.drop(columns='drugbank_id').to_csv(
        '../data/Yamanishi/Unsplitted_Data/DrugBankID_Mapped/bind_orfhsa_drug_ic.csv', index=False)
    drug_nr.drop(columns='drugbank_id').to_csv(
        '../data/Yamanishi/Unsplitted_Data/DrugBankID_Mapped/bind_orfhsa_drug_nr.csv', index=False)
    drug_targetsAll.drop(columns='drugbank_id').to_csv(
        '../data/Yamanishi/Unsplitted_Data/DrugBankID_Mapped/bind_orfhsa_drug_targetsAll.csv', index=False)

    # Check number of unique drugs that don't have a DrugBankID
    non_mapped = drug_targetsAll.drop(columns='protein')
    non_mapped = non_mapped[non_mapped['drugbank_id'].isnull()]
    non_mapped_amount = len(non_mapped.drop_duplicates())
    # Last time checked, the amount of unique drugs that don't have a DrugBankID was 217.
    print(f'The Yamanishi dataset contains 791 unique drugs, out of which {non_mapped_amount} do not have a DrugBankID.')
