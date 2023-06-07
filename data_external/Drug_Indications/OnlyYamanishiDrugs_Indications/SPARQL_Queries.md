# SPARQL-Queries

This is a collection of the SPARQL-Queries, used to extract the relevant data for each 
Yamanishi subset {enzyme, gpcr, ion_channel, nuclear_receptor, whole_yamanishi}. 
The RML-Mapping file (.ttl), that was used to create the Knowledge Graph, can also be found in this directory.

## Enzyme
```
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX ex: <http://example.com/>
PREFIX db: <https://go.drugbank.com/drugs/> 
PREFIX kegg: <https://www.kegg.jp/entry/>

SELECT DISTINCT ?s ?o
WHERE {
    ?s ex:hasIndication ?o .
    ?x ex:hasIndication ?o .
    
    ?s rdf:type db:drug .
    ?x rdf:type db:drug .
    
    ?s ex:fromSource ex:YamanishiEnzyme .
    ?x ex:fromSource ex:YamanishiEnzyme .
    
    FILTER ( ?s != ?x ) 
} LIMIT 100000
```

## GPCR
```
    ...
    ?s ex:fromSource ex:YamanishiGPCR .
    ?x ex:fromSource ex:YamanishiGPCR .
    ...
```

## Ion Channel
```
    ...
    ?s ex:fromSource ex:YamanishiIonChannel .
    ?x ex:fromSource ex:YamanishiIonChannel .
    ...
```

## Nuclear Receptor
```
    ...
    ?s ex:fromSource ex:YamanishiNuclearReceptor .
    ?x ex:fromSource ex:YamanishiNuclearReceptor .
    ...
```

## Whole Yamanishi
```
    ...
    ?s ex:fromSource ex:Yamanishi .
    ?x ex:fromSource ex:Yamanishi .
    ...
```

