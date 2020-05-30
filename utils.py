"""
utils.py
Utiliy functions for DeepND 
Bilkent University, Department of Computer Engineering
Ankara, 2020
"""

import numpy as np
import pandas as pd

def intersect_lists(source_list, target_list, lookup):
    source_indices = []
    target_indices = []
    result = []
    not_found_indices = []
    not_found_item = []
    not_found_count = 0
    for source_index, source_item in enumerate(source_list):
        found = 0
        for target_index, target_item in enumerate(target_list):
            if source_item.lower() == target_item.lower():
                source_indices.append(source_index)
                target_indices.append(target_index)
                result.append(target_item.lower())
                found = 1
                break
        if found == 0:
            for target_index, target_item in enumerate(target_list):
                #if source_item.lower() in lookup and target_item.lower() in lookup and does_intersect(lookup[source_item.lower()], lookup[target_item.lower()]): #Dictionary search
                if source_item.lower() in lookup and target_item.lower() in lookup and lookup[source_item.lower()] and  lookup[target_item.lower()] and  lookup[source_item.lower()][-1] ==  lookup[target_item.lower()][-1] : #Dictionary search
                    source_indices.append(source_index)
                    target_indices.append(target_index)
                    result.append(target_item.lower())
                    found = 1
                    print("Found in Dictionary!", source_item , target_item)
                    break
        if found == 0:
            not_found_indices.append(source_index)
            not_found_item.append(source_item)
            not_found_count += 1
            #print("The gene {0} is not found. Not Found Count:{1}".format(source_item, not_found_count))
       
    return result, source_indices, not_found_indices, target_indices
            
def does_intersect(source_list, target_list):
    for source_item in source_list:
        for target_item in target_list:
            if source_item.lower() == target_item.lower():
                return True
    
    return False

def constructGeneDictionary(path):
    genes = dict()
    lineCount = 1
    with open(path) as tsv:
        for line in csv.reader(tsv, dialect = csv.excel_tab, delimiter  = "\t"): #You can also use delimiter="\t" rather than giving a dialect.
            if line[0] == "Approved symbol":
                continue
            
            for item in line:
                if item == "":
                    continue
                gene_item = item.split(", ")
                #if lineCount == 10282:
                  #print(gene_item)
                for comma_item in gene_item:
                    gene_list = []
                    for item2 in line:
                        if item2 == "":
                            continue
                        gene_item2 = item2.split(", ")
                        for comma_item2 in gene_item2:
                            if comma_item2 == comma_item:
                                continue
                            gene_list.append(comma_item2.lower())
                    
                    genes[comma_item.lower()] = gene_list
            lineCount += 1
            
    return genes
