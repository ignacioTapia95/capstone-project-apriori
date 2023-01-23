import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import fpgrowth

class AprioriSys():

    def __init__(self, df, porduct_column:str, min_support:float):
        self.df = df
        self.porduct_column = porduct_column
        self.product_index = self.df.columns.get_loc(self.porduct_column)
        self.min_support = min_support
        self.df_full_dummy = None
        self.df_dummy = None
        self.frq_items = None
        self.rules = None

    def trx_2_dummy(self):

        # Generar Variables Dummy (Productos -> Dummy)
        self.df_dummy = pd.get_dummies(self.df, prefix='', columns=[self.porduct_column])
        
        colname_tmp = []
        for colname in self.df_dummy.columns:
            if colname[0] == '_':
                colname_tmp.append(colname[1:])
            else:
                colname_tmp.append(colname)
        
        self.df_dummy.columns = colname_tmp

        # Agrupar Transacciones por boleta -> Genera una sola fila por boleta
        group_columns = list(self.df_dummy.columns[:self.product_index])
        self.df_dummy = self.df_dummy.groupby(group_columns).sum().reset_index()

        # Filtrar Compras con un sólo item
        self.df_dummy = self.df_dummy[self.df_dummy.iloc[:,self.product_index:].sum(axis='columns') > 1]

        # Transformar dataframe de numérico a booleano (Recomendación de mlxtend.frequent_patterns.apriori)
        self.df_dummy.iloc[:,self.product_index:] = self.df_dummy.iloc[:,self.product_index:].apply(lambda x: [True if y >= 1 else False for y in x])

        # Reordenar DF -> sólo considera columnas booleanas
        self.df_dummy.reset_index(drop=True, inplace=True)

        self.df_full_dummy = self.df_dummy #Contiene fecha, numero de trx, y id_cliente
        self.df_dummy = self.df_dummy.iloc[:,self.product_index:]

        return self.df_dummy


    def fit_rules(self):

        df = self.trx_2_dummy()

        # Building the model
        self.frq_items = apriori(df, min_support = self.min_support, use_colnames = True)

        # Collecting the inferred rules in a dataframe
        self.rules = association_rules(self.frq_items, metric ="lift", min_threshold = 1)
        return self.rules

    def predict(self, basket, metric, n_recommendations):
                
        product_combinations = fpgrowth(basket, use_colnames=True)
        product_combinations['n_items'] = product_combinations['itemsets'].apply(lambda x: len(x))

        basket_combinations = product_combinations.loc[max(product_combinations.index),'itemsets']
        basket_combinations = set(basket_combinations)

        itemset = basket.reset_index(drop=True).T
        itemset = list(itemset[itemset[0] == True].index)

        recommendations = product_combinations.merge(
                                self.rules,
                                left_on='itemsets',
                                right_on='antecedents'
                                )\
                            .sort_values(['n_items', metric], ascending=[False, False])\
                            .reset_index(drop=True)

        top_rec = [x for x in recommendations['consequents']]

        # Flat List
        top_rec = [x for i in top_rec for x in i]
        
        final_rec = []
        for item in top_rec:
            if item in final_rec:
                continue
            elif item in itemset:
                continue
            else:
                final_rec.append(item)


        
        final_rec = final_rec[:n_recommendations]


        rec_dict = {f'rec_{y}':x for x,y in zip(final_rec,range(n_recommendations))}

        return rec_dict