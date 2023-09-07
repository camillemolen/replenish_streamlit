# def final_dataframe(recipe_index:list):

#     df = cleaning_df(recipe_index)
#     try:
#         df1 = pd.DataFrame(testing_api(df['modified'][recipe_index[0]]))
#     except:
#         df1=pd.DataFrame()
#     try:
#         df2 = pd.DataFrame(testing_api(df['modified'][recipe_index[1]]))
#     except:
#         df2=pd.DataFrame()
#     try:
#         df3 = pd.DataFrame(testing_api(df['modified'][recipe_index[2]]))
#     except:
#         df3=pd.DataFrame()

#     #final_df = pd.concat(all_df)
#     final_df = pd.concat([df1, df2, df3])

#     final_df2 = final_df.groupby('product').sum()[['quantity']].reset_index().sort_values('quantity', ascending=False)

#     final_df3 = final_df2.merge(final_df, on='product', how='inner')
#     final_df3 = final_df3.drop('quantity_y', axis=1)
#     final_df3.drop_duplicates(inplace=True)
#     # final_df3 = final_df3['product'].drop_duplicates()
#     final_df3['unit'].fillna(' ', inplace=True)
#     #final_df3 = final_df3.groupby(['product']).sum().reset_index()

#     return final_df3



