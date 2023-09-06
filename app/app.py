import streamlit as st
import datetime
import requests
import random
from PIL import Image
import pandas as pd
import numpy as np
import os
from functions import func
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from functions import shopping_list

#Setting Website Configuration
st.set_page_config(
            page_title="Replenish", # => Quick reference - Streamlit
            page_icon="🐍",
            layout="centered", # wide
            initial_sidebar_state="auto") # collapsed

#Labels

cuisines = ['Select','american','australian', 'asian','brazilian','british','cajun-creole',
            'caribbean', 'chinese', 'eastern-european', 'english',
           'french', 'german', 'greek', 'hungarian', 'indian', 'irish', 'italian',
            'japanese', 'jewish', 'korean', 'latin-american',  'mediterranean', 'mexican',
            'middle-eastern', 'moroccan',  'north-african', 'persian',  'polish', 'portuguese',
            'scandinavian', 'scottish', 'southern-soul', 'spanish', 'swedish',
            'thai', 'turkish', 'vietnamese']
#cuisines = [cuisine.title() for cuisine in cuisines]

dietary= ['Select','vegetarian','vegan', 'gluten-free','nut-free','healthy', 'dairy-free', 'egg-free', 'low-calorie', 'low-sugar',
           'high-protein', 'low-fat', 'high-fibre', 'keto', 'low-carb']
#dietary = [diet.title() for diet in dietary]


#path = os.path.join(os.path.dirname(os.getcwd()),'raw_data')
#almost_df = pd.read_csv('/Users/camillemolen/code/mfaruki/replenish_frontend/raw_data/bbc_final_df.csv')
#processed_df = func.k_means(almost_df)

#Actual DataFrame
#processed_df = pd.read_csv('/Users/camillemolen/code/mfaruki/replenish_frontend/raw_data/model_df_final.csv')
processed_df = pd.read_csv('/Users/camillemolen/code/replenish_streamlit/replenish_streamlit/raw_data/model_df_final.csv')
fail_safe_statement= "No Other Recipe"


#DataFrame for Visuals
vis_df = processed_df[['recipe_title','combined','preference','final_ingredients']]
vis_df.rename(columns={'preference':'cuisine'}, inplace=True)
vis_df.rename(columns={'final_ingredients':'Ingredients'}, inplace=True)
vis_df.rename(columns={'recipe_title':'Recipe'}, inplace=True)
vis_df['Preference']= vis_df.cuisine + " " + vis_df.combined




#Dummy Data
df= pd.DataFrame.from_dict({'title':['chili con carne', 'meatballs'],
                                 'ingredients':[['cool','stuff','wow','hungry'],['chili','meat','balls']],
                                 'cluster':[2,1],'pref':['italian', 'german']})


####################
####   WEBSITE  ####
### CONFIGURATION###
####################

#######################################         PAGE 1           ##########################
def intro():
    """Website page 1: Replenish introduction page for multifaced website."""
    ####################
    #### INTRO PAGE ####
    ####################

    row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns((.1, 2.2, .05, 1.3, .1))
    with row0_1:
        st.title('Replenish - A Recipe Optimizer')
        st.caption('Streamlit App by [Maaviya Faruki, Camille Molen, Jayesh Mistry, Jonas Korganas](https://www.linkedin.com/in/camille-molen/)')
    with row0_2:
        st.text("")
        # path = os.path.join(os.path.dirname(os.getcwd()),'raw_data')
        imagelogo = Image.open('/Users/camillemolen/code/mfaruki/replenish_frontend/raw_data/logo.jpg')
        st.image(imagelogo, use_column_width=True)

        #blue back = #F0F2F6
        # blue text = #1A2256  - white secondary back and black prim

        # [theme]
        # primaryColor="#121111"
        # backgroundColor="#EEF3EF"
        # secondaryBackgroundColor="#DEE0DA"
        # textColor="#274724"
        # font="serif"


    st.text("")
    row2_spacer1, row2_1, row2_spacer2 = st.columns((.1, 3.2, .1))
    with row2_1:
        st.markdown("""*Have you ever struggled with groceries and found yourself wondering:
                    what do I need to buy, do I really need this much broccoli if it is only for one recipe?
                    Have you ever felt like you are repeatedly throwing away expired food that you only used
                    for one meal but never had the chance to incorporate into another?
                    Well Replenish is here to solve these problems and minimise both your shopping expenses,
                    food wastage, and contribution to Carbon Footprint!*""")


    ####################
    ### CHECKOUT DATA ##
    ####################

    row3_spacer1, row3_1, row3_spacer2 = st.columns((.2, 7.1, .2))
    with row3_1:
        st.subheader("Currently selected recipes:")

    row4_spacer1, row4_1, row4_spacer2, row4_2, row4_spacer3, row4_3, row4_spacer4   = st.columns((.2, 1.6, .2, 1.6, .2, 1.6, .2))
    with row4_1:
        unique_recipes_in_df = len(processed_df.recipe_title)
        str_games = str(unique_recipes_in_df) + " Recipes"
        st.markdown(str_games)
    with row4_2:
        num_clusters_in_df = max(np.unique(processed_df.cluster))
        str_clusters = str(num_clusters_in_df) + " Ingredient Clusters"
        st.markdown(str_clusters)
    with row4_3:
        total_preferences_in_df = len(np.unique(processed_df.preference)) + len(dietary)
        str_preferences =str(total_preferences_in_df) + " Preferences"
        st.markdown(str_preferences)

    #Click to see data button

    st.markdown("")
    see_data = st.expander('Click here for an insight into the dataset 👉')
    with see_data:
        st.dataframe(data=vis_df[['Recipe','Preference','Ingredients']].drop_duplicates().set_index('Recipe').head())
    st.text("")

    ####################
    ####  FOUNDERS  ####
    ####################

    text = 'Meet The Founders'
    st.markdown(
            f"<h1 style='text-align: center;'>{text}</h1>",
            unsafe_allow_html=True)
    st.write("--------------")

    row5_spacer1,row5_1,row5_spacer2,row_5_2,row5_spacer3,row5_3,row5_spacer4,row5_4,row5_spacer5 = st.columns((.1, 1.6, .1, 1.6, .1, 1.6, .1, 1.6, .1))

    path = os.path.join(os.path.dirname(os.getcwd()),'raw_data')

    with row5_1:
        image = Image.open(os.path.join((path),'maaviya.jpg'))
        #st.subheader("***Maaviya Faruk***")
        maaviya = 'Maaviya Faruk'
        st.markdown(
            f"<h3 style='text-align: center;'><i>{maaviya}</i></h1>",
            unsafe_allow_html=True)
        st.image(image, use_column_width=True)
        st.caption("maaviya emial")

    with row_5_2:
        image1 = Image.open(os.path.join((path),'jay.jpeg'))
        #st.subheader("***Jayesh Mistry***")
        jay = 'Jayesh Mistry'
        st.markdown(
            f"<h3 style='text-align: center;'><i>{jay}</i></h1>",
            unsafe_allow_html=True)
        st.image(image1, use_column_width=True)
        st.caption("jayemail")

    with row5_3:
        image2 = Image.open(os.path.join((path),'jonas_work.png'))
        #st.subheader("***Jonas Korganas***")
        jonas = 'Jonas Korganas'
        st.markdown(
            f"<h3 style='text-align: center;'><i>{jonas}</i></h1>",
            unsafe_allow_html=True)
        st.image(image2, use_column_width=True)
        st.caption("jonasemail")

    with row5_4:
        image3 = Image.open(os.path.join((path),'camille.jpg'))
        #st.subheader("***Camille Molen***")
        camille = 'Camille Molen'
        st.markdown(
            f"<h3 style='text-align: center;'><i>{camille}</i></h1>",
            unsafe_allow_html=True)
        st.image(image3, use_column_width=True)
        st.caption("calinked in")

    st.write("--------------")

    st.markdown("""*You can find the source code in the
                [Replenish Repository](https://github.com/mfaruki/replenish)*""")
    st.markdown("""*If you are interested in investing in the Replenish
                goal feel free to contact the team!*""")











#########################################################################         PAGE 2           ##########################

def output():
    """ Website page 2:
    Given the user's food-preferences the user is returned a vareity of recipes sorted by star rating/
    User then states which recipe they chose and then they want recipes of a similar genre or very different.
    The user is then given, according their choice of similar or different recipes,
    recipes of which use most similar ingredients.
    Finally the user is returned a shopping list """

    st.title('Find Waste-Minimizing Recipes with Replenish!')

    row6_spacer1, row6_1, row6_spacer2 = st.columns((.2, 7.1, .2))
    with row6_1:
        st.subheader('Recipe Finder')
        st.markdown('Find recipes with similar ingredients according to the preference(s)...')

    ####################
    ###  PREFERENCES ###
    ####################

    row7_spacer1, row7_1, row7_spacer2, row7_2, row7_spacer3 = st.columns((.2, 2.3, .2, 2.3, .2))
    with row7_1:
        cuisine_pref = st.selectbox ("Cuisines", cuisines,key = 'cuis')
        if cuisine_pref != 'Select':
            cuis_df= (processed_df[processed_df.preference ==cuisine_pref])
            cuis_star_sorted_df=cuis_df[cuis_df['stars']!='n'].reset_index(drop=True).sort_values(by='stars', ascending=False)
        else:
            cuis_star_sorted_df=processed_df[processed_df['stars']!='n'].reset_index(drop=True).sort_values(by='stars', ascending=False)

        #st.write(cuis_star_sorted_df)


    with row7_2:
        diet_pref = st.selectbox ("Dietary", dietary,key = 'diet')
        if diet_pref != 'Select':
            diet_df= (processed_df[processed_df.combined.str.contains(diet_pref)])
            diet_star_sorted_df=diet_df[diet_df['stars']!='n'].reset_index(drop=True).sort_values(by='stars', ascending=False)
        else:
            diet_star_sorted_df=processed_df[processed_df['stars']!='n'].reset_index(drop=True).sort_values(by='stars', ascending=False)

        #st.write(diet_star_sorted_df)


    st.write("--------------")

    #centralized button for generating first pick
    col1, col2, col3 , col4, col5 = st.columns(5)

    with col1:
        pass
    with col2:
        pass
    with col4:
        pass
    with col5:
        pass
    with col3 :
        center_button = st.button('Generate Top Picks')

    ####################
    ###  1ST RECIPES ###
    ### SEARCH FUNC  ###
    ####################

    carb=1000  ######------------------DELETE LATER!!!!

    if center_button:

        #title, plot, list carbon

        #title, ingredients for 3 recipes of top stars for chosen category

        if cuisine_pref != 'Select' and diet_pref != 'Select':
            initial_df= diet_star_sorted_df
            initial_df = (initial_df[initial_df.preference ==cuisine_pref]).reset_index(drop=True)

        if cuisine_pref == 'Select' and diet_pref == 'Select':
            initial_df = cuis_star_sorted_df

        if cuisine_pref != 'Select' and diet_pref == 'Select':
            initial_df = cuis_star_sorted_df

        if diet_pref != 'Select' and cuisine_pref == 'Select':
            initial_df= diet_star_sorted_df

        initial_df.reset_index(drop=True,inplace=True)

        #st.write(initial_df)

        row8_spacer1,row8_1,row8_spacer2,row8_2,row8_spacer3,row8_3,row8_spacer4 = st.columns((.05, 1, .05, 1, .05, 1, .05))

        with row8_1:
            recipe_1 = f'1. {str(initial_df.recipe_title[0])}'
            st.subheader(recipe_1)
            st.text(f"Carbon Footprint:{carb}")
            for item in initial_df.ingredients[0].split(','):
                if item != " ":
                    st.write(f"- {item}")

        with row8_2:
            try:

                recipe_2 = f'2. {str(initial_df.recipe_title[1])}'
                st.subheader(recipe_2)
                st.text(f"Carbon Footprint:{carb}")
                for item in initial_df.ingredients[1].split(','):
                    if item != " ":
                        st.write(f"- {item}")
            except:
                st.subheader(fail_safe_statement)

        with row8_3:
            try:
                recipe_2 = f'3. {str(initial_df.recipe_title[2])}'
                st.subheader(recipe_2)
                st.text(f"Carbon Footprint:{carb}")
                for item in initial_df.ingredients[2].split(','):
                    if item != " ":
                        st.write(f"- {item}")
            except:
                st.subheader(fail_safe_statement)

    st.write("--------------")


    col2_1, col2_2, col2_3 , col2_4, col2_5 = st.columns(5)

    with col2_1:
        pass
    with col2_2:
        pass
    with col2_4:
        pass
    with col2_5:
        pass
    with col2_3 :
        recipe_pick = st.selectbox('Pick a Recipe', [1,2,3])


    st.write("--------------")

    ####################
    ###    SIMILAR   ###
    ###    RECIPES & ###
    ###  INGREDIENTS ###
    ####################

    st.text("Click below for more recipes!")

    if st.button("Stay in your comfort zone"):

        row10_spacer1,row10_1,row10_spacer2,row10_2,row10_spacer3,row10_3,row10_spacer4 = st.columns((.05, 1, .05, 1, .05, 1, .05))
        if cuisine_pref != 'Select' and diet_pref != 'Select':
            similar_df= diet_star_sorted_df
            similar_df = (similar_df[similar_df.preference ==cuisine_pref]).reset_index(drop=True)

        if cuisine_pref == 'Select' and diet_pref == 'Select':
            similar_df = cuis_star_sorted_df

        if cuisine_pref != 'Select' and diet_pref == 'Select':
            similar_df = cuis_star_sorted_df

        if diet_pref != 'Select' and cuisine_pref == 'Select':
            similar_df= diet_star_sorted_df


        similar_df.reset_index(drop=True,inplace=True)
        #st.write(similar_df)

        final_similar_df=(similar_df[similar_df.cluster== similar_df.cluster[recipe_pick-1]]).reset_index(drop=True)
        index1 = final_similar_df[final_similar_df['recipe_title'] == similar_df.recipe_title[recipe_pick-1]].index.tolist()[0]

        #############################################################
        ################# COSINE SIMILARITY SORTING #################
        #############################################################

        ingredients_cluster_list1 = final_similar_df.clean_text.tolist()
        vectorizer1 = TfidfVectorizer()
        vectors1 = vectorizer1.fit_transform(ingredients_cluster_list1)
        similarity1 = cosine_similarity(vectors1)
        similar1= pd.DataFrame(similarity1)
        final_similar_df['sim']= similar1[index1]

        final_similar_df=final_similar_df.sort_values(by='sim',ascending=False)


#       ################# TITLE INGREDIENT DEETS per recipe
        with row10_1:
            new_recipe1 = str(final_similar_df.recipe_title[0])
            st.subheader(new_recipe1)
            st.text(f"Carbon Footprint:{carb}")
            for item in final_similar_df.ingredients[0].split(','):
                if item != " ":
                    st.write(f"- {item}")

        with row10_2:
            try:
                new_recipe2 = str(final_similar_df.recipe_title[1])
                st.subheader(new_recipe2)
                st.text(f"Carbon Footprint:{carb}")
                for item in final_similar_df.ingredients[1].split(','):
                    if item != " ":
                        st.write(f"- {item}")
            except:
                st.subheader(fail_safe_statement)

        with row10_3:
            try:
                new_recipe3 = str(final_similar_df.recipe_title[2])
                st.subheader(new_recipe3)
                st.text(f"Carbon Footprint:{carb}")
                for item in final_similar_df.ingredients[2].split(','):
                    if item != " ":
                        st.write(f"- {item}")
            except:
                st.subheader(fail_safe_statement)


        ####################
        ## SHOPPING LIST ###
        ####################
        row11_spacer1,row11_1,row11_spacer2,row11_2, row11_spacer3 = st.columns((.2, 1.6, .2, 1.6, .2))

        shopping_index_list = []
        try:
            ind1 = processed_df[processed_df.recipe_title == final_similar_df.recipe_title[0]].index.tolist()[0]
            shopping_index_list.append(ind1)

        except:
            None
        try:
            ind2 = processed_df[processed_df.recipe_title == final_similar_df.recipe_title[1]].index.tolist()[0]
            shopping_index_list.append(ind2)
        except:
            None
        try:
            ind3 = processed_df[processed_df.recipe_title == final_similar_df.recipe_title[2]].index.tolist()[0]
            shopping_index_list.append(ind3)
        except:
            None



        # fridge = shopping_list.final_dataframe(shopping_index_list)
        fridge = pd.DataFrame(shopping_list.final_dataframe(shopping_index_list))
        fridge = fridge[fridge['quantity_x']!=0]
        with row11_1:
            st.subheader("Your Shopping List:")
            for idx, row in fridge.iterrows():
                # st.write(row)
                st.write(f"- {row['product']} : {row['quantity_x']} {row['unit']}")

        with row11_2:
            st.subheader(f"Your Shopping List's Carbon Foodprint is: {carb}")

        st.write("---------")


    ####################
    # DIVERSE RECIPES ##
    ###    SIMILAR   ###
    ###  INGREDIENTS ###
    ####################

    if st.button("Experiment with your tastebuds"): ################################################ #### ##### #######

        if cuisine_pref != 'Select' and diet_pref != 'Select':
            diff_df= diet_star_sorted_df
            different_df = (diff_df[diff_df.dietary !=cuisine_pref]).reset_index(drop=True)
            fun_df = (diff_df[diff_df.preference ==cuisine_pref]).reset_index(drop=True)

        if cuisine_pref == 'Select' and diet_pref == 'Select':
            different_df = cuis_star_sorted_df
            fun_df = cuis_star_sorted_df

        if cuisine_pref != 'Select' and diet_pref == 'Select':
            different_df = (processed_df[processed_df.preference !=cuisine_pref])
            different_df=different_df[different_df['stars']!='n'].reset_index(drop=True).sort_values(by='stars', ascending=False)
            fun_df = cuis_star_sorted_df


        if diet_pref != 'Select' and cuisine_pref == 'Select':
            different_df= diet_star_sorted_df
            fun_df = diet_star_sorted_df


        chosen_recipe_row = pd.DataFrame(fun_df.iloc[recipe_pick-1]).T
        diff_with_original = pd.concat([different_df,chosen_recipe_row])
        diff_with_original.reset_index(drop=True,inplace=True)
        fun_df.reset_index(drop=True,inplace=True)


        row10_spacer1,row10_1,row10_spacer2,row10_2,row10_spacer3,row10_3,row10_spacer4 = st.columns((.05, 1, .05, 1, .05, 1, .05))

        final_diff_df=(diff_with_original[diff_with_original.cluster==fun_df.cluster[recipe_pick-1]]).reset_index(drop=True).drop_duplicates()
        index2 = final_diff_df[final_diff_df.recipe_title == fun_df.recipe_title[recipe_pick-1]].index.tolist()[0]

        #############################################################
        ################# COSINE SIMILARITY SORTING #################
        #############################################################

        ingredients_cluster_list2 = final_diff_df.clean_text.tolist()
        vectorizer2 = TfidfVectorizer()
        vectors2 = vectorizer2.fit_transform(ingredients_cluster_list2)
        similarity2 = cosine_similarity(vectors2)
        similar2= pd.DataFrame(similarity2)
        final_diff_df['sim']= similar2[index2]

        final_diff_df=final_diff_df.sort_values(by='sim',ascending=False).reset_index(drop=True)

        #st.write(final_diff_df)


        ################## TITLE INGREDIENT DEETS per recipe
        with row10_1:
            diff_recipe1 = str(final_diff_df.recipe_title[0])
            st.subheader(diff_recipe1)
            st.text(f"Carbon Footprint:{carb}")
            for item in final_diff_df.ingredients[0].split(','):
                if item != " ":
                    st.write(f"- {item}")

        with row10_2:
            try:
                diff_recipe2 = str(final_diff_df.recipe_title[1])
                st.subheader(diff_recipe2)
                st.text(f"Carbon Footprint:{carb}")
                for item in final_diff_df.ingredients[1].split(','):
                    if item != " ":
                        st.write(f"- {item}")
            except:
                st.subheader(fail_safe_statement)

        with row10_3:
            try:

                diff_recipe3 = str(final_diff_df.recipe_title[2])
                st.subheader(diff_recipe3)
                st.text(f"Carbon Footprint:{carb}")
                for item in final_diff_df.ingredients[2].split(','):
                    if item != " ":
                        st.write(f"- {item}")
            except:
                st.subheader(fail_safe_statement)

        st.write("------")

    ####################
    ## SHOPPING LIST ###
    ####################

        row11_spacer1,row11_1,row11_spacer2,row11_2, row11_spacer3 = st.columns((.2, 1.6, .2, 1.6, .2))

        shopping_index_list = []
        try:
            ind1 = processed_df[processed_df.recipe_title == final_diff_df.recipe_title[0]].index.tolist()[0]
            shopping_index_list.append(ind1)

        except:
            None
        try:
            ind2 = processed_df[processed_df.recipe_title == final_diff_df.recipe_title[1]].index.tolist()[0]
            shopping_index_list.append(ind2)
        except:
            None
        try:
            ind3 = processed_df[processed_df.recipe_title == final_diff_df.recipe_title[2]].index.tolist()[0]
            shopping_index_list.append(ind3)
        except:
            None



        # fridge = shopping_list.final_dataframe(shopping_index_list)
        fridge = pd.DataFrame(shopping_list.final_dataframe(shopping_index_list))
        fridge = fridge[fridge['quantity_x']!=0]
        st.write(fridge)
        with row11_1:
            st.subheader("Your Shopping List:")
            for idx, row in fridge.iterrows():
                # st.write(row)
                st.write(f"- {row['product']} : {row['quantity_x']} {row['unit']}")

        with row11_2:
            st.subheader(f"Your Shopping List's Carbon Foodprint is: {carb}")

        st.write("---------")

################################################################################         PAGE 3           ##########################

def graphing():
    """Third website page will show the distribution of preferences across the clusters
    determined by Replenish's model, where the cluster to be seen is inputted/chosen by
    the viewer."""
    st.title("Graphing Replenish's Cluster Distributions!")
    select = st.selectbox("Select a cluster group you would like to observe", range(1, max(processed_df.cluster)+2))

    if st.button("Submit"):
        st.text("Below is the distribution of preference categories")
        st.text(f"within the ingredient-based cluster {select}")

        cluster_counts = processed_df.groupby('cluster')['recipe_title'].count().reset_index()
        cluster_counts.rename(columns={'recipe_title': 'Number of Recipes'}, inplace=True)

        cluster_df = processed_df[processed_df['cluster'] == select]
        cuisine_counts = cluster_df['preference'].value_counts()
        fig_df = pd.DataFrame(cuisine_counts).reset_index()

        fig= plt.figure(figsize=(20, 10), facecolor=(0,0,0,0))
        sns.barplot(data=fig_df, x = 'index', y = 'preference')
        plt.xlabel('Preference', fontsize=25, fontname="Times New Roman",fontweight="bold")
        plt.ylabel('Number of Recipes', fontsize=25, fontname="Times New Roman",fontweight="bold")
        plt.title(f'Cluster {select} - Preference vs. Number of Recipes', fontsize=40, fontname="Times New Roman",fontweight="bold")
        plt.xticks(rotation=45, fontsize=15, fontname="Times New Roman")
        plt.yticks(fontsize=15, fontname="Times New Roman")

        st.pyplot(fig)

page_names_to_funcs = {
    "About Us": intro,
    "Recipe Generator": output,
    "Data Graphing": graphing,
}

demo_name = st.sidebar.selectbox("Click to find out more 👉", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()
