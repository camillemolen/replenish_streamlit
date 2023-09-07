import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.font_manager as fm

# Set Matplotlib style
style.use('default')

import seaborn as sns
from functions import shopping_list, func
from wordcloud import WordCloud, ImageColorGenerator
import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

#Setting Website Configuration
st.set_page_config(
            page_title="Replenish", # => Quick reference - Streamlit
            page_icon="🐍",
            layout="centered", # wide
            initial_sidebar_state="auto") # collapsed

with open(os.path.join(os.getcwd(), 'app','style.css')) as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

#Labels

cuisines = ['Select','american','australian', 'asian','brazilian','british','cajun-creole',
            'caribbean', 'chinese', 'eastern-european', 'english',
           'french', 'german', 'greek', 'hungarian', 'indian', 'irish', 'italian',
            'japanese', 'jewish', 'korean', 'latin-american',  'mediterranean', 'mexican',
            'middle-eastern', 'moroccan',  'north-african', 'persian',  'polish', 'portuguese',
            'scandinavian', 'scottish', 'southern-soul', 'spanish', 'swedish',
            'thai', 'turkish', 'vietnamese']
#cuisines = [cuisine.title() for cuisine in cuisines]

dietary= ['Select','vegetarian','vegan', 'gluten-free','nut-free','healthy', 'dairy-free', 'egg-free', 'low-calorie', 'low-sugar', 'low-fat', 'high-fibre', 'keto', 'low-carb']
#dietary = [diet.title() for diet in dietary]


#path = os.path.join(os.path.dirname(os.getcwd()),'raw_data')
#almost_df = pd.read_csv('/Users/camillemolen/code/mfaruki/replenish_frontend/raw_data/bbc_final_df.csv')
#processed_df = func.k_means(almost_df)

#Actual DataFrame
#processed_df = pd.read_csv('/Users/camillemolen/code/mfaruki/replenish_frontend/raw_data/model_df_final.csv')

path = os.path.join(os.getcwd(), 'raw_data')
processed_df = pd.read_csv(os.path.join(path, 'model_df_final.csv'))
#processed_df = pd.read_csv('/Users/camillemolen/code/replenish_streamlit/replenish_streamlit/raw_data/model_df_final.csv')
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
    path = os.path.join(os.getcwd(), 'raw_data')
    with row0_1:
        st.title('Maximise Taste, Minimise Waste!')
        st.caption('Streamlit App by [Maaviya Faruki, Camille Molen, Jayesh Mistry, Jonas Korganas](https://github.com/mfaruki/replenish)')
    with row0_2:
        st.text("")
        imagelogo = Image.open(os.path.join((path), 'new_logo.png'))
        st.image(imagelogo, use_column_width=True)



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
        st.header("Currently selected recipes:")

    row4_spacer1, row4_1, row4_spacer2, row4_2, row4_spacer3, row4_3, row4_spacer4   = st.columns((.2, 1.6, .2, 1.6, .2, 1.6, .2))
    with row4_1:
        unique_recipes_in_df = len(processed_df.recipe_title)
        str_games = str(unique_recipes_in_df) + " Recipes🥕"
        st.markdown(str_games)
    with row4_2:
        num_clusters_in_df = max(np.unique(processed_df.cluster))
        str_clusters = str(num_clusters_in_df + 1) + " Ingredient Clusters🍡"
        st.markdown(str_clusters)
    with row4_3:
        total_preferences_in_df = len(np.unique(processed_df.preference)) + len(dietary)
        str_preferences =str(total_preferences_in_df) + " Preferences🍣"
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

    path = os.path.join(os.getcwd(), 'raw_data')

    with row5_1:
        image = Image.open(os.path.join((path),'maaviya.jpg'))
        #st.subheader("***Maaviya Faruk***")
        maaviya = 'Maaviya Faruk'
        st.markdown(
            f"<h3 style='text-align: center;'><i>{maaviya}</i></h1>",
            unsafe_allow_html=True)
        st.image(image, use_column_width=True)
        st.caption("The Idea Master ")

    with row_5_2:
        image1 = Image.open(os.path.join((path),'jay.jpeg'))
        #st.subheader("***Jayesh Mistry***")
        jay = 'Jayesh Mistry'
        st.markdown(
            f"<h3 style='text-align: center;'><i>{jay}</i></h1>",
            unsafe_allow_html=True)
        st.image(image1, use_column_width=True)
        st.caption("The Modeller")

    with row5_3:
        image2 = Image.open(os.path.join((path),'jonas_work.png'))
        #st.subheader("***Jonas Korganas***")
        jonas = 'Jonas Korganas'
        st.markdown(
            f"<h3 style='text-align: center;'><i>{jonas}</i></h1>",
            unsafe_allow_html=True)
        st.image(image2, use_column_width=True)
        st.caption("The Preprocessor")

    with row5_4:
        image3 = Image.open(os.path.join((path),'camille.jpg'))
        #st.subheader("***Camille Molen***")
        camille = 'Camille Molen'
        st.markdown(
            f"<h3 style='text-align: center;'><i>{camille}</i></h1>",
            unsafe_allow_html=True)
        st.image(image3, use_column_width=True)
        st.caption("The Frontend ")

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
        st.subheader('Recipe Finder 🔍')
        st.markdown('Find recipes with similar ingredients according to the preference(s)...')

    ####################
    ###  PREFERENCES ###
    ####################

    row7_spacer1, row7_1, row7_spacer2, row7_2, row7_spacer3 = st.columns((.2, 2.3, .2, 2.3, .2))
    with row7_1:
        cuisine_pref = st.selectbox ("Cuisines 🥘", cuisines,key = 'cuis')
        if cuisine_pref != 'Select':
            cuis_df= (processed_df[processed_df.preference ==cuisine_pref])
            cuis_star_sorted_df=cuis_df[cuis_df['stars']!='n'].reset_index(drop=True).sort_values(by='stars', ascending=False)
        else:
            cuis_star_sorted_df=processed_df[processed_df['stars']!='n'].reset_index(drop=True).sort_values(by='stars', ascending=False)

        #st.write(cuis_star_sorted_df)


    with row7_2:
        diet_pref = st.selectbox ("Dietary 🍜", dietary,key = 'diet')
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
        center_button = st.button('Generate Top Picks🍋')

    ####################
    ###  1ST RECIPES ###
    ### SEARCH FUNC  ###
    ####################

    ######------------------DELETE LATER!!!!

    if center_button:

        #title, plot, list carbon

        #title, ingredients for 3 recipes of top stars for chosen category

        if cuisine_pref != 'Select' and diet_pref != 'Select':
            # initial_df= diet_star_sorted_df
            # initial_df = (initial_df[initial_df.preference ==cuisine_pref]).reset_index(drop=True)

            try:
                initial_df= diet_star_sorted_df
                initial_df = (initial_df[initial_df.preference ==cuisine_pref]).reset_index(drop=True)
                tester = initial_df.preference[0]

                # #initial_df = (diet_star_sorted_df[diet_star_sorted_df.preference ==cuisine_pref]).reset_index(drop=True)
                # ini= (processed_df[processed_df.combined.str.contains(diet_pref)])
                # if ini[ini['preference'] ==cuisine_pref]:
                #     initial_df = (ini[ini.preference ==cuisine_pref]).reset_index(drop=True)
                # else:
                #     initial_df= cuis_star_sorted_df


            except:
                initial_df = cuis_star_sorted_df
                st.markdown(f"""<span style='color:red'>No {diet_pref} recipes found for this cuisine</span>""", unsafe_allow_html=True)

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
            st.text(f"Difficulty: {initial_df.difficulty_level[0]}")
            for item in initial_df.ingredients[0].split(','):
                if item != " ":
                    st.write(f"- {item}")

        with row8_2:
            try:

                recipe_2 = f'2. {str(initial_df.recipe_title[1])}'
                st.subheader(recipe_2)
                st.text(f"Difficulty: {initial_df.difficulty_level[1]}")
                for item in initial_df.ingredients[1].split(','):
                    if item != " ":
                        st.write(f"- {item}")
            except:
                st.subheader(fail_safe_statement)

        with row8_3:
            try:
                recipe_2 = f'3. {str(initial_df.recipe_title[2])}'
                st.subheader(recipe_2)
                st.text(f"Difficulty: {initial_df.difficulty_level[2]}")
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
        recipe_pick = st.selectbox('Pick a Recipe \n 1️⃣, 2️⃣ or 3️⃣ ?', [1,2,3])


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
            try:
                similar_df= diet_star_sorted_df
                similar_df = (similar_df[similar_df.preference ==cuisine_pref]).reset_index(drop=True)
                tester = similar_df.preference[0]
            except:
                similar_df = cuis_star_sorted_df
                st.markdown(f"""<span style='color:red'>No {diet_pref} recipes found for this cuisine</span>""", unsafe_allow_html=True)

            # similar_df= diet_star_sorted_df
            # similar_df = (similar_df[similar_df.preference ==cuisine_pref]).reset_index(drop=True)

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

        #st.write(final_similar_df)

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
            st.text(f"Difficulty: {final_similar_df.difficulty_level[0]}")
            for item in final_similar_df.ingredients[0].split(','):
                if item != " ":
                    st.write(f"- {item}")

        with row10_2:
            try:
                new_recipe2 = str(final_similar_df.recipe_title[1])
                st.subheader(new_recipe2)
                st.text(f"Difficulty: {final_similar_df.difficulty_level[1]}")
                for item in final_similar_df.ingredients[1].split(','):
                    if item != " ":
                        st.write(f"- {item}")
            except:
                st.subheader(fail_safe_statement)

        with row10_3:
            try:
                new_recipe3 = str(final_similar_df.recipe_title[2])
                st.subheader(new_recipe3)
                st.text(f"Difficulty: {final_similar_df.difficulty_level[2]}")
                for item in final_similar_df.ingredients[2].split(','):
                    if item != " ":
                        st.write(f"- {item}")
            except:
                st.subheader(fail_safe_statement)


        st.write("---------")

        ####################
        ## SHOPPING LIST ###
        ####################

        spac1,col4_1,spac2, col4_2,spac3, col4_3,spac4 = st.columns((.05, 1, .05, 2.2, .05, 1, .05))

        with col4_1:
            pass
        with col4_2:
            st.header("Your Shopping List 🛒:")
        with col4_3:
            pass



        row11_spacer1,row11_1,row11_spacer2,row11_2, row11_spacer3 = st.columns((.2, 1.6, .2, 1.6, .2))

        shopping_index_list = []
        try:
            ind1 = processed_df[processed_df.recipe_title == final_similar_df.recipe_title[0]].index.tolist()[0]
            shopping_index_list.append(ind1)

        except:
            pass
        try:
            ind2 = processed_df[processed_df.recipe_title == final_similar_df.recipe_title[1]].index.tolist()[0]
            shopping_index_list.append(ind2)
        except:
            pass
        try:
            ind3 = processed_df[processed_df.recipe_title == final_similar_df.recipe_title[2]].index.tolist()[0]
            shopping_index_list.append(ind3)
        except:
            pass


        # fridge = shopping_list.final_dataframe(shopping_index_list)
        fridge = pd.DataFrame(shopping_list.final_dataframe(shopping_index_list))
        fridge = fridge[~((fridge['quantity_x']>50) & (fridge['unit']==' '))]
        #df.drop((df.acol <= df.bcol) | (df.acol <= 10), axis=0)
        fridge = fridge[fridge['quantity_x']!=0]


        #st.write(fridge)


        with row11_1:
            #st.subheader("Your Shopping List:")
            for idx, row in fridge.iterrows():
                # if idx<(len(fridge.index)/2) or idx==(len(fridge.index)/2) :
                if idx<((fridge.shape[0])/2 +1) or idx==((fridge.shape[0])/2+1) :
                # st.write(row)
                    st.write(f"- {row['product']} : {row['quantity_x']} {row['unit']}")

        with row11_2:
            # st.subheader("")
            # st.subheader("")
            for idx, row in fridge.iterrows():
                if idx > ((fridge.shape[0])/2 +1):
                # st.write(row)
                    st.write(f"- {row['product']} : {row['quantity_x']} {row['unit']}")


        model_df=processed_df.copy().reset_index(drop=True)
        word_cloud_index = model_df[model_df.recipe_title == final_similar_df.recipe_title[0]].index.tolist()[0]
        clus = model_df['cluster'][word_cloud_index]
        model_df['final_ingredients'] = model_df['final_ingredients'].apply(func.ing_list2)

        list_of_ing = model_df.groupby('cluster')['final_ingredients'].sum()[clus]
        string_of_ing = ', '.join(list_of_ing)

            # path = os.path.join(os.getcwd(), 'raw_data')
            # font_path_ = os.path.join(path, 'Helvetica.ttc')
        path = os.path.join(os.getcwd(), 'raw_data')
        font_path_temp = os.path.join(path,'MagniveraTrial-HeavyItalic.otf')


        # wordcloud = WordCloud(background_color='white', width=350, height=250).generate(string_of_ing)
            # Display the generated image:
        # fig= plt.figure(figsize=(20, 10))

        custom_mask = np.array(Image.open(os.path.join(path,'shopping_nobg.png')))
        # custom_color =  np.array(Image.open(os.path.join(path,'rainbow.png')))
        wordcloud = WordCloud(background_color='#EEF3EF', colormap='summer', font_path= font_path_temp).generate(string_of_ing)
        # image_colors = ImageColorGenerator(custom_color)
        # wordcloud.recolor(color_func=image_colors)
        fig= plt.figure(figsize=(8, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        #plt.tight_layout(pad=0)
        plt.gcf().set_facecolor("white")
        plt.axis("off")
        plt.show()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(fig)

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
            st.text(f"Difficulty: {final_diff_df.difficulty_level[0]}")
            for item in final_diff_df.ingredients[0].split(','):
                if item != " ":
                    st.write(f"- {item}")

        with row10_2:
            try:
                diff_recipe2 = str(final_diff_df.recipe_title[1])
                st.subheader(diff_recipe2)
                st.text(f"Difficulty: {final_diff_df.difficulty_level[1]}")
                for item in final_diff_df.ingredients[1].split(','):
                    if item != " ":
                        st.write(f"- {item}")
            except:
                st.subheader(fail_safe_statement)

        with row10_3:
            try:

                diff_recipe3 = str(final_diff_df.recipe_title[2])
                st.subheader(diff_recipe3)
                st.text(f"Difficulty: {final_diff_df.difficulty_level[2]}")
                for item in final_diff_df.ingredients[2].split(','):
                    if item != " ":
                        st.write(f"- {item}")
            except:
                st.subheader(fail_safe_statement)

        st.write("------")

    ####################
    ## SHOPPING LIST ###
    ####################

        spac1,col4_1,spac2, col4_2,spac3, col4_3,spac4 = st.columns((.05, 1, .05, 2.2, .05, 1, .05))

        with col4_1:
            pass
        with col4_2:
            st.header("Your Shopping List 🛒:")

        with col4_3:
            pass

        row11_spacer1,row11_1,row11_spacer2,row11_2, row11_spacer3 = st.columns((.2, 1.6, .2, 1.6, .2))

        shopping_index_list = []
        try:
            ind1 = processed_df[processed_df.recipe_title == final_diff_df.recipe_title[0]].index.tolist()[0]
            shopping_index_list.append(ind1)

        except:
            pass
        try:
            ind2 = processed_df[processed_df.recipe_title == final_diff_df.recipe_title[1]].index.tolist()[0]
            shopping_index_list.append(ind2)
        except:
            pass
        try:
            ind3 = processed_df[processed_df.recipe_title == final_diff_df.recipe_title[2]].index.tolist()[0]
            shopping_index_list.append(ind3)
        except:
            pass


        # fridge = shopping_list.final_dataframe(shopping_index_list)
        fridge = pd.DataFrame(shopping_list.final_dataframe(shopping_index_list))
        fridge = fridge[fridge['quantity_x']!=0]

        #st.write(fridge)


        with row11_1:
            #st.subheader("Your Shopping List:")
            for idx, row in fridge.iterrows():
                # if idx<(len(fridge.index)/2) or idx==(len(fridge.index)/2) :
                if idx<((fridge.shape[0])/2 +1) or idx==((fridge.shape[0])/2+1) :
                # st.write(row)
                    st.write(f"- {row['product']} : {row['quantity_x']} {row['unit']}")

        with row11_2:
            # st.subheader("")
            # st.subheader("")
            for idx, row in fridge.iterrows():
                if idx > ((fridge.shape[0])/2 +1):
                # st.write(row)
                    st.write(f"- {row['product']} : {row['quantity_x']} {row['unit']}")


        model_df=processed_df.copy().reset_index(drop=True)
        word_cloud_index = model_df[model_df.recipe_title == final_diff_df.recipe_title[0]].index.tolist()[0]
        clus = model_df['cluster'][word_cloud_index]
        model_df['final_ingredients'] = model_df['final_ingredients'].apply(func.ing_list2)

        list_of_ing = model_df.groupby('cluster')['final_ingredients'].sum()[clus]
        string_of_ing = ', '.join(list_of_ing)

            # path = os.path.join(os.getcwd(), 'raw_data')
            # font_path_ = os.path.join(path, 'Helvetica.ttc')

        path = os.path.join(os.getcwd(), 'raw_data')
        font_path_temp = os.path.join(path,'MagniveraTrial-HeavyItalic.otf')


        # wordcloud = WordCloud(background_color='white', width=350, height=250).generate(string_of_ing)
            # Display the generated image:
        # fig= plt.figure(figsize=(20, 10))

        custom_mask = np.array(Image.open(os.path.join(path,'shopping_nobg.png')))
        # custom_color =  np.array(Image.open(os.path.join(path,'rainbow.png')))
        wordcloud = WordCloud(background_color='#EEF3EF', colormap='summer', font_path= font_path_temp).generate(string_of_ing)
        # image_colors = ImageColorGenerator(custom_color)
        # wordcloud.recolor(color_func=image_colors)
        fig= plt.figure(figsize=(8, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        #plt.tight_layout(pad=0)
        plt.gcf().set_facecolor("white")
        plt.axis("off")
        plt.show()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(fig)

        st.write("---------")

################################################################################         PAGE 3           ##########################

def graphing():
    """Third website page will show the distribution of preferences across the clusters
    determined by Replenish's model, where the cluster to be seen is inputted/chosen by
    the viewer."""
    st.title("Graphing Replenish's Cluster Distributions!")
    select = st.selectbox(" 🍎 Select a cluster group you would like to observe 🍎 ", range(1, max(processed_df.cluster)+1))

    if st.button("Submit"):
        st.text("Below is the distribution of preference categories")
        st.text(f"within the ingredient-based cluster {select}")

        cluster_counts = processed_df.groupby('cluster')['recipe_title'].count().reset_index()
        cluster_counts.rename(columns={'recipe_title': 'Number of Recipes'}, inplace=True)

        cluster_df = processed_df[processed_df['cluster'] == select]
        cuisine_counts = cluster_df['preference'].value_counts()
        fig_df = pd.DataFrame(cuisine_counts).reset_index()
        # st.write(fig_df)



# Set axes background to transparent

        path = os.path.join(os.getcwd(), 'raw_data')
        font_path_sns = os.path.join(path,'times.ttf')
        font_path_sns_bold = os.path.join(path,'times-new-roman-grassetto.ttf')
        font_properties = fm.FontProperties(fname = font_path_sns)
        font_properties_bold = fm.FontProperties(fname = font_path_sns_bold)
        fig = plt.figure(figsize=(20, 10))
        ax = plt.gca()
        ax.set_facecolor((0, 0, 0, 0))
        sns.barplot(x=fig_df['preference'], y=fig_df['count'])

        plt.xlabel('Preference', fontsize=25, fontweight="bold", fontproperties = font_properties)
        plt.ylabel('Number of Recipes', fontsize=25, fontweight="bold", fontproperties = font_properties)
        plt.title(f'Cluster {select} - Preference vs. Number of Recipes', fontsize=40, fontweight="bold", fontproperties = font_properties_bold)
        plt.xticks(rotation=45, fontsize=15)
        plt.yticks(fontsize=15)

        fig.patch.set_facecolor((0, 0, 0, 0))
        # st.pyplot(fig)




        #fig = plt.figure(figsize=(20, 10), facecolor='none')  # Set background color to transparent
        #sns.set(style='whitegrid', font='Times New Roman')  # Set font to 'Times New Roman'
        #sns.barplot(x=fig_df['preference'], y=fig_df['count'])
        #plt.xlabel('Preference', fontsize=25, fontweight="bold")
        #plt.ylabel('Number of Recipes', fontsize=25, fontweight="bold")
        #plt.title(f'Cluster {select} - Preference vs. Number of Recipes', fontsize=40, fontweight="bold")
        #plt.xticks(rotation=45, fontsize=15)
        #plt.yticks(fontsize=15)

        # fig= plt.figure(figsize=(20, 10), facecolor=(0,0,0,0))
        # sns.barplot(x = fig_df['preference'], y = fig_df['count'])
        # plt.xlabel('Preference', fontsize=25, fontname="Times New Roman",fontweight="bold")
        # plt.ylabel('Number of Recipes', fontsize=25, fontname="Times New Roman",fontweight="bold")
        # plt.title(f'Cluster {select} - Preference vs. Number of Recipes', fontsize=40, fontname="Times New Roman",fontweight="bold")
        # plt.xticks(rotation=45, fontsize=15, fontname="Times New Roman")
        # plt.yticks(fontsize=15, fontname="Times New Roman")

        st.pyplot(fig, clear_figure=True)

page_names_to_funcs = {
    "About Us": intro,
    "Recipe Generator": output,
    "Data Graphing": graphing,
}

demo_name = st.sidebar.selectbox("Click to find out more 👉", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()
