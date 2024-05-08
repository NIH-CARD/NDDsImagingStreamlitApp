import pickle
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import shap
import hashlib
# import plotly.express as px
# import plotly
import copy
import matplotlib.pyplot as plt
import gzip
import numpy
# import anndata as ad
# st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_option('deprecation.showPyplotGlobalUse', False)


def app():
    st.write("## Introduction")
    st.write(
        """
        The Shapley additive explanations (SHAP) approach was used to evaluate each feature’s influence in the ensemble learning. This approach, used in game theory, assigned an importance (Shapley) value to each feature to determine a player’s contribution to success. Shapley explanations enhance understanding by creating accurate explanations for each observation in a dataset. They bolster trust when the critical variables for specific records conform to human domain knowledge and reasonable expectations.
        We used the one-vs-rest technique for multiclass classification. Based on that, we trained a separate binary classification model for each class.
        """
    )
        st.write(
            """
            "In our model, we use an approach called Shapley's Additive Explanations (SHAP) to evaluate how different features affect learning. This method assigns a value of importance to each feature. This is similar to how we assign a value to each player's contribution to the team's success in a game. Shapley's explanations help us better understand how each trait affects the model's predictions for each patient in our data set. This is useful because if the most important variables for a particular patient match what we know in medical practice and our reasonable expectations, we have more confidence in the model's predictions. For multi-class classification, we use the one-versus-the-rest technique, where we train a separate model for each class of AD we want to predict.
            The X axis in the SHAP graph below corresponds to the SHAP value which can be positive or negative regarding the value of that feature in a patient. A vertical line mark the 0 SHAP value. While the Y axis in the graph correspond to the value of the feature.
            In example: in AD the most relevant feature for prediction was “thk left hippocampushipplr” which corresponds to the thickness of the left hippocampus. The long tail to to the right is mostly filled with blue colors. This means that low values (less hippocampus thickness) are associated with great impact in the models prediction. In the other hand, a short left tail, mostly filled with red colors, means that higher hippocampus thickness is associated with less impact in the model prediction.

            Rodríguez-Pérez R, Bajorath J. Interpretation of Compound Activity Predictions from Complex Machine Learning Models Using Local Approximations and Shapley Values. J Med Chem. 2020
            """
        )
    select_disease = st.selectbox("Select the disease  ", options=['ADRD', 'PD'])
    st.write('## Summary Plot')
    st.write("""Shows top-20 features that have the most significant impact on the classification model.""")
    if select_disease == 'ADRD':
        @st.cache_data
        def get_cache_data_ad():
            f = gzip.GzipFile("ad_shap_object_values.npy.gz", "r")
            values = np.load(f)
            f.close()
            f = gzip.GzipFile("ad_shap_object_base_values.npy.gz", "r")
            base_values = np.load(f)
            f.close()
            f = gzip.GzipFile("ad_shap_object_data.npy.gz", "r")
            data = np.load(f)
            f.close()
            f = gzip.GzipFile("ad_shap_object_feature_names.npy.gz", "r")
            feature_names = np.load(f)
            f.close()
            ad_shap_obj = shap.Explanation(values=np.array(values), base_values=base_values, data=data, feature_names=feature_names)
            with open("ad_top20_feature_list.txt", 'r') as f:
                top20_features = f.read().split('\n')
            return ad_shap_obj, top20_features

        ad_shap_obj, top20_features = get_cache_data_ad()
    else:
        @st.cache_data
        def get_cache_data_pd():
            f = gzip.GzipFile("pd_shap_object_values.npy.gz", "r")
            values = np.load(f)
            f.close()
            f = gzip.GzipFile("pd_shap_object_base_values.npy.gz", "r")
            base_values = np.load(f)
            f.close()
            f = gzip.GzipFile("pd_shap_object_data.npy.gz", "r")
            data = np.load(f)
            f.close()
            f = gzip.GzipFile("pd_shap_object_feature_names.npy.gz", "r")
            feature_names = np.load(f)
            f.close()
            ad_shap_obj = shap.Explanation(values=np.array(values), base_values=base_values, data=data, feature_names=feature_names)
            with open("pd_top20_feature_list.txt", 'r') as f:
                top20_features = f.read().split('\n')
            return ad_shap_obj, top20_features

        ad_shap_obj, top20_features = get_cache_data_pd()

    # fname = "lightgbm_pd_shap.pkl"
    # with open(fname, "rb") as f:
    #     ad_shap_data = pickle.load(f)

    # ad_shap_obj = ad_shap_data['shap_values']
    # def process_name(x):
    #     return x.replace('id_invicrot1_', '').replace('_', ' ')

    # with open("feature_list.txt", 'r') as f:
    #     feature_names = f.read().split()
    feature_names = ad_shap_obj.feature_names

    if True: # st.checkbox("Show Summary Plot"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write('---')
            # temp = shap.Explanation(values=np.array(shap_values), base_values=np.array([exval]*len(X)), data=np.array(X.values), feature_names=X.columns)
            @st.cache_data
            def get_fig_bee():
                fig, ax = plt.subplots(figsize=(10,15))
                shap.plots.beeswarm(ad_shap_obj, show=False, max_display=20, plot_size=(10, 13))
                st.pyplot(fig)

            get_fig_bee()
            st.write('---')
        with col2:
            st.write('---')
            @st.cache_data
            def get_fig_bar():
                fig, ax = plt.subplots(figsize=(10,15))
                shap.plots.bar(ad_shap_obj, show=False, max_display=20)
                st.pyplot(fig)

            get_fig_bar()
            st.write('---')

    st.write('## Dependence Plots')
    st.write("""We can observe the interaction effects of different features in for predictions. To help reveal these interactions dependence_plot automatically lists (top-3) potential features for coloring.
    Furthermore, we can observe the relationship between features and SHAP values for prediction using the dependence plots, which compares the actual feature value (x-axis) against the SHAP score (y-axis).
    It shows that the effect of feature values is not a simple relationship where increase in the feature value leads to consistent changes in model output but a complicated non-linear relationship.""")
    inds = []
    if True: # st.checkbox("Show Dependence Plots"):
        feature_name = st.selectbox('Select a feature for dependence plot', options=top20_features)

        @st.cache_data
        def get_inds(feature_name):
            try:
                return shap.utils.potential_interactions(ad_shap_obj[:, feature_name], ad_shap_obj)
            except:
                return []

        inds = get_inds(feature_name)
        if len(inds) <= 3:
            st.info("Select Another Feature")
        else:

            st.write('Top-3 Potential Interactions for ***{}***'.format(feature_name))

            @st.cache_data
            def get_interaction_plot(feature_name, inds):
                col3, col4, col5 = st.columns(3)
                with col3:
                    # fig, ax = plt.subplots(figsize=(8, 8))
                    shap.plots.scatter(ad_shap_obj[:, feature_name], color=ad_shap_obj[:, inds[0]])
                    # shap.dependence_plot(feature_name, np.array(ad_shap_obj.values), ad_shap_obj.data, interaction_index=list(ad_shap_obj.feature_names).index(list(ad_shap_obj.feature_names)[inds[0]]))
                    st.pyplot()
                with col4:
                    shap.plots.scatter(ad_shap_obj[:, feature_name], color=ad_shap_obj[:, inds[1]], show=False)
                    # shap.dependence_plot(feature_name, np.array(ad_shap_obj.values), ad_shap_obj.data, interaction_index=list(ad_shap_obj.feature_names).index(list(ad_shap_obj.feature_names)[inds[1]]))
                    st.pyplot()
                with col5:
                    shap.plots.scatter(ad_shap_obj[:, feature_name], color=ad_shap_obj[:, inds[2]], show=False)
                    # shap.dependence_plot(feature_name, np.array(ad_shap_obj.values), ad_shap_obj.data, interaction_index=list(ad_shap_obj.feature_names).index(list(ad_shap_obj.feature_names)[inds[2]]))
                    st.pyplot()

            get_interaction_plot(feature_name, inds)
