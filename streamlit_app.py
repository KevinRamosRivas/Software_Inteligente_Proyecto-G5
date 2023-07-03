
#importar librerias para scraping
#importamos las librerias necesarias
import pandas as pd
#usamos una matriz dispersa para ahorrar memoria
from scipy.sparse import csr_matrix

import streamlit as st
#guardar anime_features_df_matrix en un archivo para usarlo en el notebook de recomendacion
from scipy import sparse
import sklearn
# importamos la libreria para el modelo de recomendacion
from sklearn.neighbors import NearestNeighbors
#guardar el modelo para usarlo en el futuro
import joblib



not_found_img = 'not_found.png'


#hacer un footer
st.write('Hecho por [Kevin Ramos Rivas](https://github.com/KevinRamosRivas) 👨‍💻')











