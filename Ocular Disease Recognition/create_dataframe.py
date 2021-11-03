import pandas as pd
import numpy as np

df= pd.read_csv('full_df.csv')

temp_df = df.iloc[:,1:7]

# Extracting Normal eye image's filename
df_norm_on_left = temp_df[temp_df['Left-Diagnostic Keywords'].str.match('normal')]
df_norm_on_right = temp_df[temp_df['Right-Diagnostic Keywords'].str.match('normal')]

df_norm = df_norm_on_left['Left-Fundus'].append(df_norm_on_right['Right-Fundus'], ignore_index=True)
df_normal = pd.DataFrame(df_norm, columns = ["filename"])
df_normal["label"] = "Normal"
df_normal=df_normal[0:500]

# Extracting Cataract eye image's filename
df_cat_on_lefteye = temp_df[temp_df['Left-Diagnostic Keywords'].str.match('cataract')]
df_cat_on_righteye = temp_df[temp_df['Right-Diagnostic Keywords'].str.match('cataract')]

df_cat = df_cat_on_lefteye['Left-Fundus'].append(df_cat_on_righteye['Right-Fundus'], ignore_index=True)
df_cataract = pd.DataFrame(df_cat, columns = ["filename"])
df_cataract["label"] = "Cataract"
df_cataract = df_cataract[0:500]

# Extracting Glaucoma image's filename
df_glaucoma_on_lefteye = temp_df[temp_df['Left-Diagnostic Keywords'].str.match('glaucoma')]
df_glaucoma_on_righteye = temp_df[temp_df['Right-Diagnostic Keywords'].str.match('glaucoma')]

df_glauc = df_glaucoma_on_lefteye['Left-Fundus'].append(df_glaucoma_on_righteye['Right-Fundus'], ignore_index=True)
df_glaucoma = pd.DataFrame(df_glauc, columns = ["filename"])
df_glaucoma["label"] = "Glaucoma"

dataframe = df_normal.append([df_cataract, df_glaucoma], ignore_index=True)
