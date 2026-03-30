import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def label_norm(): 
    sexo_binario = df['sexo']

    encoder = LabelEncoder()

    sexo_codificado = encoder.fit_transform(sexo_binario)

    #Manda o sexo codificado pro dataframe como uma tabela
    df['sexo_codificado'] = sexo_codificado

    return df

df = pd.read_csv("dados_normalizar.csv", sep=';', decimal=',')
scaler = MinMaxScaler(feature_range=(0, 1))
df = label_norm()

dados_coluna = df[['idade', 'altura', 'Peso', 'sexo_codificado']]

dados_normalizados = scaler.fit_transform(dados_coluna)

print(dados_coluna)
print(dados_normalizados)

#Sugestão da IA
print("Mínimos:", dados_normalizados.min(axis=0))
print("Máximos:", dados_normalizados.max(axis=0))

# Reversão 
# dados_revertidos = scaler.inverse_transform(dados_normalizados)
