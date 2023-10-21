import pandas as pd
import numpy as np

def type_cols(df):
    '''
    Funcao para retornar os tipos de variaveis da data frame
    Input:
        df - pandas DataFrame
    Output:
        numeric_cols - lista com o nome das colunas numéricas
        object_cols - lista com o nome das colunas categoricas
    '''
    numeric_cols = df.select_dtypes(include = ['number']).columns.tolist()
    object_cols = df.select_dtypes(include = ['object']).columns.tolist()
    assert len(numeric_cols + object_cols) == len(df.columns) , 'Colunas faltantes a serem consideras'
    
    return numeric_cols, object_cols

def miss_vars(df):
    miss_ser = df.isna().sum()
    return miss_ser[miss_ser>0]

def check_low_categorical(df ,
                          object_cols,
                          min_percent):
    '''
    Funcao para localizar classes das variaveis categorias com poucas observacoes
    Input:
        df: pandas DataFrame
        object_cols: lista contendo o nome das colunas categoricas
        min_percent: float com a percentagem minima
        
    Output:
        low_percent: pandas Dataframe contendo o nome da variavel, categoria e percentagem 
    '''
    
    low_percent = pd.DataFrame(columns = ['percent',
                                           'category_name'])
    
    for col in object_cols:
        percent = df[col].value_counts(normalize = True)[-1]
        category_name = df[col].value_counts(normalize = True).index[-1]
        if percent < min_percent:
            low_percent.loc[col, 'percent'] = percent
            low_percent.loc[col, 'category_name'] = category_name
        
    assert len(low_percent) > 0, f'Nenhuma coluna possui categoria com menos de {min_percent*100}%'
    return low_percent

def treat_category(df, object_cols):
    '''
    Funcao para tratar variaveis categoricas
    '''
    # aglumas variaveis contem 3 categorias, porem podem reduzir apenas para duas
    # pois alguma outra variavel já vai trazer esta informacao
    df.loc[(df.MultipleLines == 'No phone service'), 'MultipleLines'] = 'No'
    df.loc[(df.OnlineSecurity == 'No internet service'), 'OnlineSecurity'] = 'No'
    df.loc[(df.OnlineBackup == 'No internet service'), 'OnlineBackup'] = 'No'
    df.loc[(df.DeviceProtection == 'No internet service'), 'DeviceProtection'] = 'No'
    df.loc[(df.TechSupport == 'No internet service'), 'TechSupport'] = 'No'
    df.loc[(df.StreamingTV == 'No internet service'), 'StreamingTV'] = 'No'
    df.loc[(df.StreamingMovies == 'No internet service'), 'StreamingMovies'] = 'No'
    
    df[object_cols] = df[object_cols].apply(lambda x: x.str.replace(r'[-\s]', '_', regex=True))
    df[object_cols] = df[object_cols].apply(lambda x: x.str.replace(r'[\(\)]', '', regex=True))
    
    return df

def binary_transform(df, object_cols):
    '''
    Funcao para transformar variaveis categorias em binarias
    Input:
        df: pandas DataFrame
        object_cols: lista contendo o nome das colunas categoricas
        
    Output:
        df: pandas Dataframe contendo as variaveis tratadas e sem as originais
    '''
    df_transf = df.copy()
    describe_df = df_transf[object_cols].describe().T
    describe_df = describe_df[describe_df.unique == 2]
    for col in list(describe_df.index):
        aux = df_transf[col].value_counts().index[0]
        # criando diferente tratamento dependento do conteudo da variavel
        
        if aux in ['Yes','No']:
            flag_col = 'flag_'+ col
            df_transf[flag_col] = (np.where(df_transf[col]=='Yes',
                                           1,
                                           0))
        else:
            flag_col = 'flag_' + col + '_' + aux 
            df_transf[flag_col] = (np.where(df_transf[col]==aux,
                                           1,
                                           0))
            
    df_transf = df_transf.drop(columns = list(describe_df.index))
    return df_transf

def create_variables(df):
    '''
    Funcao para criar variaveis 
    Input:
        df: pandas DataFrame
    Output:
        df: pandas Dataframe contendo as variaveis tratadas
    '''
    # como a variavel de servico de internet tem o produto e se contem ou nao,
    # vamos criar uma flag de internet
    
    df['flag_InternetService'] = np.where(df['InternetService']=='No',
                                          0,
                                          1)
    # criando variavel de pagamento automatico
    df['flag_automatic_payment'] = (np.where( df.PaymentMethod.str.contains('automatic'),
                                    0,
                                    1))

    # selecionando variaveis de produtos
    product_vars =  ['flag_PhoneService',
                     'flag_MultipleLines',
                     'flag_OnlineSecurity',
                     'flag_OnlineBackup',
                     'flag_DeviceProtection',
                     'flag_TechSupport',
                     'flag_StreamingTV',
                     'flag_StreamingMovies',
                     'flag_InternetService']
    
    # como as variaveis sao flags podemos somar para ter o total de produtos
    df['num_produtos'] = df[product_vars].sum(axis=1)
    
    # fazendo media de preco por produto
    df['media_mothly_charges'] = df.MonthlyCharges/df.num_produtos
    
    # selecionando variaveis relacionados a seguranca
    security_vars = ['flag_OnlineSecurity',
                     'flag_OnlineBackup',
                     'flag_DeviceProtection']
    
    df['num_produtos_seguranca'] = df[security_vars].sum(axis=1)
   
    return df

def save_data_frame(df, file_name, version):
    df.to_csv(f'data/processado/{file_name}_{version}.csv', index = True)