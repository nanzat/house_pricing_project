import pandas as pd
import numpy as np
import sklearn
sklearn.set_config(transform_output="pandas")
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error


train = pd.read_csv('./data/house/train.csv')

# Разделение данных
X, y = train.drop('SalePrice', axis=1), train['SalePrice']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Удаление столбцов
cdrop = ['YrSold', 'MoSold', '3SsnPorch', 'BsmtFinType2', 'LowQualFinSF', 'BsmtHalfBath', 'Alley', 'PoolQC', 'Fence', 'MiscFeature', 'Id']
drops = ColumnTransformer(
    transformers=[
        ('drop', 'drop', cdrop)
    ],
    verbose_feature_names_out=False,
    remainder='passthrough'
)

# Применение удаления столбцов
df = drops.fit_transform(X_train)

# Выбор числовых и категориальных столбцов
df_cat = pd.DataFrame(df).select_dtypes(include=['object'])
categorical_features = df_cat.columns.tolist()

df_num = pd.DataFrame(df).select_dtypes(include=['float64', 'int64'])
numeric_features = df_num.columns.tolist()

# Определение трансформеров для числовых и категориальных столбцов
numeric_transformer = SimpleImputer(strategy='constant', fill_value=0)
categorical_transformer = SimpleImputer(strategy='constant', fill_value='no value')

inputer = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    verbose_feature_names_out=False,
    remainder='passthrough'
)

scaler = ColumnTransformer(
    transformers=[
        ('scaler', StandardScaler(), numeric_features),
        ('target_encoding', TargetEncoder(), categorical_features)
    ],
    verbose_feature_names_out=False,
    remainder='passthrough'
)

# Создание конвейера
preprocessor = Pipeline([
    ('drops', drops),
    ('inputer', inputer),
    ('scaler', scaler ),
    ('model', CatBoostRegressor())
])

# Обучение модели
preprocessor.fit(X_train, np.log(y_train))

y_preds = preprocessor.predict(X_valid)

test = pd.read_csv('./data/house/test.csv')
test2 = pd.read_csv('./data/house/sample_submission.csv')

testy_preds = preprocessor.predict(test)

print('MSLE:', mean_squared_error(np.log(test2['SalePrice']), testy_preds))

itog = pd.DataFrame({'Id': test['Id'], 'SalePrice':np.exp(testy_preds) })

print(itog)
#itog.to_csv('output3.csv', index=False)