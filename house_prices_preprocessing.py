import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class HousePricesPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = RobustScaler()
        self.imputer_numeric = SimpleImputer(strategy='median')
        self.imputer_categorical = SimpleImputer(strategy='most_frequent')
        
    def load_data(self, train_path, test_path):
        self.train = pd.read_csv(train_path)
        self.test = pd.read_csv(test_path)
        self.train_id = self.train['Id']
        self.test_id = self.test['Id']
        self.y_train = self.train['SalePrice']
        
        self.train = self.train.drop(['Id', 'SalePrice'], axis=1)
        self.test = self.test.drop(['Id'], axis=1)
        
        return self.train, self.test, self.y_train
    
    def handle_missing_values(self, df):
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        df[numeric_columns] = self.imputer_numeric.fit_transform(df[numeric_columns])
        df[categorical_columns] = self.imputer_categorical.fit_transform(df[categorical_columns])
        
        return df
    
    def create_features(self, df):
        df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
        df['TotalBathrooms'] = df['FullBath'] + 0.5 * df['HalfBath'] + df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']
        df['TotalPorchSF'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']
        df['Age'] = df['YrSold'] - df['YearBuilt']
        df['Remodeled'] = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)
        df['NewHouse'] = (df['YrSold'] == df['YearBuilt']).astype(int)
        
        df['HasPool'] = (df['PoolArea'] > 0).astype(int)
        df['Has2ndFloor'] = (df['2ndFlrSF'] > 0).astype(int)
        df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
        df['HasBasement'] = (df['TotalBsmtSF'] > 0).astype(int)
        df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)
        
        df['OverallGrade'] = df['OverallQual'] * df['OverallCond']
        df['ExterGrade'] = df['ExterQual'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1})
        df['KitchenGrade'] = df['KitchenQual'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1})
        
        df['LotShape_Score'] = df['LotShape'].map({'Reg': 4, 'IR1': 3, 'IR2': 2, 'IR3': 1})
        df['LandContour_Score'] = df['LandContour'].map({'Lvl': 4, 'Bnk': 3, 'HLS': 2, 'Low': 1})
        df['LandSlope_Score'] = df['LandSlope'].map({'Gtl': 3, 'Mod': 2, 'Sev': 1})
        
        df['BsmtExposure_Score'] = df['BsmtExposure'].map({'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1})
        df['BsmtFinType1_Score'] = df['BsmtFinType1'].map({'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1})
        df['BsmtFinType2_Score'] = df['BsmtFinType2'].map({'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1})
        
        df['GarageFinish_Score'] = df['GarageFinish'].map({'Fin': 3, 'RFn': 2, 'Unf': 1})
        df['GarageQual_Score'] = df['GarageQual'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1})
        df['GarageCond_Score'] = df['GarageCond'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1})
        
        df['Fence_Score'] = df['Fence'].map({'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1})
        
        df['MSSubClass_New'] = df['MSSubClass'].map({
            20: 1, 30: 1, 40: 1, 45: 1, 50: 1, 60: 2, 70: 2, 75: 2, 80: 2, 85: 2,
            90: 3, 120: 4, 150: 4, 160: 4, 180: 4, 190: 4
        })
        
        df['Neighborhood_Score'] = df['Neighborhood'].map({
            'Blmngtn': 1, 'Blueste': 1, 'BrDale': 1, 'BrkSide': 1, 'ClearCr': 2,
            'CollgCr': 2, 'Crawfor': 2, 'Edwards': 1, 'Gilbert': 2, 'IDOTRR': 1,
            'MeadowV': 1, 'Mitchel': 2, 'NAmes': 2, 'NoRidge': 3, 'NPkVill': 2,
            'NridgHt': 3, 'NWAmes': 2, 'OldTown': 1, 'SWISU': 2, 'Sawyer': 2,
            'SawyerW': 2, 'Somerst': 3, 'StoneBr': 3, 'Timber': 2, 'Veenker': 3
        })
        
        df['Condition1_Score'] = df['Condition1'].map({
            'Artery': 1, 'Feedr': 2, 'Norm': 3, 'RRNn': 1, 'RRAn': 1,
            'PosN': 4, 'PosA': 4, 'RRNe': 1, 'RRAe': 1
        })
        
        df['Condition2_Score'] = df['Condition2'].map({
            'Artery': 1, 'Feedr': 2, 'Norm': 3, 'RRNn': 1, 'RRAn': 1,
            'PosN': 4, 'PosA': 4, 'RRNe': 1, 'RRAe': 1
        })
        
        df['BldgType_Score'] = df['BldgType'].map({
            '1Fam': 3, '2FmCon': 2, 'Duplex': 2, 'TwnhsE': 2, 'TwnhsI': 1
        })
        
        df['HouseStyle_Score'] = df['HouseStyle'].map({
            '1Story': 1, '1.5Fin': 2, '1.5Unf': 1, '2Story': 3, '2.5Fin': 3,
            '2.5Unf': 2, 'SFoyer': 2, 'SLvl': 2
        })
        
        df['RoofStyle_Score'] = df['RoofStyle'].map({
            'Flat': 1, 'Gable': 2, 'Gambrel': 3, 'Hip': 3, 'Mansard': 4, 'Shed': 1
        })
        
        df['RoofMatl_Score'] = df['RoofMatl'].map({
            'ClyTile': 5, 'CompShg': 3, 'Membran': 4, 'Metal': 4, 'Roll': 2,
            'Tar&Grv': 2, 'WdShake': 4, 'WdShngl': 3
        })
        
        df['Foundation_Score'] = df['Foundation'].map({
            'BrkTil': 2, 'CBlock': 2, 'PConc': 4, 'Slab': 1, 'Stone': 3, 'Wood': 1
        })
        
        df['Heating_Score'] = df['Heating'].map({
            'Floor': 1, 'GasA': 3, 'GasW': 3, 'Grav': 1, 'OthW': 2, 'Wall': 1
        })
        
        df['HeatingQC_Score'] = df['HeatingQC'].map({
            'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1
        })
        
        df['Electrical_Score'] = df['Electrical'].map({
            'SBrkr': 4, 'FuseA': 3, 'FuseF': 2, 'FuseP': 1, 'Mix': 1
        })
        
        df['Functional_Score'] = df['Functional'].map({
            'Typ': 7, 'Min1': 6, 'Min2': 5, 'Mod': 4, 'Maj1': 3, 'Maj2': 2, 'Sev': 1, 'Sal': 0
        })
        
        df['PavedDrive_Score'] = df['PavedDrive'].map({'Y': 3, 'P': 2, 'N': 1})
        
        df['SaleType_Score'] = df['SaleType'].map({
            'WD': 4, 'CWD': 5, 'VWD': 5, 'New': 5, 'COD': 3, 'Con': 2, 'ConLw': 1, 'ConLI': 1, 'ConLD': 1, 'Oth': 1
        })
        
        df['SaleCondition_Score'] = df['SaleCondition'].map({
            'Normal': 4, 'Abnorml': 2, 'AdjLand': 1, 'Alloca': 3, 'Family': 3, 'Partial': 5
        })
        
        return df
    
    def encode_categorical(self, df):
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        return df
    
    def scale_features(self, df):
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = self.scaler.fit_transform(df[numeric_columns])
        return df
    
    def remove_outliers(self, df, y, threshold=3):
        z_scores = np.abs((df - df.mean()) / df.std())
        outlier_mask = (z_scores > threshold).any(axis=1)
        
        df_clean = df[~outlier_mask]
        y_clean = y[~outlier_mask]
        
        return df_clean, y_clean
    
    def select_features(self, df, method='correlation', threshold=0.1):
        if method == 'correlation':
            correlation_matrix = df.corr().abs()
            upper_triangle = correlation_matrix.where(
                np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
            )
            to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
            df = df.drop(columns=to_drop)
        
        return df
    
    def preprocess(self, train_path, test_path, remove_outliers=True, feature_selection=True):
        train, test, y_train = self.load_data(train_path, test_path)
        
        train = self.handle_missing_values(train)
        test = self.handle_missing_values(test)
        
        train = self.create_features(train)
        test = self.create_features(test)
        
        train = self.encode_categorical(train)
        test = self.encode_categorical(test)
        
        if remove_outliers:
            train, y_train = self.remove_outliers(train, y_train)
        
        if feature_selection:
            train = self.select_features(train)
            test = test[train.columns]
        
        train = self.scale_features(train)
        test = self.scale_features(test)
        
        return train, test, y_train
    
    def get_feature_importance_ranking(self, train, y_train):
        correlation_with_target = train.corrwith(y_train).abs().sort_values(ascending=False)
        return correlation_with_target
    
    def create_polynomial_features(self, df, degree=2, max_features=50):
        from sklearn.preprocessing import PolynomialFeatures
        
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly_features = poly.fit_transform(df)
        
        feature_names = poly.get_feature_names_out(df.columns)
        poly_df = pd.DataFrame(poly_features, columns=feature_names)
        
        correlation_with_target = poly_df.corrwith(y_train).abs().sort_values(ascending=False)
        top_features = correlation_with_target.head(max_features).index
        
        return poly_df[top_features]
    
    def create_interaction_features(self, df, feature_pairs):
        for feat1, feat2 in feature_pairs:
            df[f'{feat1}_{feat2}_interaction'] = df[feat1] * df[feat2]
        return df

def main():
    preprocessor = HousePricesPreprocessor()
    
    train_path = '/Users/namburirammohanrao/Documents/house-prices-advanced-regression-techniques/train.csv'
    test_path = '/Users/namburirammohanrao/Documents/house-prices-advanced-regression-techniques/test.csv'
    
    train_processed, test_processed, y_train = preprocessor.preprocess(
        train_path, test_path, remove_outliers=True, feature_selection=True
    )
    
    print(f"Processed training data shape: {train_processed.shape}")
    print(f"Processed test data shape: {test_processed.shape}")
    print(f"Target variable shape: {y_train.shape}")
    
    feature_importance = preprocessor.get_feature_importance_ranking(train_processed, y_train)
    print("\nTop 20 most important features:")
    print(feature_importance.head(20))
    
    train_processed.to_csv('train_processed.csv', index=False)
    test_processed.to_csv('test_processed.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    
    print("\nPreprocessed data saved to CSV files.")

if __name__ == "__main__":
    main() 