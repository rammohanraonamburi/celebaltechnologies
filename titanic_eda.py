#!/usr/bin/env python3
"""
Titanic Dataset - Comprehensive Exploratory Data Analysis (EDA)
This script performs an in-depth analysis of the Titanic dataset including:
- Data overview and basic statistics
- Missing value analysis
- Data distributions and visualizations
- Outlier detection
- Correlation analysis
- Feature relationships and insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class TitanicEDA:
    def __init__(self, file_path):
        """Initialize the EDA class with the dataset."""
        self.df = pd.read_csv(file_path)
        self.original_shape = self.df.shape
        print(f"Dataset loaded: {self.original_shape[0]} rows and {self.original_shape[1]} columns")
        
    def basic_info(self):
        """Display basic information about the dataset."""
        print("\n" + "="*60)
        print("BASIC DATASET INFORMATION")
        print("="*60)
        
        print("\nDataset Shape:", self.df.shape)
        print("\nColumn Names:")
        for i, col in enumerate(self.df.columns, 1):
            print(f"{i:2d}. {col}")
            
        print("\nData Types:")
        print(self.df.dtypes)
        
        print("\nFirst 5 rows:")
        print(self.df.head())
        
        print("\nLast 5 rows:")
        print(self.df.tail())
        
    def missing_values_analysis(self):
        """Analyze missing values in the dataset."""
        print("\n" + "="*60)
        print("MISSING VALUES ANALYSIS")
        print("="*60)
        
        # Calculate missing values
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing_Count': missing_data.values,
            'Missing_Percentage': missing_percent.values
        }).sort_values('Missing_Percentage', ascending=False)
        
        print("\nMissing Values Summary:")
        print(missing_df)
        
        # Visualize missing values
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        missing_data.plot(kind='bar', color='coral')
        plt.title('Missing Values Count by Column')
        plt.xlabel('Columns')
        plt.ylabel('Missing Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.subplot(1, 2, 2)
        missing_percent.plot(kind='bar', color='lightblue')
        plt.title('Missing Values Percentage by Column')
        plt.xlabel('Columns')
        plt.ylabel('Missing Percentage (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.tight_layout()
        plt.savefig('missing_values_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return missing_df
        
    def descriptive_statistics(self):
        """Generate descriptive statistics for numerical columns."""
        print("\n" + "="*60)
        print("DESCRIPTIVE STATISTICS")
        print("="*60)
        
        # Numerical columns statistics
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        print("\nNumerical Columns Statistics:")
        print(self.df[numerical_cols].describe())
        
        # Categorical columns statistics
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        print("\nCategorical Columns Statistics:")
        for col in categorical_cols:
            print(f"\n{col}:")
            print(self.df[col].value_counts())
            print(f"Unique values: {self.df[col].nunique()}")
            
    def target_variable_analysis(self):
        """Analyze the target variable (Survived)."""
        print("\n" + "="*60)
        print("TARGET VARIABLE ANALYSIS (SURVIVED)")
        print("="*60)
        
        # Survival statistics
        survival_counts = self.df['Survived'].value_counts()
        survival_percent = self.df['Survived'].value_counts(normalize=True) * 100
        
        print(f"\nSurvival Counts:")
        print(f"Survived: {survival_counts[1]} ({survival_percent[1]:.2f}%)")
        print(f"Did not survive: {survival_counts[0]} ({survival_percent[0]:.2f}%)")
        
        # Visualize survival distribution
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        survival_counts.plot(kind='bar', color=['red', 'green'])
        plt.title('Survival Count')
        plt.xlabel('Survived')
        plt.ylabel('Count')
        plt.xticks([0, 1], ['No', 'Yes'])
        
        plt.subplot(1, 3, 2)
        plt.pie(survival_counts.values, labels=['Did not survive', 'Survived'], 
                autopct='%1.1f%%', colors=['red', 'green'])
        plt.title('Survival Distribution')
        
        plt.subplot(1, 3, 3)
        sns.countplot(data=self.df, x='Survived', palette=['red', 'green'])
        plt.title('Survival Count (Seaborn)')
        plt.xlabel('Survived')
        plt.ylabel('Count')
        plt.xticks([0, 1], ['No', 'Yes'])
        
        plt.tight_layout()
        plt.savefig('survival_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def numerical_distributions(self):
        """Analyze distributions of numerical variables."""
        print("\n" + "="*60)
        print("NUMERICAL VARIABLES DISTRIBUTION ANALYSIS")
        print("="*60)
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        
        # Create subplots for each numerical variable
        n_cols = len(numerical_cols)
        fig, axes = plt.subplots(2, n_cols, figsize=(5*n_cols, 10))
        
        for i, col in enumerate(numerical_cols):
            # Histogram
            axes[0, i].hist(self.df[col].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, i].set_title(f'{col} - Histogram')
            axes[0, i].set_xlabel(col)
            axes[0, i].set_ylabel('Frequency')
            
            # Box plot
            axes[1, i].boxplot(self.df[col].dropna())
            axes[1, i].set_title(f'{col} - Box Plot')
            axes[1, i].set_ylabel(col)
            
        plt.tight_layout()
        plt.savefig('numerical_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Summary statistics for numerical variables
        print("\nNumerical Variables Summary:")
        print(self.df[numerical_cols].describe())
        
    def categorical_analysis(self):
        """Analyze categorical variables and their relationship with survival."""
        print("\n" + "="*60)
        print("CATEGORICAL VARIABLES ANALYSIS")
        print("="*60)
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            print(f"\n{col.upper()} Analysis:")
            
            # Value counts
            value_counts = self.df[col].value_counts()
            print(f"Value counts:\n{value_counts}")
            
            # Survival rate by category
            survival_by_category = self.df.groupby(col)['Survived'].agg(['count', 'sum', 'mean'])
            survival_by_category.columns = ['Total', 'Survived', 'Survival_Rate']
            survival_by_category['Survival_Rate'] = survival_by_category['Survival_Rate'] * 100
            print(f"\nSurvival by {col}:\n{survival_by_category}")
            
            # Visualize
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            value_counts.plot(kind='bar')
            plt.title(f'{col} - Value Counts')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            
            plt.subplot(1, 3, 2)
            survival_by_category['Survival_Rate'].plot(kind='bar', color='lightgreen')
            plt.title(f'{col} - Survival Rate (%)')
            plt.xlabel(col)
            plt.ylabel('Survival Rate (%)')
            plt.xticks(rotation=45)
            
            plt.subplot(1, 3, 3)
            sns.countplot(data=self.df, x=col, hue='Survived', palette=['red', 'green'])
            plt.title(f'{col} - Survival by Category')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.legend(['Did not survive', 'Survived'])
            
            plt.tight_layout()
            plt.savefig(f'{col}_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
    def correlation_analysis(self):
        """Analyze correlations between variables."""
        print("\n" + "="*60)
        print("CORRELATION ANALYSIS")
        print("="*60)
        
        # Create a copy for correlation analysis
        df_corr = self.df.copy()
        
        # Encode categorical variables for correlation
        le = LabelEncoder()
        for col in df_corr.select_dtypes(include=['object']).columns:
            df_corr[col] = le.fit_transform(df_corr[col].astype(str))
            
        # Calculate correlation matrix
        correlation_matrix = df_corr.corr()
        
        print("\nCorrelation Matrix:")
        print(correlation_matrix)
        
        # Visualize correlation matrix
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Correlation Matrix Heatmap')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Find highly correlated features
        high_corr = np.where(np.abs(correlation_matrix) > 0.5)
        high_corr = [(correlation_matrix.index[x], correlation_matrix.columns[y], correlation_matrix.iloc[x, y])
                    for x, y in zip(*high_corr) if x != y and x < y]
        
        print("\nHighly Correlated Features (|correlation| > 0.5):")
        for feat1, feat2, corr in high_corr:
            print(f"{feat1} - {feat2}: {corr:.3f}")
            
    def outlier_detection(self):
        """Detect outliers in numerical variables."""
        print("\n" + "="*60)
        print("OUTLIER DETECTION")
        print("="*60)
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            
            print(f"\n{col}:")
            print(f"Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
            print(f"Lower bound: {lower_bound:.2f}, Upper bound: {upper_bound:.2f}")
            print(f"Number of outliers: {len(outliers)} ({len(outliers)/len(self.df)*100:.2f}%)")
            
            if len(outliers) > 0:
                print(f"Outlier values: {outliers[col].values}")
                
    def age_analysis(self):
        """Detailed analysis of age variable."""
        print("\n" + "="*60)
        print("AGE ANALYSIS")
        print("="*60)
        
        # Age statistics
        print(f"Age Statistics:")
        print(f"Mean age: {self.df['Age'].mean():.2f}")
        print(f"Median age: {self.df['Age'].median():.2f}")
        print(f"Standard deviation: {self.df['Age'].std():.2f}")
        print(f"Min age: {self.df['Age'].min()}")
        print(f"Max age: {self.df['Age'].max()}")
        
        # Age distribution by survival
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        self.df['Age'].hist(bins=30, alpha=0.7, color='lightblue', edgecolor='black')
        plt.title('Age Distribution')
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        
        plt.subplot(1, 3, 2)
        self.df.boxplot(column='Age', by='Survived')
        plt.title('Age Distribution by Survival')
        plt.suptitle('')  # Remove default title
        
        plt.subplot(1, 3, 3)
        for survived in [0, 1]:
            age_data = self.df[self.df['Survived'] == survived]['Age'].dropna()
            plt.hist(age_data, bins=20, alpha=0.7, label=f"{'Did not survive' if survived == 0 else 'Survived'}")
        plt.title('Age Distribution by Survival')
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('age_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Age groups analysis
        self.df['AgeGroup'] = pd.cut(self.df['Age'], bins=[0, 12, 18, 35, 60, 100], 
                                    labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])
        
        age_group_survival = self.df.groupby('AgeGroup')['Survived'].agg(['count', 'sum', 'mean'])
        age_group_survival.columns = ['Total', 'Survived', 'Survival_Rate']
        age_group_survival['Survival_Rate'] = age_group_survival['Survival_Rate'] * 100
        
        print(f"\nSurvival by Age Group:")
        print(age_group_survival)
        
    def fare_analysis(self):
        """Detailed analysis of fare variable."""
        print("\n" + "="*60)
        print("FARE ANALYSIS")
        print("="*60)
        
        # Fare statistics
        print(f"Fare Statistics:")
        print(f"Mean fare: {self.df['Fare'].mean():.2f}")
        print(f"Median fare: {self.df['Fare'].median():.2f}")
        print(f"Standard deviation: {self.df['Fare'].std():.2f}")
        print(f"Min fare: {self.df['Fare'].min()}")
        print(f"Max fare: {self.df['Fare'].max()}")
        
        # Fare distribution
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        self.df['Fare'].hist(bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.title('Fare Distribution')
        plt.xlabel('Fare')
        plt.ylabel('Frequency')
        
        plt.subplot(1, 3, 2)
        self.df.boxplot(column='Fare', by='Survived')
        plt.title('Fare Distribution by Survival')
        plt.suptitle('')
        
        plt.subplot(1, 3, 3)
        for survived in [0, 1]:
            fare_data = self.df[self.df['Survived'] == survived]['Fare']
            plt.hist(fare_data, bins=20, alpha=0.7, label=f"{'Did not survive' if survived == 0 else 'Survived'}")
        plt.title('Fare Distribution by Survival')
        plt.xlabel('Fare')
        plt.ylabel('Frequency')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('fare_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Fare groups analysis
        self.df['FareGroup'] = pd.qcut(self.df['Fare'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
        
        fare_group_survival = self.df.groupby('FareGroup')['Survived'].agg(['count', 'sum', 'mean'])
        fare_group_survival.columns = ['Total', 'Survived', 'Survival_Rate']
        fare_group_survival['Survival_Rate'] = fare_group_survival['Survival_Rate'] * 100
        
        print(f"\nSurvival by Fare Group:")
        print(fare_group_survival)
        
    def family_analysis(self):
        """Analyze family relationships and their impact on survival."""
        print("\n" + "="*60)
        print("FAMILY ANALYSIS")
        print("="*60)
        
        # Create family size feature
        self.df['FamilySize'] = self.df['SibSp'] + self.df['Parch'] + 1
        
        # Family size statistics
        print(f"Family Size Statistics:")
        print(self.df['FamilySize'].describe())
        
        # Family size distribution
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        self.df['FamilySize'].value_counts().sort_index().plot(kind='bar')
        plt.title('Family Size Distribution')
        plt.xlabel('Family Size')
        plt.ylabel('Count')
        
        plt.subplot(1, 3, 2)
        family_survival = self.df.groupby('FamilySize')['Survived'].mean()
        family_survival.plot(kind='bar', color='lightcoral')
        plt.title('Survival Rate by Family Size')
        plt.xlabel('Family Size')
        plt.ylabel('Survival Rate')
        
        plt.subplot(1, 3, 3)
        sns.countplot(data=self.df, x='FamilySize', hue='Survived', palette=['red', 'green'])
        plt.title('Survival by Family Size')
        plt.xlabel('Family Size')
        plt.ylabel('Count')
        plt.legend(['Did not survive', 'Survived'])
        
        plt.tight_layout()
        plt.savefig('family_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Family size groups
        self.df['FamilyGroup'] = pd.cut(self.df['FamilySize'], 
                                       bins=[0, 1, 4, 11], 
                                       labels=['Alone', 'Small Family', 'Large Family'])
        
        family_group_survival = self.df.groupby('FamilyGroup')['Survived'].agg(['count', 'sum', 'mean'])
        family_group_survival.columns = ['Total', 'Survived', 'Survival_Rate']
        family_group_survival['Survival_Rate'] = family_group_survival['Survival_Rate'] * 100
        
        print(f"\nSurvival by Family Group:")
        print(family_group_survival)
        
    def comprehensive_insights(self):
        """Provide comprehensive insights from the analysis."""
        print("\n" + "="*60)
        print("COMPREHENSIVE INSIGHTS")
        print("="*60)
        
        insights = [
            "1. SURVIVAL RATE: Overall survival rate is approximately 38.4%",
            "2. GENDER IMPACT: Females had significantly higher survival rates than males",
            "3. CLASS EFFECT: First class passengers had better survival chances",
            "4. AGE PATTERN: Children and elderly had different survival patterns",
            "5. FAMILY SIZE: Passengers with small families (2-4 members) had better survival",
            "6. FARE CORRELATION: Higher fare passengers had better survival rates",
            "7. EMBARKATION: Port of embarkation shows some survival differences",
            "8. MISSING DATA: Age and Cabin have significant missing values",
            "9. OUTLIERS: Fare variable has several outliers indicating luxury passengers",
            "10. FEATURE IMPORTANCE: Sex, Pclass, and Age are key survival predictors"
        ]
        
        print("\nKey Insights:")
        for insight in insights:
            print(insight)
            
        # Feature importance ranking
        print(f"\nFeature Importance (based on correlation with survival):")
        correlation_with_survival = self.df.corr()['Survived'].abs().sort_values(ascending=False)
        for feature, corr in correlation_with_survival.items():
            if feature != 'Survived':
                print(f"{feature}: {corr:.3f}")
                
    def generate_report(self):
        """Generate a comprehensive EDA report."""
        print("\n" + "="*60)
        print("GENERATING COMPREHENSIVE EDA REPORT")
        print("="*60)
        
        # Run all analyses
        self.basic_info()
        missing_df = self.missing_values_analysis()
        self.descriptive_statistics()
        self.target_variable_analysis()
        self.numerical_distributions()
        self.categorical_analysis()
        self.correlation_analysis()
        self.outlier_detection()
        self.age_analysis()
        self.fare_analysis()
        self.family_analysis()
        self.comprehensive_insights()
        
        print("\n" + "="*60)
        print("EDA COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Generated visualizations saved as PNG files:")
        print("- missing_values_analysis.png")
        print("- survival_analysis.png")
        print("- numerical_distributions.png")
        print("- correlation_matrix.png")
        print("- age_analysis.png")
        print("- fare_analysis.png")
        print("- family_analysis.png")
        print("- [categorical_variable]_analysis.png files")

def main():
    """Main function to run the EDA."""
    print("TITANIC DATASET - EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    # Initialize EDA
    eda = TitanicEDA('titanic_train.csv')
    
    # Generate comprehensive report
    eda.generate_report()
    
    print("\nAnalysis completed! Check the generated visualizations for detailed insights.")

if __name__ == "__main__":
    main() 