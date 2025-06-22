import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load the data
print("Loading FIFA dataset...")
df = pd.read_csv('fifa_data.csv')

# Clean the data
print("Cleaning data...")
# Remove rows with missing values in key columns
df = df.dropna(subset=['Overall', 'Age', 'Value', 'Wage'])

# Convert Value and Wage to numeric (remove € and M/K symbols)
def convert_value(value_str):
    if pd.isna(value_str) or value_str == '':
        return 0
    value_str = str(value_str).replace('€', '').replace(',', '')
    if 'M' in value_str:
        return float(value_str.replace('M', '')) * 1000000
    elif 'K' in value_str:
        return float(value_str.replace('K', '')) * 1000
    else:
        return float(value_str)

df['Value_Numeric'] = df['Value'].apply(convert_value)
df['Wage_Numeric'] = df['Wage'].apply(convert_value)

# Convert Height to cm
def height_to_cm(height_str):
    if pd.isna(height_str) or height_str == '':
        return 0
    height_str = str(height_str)
    if "'" in height_str:
        feet, inches = height_str.split("'")
        inches = inches.replace('"', '')
        return int(feet) * 30.48 + int(inches) * 2.54
    return 0

df['Height_cm'] = df['Height'].apply(height_to_cm)

# Convert Weight to kg
def weight_to_kg(weight_str):
    if pd.isna(weight_str) or weight_str == '':
        return 0
    weight_str = str(weight_str)
    if 'lbs' in weight_str:
        lbs = float(weight_str.replace('lbs', ''))
        return lbs * 0.453592
    return 0

df['Weight_kg'] = df['Weight'].apply(weight_to_kg)

print(f"Dataset loaded with {len(df)} players")

# Create a comprehensive visualization dashboard
fig = plt.figure(figsize=(20, 24))

# 1. Overall Rating Distribution
plt.subplot(4, 3, 1)
sns.histplot(data=df, x='Overall', bins=30, kde=True, color='skyblue', alpha=0.7)
plt.title('Distribution of Player Overall Ratings', fontsize=14, fontweight='bold')
plt.xlabel('Overall Rating')
plt.ylabel('Number of Players')

# 2. Age Distribution by Position
plt.subplot(4, 3, 2)
position_age = df.groupby('Position')['Age'].mean().sort_values(ascending=False)
colors = plt.cm.viridis(np.linspace(0, 1, len(position_age)))
plt.bar(range(len(position_age)), position_age.values, color=colors)
plt.title('Average Age by Position', fontsize=14, fontweight='bold')
plt.xlabel('Position')
plt.ylabel('Average Age')
plt.xticks(range(len(position_age)), position_age.index, rotation=45)

# 3. Top 10 Players by Overall Rating
plt.subplot(4, 3, 3)
top_players = df.nlargest(10, 'Overall')[['Name', 'Overall', 'Club']]
bars = plt.barh(range(len(top_players)), top_players['Overall'], 
                color=plt.cm.plasma(np.linspace(0, 1, len(top_players))))
plt.title('Top 10 Players by Overall Rating', fontsize=14, fontweight='bold')
plt.xlabel('Overall Rating')
plt.yticks(range(len(top_players)), top_players['Name'], fontsize=8)
for i, (bar, overall) in enumerate(zip(bars, top_players['Overall'])):
    plt.text(overall + 0.5, i, str(overall), va='center', fontweight='bold')

# 4. Value vs Overall Rating Scatter Plot
plt.subplot(4, 3, 4)
plt.scatter(df['Overall'], df['Value_Numeric'] / 1000000, alpha=0.6, s=20, c=df['Age'], cmap='viridis')
plt.colorbar(label='Age')
plt.title('Player Value vs Overall Rating', fontsize=14, fontweight='bold')
plt.xlabel('Overall Rating')
plt.ylabel('Value (Million €)')

# 5. Top 10 Clubs by Average Overall Rating
plt.subplot(4, 3, 5)
club_avg = df.groupby('Club')['Overall'].mean().sort_values(ascending=False).head(10)
colors = plt.cm.Set3(np.linspace(0, 1, len(club_avg)))
plt.barh(range(len(club_avg)), club_avg.values, color=colors)
plt.title('Top 10 Clubs by Average Overall Rating', fontsize=14, fontweight='bold')
plt.xlabel('Average Overall Rating')
plt.yticks(range(len(club_avg)), club_avg.index, fontsize=8)

# 6. Nationality Distribution (Top 15)
plt.subplot(4, 3, 6)
nationality_counts = df['Nationality'].value_counts().head(15)
plt.pie(nationality_counts.values, labels=nationality_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Top 15 Nationalities', fontsize=14, fontweight='bold')

# 7. Height vs Weight Scatter Plot
plt.subplot(4, 3, 7)
valid_height_weight = df[(df['Height_cm'] > 0) & (df['Weight_kg'] > 0)]
plt.scatter(valid_height_weight['Height_cm'], valid_height_weight['Weight_kg'], 
           alpha=0.6, s=20, c=valid_height_weight['Overall'], cmap='plasma')
plt.colorbar(label='Overall Rating')
plt.title('Height vs Weight by Overall Rating', fontsize=14, fontweight='bold')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')

# 8. Preferred Foot Distribution
plt.subplot(4, 3, 8)
foot_counts = df['Preferred Foot'].value_counts()
colors = ['#ff9999', '#66b3ff']
plt.pie(foot_counts.values, labels=foot_counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
plt.title('Preferred Foot Distribution', fontsize=14, fontweight='bold')

# 9. Work Rate Analysis
plt.subplot(4, 3, 9)
work_rate_counts = df['Work Rate'].value_counts().head(10)
plt.bar(range(len(work_rate_counts)), work_rate_counts.values, 
        color=plt.cm.tab10(np.linspace(0, 1, len(work_rate_counts))))
plt.title('Top 10 Work Rate Combinations', fontsize=14, fontweight='bold')
plt.xlabel('Work Rate')
plt.ylabel('Number of Players')
plt.xticks(range(len(work_rate_counts)), work_rate_counts.index, rotation=45, fontsize=8)

# 10. Skill Moves vs Weak Foot
plt.subplot(4, 3, 10)
skill_weak = df.groupby(['Skill Moves', 'Weak Foot']).size().unstack(fill_value=0)
sns.heatmap(skill_weak, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': 'Number of Players'})
plt.title('Skill Moves vs Weak Foot', fontsize=14, fontweight='bold')
plt.xlabel('Weak Foot Rating')
plt.ylabel('Skill Moves Rating')

# 11. Age vs Potential Scatter Plot
plt.subplot(4, 3, 11)
plt.scatter(df['Age'], df['Potential'], alpha=0.6, s=20, c=df['Overall'], cmap='viridis')
plt.colorbar(label='Overall Rating')
plt.title('Age vs Potential by Overall Rating', fontsize=14, fontweight='bold')
plt.xlabel('Age')
plt.ylabel('Potential Rating')

# 12. Wage Distribution by Position
plt.subplot(4, 3, 12)
position_wage = df.groupby('Position')['Wage_Numeric'].mean().sort_values(ascending=False)
colors = plt.cm.coolwarm(np.linspace(0, 1, len(position_wage)))
plt.bar(range(len(position_wage)), position_wage.values / 1000, color=colors)
plt.title('Average Wage by Position', fontsize=14, fontweight='bold')
plt.xlabel('Position')
plt.ylabel('Average Wage (K €)')
plt.xticks(range(len(position_wage)), position_wage.index, rotation=45)

plt.tight_layout()
plt.savefig('fifa_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Create additional detailed analysis
fig2, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Performance Attributes Radar Chart for Top Players
ax1 = plt.subplot(2, 2, 1, projection='polar')
top_5_players = df.nlargest(5, 'Overall')
attributes = ['Finishing', 'Dribbling', 'BallControl', 'Acceleration', 'SprintSpeed', 'Stamina']
angles = np.linspace(0, 2 * np.pi, len(attributes), endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

for i, player in top_5_players.iterrows():
    values = [player[attr] for attr in attributes]
    values += values[:1]  # Complete the circle
    ax1.plot(angles, values, 'o-', linewidth=2, label=player['Name'], alpha=0.7)
    ax1.fill(angles, values, alpha=0.1)

ax1.set_xticks(angles[:-1])
ax1.set_xticklabels(attributes)
ax1.set_ylim(0, 100)
ax1.set_title('Performance Attributes - Top 5 Players', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

# 2. Club Value Analysis
ax2 = plt.subplot(2, 2, 2)
club_value = df.groupby('Club')['Value_Numeric'].sum().sort_values(ascending=False).head(15)
colors = plt.cm.viridis(np.linspace(0, 1, len(club_value)))
bars = ax2.bar(range(len(club_value)), club_value.values / 1000000, color=colors)
ax2.set_title('Total Squad Value by Club (Top 15)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Club')
ax2.set_ylabel('Total Value (Million €)')
ax2.set_xticks(range(len(club_value)))
ax2.set_xticklabels(club_value.index, rotation=45, ha='right', fontsize=8)

# Add value labels on bars
for i, (bar, value) in enumerate(zip(bars, club_value.values)):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
             f'{value/1000000:.0f}M', ha='center', va='bottom', fontsize=8)

# 3. Age Distribution by Overall Rating
ax3 = plt.subplot(2, 2, 3)
# Create age bins
df['Age_Bin'] = pd.cut(df['Age'], bins=[15, 20, 25, 30, 35, 40, 50], labels=['15-20', '21-25', '26-30', '31-35', '36-40', '40+'])
age_overall = df.groupby('Age_Bin')['Overall'].mean()
colors = plt.cm.plasma(np.linspace(0, 1, len(age_overall)))
bars = ax3.bar(range(len(age_overall)), age_overall.values, color=colors)
ax3.set_title('Average Overall Rating by Age Group', fontsize=14, fontweight='bold')
ax3.set_xlabel('Age Group')
ax3.set_ylabel('Average Overall Rating')
ax3.set_xticks(range(len(age_overall)))
ax3.set_xticklabels(age_overall.index)

# Add rating labels on bars
for i, (bar, rating) in enumerate(zip(bars, age_overall.values)):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'{rating:.1f}', ha='center', va='bottom', fontweight='bold')

# 4. Correlation Heatmap of Key Attributes
ax4 = plt.subplot(2, 2, 4)
key_attributes = ['Overall', 'Potential', 'Age', 'Value_Numeric', 'Wage_Numeric', 
                  'Finishing', 'Dribbling', 'BallControl', 'Acceleration', 'SprintSpeed']
correlation_matrix = df[key_attributes].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, ax=ax4, fmt='.2f', cbar_kws={'label': 'Correlation Coefficient'})
ax4.set_title('Correlation Matrix of Key Attributes', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('fifa_detailed_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\n" + "="*50)
print("FIFA DATASET ANALYSIS SUMMARY")
print("="*50)
print(f"Total Players: {len(df):,}")
print(f"Average Overall Rating: {df['Overall'].mean():.2f}")
print(f"Average Age: {df['Age'].mean():.2f}")
print(f"Total Market Value: €{df['Value_Numeric'].sum()/1000000000:.2f}B")
print(f"Average Player Value: €{df['Value_Numeric'].mean()/1000000:.2f}M")
print(f"Top 5 Players by Overall Rating:")
for i, player in df.nlargest(5, 'Overall')[['Name', 'Overall', 'Club']].iterrows():
    print(f"  {player['Name']} - {player['Overall']} ({player['Club']})")
print(f"Most Common Nationality: {df['Nationality'].mode()[0]}")
print(f"Most Common Position: {df['Position'].mode()[0]}")
print("="*50)

print("\nVisualization files saved:")
print("- fifa_comprehensive_analysis.png")
print("- fifa_detailed_analysis.png") 