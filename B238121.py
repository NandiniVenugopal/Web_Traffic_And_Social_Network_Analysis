import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pprint as pp

data = pd.read_csv('visitor_data_clickstream.csv', index_col = 0)
print(data)

# Extract unique values from the first column
unique_origins = data.iloc[:,0].unique()

# Initialize lists to store campaign metrics
campaign_metrics = []

# Initialize dictionaries to store campaign metrics
campaign_clicks = {}
campaign_conversions = {}
campaign_bounce_rates = {}

# Calculate metrics for each origin
for origin in unique_origins:
    # Filter data for the current origin
    origin_data = data[data.iloc[:,0] == origin]
    
    # Calculate total clicks for the origin
    total_clicks = len(origin_data)
    campaign_clicks[origin] = total_clicks
    
    # Calculate conversions (purchase_success) for the origin
    conversions = len(origin_data[origin_data.iloc[:,7].notnull()])
    campaign_conversions[origin] = conversions
    
    # Calculate conversion rate
    conversion_rate = conversions / total_clicks if total_clicks > 0 else 0

    # Calculate bounce rate
    bounce_rate = (len(origin_data[origin_data['purchase_start'].isnull()]) / total_clicks) * 100

    # Append the metrics to the list
    campaign_metrics.append({
        'Origin': origin,
        'Total Clicks': total_clicks,
        'Conversions': conversions,
        'Conversion Rate': conversion_rate,
        'Bounce Rate': bounce_rate
    })
    
    print(f"Results for {origin}:")
    print(f"Total clicks: {total_clicks}")
    print(f"Conversions: {conversions}")
    print(f"Conversion rate: {conversion_rate:.2%}\n")
    print(f"Bounce rate: {bounce_rate:.2f}%\n")

# Create a DataFrame from the list of metrics
metrics_df = pd.DataFrame(campaign_metrics)

# Display the DataFrame
print(metrics_df)

class Visit:
    def __init__(self, line_of_file):
        line_of_file = line_of_file.rstrip("\n")
        data_items = line_of_file.split(",")
        self.source = data_items[0]
        self.platform = data_items[1]
        self.clickstream = data_items[2:]

    def did_purchase(self):
        return 'purchase_success' in self.clickstream

    def visit_duration(self):
        return len(self.clickstream)

def calculate_metrics(visits):
    total_visits = len(visits)
    total_purchases = sum(visit.did_purchase() for visit in visits)
    total_visit_duration = sum(visit.visit_duration() for visit in visits)

    bounce_rate = (total_visits - total_purchases) / total_visits * 100
    conversion_rate = total_purchases / total_visits * 100
    average_visit_duration = total_visit_duration / total_visits

    return {
        "bounce_rate": bounce_rate,
        "conversion_rate": conversion_rate,
        "average_visit_duration": average_visit_duration
    }

# Read data from file
all_data = []
with open('visitor_data_clickstream.csv', 'r') as file:
    lines = file.readlines()[1:]  # Ignore first line
    for line in lines:
        data_item_object = Visit(line)
        all_data.append(data_item_object)

# Define classes and functions for data processing
class Visit:
    def __init__(self, line_of_file):
        line_of_file = line_of_file.rstrip("\n")
        data_items = line_of_file.split(",")
        self.source = data_items[0]
        self.platform = data_items[1]
        self.clickstream = data_items[2:]

    def did_purchase(self):
        return 'purchase_success' in self.clickstream

    def visit_duration(self):
        return len(self.clickstream)

def calculate_metrics(visits):
    total_visits = len(visits)
    total_purchases = sum(visit.did_purchase() for visit in visits)
    total_visit_duration = sum(visit.visit_duration() for visit in visits)

    bounce_rate = (total_visits - total_purchases) / total_visits * 100
    conversion_rate = total_purchases / total_visits * 100
    average_visit_duration = total_visit_duration / total_visits

    return {
        "bounce_rate": bounce_rate,
        "conversion_rate": conversion_rate,
        "average_visit_duration": average_visit_duration
    }

# Read data from file
all_data = []
with open('visitor_data_clickstream.csv', 'r') as file:
    lines = file.readlines()[1:]  # Ignore first line
    for line in lines:
        data_item_object = Visit(line)
        all_data.append(data_item_object)

# Segregate data by platform and source
platforms = ['android', 'ios', 'windows', 'mac']
sources = ['direct', 'search', 'facebook_share', 'linkedin_share', 'linkedin_advert', 'facebook_advert', 'partner_advert']  # Add more sources as needed

metrics_by_platform = {}
metrics_by_source = {}

for platform in platforms:
    platform_visits = [visit for visit in all_data if visit.platform == platform]
    metrics_by_platform[platform] = calculate_metrics(platform_visits)

for source in sources:
    source_visits = [visit for visit in all_data if visit.source == source]
    metrics_by_source[source] = calculate_metrics(source_visits)

print("Metrics by Platform:")
for platform, metrics in metrics_by_platform.items():
    print(f"{platform.capitalize()} Metrics:")
    print(metrics)

print("\nMetrics by Source:")
for source, metrics in metrics_by_source.items():
    print(f"{source.capitalize()} Metrics:")
    print(metrics)

import matplotlib.pyplot as plt

# Metrics by Platform
platform_metrics = {
    'Android': {'bounce_rate': 85.08, 'conversion_rate': 14.92, 'average_visit_duration': 18.0},
    'iOS': {'bounce_rate': 87.31, 'conversion_rate': 12.69, 'average_visit_duration': 18.0},
    'Windows': {'bounce_rate': 83.6, 'conversion_rate': 16.4, 'average_visit_duration': 18.0},
    'Mac': {'bounce_rate': 83.99, 'conversion_rate': 16.00, 'average_visit_duration': 18.0}
}

plt.figure(figsize=(10, 6))

# Plot bounce rate
plt.subplot(2, 1, 1)
plt.bar(platform_metrics.keys(), [metrics['bounce_rate'] for metrics in platform_metrics.values()], color='skyblue')
plt.title('Bounce Rate by Platform')
plt.ylabel('Bounce Rate (%)')

# Metrics by Source
source_metrics = {
    'Direct': {'bounce_rate': 68.33, 'conversion_rate': 31.67, 'average_visit_duration': 18.0},
    'Search': {'bounce_rate': 84.81, 'conversion_rate': 15.19, 'average_visit_duration': 18.0},
    'Facebook Share': {'bounce_rate': 87.02, 'conversion_rate': 12.98, 'average_visit_duration': 18.0},
    'LinkedIn Share': {'bounce_rate': 87.99, 'conversion_rate': 12.01, 'average_visit_duration': 18.0},
    'LinkedIn Advert': {'bounce_rate': 74.8, 'conversion_rate': 25.2, 'average_visit_duration': 18.0},
    'Facebook Advert': {'bounce_rate': 99.07, 'conversion_rate': 0.93, 'average_visit_duration': 18.0},
    'Partner Advert': {'bounce_rate': 89.1, 'conversion_rate': 10.9, 'average_visit_duration': 18.0}
}

plt.figure(figsize=(10, 6))

# Plot bounce rate
plt.subplot(2, 1, 1)
plt.bar(source_metrics.keys(), [metrics['bounce_rate'] for metrics in source_metrics.values()], color='skyblue')
plt.title('Bounce Rate by Source')
plt.ylabel('Bounce Rate (%)')


plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Create DataFrame for metrics by platform
platform_df = pd.DataFrame(platform_metrics).transpose()

# Display DataFrame
print("Metrics by Platform:")
print(platform_df)

# Create DataFrame for metrics by source
source_df = pd.DataFrame(source_metrics).transpose()

# Display DataFrame
print("\nMetrics by Source:")
print(source_df)

# Filter the dataset for paths containing only 'blog_1'
blog_1_data = data[data.apply(lambda row: 'blog_1' in row.values and 'blog_2' not in row.values, axis=1)]

# Convert non-numeric columns to numeric, ignoring errors
blog_1_data_numeric = blog_1_data.apply(pd.to_numeric, errors='ignore')

# Drop non-numeric columns if any
blog_1_data_numeric = blog_1_data_numeric.select_dtypes(include=np.number)

# Calculate metrics for paths containing only 'blog_1'
total_clicks_blog_1 = blog_1_data_numeric.sum().sum()  # Sum all the values in the filtered dataset
conversions_blog_1 = len(blog_1_data[blog_1_data['purchase_success'].notnull()])
total_visits_blog_1 = len(blog_1_data)

conversion_rate_blog_1 = (conversions_blog_1 / total_visits_blog_1) * 100 if total_visits_blog_1 > 0 else 0
bounce_rate_blog_1 = ((total_visits_blog_1 - conversions_blog_1) / total_visits_blog_1) * 100
average_visit_duration_blog_1 = total_clicks_blog_1 / total_visits_blog_1

# Display the results
print("Metrics for paths containing only 'blog_1':")
print(f"Total Clicks: {total_clicks_blog_1}")
print(f"Conversions: {conversions_blog_1}")
print(f"Conversion Rate: {conversion_rate_blog_1:.2f}%")
print(f"Bounce Rate: {bounce_rate_blog_1:.2f}%")
print(f"Average Visit Duration: {average_visit_duration_blog_1:.2f}")

# Filter the dataset for paths containing only 'blog_2'
blog_2_data = data[data.apply(lambda row: 'blog_2' in row.values and 'blog_1' not in row.values, axis=1)]

# Convert non-numeric columns to numeric, ignoring errors
blog_2_data_numeric = blog_2_data.apply(pd.to_numeric, errors='ignore')

# Drop non-numeric columns if any
blog_2_data_numeric = blog_2_data_numeric.select_dtypes(include=np.number)

# Calculate metrics for paths containing only 'blog_2'
total_clicks_blog_2 = blog_2_data_numeric.sum().sum()  # Sum all the values in the filtered dataset
conversions_blog_2 = len(blog_2_data[blog_2_data['purchase_success'].notnull()])
total_visits_blog_2 = len(blog_2_data)

conversion_rate_blog_2 = (conversions_blog_2 / total_visits_blog_2) * 100 if total_visits_blog_2 > 0 else 0
bounce_rate_blog_2 = ((total_visits_blog_2 - conversions_blog_2) / total_visits_blog_2) * 100
average_visit_duration_blog_2 = total_clicks_blog_2 / total_visits_blog_2

# Display the results
print("Metrics for paths containing only 'blog_2':")
print(f"Total Clicks: {total_clicks_blog_2}")
print(f"Conversions: {conversions_blog_2}")
print(f"Conversion Rate: {conversion_rate_blog_2:.2f}%")
print(f"Bounce Rate: {bounce_rate_blog_2:.2f}%")
print(f"Average Visit Duration: {average_visit_duration_blog_2:.2f}")
import matplotlib.pyplot as plt

# Metrics for 'blog_1'
metrics_blog_1 = {
    'Bounce Rate': bounce_rate_blog_1,
    'Conversion Rate': conversion_rate_blog_1,
}

# Metrics for 'blog_2'
metrics_blog_2 = {
    'Bounce Rate': bounce_rate_blog_2,
    'Conversion Rate': conversion_rate_blog_2,
}

# Create lists of metric names and values for both 'blog_1' and 'blog_2'
metric_names = list(metrics_blog_1.keys())
metrics_values_blog_1 = list(metrics_blog_1.values())
metrics_values_blog_2 = list(metrics_blog_2.values())

# Create a bar plot to compare metrics of 'blog_1' and 'blog_2'
fig, ax = plt.subplots(figsize=(10, 6))

# Define width of bars
bar_width = 0.35

# Define the index for the bars
index = range(len(metric_names))

# Plot 'blog_2' metrics
bar1 = ax.bar(index, metrics_values_blog_2, bar_width, label='Blog 2', color='blue')

# Plot 'blog_1' metrics
bar2 = ax.bar([i + bar_width for i in index], metrics_values_blog_1, bar_width, label='Blog 1', color='orange')

# Set labels and title
ax.set_xlabel('Metrics')
ax.set_ylabel('Values')
ax.set_title('Comparison of Bounce Rate, Conversion Rate for Blog 1 and Blog 2')
ax.set_xticks([i + bar_width / 2 for i in index])
ax.set_xticklabels(metric_names)
ax.legend()

# Add text labels for each bar
for rect in bar1 + bar2:
    height = rect.get_height()
    ax.annotate('{}'.format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

# Show plot
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Define the data for the pie chart
labels = list(metrics_by_source.keys())
values = [metrics['conversion_rate'] for metrics in metrics_by_source.values()]

# Create the pie chart
plt.figure(figsize=(8, 8))
plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Conversion Rate by Source')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

# Show the pie chart
plt.show()

# Define the data for the pie chart
labels = list(metrics_by_source.keys())
values = [metrics['conversion_rate'] for metrics in metrics_by_source.values()]

# Create the pie chart
plt.figure(figsize=(8, 8))
plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Conversion Rate by Source')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

# Show the pie chart
plt.show()

# Define the data for the bar graph
platforms = list(metrics_by_platform.keys())
conversion_rates = [metrics['conversion_rate'] for metrics in metrics_by_platform.values()]

# Create the bar graph
plt.figure(figsize=(10, 6))
plt.bar(platforms, conversion_rates, color='skyblue')
plt.title('Conversion Rate by Platform')
plt.xlabel('Platform')
plt.ylabel('Conversion Rate (%)')

# Show the bar graph
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Define the data for the bar graph
platforms = list(metrics_by_platform.keys())
conversion_rates = [metrics['conversion_rate'] for metrics in metrics_by_platform.values()]

# Create the bar graph
plt.figure(figsize=(10, 6))
plt.bar(platforms, conversion_rates, color='skyblue')
plt.title('Conversion Rate by Platform')
plt.xlabel('Platform')
plt.ylabel('Conversion Rate (%)')

# Set the y-axis ticks to display percentages
plt.gca().set_yticklabels(['{:.0f}%'.format(x) for x in plt.gca().get_yticks()])

# Show the bar graph
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Define the data for the bar graph
origins = list(metrics_by_source.keys())
conversion_rates = [metrics['conversion_rate'] for metrics in metrics_by_source.values()]

# Create the bar graph
plt.figure(figsize=(10, 6))
plt.bar(origins, conversion_rates, color='skyblue')
plt.title('Conversion Rate by Origin')
plt.xlabel('Origin')
plt.ylabel('Conversion Rate')

# Set the y-axis ticks to display percentages
plt.gca().set_yticklabels(['{}%'.format(int(x)) for x in plt.gca().get_yticks()])

# Show the bar graph
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Define the data for the bar graph
campaign_origins = list(metrics_by_platform.keys())
conversion_rates = [metrics['conversion_rate'] for metrics in metrics_by_platform.values()]

# Create the bar graph
plt.figure(figsize=(10, 6))
plt.bar(campaign_origins, conversion_rates, color='skyblue')
plt.title('Conversion Rate by Campaign Origin')
plt.xlabel('Campaign Origin')
plt.ylabel('Conversion Rate (%)')

# Set the y-axis ticks to display percentages
plt.yticks(np.arange(0, max(conversion_rates) + 10, 10))

# Show the bar graph
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Create lists of metric names and values for both 'blog_1' and 'blog_2'
metric_names = list(metrics_blog_1.keys())
metrics_values_blog_1 = list(metrics_blog_1.values())
metrics_values_blog_2 = list(metrics_blog_2.values())

# Create a bar plot to compare metrics of 'blog_1' and 'blog_2'
fig, ax = plt.subplots(figsize=(10, 6))

# Define width of bars
bar_width = 0.35

# Define the index for the bars
index = range(len(metric_names))

# Plot 'blog_1' metrics
bar1 = ax.bar(index, metrics_values_blog_1, bar_width, label='Blog 1', color='orange')

# Plot 'blog_2' metrics
bar2 = ax.bar([i + bar_width for i in index], metrics_values_blog_2, bar_width, label='Blog 2', color='blue')

# Set labels and title
ax.set_xlabel('Metrics')
ax.set_ylabel('Values')
ax.set_title('Comparison of Bounce Rate, Conversion Rate for Blog 1 and Blog 2')
ax.set_xticks([i + bar_width / 2 for i in index])
ax.set_xticklabels(metric_names)
ax.legend()

# Add text labels for each bar
for rect in bar1 + bar2:
    height = rect.get_height()
    ax.annotate('{}'.format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

# Show plot
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()