
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
from config import *


def plot_class_explanations(aggregated_explanations, method, label_names, extra_str):

    min_importance = min(min(importance.values()) for importance in aggregated_explanations.values()) * 1.1
    max_importance = max(max(importance.values()) for importance in aggregated_explanations.values()) * 1.1
    
    for cluster, feature_importances in aggregated_explanations.items():
        features = list(feature_importances.keys())
        importances = list(feature_importances.values())
        sorted_indices = np.argsort(importances)
        sorted_features = [features[i] for i in sorted_indices]
        sorted_importances = [importances[i] for i in sorted_indices]
        
        plt.figure(figsize=(15, 10))  
        plt.barh(sorted_features, sorted_importances, color='blue', alpha=0.7)#, label=label_names[cluster])
        plt.title(f'{method} Explanations for \"{label_names[cluster]}\"', fontsize=16)
        plt.xlabel('Influence on Prediction', fontsize=14)
        plt.ylabel('Features', fontsize=14)
        plt.xticks(fontsize=16)  
        plt.xlim(min_importance, max_importance)
        plt.gca().invert_yaxis() 
        plt.legend(loc='best')
        plt.tight_layout() 
        plt.savefig(IMAGES_PATH + f'rf80_{method}_explanations_{label_names[cluster]}_{extra_str}.png', dpi=300)
        plt.savefig(IMAGES_PATH + f'rf80_{method}_explanations_{label_names[cluster]}_{extra_str}.eps', dpi=300)
        plt.show()

def plot_aggregated_class_explanations(aggregated_explanations, method, label_names, extra_str):

    min_importance = min(min(importance.values()) for importance in aggregated_explanations.values()) * 1.1
    max_importance = max(max(importance.values()) for importance in aggregated_explanations.values()) * 1.1
    
    fig, ax = plt.subplots(figsize=(15, 10))
    colors = ['red', 'yellow', 'blue']
    
    for cluster, feature_importances in aggregated_explanations.items():
        features = list(feature_importances.keys())
        importances = list(feature_importances.values())
        sorted_indices = np.argsort(importances)
        sorted_features = [features[i] for i in sorted_indices]
        sorted_importances = [importances[i] for i in sorted_indices]
        ax.barh(sorted_features, sorted_importances, color=colors[cluster], alpha=0.7, label=label_names[cluster])
    
    ax.set_title(f'{method} Explanations', fontsize=16)
    ax.set_xlabel('Influence on Prediction', fontsize=14)
    ax.set_ylabel('Features', fontsize=14)
    # ax.set_xticks(fontsize=16)  
    ax.set_xlim(min_importance, max_importance)
    ax.invert_yaxis() 
    ax.legend(['"Not Ready', 'Partially Ready', 'Ready'])
    ax.legend(loc='best')
    plt.grid(True)  # Set grid on
    plt.tight_layout() 
    plt.savefig(IMAGES_PATH + f'rf80_{method}_explanations_aggregated_{extra_str}.png', dpi=300)
    plt.savefig(IMAGES_PATH + f'rf80_{method}_explanations_aggregated_{extra_str}.eps', dpi=300)
    plt.show()

def plot_aggregated_class_explanations_horiz(aggregated_explanations, method, label_names, extra_str):
    
    # Determine all unique features across clusters
    all_features = set()
    for feature_importances in aggregated_explanations.values():
        all_features.update(feature_importances.keys())
    all_features = sorted(all_features)  # Sort for consistent ordering
    
    num_clusters = len(aggregated_explanations)
    bar_width = 0.25
    colors = ['red', 'yellow', 'blue']
    
    fig, ax = plt.subplots(figsize=(15, 10))
    
    for cluster_idx, (cluster, feature_importances) in enumerate(aggregated_explanations.items()):
        importances = [feature_importances.get(feature, 0) for feature in all_features]
        
        # Calculate x-axis positions for each cluster, shifted by bar_width
        x_positions = np.arange(len(all_features)) + cluster_idx * bar_width
        
        # Plot the bars with the x-positions for this cluster
        ax.bar(x_positions, importances, bar_width, color=colors[cluster_idx], alpha=0.7, label=label_names[cluster])
    
    # Adjust x-axis ticks to be in the center of the grouped bars
    ax.set_xticks(np.arange(len(all_features)) + (num_clusters - 1) * bar_width / 2)
    ax.set_xticklabels(all_features, fontsize=16)
    
    ax.set_title(f'{method} Explanations', fontsize=16)
    ax.set_xlabel('Features', fontsize=14)
    ax.set_ylabel('Influence on Prediction', fontsize=14)
    ax.legend(loc='best')
    
    plt.xticks(rotation=45, ha="right")
    plt.grid(True)  # Set grid on
    plt.tight_layout() 
    plt.savefig(IMAGES_PATH + f'rf80_{method}_explanations_aggregated_{extra_str}.png', dpi=300)
    plt.savefig(IMAGES_PATH + f'rf80_{method}_explanations_aggregated_{extra_str}.eps', dpi=300)
    plt.show()

def plot_aggregated_class_explanations_horiz_ordered_by_sum_of_absolutes(aggregated_explanations, method, label_names, extra_str):
    
    # Determine all unique features across clusters and calculate the sum of absolute importance
    all_features = set()
    feature_total_importance = {}
    
    for feature_importances in aggregated_explanations.values():
        for feature, importance in feature_importances.items():
            all_features.add(feature)
            if feature in feature_total_importance:
                feature_total_importance[feature] += abs(importance)
            else:
                feature_total_importance[feature] = abs(importance)
    
    # Sort features by their total absolute importance
    sorted_features = sorted(all_features, key=lambda x: feature_total_importance[x], reverse=True)
    
    num_clusters = len(aggregated_explanations)
    bar_width = 0.25
    colors = ['red', 'yellow', 'blue']
    
    fig, ax = plt.subplots(figsize=(15, 10))
    
    for cluster_idx, (cluster, feature_importances) in enumerate(aggregated_explanations.items()):
        importances = [feature_importances.get(feature, 0) for feature in sorted_features]
        
        # Calculate x-axis positions for each cluster, shifted by bar_width
        x_positions = np.arange(len(sorted_features)) + cluster_idx * bar_width
        
        # Plot the bars with the x-positions for this cluster
        ax.bar(x_positions, importances, bar_width, color=colors[cluster_idx], alpha=0.7, label=label_names[cluster])
    
    # Adjust x-axis ticks to be in the center of the grouped bars
    ax.set_xticks(np.arange(len(sorted_features)) + (num_clusters - 1) * bar_width / 2)
    ax.set_xticklabels(sorted_features, fontsize=16)  # Set a bigger font size for xticks
    
    ax.set_title(f'{method} Explanations', fontsize=16)
    ax.set_xlabel('Features', fontsize=14)
    ax.set_ylabel('Influence on Prediction', fontsize=14)
    ax.legend(loc='best')
    
    plt.xticks(rotation=45, ha="right")
    plt.grid(True)  # Set grid on
    plt.tight_layout() 
    plt.savefig(IMAGES_PATH + f'rf80_{method}_explanations_aggregated_{extra_str}.png', dpi=300)
    plt.savefig(IMAGES_PATH + f'rf80_{method}_explanations_aggregated_{extra_str}.eps', dpi=300)
    plt.show()

def aggregate_shap_values(shap_explanations, feature_names):
    aggregated_explanations = {0: {}, 1: {}, 2: {}}
    
    for cluster in shap_explanations.keys():
        feature_importances = {feature: [] for feature in feature_names}
        
        for shap_values in shap_explanations[cluster]:
            for feature_index, feature_value in enumerate(shap_values.values[0]):
                feature_name = feature_names[feature_index]
                feature_importances[feature_name].append(feature_value)
        
        for feature_name in feature_importances.keys():
            aggregated_explanations[cluster][feature_name] = np.mean(feature_importances[feature_name])

    return aggregated_explanations


def aggregate_lime_explanations(lime_explanations):
    aggregated_explanations = {}
    for cluster, explanations in lime_explanations.items():
        feature_importances = {}
        for exp in explanations:
            for feature, importance in exp.as_list():
                feature_name = feature.split(' <= ')[0] 
                if feature_name in feature_importances:
                    feature_importances[feature_name] += importance
                else:
                    feature_importances[feature_name] = importance
        for feature in feature_importances:
            feature_importances[feature] /= len(explanations)
        aggregated_explanations[cluster] = feature_importances
    return aggregated_explanations