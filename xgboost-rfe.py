import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import xgboost as xgb  # 修改：从catboost改为xgboost
import time
import warnings
import os
import shap

warnings.filterwarnings('ignore')
n_estimators = 100
n_folds = 10
random_state = 32
test_size = 0.4


class XGBoost_RFE_FeatureSelector:  # 修改：类名从CatBoost改为XGBoost
    """
    XGBoost Recursive Feature Elimination for feature selection with hyperparameter optimization
    """

    def __init__(self, n_estimators=n_estimators, n_folds=n_folds, random_state=random_state,
                 optimize_params=True, n_iter=10):
        """
        Initialize the XGBoost feature selector  # 修改：注释

        Parameters:
        -----------
        n_estimators : int
            Number of trees in XGBoost (will be overridden if optimize_params=True)  # 修改：注释
        n_folds : int
            Number of folds for cross-validation
        random_state : int
            Random seed for reproducibility
        optimize_params : bool
            Whether to optimize hyperparameters using RandomizedSearchCV
        n_iter : int
            Number of iterations for hyperparameter optimization
        """
        self.n_estimators = n_estimators
        self.n_folds = n_folds
        self.random_state = random_state
        self.optimize_params = optimize_params
        self.n_iter = n_iter
        self.results = {
            'n_features': [],
            'features': [],
            'avg_mse': [],
            'avg_r2': [],
            'avg_mae': [],
            'avg_mape': [],
            'importance_scores': [],
            'feature_ranking': []
        }
        self.optimal_features = None
        self.best_params = None

    def _optimize_hyperparameters(self, X, y, verbose=True):
        """
        Optimize XGBoost hyperparameters using RandomizedSearchCV
        """
        if verbose:
            print("Optimizing XGBoost hyperparameters...")

        # Create base model
        base_model = xgb.XGBRegressor(
            random_state=self.random_state,
            verbosity=0,
            n_jobs=-1
        )

        param_grid = {
            'n_estimators': [100, 200, 300, 400, 500],
            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
            'max_depth': [3, 5, 7, 9],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.01, 0.1, 0.5, 1.0],
            'reg_lambda': [0.1, 0.5, 1.0, 1.5, 2.0],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2, 0.3]
        }

        # Set up cross-validation
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        # Perform randomized search
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=self.n_iter,
            scoring='neg_root_mean_squared_error',
            cv=kf,
            random_state=self.random_state,
            n_jobs=-1
        )

        random_search.fit(X, y)
        self.best_params = random_search.best_params_

        if verbose:
            print(f"Best parameters: {self.best_params}")
            print(f"Best RMSE: {abs(random_search.best_score_):.6f}")

        return self.best_params

    def fit(self, X, y, verbose=True):
        """
        Perform the XGBoost-RFE feature selection process  # 修改：注释

        Parameters:
        -----------
        X : pandas DataFrame
            Features
        y : pandas Series
            Target variable
        verbose : bool
            Whether to print progress information

        Returns:
        --------
        self : object
            Returns self
        """
        start_time = time.time()

        feature_names = list(X.columns)
        n_features = len(feature_names)

        if verbose:
            print(f"Starting XGBoost-RFE feature selection with {n_features} features")  # 修改：打印信息
            print("=" * 60)

        # Optimize hyperparameters on full dataset if requested
        if self.optimize_params:
            self._optimize_hyperparameters(X, y, verbose)

            # Start with all features
            current_features = feature_names.copy()
            X_current = X.copy()

            # Track the best model and its performance
            best_mse = float('inf')
            best_features = None

            # Initialize feature ranking dictionary
            feature_ranking = {feature: 0 for feature in feature_names}

            # Recursive Feature Elimination
            iteration = 1
            while len(current_features) >= 1:
                if verbose:
                    print(f"\nIteration {iteration}: Evaluating {len(current_features)} features")

                # Set up k-fold cross-validation
                kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
                fold_mse = []
                fold_r2 = []
                fold_mae = []
                fold_mape = []
                fold_importance = []

                # Perform k-fold cross-validation
                for fold, (train_idx, val_idx) in enumerate(kf.split(X_current)):
                    # Get training and validation data for this fold
                    X_fold_train = X_current.iloc[train_idx]
                    y_fold_train = y.iloc[train_idx]
                    X_fold_val = X_current.iloc[val_idx]
                    y_fold_val = y.iloc[val_idx]

                    # Create XGBoost model with optimized or default parameters
                    if self.best_params:
                        xgb_model = xgb.XGBRegressor(  # 修改：模型创建
                            random_state=self.random_state,
                            verbosity=0,
                            n_jobs=-1,
                            **self.best_params
                        )
                    else:
                        xgb_model = xgb.XGBRegressor(  # 修改：模型创建
                            n_estimators=self.n_estimators,
                            random_state=self.random_state,
                            verbosity=0,
                            n_jobs=-1
                        )

                    # Train model
                    xgb_model.fit(X_fold_train, y_fold_train)  # 修改：变量名

                    # Evaluate on validation set
                    y_pred = xgb_model.predict(X_fold_val)  # 修改：变量名
                    mse = mean_squared_error(y_fold_val, y_pred)
                    r2 = r2_score(y_fold_val, y_pred)
                    mae = mean_absolute_error(y_fold_val, y_pred)
                    mape = mean_absolute_percentage_error(y_fold_val, y_pred)

                    fold_mse.append(mse)
                    fold_r2.append(r2)
                    fold_mae.append(mae)
                    fold_mape.append(mape)
                    fold_importance.append(xgb_model.feature_importances_)
                # Calculate average performance across all folds
                avg_mse = np.mean(fold_mse)
                avg_r2 = np.mean(fold_r2)
                avg_mae = np.mean(fold_mae)
                avg_mape = np.mean(fold_mape)
                avg_importance = np.mean(fold_importance, axis=0)

                # Store results for this feature subset
                self.results['n_features'].append(len(current_features))
                self.results['features'].append(current_features.copy())
                self.results['avg_mse'].append(avg_mse)
                self.results['avg_r2'].append(avg_r2)
                self.results['avg_mae'].append(avg_mae)
                self.results['avg_mape'].append(avg_mape)
                self.results['importance_scores'].append(dict(zip(current_features, avg_importance)))

                if verbose:
                    print(f"  Average MSE: {avg_mse:.6f}, R²: {avg_r2:.6f}, MAE: {avg_mae:.6f}, MAPE: {avg_mape:.6f}")

                # Update best model if current model is better
                if avg_mse < best_mse:
                    best_mse = avg_mse
                    best_features = current_features.copy()
                    if verbose:
                        print(f"  New best model found with {len(current_features)} features")

                # Break if only one feature remains
                if len(current_features) == 1:
                    break

                # Find the least important feature
                least_important_idx = np.argmin(avg_importance)
                least_important_feature = current_features[least_important_idx]

                # Update feature ranking (last eliminated = lowest rank)
                feature_ranking[least_important_feature] = n_features - len(current_features) + 1

                if verbose:
                    print(f"  Removing least important feature: '{least_important_feature}'")

                # Remove the least important feature
                current_features.pop(least_important_idx)
                X_current = X_current.drop(least_important_feature, axis=1)

                iteration += 1

            # Update feature ranking for remaining features
        for i, feature in enumerate(current_features):
            feature_ranking[feature] = n_features

        self.results['feature_ranking'] = feature_ranking

        # Find the optimal subset (with minimum avgMSE)
        optimal_idx = np.argmin(self.results['avg_mse'])
        self.optimal_features = self.results['features'][optimal_idx]
        optimal_n_features = self.results['n_features'][optimal_idx]
        min_mse = self.results['avg_mse'][optimal_idx]

        elapsed_time = time.time() - start_time

        if verbose:
            print("\n" + "=" * 60)
            print(f"Feature selection completed in {elapsed_time:.2f} seconds")
            print(f"Optimal feature subset contains {optimal_n_features} features with avgMSE = {min_mse:.6f}")
            print(f"Optimal features: {self.optimal_features}")

        return self

    def transform(self, X):
        """Transform the input data using the optimal feature subset"""
        if self.optimal_features is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return X[self.optimal_features]

    def fit_transform(self, X, y, verbose=True):
        """Fit the feature selector and transform the input data"""
        self.fit(X, y, verbose=verbose)
        return self.transform(X)

    def get_optimal_features(self):
        """Get the optimal feature subset"""
        if self.optimal_features is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self.optimal_features

    def plot_results(self):
        """Visualize the feature selection process results"""
        if not self.results['n_features']:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(20, 12))

        # Plot 1: MSE and R² vs number of features
        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(self.results['n_features'], self.results['avg_mse'], marker='o', color='blue', label='MSE')
        ax1.set_xlabel('Number of Features')
        ax1.set_ylabel('Average MSE', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True, linestyle='--', alpha=0.7)

        ax1_r2 = ax1.twinx()
        ax1_r2.plot(self.results['n_features'], self.results['avg_r2'], marker='s', color='red', label='R²')
        ax1_r2.set_ylabel('Average R²', color='red')
        ax1_r2.tick_params(axis='y', labelcolor='red')

        # Highlight optimal point
        optimal_idx = np.argmin(self.results['avg_mse'])
        optimal_n_features = self.results['n_features'][optimal_idx]
        optimal_mse = self.results['avg_mse'][optimal_idx]
        optimal_r2 = self.results['avg_r2'][optimal_idx]

        ax1.scatter(optimal_n_features, optimal_mse, color='green', s=100, zorder=5)
        ax1_r2.scatter(optimal_n_features, optimal_r2, color='purple', s=100, zorder=5)

        ax1.set_title('Model Performance vs Number of Features')

        # Plot 2: MAE and MAPE vs number of features
        ax2 = plt.subplot(2, 2, 2)
        ax2.plot(self.results['n_features'], self.results['avg_mae'], marker='o', color='orange', label='MAE')
        ax2.set_xlabel('Number of Features')
        ax2.set_ylabel('Average MAE', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        ax2.grid(True, linestyle='--', alpha=0.7)

        ax2_mape = ax2.twinx()
        ax2_mape.plot(self.results['n_features'], self.results['avg_mape'], marker='s', color='purple', label='MAPE')
        ax2_mape.set_ylabel('Average MAPE', color='purple')
        ax2_mape.tick_params(axis='y', labelcolor='purple')

        ax2.set_title('Error Metrics vs Number of Features')

        # Plot 3: Feature importance for optimal subset
        ax3 = plt.subplot(2, 1, 2)
        optimal_importances = self.results['importance_scores'][optimal_idx]
        importance_df = pd.DataFrame({
            'Feature': list(optimal_importances.keys()),
            'Importance': list(optimal_importances.values())
        }).sort_values('Importance', ascending=True)

        sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis', ax=ax3)
        ax3.set_title(f'Feature Importance - Optimal Subset ({optimal_n_features} features)')
        ax3.grid(True, linestyle='--', alpha=0.7, axis='x')

        plt.tight_layout()
        plt.show()

        return fig

    def evaluate_final_model(self, X_train, X_test, y_train, y_test, feature_names=None,
                             save_shap=True, output_dir="xgboost_rfe_results"):  # 修改：默认输出目录名
        """
        Evaluate the final XGBoost model with SHAP analysis  # 修改：注释
        """
        if self.optimal_features is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Transform data to optimal features
        X_train_optimal = X_train[self.optimal_features]
        X_test_optimal = X_test[self.optimal_features]

        # Create final model with best parameters
        if self.best_params:
            final_model = xgb.XGBRegressor(  # 修改：模型创建
                random_state=self.random_state,
                verbosity=0,
                n_jobs=-1,
                **self.best_params
            )
        else:
            final_model = xgb.XGBRegressor(  # 修改：模型创建
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                verbosity=0,
                n_jobs=-1
            )

        # Train final model
        final_model.fit(X_train_optimal, y_train)

        # Cross-validation on training set
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        cv_scores_rmse = -cross_val_score(final_model, X_train_optimal, y_train, cv=kf,
                                          scoring='neg_root_mean_squared_error')
        cv_scores_mae = -cross_val_score(final_model, X_train_optimal, y_train, cv=kf,
                                         scoring='neg_mean_absolute_error')
        cv_scores_r2 = cross_val_score(final_model, X_train_optimal, y_train, cv=kf, scoring='r2')

        print("\n" + "=" * 60)
        print("Final Model Cross-Validation Results")
        print("=" * 60)
        for i in range(len(cv_scores_rmse)):
            print(
                f'Fold {i + 1}: RMSE = {cv_scores_rmse[i]:.4f}, MAE = {cv_scores_mae[i]:.4f}, R² = {cv_scores_r2[i]:.4f}')

        print(f'\nMean RMSE: {np.mean(cv_scores_rmse):.4f} (±{np.std(cv_scores_rmse):.4f})')
        print(f'Mean MAE: {np.mean(cv_scores_mae):.4f} (±{np.std(cv_scores_mae):.4f})')
        print(f'Mean R²: {np.mean(cv_scores_r2):.4f} (±{np.std(cv_scores_r2):.4f})')

        # Training set evaluation
        y_train_pred = final_model.predict(X_train_optimal)
        train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        train_mape = mean_absolute_percentage_error(y_train, y_train_pred)

        print("\n训练集结果：")
        print(f"RMSE: {train_rmse:.6f}")
        print(f"MAE: {train_mae:.6f}")
        print(f"R²: {train_r2:.6f}")
        print(f"MAPE: {train_mape:.6f}")

        # Test set evaluation
        y_test_pred = final_model.predict(X_test_optimal)
        test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_mape = mean_absolute_percentage_error(y_test, y_test_pred)

        print("\n测试集结果：")
        print(f"RMSE: {test_rmse:.6f}")
        print(f"MAE: {test_mae:.6f}")
        print(f"R²: {test_r2:.6f}")
        print(f"MAPE: {test_mape:.6f}")

        # Feature importance
        importance = final_model.feature_importances_
        print(f"\nFeature Importance: {importance}")

        # Visualize feature importance
        plt.figure(figsize=(10, 6))
        feature_importance = dict(zip(self.optimal_features, importance))
        plt.barh(list(feature_importance.keys()), list(feature_importance.values()))
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature')
        plt.title('XGBoost Feature Importance - Optimal Features')  # 修改：标题
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["figure.dpi"] = 200
        plt.tight_layout()
        plt.show()

        # Save training predictions
        train_df = pd.DataFrame({'真实值': y_train, '拟合值': y_train_pred})
        train_file = os.path.join(output_dir, 'train_predictions.xlsx')
        train_df.to_excel(train_file, index=False)

        # Save test predictions
        test_df = pd.DataFrame({'真实值': y_test, '拟合值': y_test_pred})
        test_file = os.path.join(output_dir, 'test_predictions.xlsx')
        test_df.to_excel(test_file, index=False)

        # Plot actual vs predicted
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Training set
        ax1.scatter(y_train, y_train_pred, alpha=0.5)
        ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
        ax1.set_xlabel('Actual Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title(f'Training Set: Actual vs Predicted\nR² = {train_r2:.4f}')
        ax1.grid(True)

        # Test set
        ax2.scatter(y_test, y_test_pred, alpha=0.5)
        ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax2.set_xlabel('Actual Values')
        ax2.set_ylabel('Predicted Values')
        ax2.set_title(f'Test Set: Actual vs Predicted\nR² = {test_r2:.4f}')
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

        # SHAP Analysis
        if save_shap:
            print("\nCalculating SHAP values...")
            try:
                # Calculate SHAP values
                explainer = shap.Explainer(final_model)
                shap_values = explainer.shap_values(X_train_optimal.values)

                # Save SHAP values
                shap_df = pd.DataFrame(shap_values, columns=self.optimal_features)
                shap_file = os.path.join(output_dir, 'shap_values.csv')
                shap_df.to_csv(shap_file, index=False)

                # SHAP visualizations
                plt.rcParams["font.family"] = "Times New Roman"
                plt.rcParams["figure.dpi"] = 200

                # Summary plot (bar)
                shap.summary_plot(shap_values, X_train_optimal.values,
                                  feature_names=self.optimal_features, plot_type="bar", show=False)
                plt.title("SHAP Feature Importance")
                plt.tight_layout()
                plt.show()

                # Summary plot (beeswarm)
                shap.summary_plot(shap_values, X_train_optimal.values,
                                  feature_names=self.optimal_features, show=False)
                plt.title("SHAP Summary Plot")
                plt.tight_layout()
                plt.show()

                # Decision plot for a subset of samples
                if len(X_train_optimal) > 20:
                    sample_indices = np.random.choice(len(X_train_optimal), 20, replace=False)
                    shap.decision_plot(explainer.expected_value,
                                       shap_values[sample_indices],
                                       feature_names=self.optimal_features, show=False)
                    plt.title("SHAP Decision Plot (Sample)")
                    plt.tight_layout()
                    plt.show()

                print(f"SHAP values saved to: {shap_file}")
            except Exception as e:
                print(f"Error calculating SHAP values: {e}")

        # Save comprehensive results
        eval_data = {
            'Metric': ['Train_RMSE', 'Train_MAE', 'Train_R2', 'Train_MAPE',
                       'Test_RMSE', 'Test_MAE', 'Test_R2', 'Test_MAPE',
                       'CV_Mean_RMSE', 'CV_Std_RMSE', 'CV_Mean_MAE', 'CV_Std_MAE',
                       'CV_Mean_R2', 'CV_Std_R2'],
            'Value': [train_rmse, train_mae, train_r2, train_mape,
                      test_rmse, test_mae, test_r2, test_mape,
                      np.mean(cv_scores_rmse), np.std(cv_scores_rmse),
                      np.mean(cv_scores_mae), np.std(cv_scores_mae),
                      np.mean(cv_scores_r2), np.std(cv_scores_r2)]
        }
        eval_df = pd.DataFrame(eval_data)
        eval_file = os.path.join(output_dir, "comprehensive_evaluation.csv")
        eval_df.to_csv(eval_file, index=False)

        # Save final feature importances
        final_imp_df = pd.DataFrame({
            'Feature': self.optimal_features,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        final_imp_file = os.path.join(output_dir, "final_feature_importances.csv")
        final_imp_df.to_csv(final_imp_file, index=False)

        print(f"\nComprehensive evaluation saved to: {eval_file}")
        print(f"Final feature importances saved to: {final_imp_file}")
        print(f"Training predictions saved to: {train_file}")
        print(f"Test predictions saved to: {test_file}")

        return {
            'final_model': final_model,
            'train_metrics': {'rmse': train_rmse, 'mae': train_mae, 'r2': train_r2, 'mape': train_mape},
            'test_metrics': {'rmse': test_rmse, 'mae': test_mae, 'r2': test_r2, 'mape': test_mape},
            'cv_metrics': {'rmse': cv_scores_rmse, 'mae': cv_scores_mae, 'r2': cv_scores_r2}
        }

    def save_results_to_csv(self, output_dir="catboost_rfe_results"):
        """Save the feature selection results to CSV files"""
        if not self.results['n_features']:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save feature ranking
        ranking_df = pd.DataFrame({
            'Feature': list(self.results['feature_ranking'].keys()),
            'Ranking': list(self.results['feature_ranking'].values())
        }).sort_values('Ranking', ascending=False)

        ranking_file = os.path.join(output_dir, "feature_ranking.csv")
        ranking_df.to_csv(ranking_file, index=False)

        # Save performance metrics for each iteration
        iterations_df = pd.DataFrame({
            'n_features': self.results['n_features'],
            'avg_mse': self.results['avg_mse'],
            'avg_r2': self.results['avg_r2'],
            'avg_mae': self.results['avg_mae'],
            'avg_mape': self.results['avg_mape']
        })

        # Add feature lists
        feature_lists = []
        for features in self.results['features']:
            feature_lists.append(','.join(features))
        iterations_df['features'] = feature_lists

        iterations_file = os.path.join(output_dir, "iteration_results.csv")
        iterations_df.to_csv(iterations_file, index=False)

        # Save optimal features
        optimal_idx = np.argmin(self.results['avg_mse'])
        optimal_df = pd.DataFrame({
            'Feature': self.optimal_features,
            'IsOptimal': [True] * len(self.optimal_features)
        })

        optimal_file = os.path.join(output_dir, "optimal_features.csv")
        optimal_df.to_csv(optimal_file, index=False)

        # Get optimal feature importance scores
        optimal_importances = self.results['importance_scores'][optimal_idx]
        importance_df = pd.DataFrame({
            'Feature': list(optimal_importances.keys()),
            'Importance': list(optimal_importances.values())
        }).sort_values('Importance', ascending=False)

        importance_file = os.path.join(output_dir, "feature_importances.csv")
        importance_df.to_csv(importance_file, index=False)

        print(f"\nResults saved to CSV files in '{output_dir}' directory:")
        print(f"  - Feature ranking: {ranking_file}")
        print(f"  - Iteration results: {iterations_file}")
        print(f"  - Optimal features: {optimal_file}")
        print(f"  - Feature importances: {importance_file}")

        return {
            'ranking': ranking_file,
            'iterations': iterations_file,
            'optimal': optimal_file,
            'importance': importance_file
        }


def load_and_preprocess_data(file_path, target_col='Utot', test_size=test_size, random_state=random_state):
    """
    Load and preprocess the dataset

    Parameters:
    -----------
    file_path : str
        Path to the CSV data file
    target_col : str
        Name of the target column
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    X_train, X_test, y_train, y_test : pandas DataFrames/Series
        Split data for training and testing
    """
    print(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)

    print(f"Dataset shape: {df.shape}")
    print(f"Feature names: {df.columns.tolist()}")

    # Check for missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"Warning: Dataset contains {missing_count} missing values")
        print(df.isnull().sum()[df.isnull().sum() > 0])

        # Fill missing values (you may want to customize this)
        df = df.fillna(df.mean())
        print("Missing values filled with column means")

    # Split features and target
    y = df[target_col]
    X = df.drop(target_col, axis=1)

    # Normalize features
    print("Normalizing features...")
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Split data
    print(f"Splitting data: {100 * (1 - test_size):.0f}% training, {100 * test_size:.0f}% testing")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )

    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test


def main():
    """Main execution function with XGBoost feature selection"""  # 修改：注释
    # File path - update with your actual file path
    file_path = '特征选择fc2.csv'
    xgboost_output_dir = "xgboost_rfe_results2"  # 修改：输出目录名

    try:
        # Step 1: Load and preprocess data
        X_train, X_test, y_train, y_test = load_and_preprocess_data(
            file_path,
            target_col='Utot',
            test_size=test_size,
            random_state=random_state
        )

        print("\n" + "=" * 80)
        print("XGBOOST RFE 特征选择")  # 修改：打印标题
        print("=" * 80)

        # XGBoost RFE
        xgboost_selector = XGBoost_RFE_FeatureSelector(
            n_estimators=n_estimators,
            n_folds=n_folds,
            random_state=random_state,
            optimize_params=True,  # Enable hyperparameter optimization
            n_iter=10  # Number of iterations for hyperparameter search
        )

        xgboost_selector.fit(X_train, y_train)  # 修改：变量名
        xgboost_optimal_features = xgboost_selector.get_optimal_features()  # 修改：变量名

        # Plot XGBoost results
        xgboost_selector.plot_results()  # 修改：变量名

        # Save XGBoost results
        xgboost_selector.save_results_to_csv(xgboost_output_dir)  # 修改：变量名

        # XGBoost comprehensive evaluation with SHAP
        xgboost_results = xgboost_selector.evaluate_final_model(  # 修改：变量名
            X_train, X_test, y_train, y_test,
            feature_names=list(X_train.columns),
            save_shap=True,
            output_dir=xgboost_output_dir
        )

        print("\n" + "=" * 80)
        print("XGBOOST FEATURE SELECTION SUMMARY")  # 修改：打印标题
        print("=" * 80)
        print(f"Optimal features ({len(xgboost_optimal_features)}): {xgboost_optimal_features}")  # 修改：变量名
        print(f"Test R²: {xgboost_results['test_metrics']['r2']:.6f}")  # 修改：变量名
        print(f"Test MSE: {xgboost_results['test_metrics']['rmse'] ** 2:.6f}")  # 修改：变量名
        print(f"Test RMSE: {xgboost_results['test_metrics']['rmse']:.6f}")  # 修改：变量名
        print(f"Test MAE: {xgboost_results['test_metrics']['mae']:.6f}")  # 修改：变量名
        print(f"Test MAPE: {xgboost_results['test_metrics']['mape']:.6f}")  # 修改：变量名

    except FileNotFoundError:
        print(f"Error: Could not find the file '{file_path}'")
        print("Please update the file path in the code to point to your dataset.")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()