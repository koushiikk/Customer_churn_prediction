def train_model(df):
    import pandas as pd
    from preprocessing import create_preprocessor
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
    
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import recall_score, f1_score, roc_auc_score, classification_report,precision_recall_curve
    from scipy.stats import uniform, randint
   

    X = df.drop(['customerID','Churn'],axis =1)
    y = df['Churn']
    df['tenure_group'] = pd.cut(df['tenure'], bins=[0,12,24,48,72],
                                 labels=['0-1yr','1-2yr','2-4yr','4-6yr'])
    
    df['monthly_to_total_ratio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)
    
    df['num_services'] = (
        (df['PhoneService'] == 'Yes').astype(int) +
        (df['MultipleLines'] == 'Yes').astype(int) +
        (df['InternetService'] != 'No').astype(int) +
        (df['OnlineSecurity'] == 'Yes').astype(int) +
        (df['OnlineBackup'] == 'Yes').astype(int) +
        (df['DeviceProtection'] == 'Yes').astype(int) +
        (df['TechSupport'] == 'Yes').astype(int) +
        (df['StreamingTV'] == 'Yes').astype(int) +
        (df['StreamingMovies'] == 'Yes').astype(int)
    )

    ## splitting train and test data
    X_train,X_test,y_train,y_test = train_test_split(
        X,
        y,
        test_size =0.2,
        random_state = 42,
        stratify=y)
    
    ## preprocessing training data
    preprocessor = create_preprocessor(X_train)
    
  
    

    ## defining models
    models  = {
        'log_reg':LogisticRegression(max_iter = 1000,class_weight = 'balanced'),
        'dec_tree':DecisionTreeClassifier(class_weight = 'balanced'),
        'random_for':RandomForestClassifier(class_weight = 'balanced'),
        'grad_boost':GradientBoostingClassifier(),
       
        
    }
    
    ## defining hyperparameter grids
    param_grids = {
    "log_reg": {
        "model__C": [0.01, 0.1, 1, 10]
    },
    "dec_tree": {
        "model__max_depth": [3, 5, 10],
        "model__min_samples_split": [2, 5, 10]
    },
    "random_for": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [5, 10, None]
    },
    "grad_boost": {
        "model__learning_rate": [0.01, 0.1],
        "model__n_estimators": [100, 200]
    },
    
}
  
    ##looping through models
    results = []

    for name, model in models.items():
        print(f"Training {name}...")

        pipeline = Pipeline([
            ('preprocessor',preprocessor),
            ('model',model)
        ])

        grid = GridSearchCV(
        estimator = pipeline,   
        param_grid = param_grids[name],
        cv=10,
        scoring='recall',   
        n_jobs=-1
    )

        grid.fit(X_train, y_train)
    
      # Add this for grad_boost after grid.fit()
        y_prob = grid.predict_proba(X_test)[:, 1]
        prec, rec, thresholds = precision_recall_curve(y_test, y_prob)
        f1s = 2 * prec * rec / (prec + rec + 1e-8)
        best_thresh = thresholds[f1s.argmax()]
        y_pred = (y_prob >= best_thresh).astype(int)

        ##evaluation
        recall = recall_score(y_test,y_pred)
        f1  = f1_score(y_test,y_pred)
        roc_auc = roc_auc_score(y_test,y_prob)
        print(classification_report(y_test,y_pred))
        results.append(
            {
                'model':name,
                'best_params':grid.best_params_,
                'recall':recall,
                'f1_score':f1,
                'roc_auc_score':roc_auc,
                'best_model':grid.best_estimator_
            }
        )
    print(f"Training XGB...")
    xgb_pipeline = Pipeline([
        ('preprocessor',preprocessor),
        ('model',XGBClassifier(
            eval_metric = 'logloss',
            random_state = 42,
            tree_method='hist'       
        ))
    ])
    xgb_params = {
        "model__learning_rate":    uniform(0.01,0.15),
        "model__n_estimators":     [500],
        "model__max_depth":        randint(3,6),
        "model__subsample":        uniform(0.7, 0.3),
        "model__colsample_bytree": uniform(0.7, 0.3),
        "model__reg_alpha":        uniform(0, 0.5),   
        "model__reg_lambda":       uniform(1, 1.5),   
        "model__min_child_weight": randint(1,6)
    }
    xgb_grid = RandomizedSearchCV(
        estimator = xgb_pipeline,
        param_distributions = xgb_params,
        n_iter = 50,
        cv = 5,
        scoring = 'recall',
        n_jobs = -1,
        random_state = 42
    )
    xgb_grid.fit(X_train,y_train)
    # Add this for grad_boost after grid.fit()
    y_prob = xgb_grid.predict_proba(X_test)[:, 1]
    prec, rec, thresholds = precision_recall_curve(y_test, y_prob)
    f1s = 2 * prec * rec / (prec + rec + 1e-8)
    best_thresh = thresholds[f1s.argmax()]
    y_pred = (y_prob >= best_thresh).astype(int)

     ##evaluation
    recall = recall_score(y_test,y_pred)
    f1  = f1_score(y_test,y_pred)
    roc_auc = roc_auc_score(y_test,y_prob)
    print(classification_report(y_test,y_pred))
    results.append(
            {
                'model':'XGB',
                'best_params':xgb_grid.best_params_,
                'recall':recall,
                'f1_score':f1,
                'roc_auc_score':roc_auc,
                'best_model':xgb_grid.best_estimator_
            }
        )
    
    
    return results
