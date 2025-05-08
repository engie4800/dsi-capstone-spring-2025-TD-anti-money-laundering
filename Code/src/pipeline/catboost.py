from pipeline import BaseModelPipeline
from sklearn.metrics import f1_score, average_precision_score
import xgboost as xgb
import numpy as np
import pandas as pd



class CatBoostPipeline(BaseModelPipeline):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        
        self.keep_acct_idx = True
    
    def should_keep_acct_idx(self):
        return True
    
    def split_train_test_val(self, X_cols=None, y_col="is_laundering", test_size=0.15, val_size=0.15, split_type="temporal"):
        return super().split_train_test_val(X_cols, y_col, test_size, val_size, split_type)
       
    def add_node_graph_feats_to_df(self, node_feat_cols=None):
        """
        Used for non-GNN model (e.g., CatBoost).
        Merge node-level graph features (e.g., pagerank, degree_centrality)
        into the transaction DataFrames (X_train, X_val, X_test) for both sender and receiver.

        Args:
            node_feat_cols (list or None): List of node feature columns to merge (excluding 'node_id').
                                        If None, will use all columns in node DataFrames except
                                        a "node_id"
        """
        if 'from_account_idx' not in self.X_train.columns:
            raise RuntimeError("To add node feats to tabular df, need from_account_idx and to_account_idx")
        
        if node_feat_cols is None:
            node_feat_cols = [col for col in self.train_nodes.columns if col != 'node_id']

        def merge_feats(txns_df, nodes_df):
            if "node_id" not in nodes_df.columns:
                raise ValueError("Each nodes_df must include a 'node_id' column")

            # Sender node features
            sender_feats = nodes_df[["node_id"] + node_feat_cols].copy()
            sender_feats = sender_feats.rename(columns={col: f"from_{col}" for col in node_feat_cols})
            sender_feats = sender_feats.rename(columns={"node_id": "from_account_idx"})

            # Receiver node features
            receiver_feats = nodes_df[["node_id"] + node_feat_cols].copy()
            receiver_feats = receiver_feats.rename(columns={col: f"to_{col}" for col in node_feat_cols})
            receiver_feats = receiver_feats.rename(columns={"node_id": "to_account_idx"})

            # Merge into transaction dataframe
            txns_df = txns_df.merge(sender_feats, on="from_account_idx", how="left")
            txns_df = txns_df.merge(receiver_feats, on="to_account_idx", how="left")
            
            txns_df = txns_df.drop(columns=['from_account_idx','to_account_idx'])

            return txns_df

        self.X_train = merge_feats(self.X_train, self.train_nodes)
        self.X_val = merge_feats(self.X_val, self.val_nodes)
        self.X_test = merge_feats(self.X_test, self.test_nodes)
            
    def scale_edge_features(self, edge_features_to_scale:list[str]):
        return super().numerical_scaling(edge_features_to_scale)
       
    def run_model_specific_steps(self, edge_features_to_scale:list[str]):
        self.scale_edge_features(edge_features_to_scale)
        self.add_node_graph_features_to_df()

    # TODO: I add this new one - need verify
    def shap_feature_selection(self, xgb_model, top_n: int = 3):
        """
        Run SHAP on the current train/val/test split, print the top-3 in each group,
        add the “other” indicators to X_train/val/test, and return:
          - shap_importance DataFrame
          - three Python lists: top3_sent, top3_received, top3_payment
        """
        # 1) Get SHAP value df
        booster = xgb_model.get_booster()
        dtest = xgb.DMatrix(self.X_test)
        shap_vals = booster.predict(dtest, pred_contribs=True)
        feature_shap = pd.DataFrame(shap_vals[:, :-1],
                                    columns=self.X_test.columns,
                                    index=self.X_test.index)

        # 2) Mean absolute SHAP per feature
        mean_abs = feature_shap.abs().mean().sort_values(ascending=False)
        shap_importance = (
            pd.DataFrame({
                'feature': self.X_test.columns,
                'mean_abs_shap': mean_abs
            })
            .sort_values('mean_abs_shap', ascending=False)
            .reset_index(drop=True)
        )

        # 3) Build full matrix X and define your groups
        X = pd.concat([self.X_train, self.X_val, self.X_test],
                      axis=0, ignore_index=True)
        currency_features_sent = [c for c in X.columns if c.startswith("sent_currency_")]
        currency_features_received = [c for c in X.columns if c.startswith("received_currency_")]
        payment_type_features = [c for c in X.columns if c.startswith("payment_type_")]

        # 4) Pull out top-n in each
        top3_sent = shap_importance[shap_importance['feature'].isin(currency_features_sent)].nlargest(top_n,
                                                                                                      'mean_abs_shap')
        top3_received = shap_importance[shap_importance['feature'].isin(currency_features_received)].nlargest(top_n,
                                                                                                              'mean_abs_shap')
        top3_payment = shap_importance[shap_importance['feature'].isin(payment_type_features)].nlargest(top_n,
                                                                                                        'mean_abs_shap')
        # 5) Print them
        print("▶ Top 3 ‘currencies sent’ by |SHAP|:")
        print(top3_sent.to_string(index=False))
        print("\n▶ Top 3 ‘currencies received’ by |SHAP|:")
        print(top3_received.to_string(index=False))
        print("\n▶ Top 3 ‘payment types’ by |SHAP|:")
        print(top3_payment.to_string(index=False))

        # 6) Add “other” indicators to each split
        #    (sum of all in group minus sum of top3 > 0)
        sent_cols, recv_cols, pay_cols = currency_features_sent, currency_features_received, payment_type_features
        t_sent, t_recv, t_pay = top3_sent['feature'], top3_received['feature'], top3_payment['feature']

        for df in (self.X_train, self.X_val, self.X_test, X):
            df['sent_currency_other'] = ((df[sent_cols].sum(axis=1) - df[t_sent].sum(axis=1)) > 0).astype(int)
            df['received_currency_other'] = ((df[recv_cols].sum(axis=1) - df[t_recv].sum(axis=1)) > 0).astype(int)
            df['payment_type_other'] = ((df[pay_cols].sum(axis=1) - df[t_pay].sum(axis=1)) > 0).astype(int)

        # 7) Convert the top3 DataFrames into plain Python lists
        top3_sent_list = top3_sent['feature'].tolist()
        top3_received_list = top3_received['feature'].tolist()
        top3_payment_list = top3_payment['feature'].tolist()

        return shap_importance, top3_sent_list, top3_received_list, top3_payment_list

    def evaluate_feature_sets_xgb(self, feature_sets: dict):
        """
        Given a dict name→feature_list, trains an XGBClassifier
        on each and prints F1 & PR-AUC. Returns a results dict.
        """
        results = {}
        for name, feats in feature_sets.items():
            model = xgb.XGBClassifier(
                n_estimators=500,
                max_depth=10,
                learning_rate=0.1,
                objective='binary:logistic',
                scale_pos_weight=12,
                eval_metric='aucpr',
                random_state=42,
            )
            # fit on train
            model.fit(self.X_train[feats], self.y_train)

            # predict on val
            Xv = self.X_val[feats].astype(np.float32).values
            y_pred = model.predict(Xv)
            y_prob = model.predict_proba(Xv)[:, 1]

            # metrics
            f1 = f1_score(self.y_val, y_pred)
            pr_auc = average_precision_score(self.y_val, y_prob)

            results[name] = {"F1": f1, "PR AUC": pr_auc}
            print(f"{name:12s} →  F1 = {f1:.4f},  PR-AUC = {pr_auc:.4f}")

        return results

    def build_grouped_feature_sets(self,
                                   top3_sent: list[str],
                                   top3_received: list[str],
                                   top3_payment: list[str]):
        """
        1) Identifies your one-hot groups from the full feature set
        2) Adds the “other” indicator columns to self.X_train/val/test
        3) Returns the feature_sets dict exactly as in your snippet
        """
        # -- rebuild full X to scan for all one-hot cols --
        X = pd.concat([self.X_train, self.X_val, self.X_test],
                      axis=0, ignore_index=True)

        # 2) Define your groups of interest
        currency_features_sent = [c for c in X.columns if c.startswith("sent_currency_")]
        currency_features_received = [c for c in X.columns if c.startswith("received_currency_")]
        payment_type_features = [c for c in X.columns if c.startswith("payment_type_")]

        # 3) Add “other” flags to each split
        for df in (self.X_train, self.X_val, self.X_test):
            df['sent_currency_other'] = ((df[currency_features_sent].sum(axis=1)
                                          - df[top3_sent].sum(axis=1)) > 0).astype(int)
            df['received_currency_other'] = ((df[currency_features_received].sum(axis=1)
                                              - df[top3_received].sum(axis=1)) > 0).astype(int)
            df['payment_type_other'] = ((df[payment_type_features].sum(axis=1)
                                         - df[top3_payment].sum(axis=1)) > 0).astype(int)

        # 4) Build base_features by removing all one-hot cols
        all_cols = self.X_train.columns.tolist()
        grouped_onehot = set(currency_features_sent
                             + currency_features_received
                             + payment_type_features)
        base_features = [c for c in all_cols if c not in grouped_onehot]

        # 5) Assemble the dictionary
        feature_sets = {
            "all_onehots": base_features
                           + currency_features_sent
                           + currency_features_received
                           + payment_type_features,
            "no_sent": base_features
                       + currency_features_received
                       + payment_type_features,
            "no_received": base_features
                           + currency_features_sent
                           + payment_type_features,
            "no_currency": base_features
                           + payment_type_features,
            "top3_all": base_features
                        + top3_sent
                        + top3_received
                        + top3_payment,
            "top3_payment_only": base_features
                                 + currency_features_sent
                                 + currency_features_received
                                 + top3_payment,
        }

        return feature_sets

    def backward_selection_xgb(self,
                               features: list[str],
                               min_features: int = 10) -> list[str]:
        """
        Iteratively drop one feature at a time from features, retrain with early stopping,
        and keep the drop if PR‐AUC does not decrease, until len(features)==min_features.
        Returns the pruned feature list.
        """
        current = features.copy()
        best_score = 0.0
        best_feats = current.copy()

        while len(current) > min_features:
            scores = {}
            for f in current:
                cand = [c for c in current if c != f]
                model = xgb.XGBClassifier(
                    n_estimators=500,
                    max_depth=4,
                    learning_rate=0.03,
                    objective='binary:logistic',
                    scale_pos_weight=12,
                    subsample=0.6,
                    colsample_bytree=0.6,
                    reg_alpha=5.0,
                    reg_lambda=5.0,
                    random_state=42,
                    eval_metric='aucpr',
                    early_stopping_rounds=50
                )
                model.fit(
                    self.X_train[cand], self.y_train,
                    eval_set=[(self.X_val[cand], self.y_val)],
                    verbose=False
                )
                y_prob = model.predict_proba(self.X_val[cand])[:, 1]
                scores[f] = average_precision_score(self.y_val, y_prob)

            drop, score = max(scores.items(), key=lambda kv: kv[1])
            print(f"Trying drop {drop}: PR‐AUC = {score:.4f}")
            if score >= best_score:
                best_score = score
                current.remove(drop)
                best_feats = current.copy()
                print(f"✅ Removed {drop}; new best PR‐AUC = {best_score:.4f}")
            else:
                print(f"🛑 Removal of {drop} hurt performance. Stopping.")
                break

        return best_feats

    def feature_selection_pipeline(self,
                                   xgb_model,
                                   top_n: int = 3,
                                   min_features: int = 10):
        """
        1) SHAP → top_n per group → add “other” flags
        2) build grouped feature‐sets & evaluate F1/PR‐AUC
        3) pick best set → backward‐select down to min_features
        4) train final XGB on that reduced set → return everything
        """

        # Stage 1: SHAP selection
        shap_imp, top3_sent, top3_recv, top3_pay = self.shap_feature_selection(xgb_model, top_n)

        # Stage 1b: build your grouped one‐hot sets
        fs = self.build_grouped_feature_sets(top3_sent, top3_recv, top3_pay)

        # Stage 1c: evaluate all of them
        evals = self.evaluate_feature_sets_xgb(fs)

        # Stage 2: pick the best initial set by PR‐AUC
        best_name = max(evals, key=lambda k: evals[k]['PR AUC'])
        initial_feats = fs[best_name]

        # Stage 2b: backward‐selection
        selected = self.backward_selection_xgb(
            initial_feats,
            min_features=min_features
        )

        # Stage 3: fit the final model
        final_model = xgb.XGBClassifier(
            n_estimators=500,  # allow more trees, we'll stop early
            max_depth=4,  # shallower trees → less variance
            learning_rate=0.03,  # smaller step size
            objective='binary:logistic',
            scale_pos_weight=12,  # keep your class‐imbalance weight
            subsample=0.6,  # each tree sees 80% of the rows
            colsample_bytree=0.6,  # each tree sees 80% of the columns
            reg_alpha=5.0,  # L1 regularization
            reg_lambda=5.0,  # L2 regularization
            random_state=42,
            eval_metric='aucpr',
            early_stopping_rounds=50  # monitor PR-AUC
        )
        final_model.fit(self.X_train[selected], self.y_train,
                        eval_set=[
                            (self.X_train[selected], self.y_train),  # monitor train + val
                            (self.X_val[selected], self.y_val)
                        ])

        # Stage 4: test‐set PR‐AUC
        test_probs = final_model.predict_proba(self.X_test[selected])[:, 1]
        test_pr = average_precision_score(self.y_test, test_probs)

        return {
            'shap_importance': shap_imp,
            'feature_sets': fs,
            'evaluation': evals,
            'best_initial_set': best_name,
            'initial_features': initial_feats,
            'selected_features': selected,
            'final_model': final_model,
            'test_pr_auc': test_pr
        }



