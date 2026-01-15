import os
import sys
import argparse
import configparser
from dataclasses import dataclass
import logging
import time
import random

import numpy as np
import pandas as pd
import allel

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.nn import MSELoss

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

import optuna
from lightning.pytorch import LightningModule

import functools

# -------------------------------
# Logger setup
# -------------------------------
def setup_logger(log_path):
    logger = logging.getLogger("cropformer")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger



def set_all_seeds(seed: int):    
    random.seed(seed)    
    np.random.seed(seed)    
    torch.manual_seed(seed)    
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@dataclass
class PipelineConfig:
    # general
    result_dir: str
    device: str
    random_state: int

    # data
    train_geno: str
    train_pheno: str
    test_geno: str | None
    test_pheno: str | None

    # model
    input_size: int
    hidden_size: int
    output_dim: int
    kernel_size: int
    hidden_dropout_prob: float
    attention_probs_dropout_prob: float
    num_attention_heads: int
    learning_rate: float

    # training
    outer_folds: int
    epochs: int
    batch_size: int
    patience: int
    optuna_trials: int


def _require_file(path: str, label: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} not found: {path}")


def _validate_genotype(path: str, label: str):
    _require_file(path, label)
    if not (path.endswith(".csv") or path.endswith(".vcf")):
        raise ValueError(f"{label} must be .csv or .vcf (got {path})")


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device not in {"cpu", "cuda"}:
        raise ValueError(f"Invalid device: {device}")
    return device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cropformer end-to-end pipeline"
    )
    parser.add_argument(
        "-c", "--config",
        required=True,
        help="Path to config.ini"
    )
    return parser.parse_args()


def load_and_validate_config(config_path: str) -> PipelineConfig:

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    parser = configparser.ConfigParser()
    parser.read(config_path)

    # ----------------------------
    # Section presence check
    # ----------------------------
    required_sections = {"general", "data", "model", "training"}
    missing = required_sections - set(parser.sections())
    if missing:
        raise ValueError(f"Missing config sections: {missing}")

    # ----------------------------
    # [general]
    # ----------------------------
    general = parser["general"]

    result_dir = general.get("result_dir")
    device = general.get("device", "auto")
    random_state = general.getint("random_state")

    if not result_dir:
        raise ValueError("[general] result_dir is required")

    device = _resolve_device(device)

    os.makedirs(result_dir, exist_ok=True)

    # ----------------------------
    # [data]
    # ----------------------------
    data = parser["data"]

    train_geno = data.get("train_geno")
    train_pheno = data.get("train_pheno")
    test_geno = data.get("test_geno", fallback=None)
    test_pheno = data.get("test_pheno", fallback=None)

    if not train_geno:
        raise ValueError("[data] train_geno is required")
    if not train_pheno:
        raise ValueError("[data] train_pheno is required")

    _validate_genotype(train_geno, "train_geno")
    _require_file(train_pheno, "train_pheno")

    if test_geno:
        _validate_genotype(test_geno, "test_geno")
    if test_pheno:
        _require_file(test_pheno, "test_pheno")

    # ----------------------------
    # [model]
    # ----------------------------
    model = parser["model"]

    input_size = model.getint("input_size")
    hidden_size = model.getint("hidden_size")
    output_dim = model.getint("output_dim")
    kernel_size = model.getint("kernel_size")
    hidden_dropout_prob = model.getfloat("hidden_dropout_prob")
    attention_probs_dropout_prob = model.getfloat("attention_probs_dropout_prob")
    num_attention_heads = model.getint("num_attention_heads")
    learning_rate = model.getfloat("learning_rate")

    if input_size <= 0:
        raise ValueError("input_size must be > 0")

    if hidden_size % num_attention_heads != 0:
        raise ValueError(
            "hidden_size must be divisible by num_attention_heads"
        )

    # ----------------------------
    # [training]
    # ----------------------------
    training = parser["training"]

    outer_folds = training.getint("outer_folds")
    epochs = training.getint("epochs")
    batch_size = training.getint("batch_size")
    patience = training.getint("patience")
    optuna_trials = training.getint("optuna_trials")

    if outer_folds < 2:
        raise ValueError("outer_folds must be >= 2")

    # ----------------------------
    # Return typed config
    # ----------------------------
    return PipelineConfig(
        result_dir=result_dir,
        device=device,
        random_state=random_state,

        train_geno=train_geno,
        train_pheno=train_pheno,
        test_geno=test_geno,
        test_pheno=test_pheno,

        input_size=input_size,
        hidden_size=hidden_size,
        output_dim=output_dim,
        kernel_size=kernel_size,
        hidden_dropout_prob=hidden_dropout_prob,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        num_attention_heads=num_attention_heads,
        learning_rate=learning_rate,

        outer_folds=outer_folds,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        optuna_trials=optuna_trials,
    )


def random_snp_select(X: np.ndarray, k: int, random_state: int):
    rng = np.random.default_rng(random_state)
    n_features = X.shape[1]

    if k >= n_features:
        idx = np.arange(n_features)
    else:
        idx = rng.choice(n_features, size=k, replace=False)

    return X[:, idx], idx

def vcf_to_numeric_matrix(vcf_path: str):
    callset = allel.read_vcf(
        vcf_path,
        fields=["samples", "calldata/GT"]
    )

    gt = allel.GenotypeArray(callset["calldata/GT"])
    X = gt.to_n_alt().astype(np.float32)
    X = np.nan_to_num(X, nan=-1)

    # (variants, samples) → (samples, variants)
    return X.T, callset["samples"].astype(str)

def load_genotype(path: str,
                  input_size: int,
                  random_state: int,
                  snp_idx: np.ndarray | None = None):
    """
    Returns:
        X          : (n_samples, input_size)
        sample_ids: np.ndarray[str] or None
        snp_idx   : indices used (train only)
    """

    if path.endswith(".csv"):
        X = pd.read_csv(path).values
        sample_ids = None

    elif path.endswith(".vcf"):
        X, sample_ids = vcf_to_numeric_matrix(path)

        if snp_idx is None:
            X, snp_idx = random_snp_select(
                X, input_size, random_state
            )
        else:
            X = X[:, snp_idx]

    else:
        raise ValueError(f"Unsupported genotype format: {path}")

    # Pad / trim
    if X.shape[1] < input_size:
        pad = input_size - X.shape[1]
        X = np.hstack([X, np.zeros((X.shape[0], pad))])
    elif X.shape[1] > input_size:
        X = X[:, :input_size]

    return X.astype(np.float32), sample_ids, snp_idx

def load_phenotype(pheno_path: str):
    """
    First column  : sample IDs (name irrelevant)
    Other columns : traits
    """
    df = pd.read_csv(pheno_path)

    if df.shape[1] < 2:
        raise ValueError(
            f"Phenotype file must contain at least one trait: {pheno_path}"
        )

    sample_ids = df.iloc[:, 0].astype(str).values
    traits_df = df.iloc[:, 1:]

    return sample_ids, traits_df

def align_geno_pheno(X, geno_ids, pheno_ids, pheno_df):
    common = np.intersect1d(geno_ids, pheno_ids)

    if len(common) == 0:
        raise ValueError("No common samples between genotype and phenotype")

    geno_mask = np.isin(geno_ids, common)
    pheno_mask = np.isin(pheno_ids, common)

    X_aligned = X[geno_mask]
    pheno_aligned = pheno_df.iloc[pheno_mask].reset_index(drop=True)

    return X_aligned, pheno_aligned

def load_all_data(cfg):
    """
    Returns:
        X_train      : np.ndarray
        train_traits : DataFrame (traits only)
        X_test       : np.ndarray or None
        test_traits  : DataFrame or None
    """

    # -------- Train phenotype --------
    train_pheno_ids, train_traits = load_phenotype(cfg.train_pheno)

    # -------- Train genotype --------
    X_train, train_geno_ids, snp_idx = load_genotype(
        cfg.train_geno,
        cfg.input_size,
        cfg.random_state
    )

    if train_geno_ids is not None:
        X_train, train_traits = align_geno_pheno(
            X_train, train_geno_ids, train_pheno_ids, train_traits
        )

    # -------- Optional test data --------
    X_test = test_traits = None

    if cfg.test_geno:
        X_test, test_geno_ids, _ = load_genotype(
            cfg.test_geno,
            cfg.input_size,
            cfg.random_state,
            snp_idx=snp_idx
        )

        if cfg.test_pheno:
            test_pheno_ids, test_traits = load_phenotype(cfg.test_pheno)

            if test_geno_ids is not None:
                X_test, test_traits = align_geno_pheno(
                    X_test, test_geno_ids, test_pheno_ids, test_traits
                )

    return X_train, train_traits, X_test, test_traits

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
        
class SelfAttention(LightningModule):
    def __init__(self, num_attention_heads, input_size, hidden_size, output_dim=1, kernel_size=3,
                 hidden_dropout_prob=0.5, attention_probs_dropout_prob=0.5, learning_rate=0.001):
        super(SelfAttention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.query = torch.nn.Linear(input_size, self.all_head_size)
        self.key = torch.nn.Linear(input_size, self.all_head_size)
        self.value = torch.nn.Linear(input_size, self.all_head_size)

        self.attn_dropout = torch.nn.Dropout(attention_probs_dropout_prob)
        self.out_dropout = torch.nn.Dropout(hidden_dropout_prob)
        self.dense = torch.nn.Linear(hidden_size, input_size)
        self.LayerNorm = torch.nn.LayerNorm(input_size, eps=1e-12)
        self.relu = torch.nn.ReLU()
        self.out = torch.nn.Linear(input_size, output_dim)
        self.cnn = torch.nn.Conv1d(1, 1, kernel_size, stride=1, padding=1)

        self.learning_rate = learning_rate
        self.loss_fn = MSELoss()

    def forward(self, input_tensor):
        input_tensor = input_tensor.to(self.device)
        self.cnn = self.cnn.to(self.device)

        cnn_hidden = self.cnn(input_tensor.view(input_tensor.size(0), 1, -1))
        input_tensor = cnn_hidden
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = mixed_query_layer
        key_layer = mixed_key_layer
        value_layer = mixed_value_layer

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / np.sqrt(self.attention_head_size)
        attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        output = self.out(self.relu(hidden_states.view(hidden_states.size(0), -1)))
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        val_loss = self.loss_fn(y_pred, y)
        return val_loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)



def objective(trial, x_train, y_train, inner_cv, DEVICE, hidden_dim, output_dim, kernel_size, learning_rate):
    num_attention_heads = trial.suggest_categorical('num_attention_heads', [4, 8])
    attention_probs_dropout_prob = trial.suggest_categorical('attention_probs_dropout_prob', [0.2, 0.5])

    fold_losses = []
    for train_idx, valid_idx in inner_cv.split(x_train):
        x_inner_train, x_inner_valid = x_train[train_idx], x_train[valid_idx]
        y_inner_train, y_inner_valid = y_train[train_idx], y_train[valid_idx]

        scaler = StandardScaler()
        x_inner_train = scaler.fit_transform(x_inner_train)
        x_inner_valid = scaler.transform(x_inner_valid)

        x_inner_train_tensor = torch.from_numpy(x_inner_train).float().to(DEVICE)
        y_inner_train_tensor = torch.from_numpy(y_inner_train).float().to(DEVICE)
        x_inner_valid_tensor = torch.from_numpy(x_inner_valid).float().to(DEVICE)
        y_inner_valid_tensor = torch.from_numpy(y_inner_valid).float().to(DEVICE)

        train_data = TensorDataset(x_inner_train_tensor, y_inner_train_tensor)
        valid_data = TensorDataset(x_inner_valid_tensor, y_inner_valid_tensor)

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        valid_loader = DataLoader(valid_data, batch_size=32, shuffle=False)

        model = SelfAttention(num_attention_heads, x_inner_train.shape[1], hidden_dim, output_dim,
                              hidden_dropout_prob=0.5, kernel_size=kernel_size,
                              attention_probs_dropout_prob=attention_probs_dropout_prob).to(DEVICE)
        loss_function = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(20):
            model.train()
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = model(x_batch)
                loss = loss_function(y_pred, y_batch.reshape(-1, 1))
                loss.backward()
                optimizer.step()

        model.eval()
        valid_losses = []
        with torch.no_grad():
            for x_batch, y_batch in valid_loader:
                y_pred = model(x_batch)
                loss = loss_function(y_pred, y_batch.reshape(-1, 1))
                valid_losses.append(loss.item())

        fold_losses.append(np.mean(valid_losses))

    return np.mean(fold_losses)


class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def run_nested_cv_with_early_stopping(data, label,outer_cv,inner_cv,learning_rate,batch_size,hidden_dim,output_dim,kernel_size, patience,DEVICE,result_dir):
    import time

    os.makedirs(result_dir, exist_ok=True)

    fold_records = []
    best_overall_corr = -float("inf")
    best_overall_fold = None

    time_start = time.time()

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(data), start=1):

        x_train, x_test = data[train_idx], data[test_idx]
        y_train, y_test = label[train_idx], label[test_idx]

        objective_with_data = functools.partial(
            objective,
            x_train=x_train,
            y_train=y_train,
            inner_cv=inner_cv,
            DEVICE=DEVICE,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            kernel_size=kernel_size,
            learning_rate=learning_rate
        )

        sampler = optuna.samplers.TPESampler(seed=cfg.random_state)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective_with_data, n_trials=cfg.optuna_trials)

        best_trial = study.best_trial
        num_attention_heads = best_trial.params["num_attention_heads"]
        attention_probs_dropout_prob = best_trial.params["attention_probs_dropout_prob"]

        model = SelfAttention(
            num_attention_heads,
            x_train.shape[1],
            hidden_dim,
            output_dim,
            hidden_dropout_prob=0.5,
            kernel_size=kernel_size,
            attention_probs_dropout_prob=attention_probs_dropout_prob
        ).to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_function = torch.nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=10, verbose=True
        )

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        g = torch.Generator()
        g.manual_seed(cfg.random_state)


        train_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(x_train).float().to(DEVICE),
                torch.from_numpy(y_train).float().to(DEVICE)
            ),
            batch_size=batch_size,
            shuffle=True,
            generator=g
        )

        test_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(x_test).float().to(DEVICE),
                torch.from_numpy(y_test).float().to(DEVICE)
            ),
            batch_size=batch_size,
            shuffle=False,
            generator=g
        )

        early_stopping = EarlyStopping(patience=patience)
        best_fold_corr = -float("inf")

        for epoch in range(100):
            model.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                loss = loss_function(model(xb), yb.reshape(-1, 1))
                loss.backward()
                optimizer.step()

            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for xb, yb in test_loader:
                    preds.extend(model(xb).cpu().numpy().ravel())
                    trues.extend(yb.cpu().numpy().ravel())

            corr_coef = np.corrcoef(preds, trues)[0, 1]
            scheduler.step(-corr_coef)

            if corr_coef > best_fold_corr:
                best_fold_corr = corr_coef
                torch.save(
                    model.state_dict(),
                    os.path.join(result_dir, f"best_model_fold_{fold}.pth")
                )

            early_stopping(corr_coef)
            if early_stopping.early_stop:
                print(f"[INFO] Early stopping at epoch {epoch + 1}")
                break

        fold_records.append({
            "fold": fold,
            "best_corr": best_fold_corr
        })

        if best_fold_corr > best_overall_corr:
            best_overall_corr = best_fold_corr
            best_overall_fold = fold

        print(
            f"Fold {fold}: "
            f"Best Correlation Coefficient: {best_fold_corr:.4f}"
        )

    elapsed = int(time.time() - time_start)

    pd.DataFrame(fold_records).to_csv(
        os.path.join(result_dir, "fold_metrics.csv"),
        index=False
    )

    return best_overall_fold, best_overall_corr, elapsed

def run_prediction_for_trait(
    model_path,
    X_test,
    cfg,
    trait_dir
):
    DEVICE = cfg.device

    model = SelfAttention(
        num_attention_heads=cfg.num_attention_heads,
        input_size=cfg.input_size,
        hidden_size=cfg.hidden_size,
        output_dim=cfg.output_dim,
        hidden_dropout_prob=cfg.hidden_dropout_prob,
        kernel_size=cfg.kernel_size,
        attention_probs_dropout_prob=cfg.attention_probs_dropout_prob
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(model_path, map_location=DEVICE)
    )

    # --- ensure correct input size (old behavior preserved)
    X = X_test
    if X.shape[1] < cfg.input_size:
        missing = cfg.input_size - X.shape[1]
        X = np.hstack([X, np.zeros((X.shape[0], missing))])
    elif X.shape[1] > cfg.input_size:
        X = X[:, :cfg.input_size]

    X_tensor = torch.from_numpy(X).float().to(DEVICE)

    model.eval()
    with torch.no_grad():
        preds = model(X_tensor).cpu().numpy().reshape(-1)

    pred_path = os.path.join(trait_dir, "predictions.csv")
    pd.DataFrame({"prediction": preds}).to_csv(pred_path, index=False)

    return pred_path

if __name__ == "__main__":

    # -------------------------------
    # Start total runtime
    # -------------------------------
    total_start_time = time.time()

    args = parse_args()

    try:
        cfg = load_and_validate_config(args.config)
        set_all_seeds(cfg.random_state)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(cfg.result_dir, exist_ok=True)

    logger = setup_logger(
        os.path.join(cfg.result_dir, "pipeline.log")
    )

    logger.info("Config loaded successfully")
    logger.info(f"Result directory: {cfg.result_dir}")
    logger.info(f"Device: {cfg.device}")

    # -------------------------------
    # Load data
    # -------------------------------
    X_train, train_traits, X_test, test_traits = load_all_data(cfg)

    logger.info(f"Train genotype shape: {X_train.shape}")
    logger.info(f"Train traits: {train_traits.columns.tolist()}")

    if X_test is not None:
        logger.info(f"Test genotype shape: {X_test.shape}")
    if test_traits is not None:
        logger.info(f"Test traits: {test_traits.columns.tolist()}")

    # -------------------------------
    # CV setup
    # -------------------------------
    outer_cv = KFold(
        n_splits=cfg.outer_folds,
        shuffle=True,
        random_state=cfg.random_state
    )

    inner_cv = KFold(
        n_splits=3,
        shuffle=True,
        random_state=cfg.random_state
    )

    # -------------------------------
    # Trait loop
    # -------------------------------
    for trait in train_traits.columns:
        trait_start_time = time.time()

        logger.info(f"Running trait: {trait}")

        trait_dir = os.path.join(cfg.result_dir, f"Trait_{trait}")
        os.makedirs(trait_dir, exist_ok=True)

        y_train = train_traits[trait].values

        best_fold, best_corr, train_time = run_nested_cv_with_early_stopping(
            data=X_train,
            label=y_train,
            outer_cv=outer_cv,
            inner_cv=inner_cv,
            learning_rate=cfg.learning_rate,
            batch_size=cfg.batch_size,
            hidden_dim=cfg.hidden_size,
            output_dim=cfg.output_dim,
            kernel_size=cfg.kernel_size,
            patience=cfg.patience,
            DEVICE=cfg.device,
            result_dir=trait_dir
        )

        logger.info(
            f"Trait {trait} training completed | "
            f"Best fold: {best_fold} | "
            f"Best corr: {best_corr:.4f} | "
            f"Training time: {train_time}s"
        )

        # -------------------------------
        # Prediction (only if valid)
        # -------------------------------
        if (
            X_test is not None and
            test_traits is not None and
            trait in test_traits.columns
        ):
            logger.info(f"Running prediction for trait: {trait}")

            model_path = os.path.join(
                trait_dir, f"best_model_fold_{best_fold}.pth"
            )

            pred_path = run_prediction_for_trait(
                model_path=model_path,
                X_test=X_test,
                cfg=cfg,
                trait_dir=trait_dir
            )

            y_true = test_traits[trait].values
            y_pred = pd.read_csv(pred_path)["prediction"].values

            r = np.corrcoef(y_true, y_pred)[0, 1]

            pd.DataFrame(
                {"pearson_r": [r]}
            ).to_csv(
                os.path.join(trait_dir, "prediction_r.csv"),
                index=False
            )

            logger.info(
                f"Trait {trait} prediction Pearson R: {r:.4f}"
            )

        trait_elapsed = int(time.time() - trait_start_time)
        logger.info(
            f"Trait {trait} total runtime: {trait_elapsed}s"
        )

    # -------------------------------
    # End total runtime
    # -------------------------------
    total_elapsed = int(time.time() - total_start_time)

    logger.info("Pipeline completed successfully")
    logger.info(f"Total pipeline runtime: {total_elapsed}s")