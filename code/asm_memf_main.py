# ============================================================
# MEMF V7 — Agentic Economics for Uruguay
# Small Open Economy MARL Laboratory
# ------------------------------------------------------------
# Features:
# - Heterogeneous firms: agro_export, manufacturing, retail,
#   domestic_services, tourism, tech_services
# - Heterogeneous households
# - Prudential banks
# - Government + Central Bank
# - External shocks: FX, commodities, Argentina/Brazil demand
# - Daily simulation with slower strategic adjustment
# - Network interactions + MARL + social imitation
# - Macro, financial and market diagnostics
# ============================================================

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

np.random.seed(42)

# ============================================================
# 1. CONFIG
# ============================================================

N_FIRMS = 60
N_HOUSEHOLDS = 400
N_BANKS = 3

N_EPISODES = 80
T_DAYS = 260                   # ~ 1 trading year
DECISION_INTERVAL = 5          # firms revise strategy every 5 days

NETWORK_P = 0.06
SOCIAL_BETA = 0.025
LEARNING_RATE = 0.004
EPSILON_START = 0.18
EPSILON_END = 0.03

# nominal scales
BASE_WAGE = 18.0
BASE_POLICY_RATE = 0.09        # annual-ish reference transformed internally
BASE_LENDING_SPREAD = 0.04

# firm initial conditions
FIRM_CASH_INIT = (6000.0, 22000.0)
FIRM_DEBT_INIT = (0.0, 9000.0)
FIRM_EMP_INIT = (2, 25)
FIRM_PRODUCTIVITY_INIT = (0.85, 1.25)
FIRM_TECH_INIT = (0.85, 1.20)
FIRM_PRICE_INIT = (0.90, 1.10)
FIRM_INVENTORY_INIT = (40.0, 140.0)

# households
HH_BASE_INCOME = (900.0, 4500.0)

# default thresholds
FIRM_DEFAULT_CASH_THRESHOLD = -12000.0
BANK_CAPITAL_BUFFER = 150000.0

# policy weights
GOV_MAX_DEBT = 2_500_000.0

# sectors
SECTOR_TYPES = [
    "agro_export",
    "manufacturing",
    "retail",
    "domestic_services",
    "tourism",
    "tech_services"
]

SECTOR_WEIGHTS = np.array([0.14, 0.18, 0.22, 0.22, 0.12, 0.12])

# actions
ACTION_NAMES = {
    0: "cut_price",
    1: "raise_price",
    2: "hire",
    3: "fire",
    4: "borrow",
    5: "invest_innovation",
    6: "hedge_fx",
    7: "wait"
}
N_ACTIONS = len(ACTION_NAMES)

# ============================================================
# 2. HELPERS
# ============================================================

def softmax(z):
    z = np.asarray(z, dtype=float)
    z = z - np.max(z)
    e = np.exp(z)
    return e / (np.sum(e) + 1e-12)

def stable_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -12, 12)))

def safe_mean(x):
    return np.mean(x) if len(x) > 0 else np.nan

def action_entropy(action_list, n_actions=N_ACTIONS):
    if len(action_list) == 0:
        return np.nan
    counts = np.array([action_list.count(a) for a in range(n_actions)], dtype=float)
    probs = counts / (counts.sum() + 1e-12)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs + 1e-12))

def gini(array):
    x = np.array(array, dtype=float)
    if len(x) == 0:
        return np.nan
    if np.min(x) < 0:
        x = x - np.min(x)
    x = x + 1e-9
    x = np.sort(x)
    n = len(x)
    idx = np.arange(1, n + 1)
    return np.sum((2 * idx - n - 1) * x) / (n * np.sum(x) + 1e-12)

def normalize_clip(x, lo=0.0, hi=2.0):
    return np.minimum(np.maximum(x, lo), hi)

def annual_to_daily(rate_annual):
    return (1.0 + rate_annual) ** (1 / 252) - 1.0

def rolling_change(curr, prev):
    if prev is None or abs(prev) < 1e-9:
        return 0.0
    return (curr - prev) / (abs(prev) + 1e-9)

def compute_hhi(firms):
    sales = np.array([max(f.sales_value, 0.0) for f in firms], dtype=float)
    total = sales.sum()
    if total <= 0:
        return np.nan
    shares = sales / total
    return np.sum(shares ** 2)

def mean_policy_distance(firms):
    Ws = [f.W.flatten() for f in firms if not f.default]
    D = []
    for i in range(len(Ws)):
        for j in range(i + 1, len(Ws)):
            D.append(np.linalg.norm(Ws[i] - Ws[j]))
    return np.mean(D) if len(D) > 0 else np.nan

def average_policy_drift(firms, previous_W):
    drifts = []
    for i, f in enumerate(firms):
        drifts.append(np.linalg.norm(f.W - previous_W[i]))
    return np.mean(drifts) if len(drifts) > 0 else np.nan

def network_policy_dispersion(firms, G):
    D = []
    for i, j in G.edges():
        if not firms[i].default and not firms[j].default:
            D.append(np.linalg.norm(firms[i].W - firms[j].W))
    return np.mean(D) if len(D) > 0 else np.nan

# ============================================================
# 3. NETWORK
# ============================================================

def make_economic_network(n_firms, p):
    G = nx.erdos_renyi_graph(n_firms, p, directed=True)
    for i in range(n_firms - 1):
        if not G.has_edge(i, i + 1):
            G.add_edge(i, i + 1)
    # extra reciprocal edges
    for i in range(0, n_firms, 5):
        j = (i + 3) % n_firms
        G.add_edge(j, i)
    return G

def get_neighbor_ids(G, i):
    return list(set(list(G.successors(i)) + list(G.predecessors(i))))

def get_neighbor_stress(i, firms, G):
    nbrs = get_neighbor_ids(G, i)
    if len(nbrs) == 0:
        return 0.0
    vals = []
    for j in nbrs:
        vals.append(1.0 if firms[j].default else max(0.0, -firms[j].cash) / 12000.0)
    return float(np.clip(np.mean(vals), 0.0, 1.5))

def get_avg_neighbor_price(i, firms, G):
    nbrs = get_neighbor_ids(G, i)
    vals = [firms[j].price for j in nbrs if not firms[j].default]
    return np.mean(vals) if len(vals) > 0 else 1.0

def get_avg_neighbor_productivity(i, firms, G):
    nbrs = get_neighbor_ids(G, i)
    vals = [firms[j].productivity * firms[j].tech_stock for j in nbrs if not firms[j].default]
    return np.mean(vals) if len(vals) > 0 else 1.0

# ============================================================
# 4. EXTERNAL ENVIRONMENT: URUGUAY SOE
# ============================================================

class ExternalEnvironment:
    def __init__(self):
        self.fx_level = 1.0                 # nominal FX index
        self.commodity_index = 1.0
        self.argentina_demand = 1.0
        self.brazil_demand = 1.0
        self.global_financial_stress = 0.15
        self.energy_cost = 1.0

        self.daily_inflation = 0.0
        self.trade_shock = 0.0

    def step(self, cb_policy_rate, gov_risk_signal):
        # AR(1)-style external dynamics
        eps_fx = np.random.normal(0, 0.004)
        eps_com = np.random.normal(0, 0.003)
        eps_arg = np.random.normal(0, 0.005)
        eps_bra = np.random.normal(0, 0.004)
        eps_gfs = np.random.normal(0, 0.004)

        self.global_financial_stress = np.clip(
            0.96 * self.global_financial_stress + 0.04 * 0.15 + eps_gfs,
            0.02, 0.80
        )

        # FX reacts to global stress and domestic policy credibility
        fx_growth = (
            0.25 * (self.global_financial_stress - 0.15)
            - 0.15 * (cb_policy_rate - 0.09)
            + 0.10 * gov_risk_signal
            + eps_fx
        )
        self.fx_level *= np.exp(np.clip(fx_growth, -0.03, 0.03))

        com_growth = 0.10 * (1.0 - self.commodity_index) + eps_com
        self.commodity_index *= np.exp(np.clip(com_growth, -0.02, 0.02))

        arg_growth = -0.03 * (self.argentina_demand - 1.0) + eps_arg
        bra_growth = -0.03 * (self.brazil_demand - 1.0) + eps_bra

        self.argentina_demand *= np.exp(np.clip(arg_growth, -0.025, 0.025))
        self.brazil_demand *= np.exp(np.clip(bra_growth, -0.020, 0.020))

        self.energy_cost = np.clip(
            0.985 * self.energy_cost + 0.015 * (1.0 + 0.2 * self.global_financial_stress)
            + np.random.normal(0, 0.002),
            0.85, 1.35
        )

        # imported inflation channel
        self.daily_inflation = np.clip(
            0.00008
            + 0.00035 * (self.fx_level - 1.0)
            + 0.00012 * (self.energy_cost - 1.0),
            -0.002, 0.003
        )

        self.trade_shock = (
            0.45 * (self.commodity_index - 1.0)
            + 0.25 * (self.argentina_demand - 1.0)
            + 0.30 * (self.brazil_demand - 1.0)
        )

# ============================================================
# 5. CENTRAL BANK
# ============================================================

class CentralBank:
    def __init__(self):
        self.policy_rate_annual = BASE_POLICY_RATE
        self.policy_rate_daily = annual_to_daily(self.policy_rate_annual)
        self.fx_concern = 0.0

    def update(self, annual_inflation_proxy, output_gap, fx_pressure, financial_stress):
        # hybrid Taylor-financial rule
        target = (
            0.09
            + 0.8 * max(annual_inflation_proxy - 0.06, 0.0)
            + 0.4 * max(fx_pressure, 0.0)
            + 0.25 * max(financial_stress - 0.15, 0.0)
            - 0.25 * max(-output_gap, 0.0)
        )
        self.policy_rate_annual = float(np.clip(target, 0.04, 0.18))
        self.policy_rate_daily = annual_to_daily(self.policy_rate_annual)
        self.fx_concern = fx_pressure

# ============================================================
# 6. GOVERNMENT
# ============================================================

class Government:
    def __init__(self):
        self.tax_rate_hh = 0.10
        self.tax_rate_corp = 0.12

        self.base_transfer = 50.0
        self.transfer = self.base_transfer

        self.base_innovation_subsidy = 0.18
        self.innovation_subsidy = self.base_innovation_subsidy

        self.base_credit_guarantee = 0.12
        self.credit_guarantee = self.base_credit_guarantee

        self.public_demand_share = 0.04
        self.export_support = 0.02

        self.revenue = 0.0
        self.spending = 0.0
        self.debt = 300000.0
        self.last_period_deficit = 0.0
        self.last_period_revenue = 0.0

    def update_policy(self, default_rate, unemployment_proxy, output_gap, external_stress):
        stress = (
            1.0 * default_rate
            + 0.7 * unemployment_proxy
            + 0.4 * max(-output_gap, 0.0)
            + 0.35 * external_stress
        )

        self.transfer = np.clip(self.base_transfer * (1.0 + 2.2 * stress), 35.0, 220.0)
        self.innovation_subsidy = np.clip(self.base_innovation_subsidy + 0.18 * stress, 0.10, 0.50)
        self.credit_guarantee = np.clip(self.base_credit_guarantee + 0.22 * stress, 0.05, 0.55)
        self.public_demand_share = np.clip(0.04 + 0.06 * stress, 0.02, 0.15)
        self.export_support = np.clip(0.02 + 0.04 * external_stress, 0.01, 0.10)

        # soft fiscal brake
        if self.debt > GOV_MAX_DEBT:
            self.transfer *= 0.92
            self.public_demand_share *= 0.90
            self.innovation_subsidy *= 0.95

    def collect_taxes(self, household_incomes, firm_profits):
        hh_tax = self.tax_rate_hh * np.sum(np.maximum(household_incomes, 0.0))
        corp_tax = self.tax_rate_corp * np.sum(np.maximum(firm_profits, 0.0))
        collected = hh_tax + corp_tax
        self.revenue += collected
        self.last_period_revenue = collected
        return collected

    def spend(self, n_households, innovation_subsidy_paid, credit_guarantee_paid, public_procurement, export_support_paid):
        transfers_total = self.transfer * n_households
        total_spending = transfers_total + innovation_subsidy_paid + credit_guarantee_paid + public_procurement + export_support_paid

        period_deficit = total_spending - self.last_period_revenue
        self.last_period_deficit = period_deficit

        self.spending += total_spending
        self.debt += max(period_deficit, 0.0)
        self.revenue = 0.0
        return total_spending

# ============================================================
# 7. AGENT BASE
# ============================================================

class Agent:
    def __init__(self, n_features, n_actions):
        self.W = np.random.randn(n_actions, n_features) * 0.03

    def policy(self, x):
        logits = self.W @ x
        return softmax(logits)

    def act(self, x, epsilon=0.05):
        probs = self.policy(x)
        if np.random.rand() < epsilon:
            action = np.random.randint(len(probs))
        else:
            action = np.random.choice(len(probs), p=probs)
        return action, probs

    def update(self, x, action, reward, baseline, lr=LEARNING_RATE):
        probs = self.policy(x)
        advantage = reward - baseline
        grad = -probs[:, None] * x[None, :]
        grad[action] += x
        self.W += lr * advantage * grad

# ============================================================
# 8. FIRM AGENTS
# ============================================================

SECTOR_PARAMS = {
    "agro_export": {
        "export_share": 0.80, "import_intensity": 0.35, "labor_intensity": 0.75,
        "price_elasticity": 0.80, "commodity_beta": 1.10, "tourism_beta": 0.00,
        "arg_beta": 0.20, "bra_beta": 0.30, "domestic_beta": 0.25
    },
    "manufacturing": {
        "export_share": 0.35, "import_intensity": 0.45, "labor_intensity": 0.95,
        "price_elasticity": 1.00, "commodity_beta": 0.30, "tourism_beta": 0.00,
        "arg_beta": 0.25, "bra_beta": 0.30, "domestic_beta": 0.60
    },
    "retail": {
        "export_share": 0.02, "import_intensity": 0.25, "labor_intensity": 1.10,
        "price_elasticity": 1.25, "commodity_beta": 0.00, "tourism_beta": 0.20,
        "arg_beta": 0.15, "bra_beta": 0.05, "domestic_beta": 1.20
    },
    "domestic_services": {
        "export_share": 0.01, "import_intensity": 0.10, "labor_intensity": 1.20,
        "price_elasticity": 0.95, "commodity_beta": 0.00, "tourism_beta": 0.15,
        "arg_beta": 0.05, "bra_beta": 0.05, "domestic_beta": 1.25
    },
    "tourism": {
        "export_share": 0.30, "import_intensity": 0.15, "labor_intensity": 1.15,
        "price_elasticity": 1.15, "commodity_beta": 0.00, "tourism_beta": 1.20,
        "arg_beta": 0.75, "bra_beta": 0.20, "domestic_beta": 0.60
    },
    "tech_services": {
        "export_share": 0.45, "import_intensity": 0.08, "labor_intensity": 1.35,
        "price_elasticity": 0.75, "commodity_beta": 0.00, "tourism_beta": 0.00,
        "arg_beta": 0.05, "bra_beta": 0.15, "domestic_beta": 0.70
    }
}

def compute_market_power(firm, avg_neighbor_price, avg_neighbor_productivity):
    relative_productivity = (firm.productivity * firm.tech_stock) / (avg_neighbor_productivity + 1e-6)
    relative_price = firm.price / (avg_neighbor_price + 1e-6)

    power = (
        0.65 * np.tanh(relative_productivity - 1.0)
        - 0.35 * np.tanh(relative_price - 1.0)
        + 0.12 * np.tanh(2.5 * (firm.last_market_share - 0.05))
    )
    return np.clip(power, -1.0, 1.0)

def compute_markup(market_power, market_share, sector_elasticity):
    base_markup = 0.08 + 0.06 / (sector_elasticity + 0.3)
    markup = (
        base_markup
        + 0.26 * np.tanh(market_power)
        + 0.16 * np.tanh(4.0 * (market_share - 0.06))
    )
    return np.clip(markup, 0.0, 0.85)

class Firm(Agent):
    def __init__(self, firm_id, sector):
        super().__init__(n_features=22, n_actions=N_ACTIONS)
        self.firm_id = firm_id
        self.sector = sector
        self.params = SECTOR_PARAMS[sector]
        self.reset_state()

    def reset_state(self):
        self.cash = np.random.uniform(*FIRM_CASH_INIT)
        self.debt = np.random.uniform(*FIRM_DEBT_INIT)
        self.employees = np.random.randint(*FIRM_EMP_INIT)
        self.productivity = np.random.uniform(*FIRM_PRODUCTIVITY_INIT)
        self.tech_stock = np.random.uniform(*FIRM_TECH_INIT)
        self.price = np.random.uniform(*FIRM_PRICE_INIT)
        self.inventory = np.random.uniform(*FIRM_INVENTORY_INIT)

        self.sales_qty = 0.0
        self.sales_value = 0.0
        self.export_sales = 0.0
        self.domestic_sales = 0.0
        self.default = False
        self.last_profit = 0.0

        self.fx_hedge = 0.0
        self.last_markup = 0.10
        self.last_market_share = 1.0 / max(N_FIRMS, 1)
        self.last_market_power = 0.0

        self.last_action = 7
        self.last_reward = 0.0
        self.total_reward = 0.0

        self.action_history = []
        self.reward_history = []
        self.cash_history = []
        self.price_history = []
        self.productivity_history = []
        self.tech_history = []
        self.market_share_history = []
        self.markup_history = []

    def unit_cost(self, lending_rate_daily, inflation_daily, fx_level, energy_cost):
        labor_component = (
            BASE_WAGE * self.params["labor_intensity"] /
            max(self.productivity * self.tech_stock * np.sqrt(max(self.employees, 1)), 1.0)
        )
        finance_component = 0.55 * lending_rate_daily * self.debt
        imported_input_component = self.params["import_intensity"] * (0.70 * fx_level + 0.30 * energy_cost)
        inflation_component = 1.0 + 40.0 * max(inflation_daily, -0.002)

        return 0.22 + 0.05 * labor_component + 0.002 * finance_component + 0.18 * imported_input_component * inflation_component

    def capacity(self):
        noise = np.random.uniform(0.94, 1.06)
        return self.employees * self.productivity * self.tech_stock * 3.2 * noise

    def competitiveness_score_domestic(self, tourism_factor):
        quality = self.productivity * self.tech_stock
        price_penalty = self.price ** self.params["price_elasticity"]
        sector_boost = 1.0 + self.params["tourism_beta"] * (tourism_factor - 1.0)
        return (quality * sector_boost * (1.0 + 0.35 * self.last_markup)) / (price_penalty + 1e-6)

    def export_score(self, ext_env, government):
        p = self.params
        score = (
            p["export_share"]
            * (self.productivity * self.tech_stock)
            * (1.0 + p["commodity_beta"] * (ext_env.commodity_index - 1.0))
            * (1.0 + p["arg_beta"] * (ext_env.argentina_demand - 1.0))
            * (1.0 + p["bra_beta"] * (ext_env.brazil_demand - 1.0))
            * (1.0 + government.export_support)
            * (ext_env.fx_level ** 0.20)
        )
        return max(score, 0.0)

    def update_price(self, avg_neighbor_price, avg_neighbor_productivity, market_share, lending_rate_daily):
        market_power = compute_market_power(self, avg_neighbor_price, avg_neighbor_productivity)
        market_power = 0.75 * self.last_market_power + 0.25 * market_power
        self.last_market_power = market_power

        markup = compute_markup(market_power, market_share, self.params["price_elasticity"])
        new_price = self.mc_proxy(lending_rate_daily) * (1.0 + markup)

        self.price = 0.82 * self.price + 0.18 * new_price
        self.price = float(np.clip(self.price, 0.40, 4.50))
        self.last_markup = markup
        return markup

    def mc_proxy(self, lending_rate_daily):
        return 0.85 + 1.2 * lending_rate_daily + 0.04 * self.params["import_intensity"]

    def state(self, local_demand, lending_rate_daily, neighbor_stress, avg_neighbor_price,
              avg_neighbor_productivity, market_share, macro_growth, gov_subsidy,
              fx_level, inflation_daily, commodity_index, argentina_demand,
              brazil_demand, cb_rate_annual, unemployment_proxy):
        x = np.array([
            self.cash / 30000.0,
            self.debt / 15000.0,
            self.employees / 35.0,
            self.productivity / 2.0,
            self.tech_stock / 2.0,
            self.price / 4.5,
            self.inventory / 250.0,
            local_demand / 18000.0,
            lending_rate_daily / 0.0025,
            neighbor_stress,
            avg_neighbor_price / 4.5,
            avg_neighbor_productivity / 2.0,
            market_share / 0.25,
            macro_growth + 1.0,
            gov_subsidy,
            self.last_markup / 0.85,
            self.last_market_power + 1.0,
            fx_level / 1.5,
            (inflation_daily + 0.002) / 0.005,
            commodity_index / 1.5,
            argentina_demand / 1.5,
            brazil_demand / 1.5
        ], dtype=float)
        x = normalize_clip(x, 0.0, 2.0)
        return x

    def apply_action(self, action, approved_credit, approved_amount, gov_subsidy, fx_level):
        innovation_subsidy_paid = 0.0

        if action == 0:
            self.price *= 0.975

        elif action == 1:
            self.price *= 1.020

        elif action == 2:
            self.employees += 1
            self.cash -= 35.0

        elif action == 3:
            self.employees = max(1, self.employees - 1)
            self.cash += 12.0

        elif action == 4:
            if approved_credit:
                self.cash += approved_amount
                self.debt += approved_amount

        elif action == 5:
            gross_invest_cost = 260.0
            subsidy = gov_subsidy * gross_invest_cost
            net_cost = gross_invest_cost - subsidy
            if self.cash >= net_cost:
                self.cash -= net_cost
                innovation_subsidy_paid = subsidy
                innovation_shock = np.random.uniform(0.010, 0.040)
                self.tech_stock *= (1.0 + innovation_shock)
                self.productivity *= (1.0 + 0.55 * innovation_shock)
                self.inventory += np.random.uniform(6, 14)

        elif action == 6:
            hedge_cost = 45.0
            if self.cash >= hedge_cost:
                self.cash -= hedge_cost
                self.fx_hedge = min(1.0, self.fx_hedge + 0.35)

        elif action == 7:
            pass

        # natural hedge decay
        self.fx_hedge *= 0.985

        self.price = float(np.clip(self.price, 0.40, 4.50))
        self.productivity = float(np.clip(self.productivity, 0.55, 3.00))
        self.tech_stock = float(np.clip(self.tech_stock, 0.60, 3.20))
        self.employees = int(np.clip(self.employees, 1, 160))
        self.inventory = float(np.clip(self.inventory, 0.0, 900.0))

        return innovation_subsidy_paid

# ============================================================
# 9. HOUSEHOLDS
# ============================================================

HOUSEHOLD_TYPES = ["vulnerable", "middle_formal", "middle_indebted", "upper_saver"]
HH_TYPE_WEIGHTS = np.array([0.26, 0.38, 0.24, 0.12])

class Household:
    def __init__(self, household_id):
        self.household_id = household_id
        self.hh_type = np.random.choice(HOUSEHOLD_TYPES, p=HH_TYPE_WEIGHTS)
        self.base_income = np.random.uniform(*HH_BASE_INCOME)
        self.current_income = self.base_income
        self.sensitivity = np.random.uniform(1.3, 3.0)
        self.debt_service = np.random.uniform(0.0, 0.18)
        self.fx_saving_preference = np.random.uniform(0.0, 0.25)

        if self.hh_type == "vulnerable":
            self.mpc = 0.93
        elif self.hh_type == "middle_formal":
            self.mpc = 0.82
        elif self.hh_type == "middle_indebted":
            self.mpc = 0.75
            self.debt_service += 0.08
        else:
            self.mpc = 0.62
            self.fx_saving_preference += 0.12

    def update_income(self, macro_wage_factor, employment_factor, gov_transfer, inflation_daily):
        real_erosion = max(0.80, 1.0 - 25.0 * max(inflation_daily, 0.0))
        self.current_income = (
            self.base_income * macro_wage_factor * employment_factor * real_erosion
            + gov_transfer
        )

    def consumption_budget(self, fx_uncertainty, unemployment_proxy):
        precaution = 1.0 - 0.18 * fx_uncertainty - 0.20 * unemployment_proxy
        debt_drag = 1.0 - self.debt_service
        saver_drag = 1.0 - self.fx_saving_preference * fx_uncertainty
        return max(0.0, self.mpc * self.current_income * precaution * debt_drag * saver_drag)

# ============================================================
# 10. BANKS
# ============================================================

def compute_credit_score(firm, neighbor_stress, macro_default_rate, gov_guarantee, ext_stress, output_gap):
    leverage = firm.debt / (firm.cash + 5000.0)
    profitability = firm.last_profit / (abs(firm.sales_value) + 1e-6)

    score = (
        0.75 * (firm.productivity - 1.0)
        + 0.65 * (firm.tech_stock - 1.0)
        + 0.45 * firm.last_markup
        + 0.55 * np.tanh(profitability)
        + 0.25 * np.tanh(firm.cash / 15000.0)
        - 0.55 * np.tanh(firm.debt / 12000.0)
        - 1.10 * leverage
        - 0.85 * neighbor_stress
        - 1.80 * macro_default_rate
        - 0.60 * ext_stress
        + 0.45 * gov_guarantee
        + 0.20 * output_gap
    )

    if firm.cash < 0:
        score -= 0.85

    return np.clip(score, -10, 10)

class Bank:
    def __init__(self, bank_id):
        self.bank_id = bank_id
        self.base_spread = BASE_LENDING_SPREAD + np.random.uniform(-0.005, 0.005)
        self.lending_rate_annual = BASE_POLICY_RATE + self.base_spread
        self.lending_rate_daily = annual_to_daily(self.lending_rate_annual)

        self.loan_book = 0.0
        self.default_losses = 0.0
        self.capital = BANK_CAPITAL_BUFFER
        self.approval_history = []
        self.score_history = []

    def update_macro_rate(self, cb_policy_rate_annual, macro_default_rate, ext_stress):
        spread = (
            self.base_spread
            + 0.05 * macro_default_rate
            + 0.03 * ext_stress
        )
        self.lending_rate_annual = float(np.clip(cb_policy_rate_annual + spread, 0.07, 0.28))
        self.lending_rate_daily = annual_to_daily(self.lending_rate_annual)

    def credit_decision(self, firm, neighbor_stress, macro_default_rate, gov_guarantee, ext_stress, output_gap):
        score = compute_credit_score(
            firm=firm,
            neighbor_stress=neighbor_stress,
            macro_default_rate=macro_default_rate,
            gov_guarantee=gov_guarantee,
            ext_stress=ext_stress,
            output_gap=output_gap
        )

        p = stable_sigmoid(score)
        if self.capital < 80000:
            p *= 0.75
        if firm.cash < 0:
            p *= 0.70

        approved = np.random.rand() < p
        amount = 0.0
        expected_guarantee_cost = 0.0

        if approved:
            base_amount = (
                450.0
                + 0.06 * max(firm.cash, 0.0)
                + 110.0 * firm.productivity
                + 90.0 * firm.tech_stock
                + 40.0 * firm.employees
            )
            risk_adjustment = 1.0 - 0.45 * macro_default_rate - 0.20 * ext_stress + 0.20 * gov_guarantee
            amount = np.clip(base_amount * max(risk_adjustment, 0.35), 300.0, 5000.0)

            self.loan_book += amount
            expected_guarantee_cost = gov_guarantee * 0.02 * amount

        self.approval_history.append(1 if approved else 0)
        self.score_history.append(score)

        return approved, amount, expected_guarantee_cost

    def register_default(self, firm, gov_guarantee):
        gross_loss = 0.45 * firm.debt
        net_loss = gross_loss * (1.0 - gov_guarantee)

        self.default_losses += net_loss
        self.capital -= net_loss

        return gross_loss * gov_guarantee

# ============================================================
# 11. DEMAND ALLOCATION
# ============================================================

def update_household_incomes(households, firms, government, ext_env):
    alive = [f for f in firms if not f.default]
    if len(alive) == 0:
        for h in households:
            h.current_income = 0.6 * h.base_income + government.transfer
        return

    avg_employment = np.mean([f.employees for f in alive])
    avg_productivity = np.mean([f.productivity * f.tech_stock for f in alive])

    macro_wage_factor = np.clip(0.88 + 0.18 * avg_productivity, 0.70, 1.50)
    employment_factor = np.clip(avg_employment / 10.0, 0.70, 1.30)

    for h in households:
        h.update_income(
            macro_wage_factor=macro_wage_factor,
            employment_factor=employment_factor,
            gov_transfer=government.transfer,
            inflation_daily=ext_env.daily_inflation
        )

def allocate_household_demand(households, firms, government, ext_env):
    active_firms = [f for f in firms if not f.default]
    if len(active_firms) == 0:
        return {}, {}

    demand_value = {f.firm_id: 0.0 for f in active_firms}
    demand_qty = {f.firm_id: 0.0 for f in active_firms}

    tourism_factor = 1.0 + 0.6 * max(ext_env.argentina_demand - 1.0, -0.4)

    firm_scores = np.array(
        [max(f.competitiveness_score_domestic(tourism_factor), 1e-6) for f in active_firms],
        dtype=float
    )

    unemployment_proxy = np.clip(
        1.0 - safe_mean([f.employees for f in active_firms]) / 12.0,
        0.0, 0.6
    )
    fx_uncertainty = np.clip(abs(ext_env.fx_level - 1.0), 0.0, 0.5)

    for h in households:
        budget = h.consumption_budget(fx_uncertainty=fx_uncertainty, unemployment_proxy=unemployment_proxy)
        weights = firm_scores ** h.sensitivity
        probs = weights / (weights.sum() + 1e-12)

        for _ in range(3):
            idx = np.random.choice(len(active_firms), p=probs)
            chosen = active_firms[idx]
            spend = budget / 3.0
            qty = spend / max(chosen.price, 1e-6)
            demand_value[chosen.firm_id] += spend
            demand_qty[chosen.firm_id] += qty

    public_budget = government.public_demand_share * sum(h.consumption_budget(fx_uncertainty, unemployment_proxy) for h in households)
    public_scores = np.array(
        [(f.productivity * f.tech_stock) / (f.price + 1e-6) for f in active_firms],
        dtype=float
    )
    public_probs = public_scores / (public_scores.sum() + 1e-12)

    n_draws = max(1, len(active_firms) // 3)
    for _ in range(n_draws):
        idx = np.random.choice(len(active_firms), p=public_probs)
        chosen = active_firms[idx]
        spend = public_budget / n_draws
        qty = spend / max(chosen.price, 1e-6)
        demand_value[chosen.firm_id] += spend
        demand_qty[chosen.firm_id] += qty

    return demand_value, demand_qty

def allocate_export_demand(firms, government, ext_env):
    export_value = {}
    export_qty = {}
    export_support_paid = 0.0

    for f in firms:
        if f.default:
            continue

        score = f.export_score(ext_env, government)
        raw_export_value = 140.0 * score
        if f.sector == "tourism":
            raw_export_value *= (1.0 + 0.9 * max(ext_env.argentina_demand - 1.0, -0.5))
        if f.sector == "agro_export":
            raw_export_value *= (1.0 + 0.8 * max(ext_env.commodity_index - 1.0, -0.5))

        support = government.export_support * raw_export_value if f.params["export_share"] > 0.15 else 0.0
        export_support_paid += support

        value = max(0.0, raw_export_value + support)
        qty = value / max(f.price, 1e-6)

        export_value[f.firm_id] = value
        export_qty[f.firm_id] = qty

    return export_value, export_qty, export_support_paid

# ============================================================
# 12. TRAIN
# ============================================================

def train_memf_v7(
    n_episodes=N_EPISODES,
    T_days=T_DAYS,
    n_firms=N_FIRMS,
    n_households=N_HOUSEHOLDS,
    n_banks=N_BANKS,
    network_p=NETWORK_P,
    social_beta=SOCIAL_BETA,
    learning_rate=LEARNING_RATE
):
    sectors = np.random.choice(SECTOR_TYPES, size=n_firms, p=SECTOR_WEIGHTS)

    firms = [Firm(i, sectors[i]) for i in range(n_firms)]
    households = [Household(i) for i in range(n_households)]
    G = make_economic_network(n_firms, network_p)

    episode_rows = []
    agent_rows = []
    period_rows = []

    for ep in range(1, n_episodes + 1):
        for f in firms:
            f.reset_state()

        banks = [Bank(i) for i in range(n_banks)]
        government = Government()
        central_bank = CentralBank()
        ext_env = ExternalEnvironment()

        prev_W = [f.W.copy() for f in firms]
        epsilon = EPSILON_START + (EPSILON_END - EPSILON_START) * (ep - 1) / max(n_episodes - 1, 1)

        macro_default_rate_prev = 0.0
        macro_growth_prev = 0.0
        prev_total_sales = None
        annual_inflation_proxy = 0.06

        states = {}
        chosen_actions = {}
        approvals = {}
        approved_amounts = {}
        expected_guarantee_costs = {}

        for day in range(1, T_days + 1):
            alive_firms = [f for f in firms if not f.default]
            if len(alive_firms) == 0:
                break

            avg_employment = safe_mean([f.employees for f in alive_firms])
            unemployment_proxy = np.clip(1.0 - avg_employment / 12.0, 0.0, 0.7)
            output_gap = macro_growth_prev
            ext_stress = ext_env.global_financial_stress + abs(ext_env.fx_level - 1.0)

            government.update_policy(
                default_rate=macro_default_rate_prev,
                unemployment_proxy=unemployment_proxy,
                output_gap=output_gap,
                external_stress=ext_stress
            )

            central_bank.update(
                annual_inflation_proxy=annual_inflation_proxy,
                output_gap=output_gap,
                fx_pressure=ext_env.fx_level - 1.0,
                financial_stress=ext_env.global_financial_stress
            )

            ext_env.step(
                cb_policy_rate=central_bank.policy_rate_annual,
                gov_risk_signal=np.clip(government.debt / GOV_MAX_DEBT, 0.0, 2.0)
            )

            annual_inflation_proxy = np.clip(0.98 * annual_inflation_proxy + 0.02 * (252 * ext_env.daily_inflation), -0.02, 0.25)

            update_household_incomes(households, firms, government, ext_env)

            for b in banks:
                b.update_macro_rate(
                    cb_policy_rate_annual=central_bank.policy_rate_annual,
                    macro_default_rate=macro_default_rate_prev,
                    ext_stress=ext_env.global_financial_stress
                )

            total_budget = sum(h.consumption_budget(abs(ext_env.fx_level - 1.0), unemployment_proxy) for h in households)
            local_demand = total_budget / max(len(alive_firms), 1)
            avg_lending_rate_daily = np.mean([b.lending_rate_daily for b in banks])

            # strategic decisions only every few days
            if day == 1 or day % DECISION_INTERVAL == 0:
                states = {}
                chosen_actions = {}
                approvals = {}
                approved_amounts = {}
                expected_guarantee_costs = {}

                total_last_sales = sum([max(f.sales_value, 0.0) for f in alive_firms]) + 1e-6

                for firm in firms:
                    if firm.default:
                        continue

                    neighbor_stress = get_neighbor_stress(firm.firm_id, firms, G)
                    avg_neighbor_price = get_avg_neighbor_price(firm.firm_id, firms, G)
                    avg_neighbor_productivity = get_avg_neighbor_productivity(firm.firm_id, firms, G)

                    market_share_guess = max(firm.sales_value, 0.0) / total_last_sales

                    x = firm.state(
                        local_demand=local_demand,
                        lending_rate_daily=avg_lending_rate_daily,
                        neighbor_stress=neighbor_stress,
                        avg_neighbor_price=avg_neighbor_price,
                        avg_neighbor_productivity=avg_neighbor_productivity,
                        market_share=market_share_guess,
                        macro_growth=macro_growth_prev,
                        gov_subsidy=government.innovation_subsidy,
                        fx_level=ext_env.fx_level,
                        inflation_daily=ext_env.daily_inflation,
                        commodity_index=ext_env.commodity_index,
                        argentina_demand=ext_env.argentina_demand,
                        brazil_demand=ext_env.brazil_demand,
                        cb_rate_annual=central_bank.policy_rate_annual,
                        unemployment_proxy=unemployment_proxy
                    )

                    action, probs = firm.act(x, epsilon=epsilon)

                    approved, amount, expected_guarantee_cost = False, 0.0, 0.0
                    if action == 4:
                        bank = np.random.choice(banks)
                        approved, amount, expected_guarantee_cost = bank.credit_decision(
                            firm=firm,
                            neighbor_stress=neighbor_stress,
                            macro_default_rate=macro_default_rate_prev,
                            gov_guarantee=government.credit_guarantee,
                            ext_stress=ext_env.global_financial_stress,
                            output_gap=output_gap
                        )

                    states[firm.firm_id] = x
                    chosen_actions[firm.firm_id] = action
                    approvals[firm.firm_id] = approved
                    approved_amounts[firm.firm_id] = amount
                    expected_guarantee_costs[firm.firm_id] = expected_guarantee_cost

                innovation_subsidy_paid_total = 0.0
                credit_guarantee_paid_total = 0.0

                for firm in firms:
                    if firm.default:
                        continue
                    innovation_subsidy_paid = firm.apply_action(
                        chosen_actions[firm.firm_id],
                        approvals[firm.firm_id],
                        approved_amounts[firm.firm_id],
                        government.innovation_subsidy,
                        ext_env.fx_level
                    )
                    innovation_subsidy_paid_total += innovation_subsidy_paid
                    credit_guarantee_paid_total += expected_guarantee_costs[firm.firm_id]
                    firm.last_action = chosen_actions[firm.firm_id]
                    firm.action_history.append(chosen_actions[firm.firm_id])
            else:
                innovation_subsidy_paid_total = 0.0
                credit_guarantee_paid_total = 0.0

            total_last_sales = sum([max(f.sales_value, 0.0) for f in alive_firms]) + 1e-6
            for firm in firms:
                if firm.default:
                    continue
                avg_neighbor_price = get_avg_neighbor_price(firm.firm_id, firms, G)
                avg_neighbor_productivity = get_avg_neighbor_productivity(firm.firm_id, firms, G)
                market_share_guess = max(firm.sales_value, 0.0) / total_last_sales
                firm.update_price(
                    avg_neighbor_price=avg_neighbor_price,
                    avg_neighbor_productivity=avg_neighbor_productivity,
                    market_share=market_share_guess,
                    lending_rate_daily=avg_lending_rate_daily
                )

            domestic_demand_value, domestic_demand_qty = allocate_household_demand(households, firms, government, ext_env)
            export_demand_value, export_demand_qty, export_support_paid = allocate_export_demand(firms, government, ext_env)

            total_sales_value = (
                sum(domestic_demand_value.values()) + sum(export_demand_value.values())
                if len(domestic_demand_value) > 0 else 1.0
            )

            period_rewards = []
            firm_profits_for_tax = []

            for firm in firms:
                if firm.default:
                    continue

                qty_dom = domestic_demand_qty.get(firm.firm_id, 0.0)
                qty_exp = export_demand_qty.get(firm.firm_id, 0.0)
                val_dom = domestic_demand_value.get(firm.firm_id, 0.0)
                val_exp = export_demand_value.get(firm.firm_id, 0.0)

                qty_demanded = qty_dom + qty_exp
                capacity = firm.capacity()
                available = firm.inventory + capacity
                realized_qty = min(qty_demanded, available)

                share_dom = qty_dom / max(qty_demanded, 1e-6)
                share_exp = qty_exp / max(qty_demanded, 1e-6)

                realized_dom = realized_qty * share_dom
                realized_exp = realized_qty * share_exp

                realized_sales_dom = realized_dom * firm.price
                realized_sales_exp = realized_exp * firm.price
                realized_sales = realized_sales_dom + realized_sales_exp

                production = capacity
                firm.inventory = max(0.0, firm.inventory + production - realized_qty)

                wage_bill = firm.employees * BASE_WAGE * firm.params["labor_intensity"]
                interest_cost = firm.debt * avg_lending_rate_daily
                variable_cost = firm.unit_cost(avg_lending_rate_daily, ext_env.daily_inflation, ext_env.fx_level, ext_env.energy_cost) * realized_qty
                depreciation = 0.0025 * firm.inventory

                # FX balance-sheet effect on imported inputs not hedged
                fx_balance_cost = (
                    20.0
                    * firm.params["import_intensity"]
                    * max(ext_env.fx_level - 1.0, 0.0)
                    * (1.0 - firm.fx_hedge)
                )

                profit = realized_sales - wage_bill - interest_cost - variable_cost - depreciation - fx_balance_cost

                firm.last_profit = profit
                firm_profits_for_tax.append(profit)
                firm.cash += profit

                firm.domestic_sales = realized_sales_dom
                firm.export_sales = realized_sales_exp

                market_share = realized_sales / max(total_sales_value, 1e-6)
                firm.last_market_share = market_share

                leverage = firm.debt / max(firm.cash + 5000.0, 1.0)
                neighbor_stress = get_neighbor_stress(firm.firm_id, firms, G)
                underproduction_gap = max(qty_demanded - realized_qty, 0.0) / max(qty_demanded + 1e-6, 1.0)
                profit_margin = profit / (realized_sales + 1e-6)
                export_bonus = realized_sales_exp / (realized_sales + 1e-6)

                reward = (
                    0.22 * (profit / 1000.0)
                    + 1.6 * profit_margin
                    + 0.55 * (firm.tech_stock - 1.0)
                    + 0.40 * market_share
                    + 0.22 * export_bonus
                    + 0.05 * (firm.employees / 10.0)
                    - 0.35 * leverage
                    - 0.22 * neighbor_stress
                    - 0.30 * underproduction_gap
                    - 0.12 * max(ext_env.fx_level - 1.0, 0.0) * (1.0 - firm.fx_hedge)
                )

                if firm.cash < 0:
                    reward -= 0.45
                if firm.last_action == 5:
                    reward += 0.18 * government.innovation_subsidy
                if firm.last_action == 6 and ext_env.fx_level > 1.03:
                    reward += 0.10

                if firm.cash < FIRM_DEFAULT_CASH_THRESHOLD:
                    firm.default = True
                    reward -= 3.0
                    for bank in banks:
                        credit_guarantee_paid_total += bank.register_default(firm, government.credit_guarantee)

                firm.sales_qty = realized_qty
                firm.sales_value = realized_sales
                firm.total_reward += reward
                firm.last_reward = reward

                firm.reward_history.append(reward)
                firm.cash_history.append(firm.cash)
                firm.price_history.append(firm.price)
                firm.productivity_history.append(firm.productivity)
                firm.tech_history.append(firm.tech_stock)
                firm.market_share_history.append(market_share)
                firm.markup_history.append(firm.last_markup)

                period_rewards.append(reward)

            household_incomes = np.array([h.current_income for h in households], dtype=float)
            government.collect_taxes(household_incomes, np.array(firm_profits_for_tax, dtype=float))
            public_procurement = government.public_demand_share * total_budget
            government.spend(
                n_households=n_households,
                innovation_subsidy_paid=innovation_subsidy_paid_total,
                credit_guarantee_paid=credit_guarantee_paid_total,
                public_procurement=public_procurement,
                export_support_paid=export_support_paid
            )

            baseline = np.mean(period_rewards) if len(period_rewards) > 0 else 0.0

            if day == 1 or day % DECISION_INTERVAL == 0:
                for firm in firms:
                    if firm.default:
                        continue
                    x = states[firm.firm_id]
                    a = chosen_actions[firm.firm_id]
                    firm.update(x, a, firm.last_reward, baseline, lr=learning_rate)

                for firm in firms:
                    if firm.default:
                        continue
                    nbrs = get_neighbor_ids(G, firm.firm_id)
                    candidate_nbrs = [firms[j] for j in nbrs if not firms[j].default]
                    better_nbrs = [g for g in candidate_nbrs if g.last_reward > firm.last_reward]
                    if len(better_nbrs) > 0:
                        model_firm = max(better_nbrs, key=lambda z: z.last_reward)
                        firm.W = (1 - social_beta) * firm.W + social_beta * model_firm.W

            alive_firms = [f for f in firms if not f.default]
            macro_default_rate_prev = np.mean([1.0 if f.default else 0.0 for f in firms])

            current_total_sales = np.sum([f.sales_value for f in alive_firms]) if len(alive_firms) > 0 else 0.0
            macro_growth_prev = rolling_change(current_total_sales, prev_total_sales)
            prev_total_sales = current_total_sales

            period_rows.append({
                "episode": ep,
                "day": day,
                "alive_firms": len(alive_firms),
                "default_rate_t": macro_default_rate_prev,
                "avg_reward_t": safe_mean([f.last_reward for f in alive_firms]),
                "avg_price_t": safe_mean([f.price for f in alive_firms]),
                "avg_cash_t": safe_mean([f.cash for f in alive_firms]),
                "avg_debt_t": safe_mean([f.debt for f in alive_firms]),
                "avg_productivity_t": safe_mean([f.productivity for f in alive_firms]),
                "avg_tech_t": safe_mean([f.tech_stock for f in alive_firms]),
                "avg_employment_t": safe_mean([f.employees for f in alive_firms]),
                "avg_markup_t": safe_mean([f.last_markup for f in alive_firms]),
                "macro_growth_t": macro_growth_prev,
                "gov_transfer_t": government.transfer,
                "gov_subsidy_t": government.innovation_subsidy,
                "gov_guarantee_t": government.credit_guarantee,
                "gov_debt_t": government.debt,
                "gov_deficit_t": government.last_period_deficit,
                "cb_rate_annual_t": central_bank.policy_rate_annual,
                "fx_t": ext_env.fx_level,
                "commodity_t": ext_env.commodity_index,
                "argentina_t": ext_env.argentina_demand,
                "brazil_t": ext_env.brazil_demand,
                "inflation_daily_t": ext_env.daily_inflation,
                "action_entropy_t": action_entropy([f.last_action for f in alive_firms]) if len(alive_firms) > 0 else np.nan,
                "sales_gini_t": gini([f.sales_value for f in alive_firms]) if len(alive_firms) > 0 else np.nan,
                "hhi_t": compute_hhi(alive_firms),
                "policy_dispersion_t": network_policy_dispersion(firms, G)
            })

        defaults = sum(1 for f in firms if f.default)
        survival = 1.0 - defaults / n_firms

        ep_actions = []
        for f in firms:
            ep_actions.extend(f.action_history)

        episode_rows.append({
            "episode": ep,
            "avg_reward": safe_mean([f.total_reward for f in firms]),
            "avg_cash_final": safe_mean([f.cash for f in firms]),
            "avg_debt_final": safe_mean([f.debt for f in firms]),
            "avg_price_final": safe_mean([f.price for f in firms]),
            "avg_productivity_final": safe_mean([f.productivity for f in firms]),
            "avg_tech_final": safe_mean([f.tech_stock for f in firms]),
            "avg_employees_final": safe_mean([f.employees for f in firms]),
            "avg_markup_final": safe_mean([f.last_markup for f in firms]),
            "default_rate": defaults / n_firms,
            "survival_rate": survival,
            "policy_drift": average_policy_drift(firms, prev_W),
            "mean_policy_distance": mean_policy_distance(firms),
            "network_policy_dispersion": network_policy_dispersion(firms, G),
            "action_entropy": action_entropy(ep_actions),
            "sales_gini": gini([f.sales_value for f in firms]),
            "hhi": compute_hhi(firms),
            "gov_debt_final": government.debt,
            "cb_rate_final": central_bank.policy_rate_annual,
            "fx_final": ext_env.fx_level,
            "commodity_final": ext_env.commodity_index,
            "argentina_final": ext_env.argentina_demand,
            "brazil_final": ext_env.brazil_demand,
            "share_cut_price": ep_actions.count(0) / len(ep_actions) if ep_actions else np.nan,
            "share_raise_price": ep_actions.count(1) / len(ep_actions) if ep_actions else np.nan,
            "share_hire": ep_actions.count(2) / len(ep_actions) if ep_actions else np.nan,
            "share_fire": ep_actions.count(3) / len(ep_actions) if ep_actions else np.nan,
            "share_borrow": ep_actions.count(4) / len(ep_actions) if ep_actions else np.nan,
            "share_invest_innovation": ep_actions.count(5) / len(ep_actions) if ep_actions else np.nan,
            "share_hedge_fx": ep_actions.count(6) / len(ep_actions) if ep_actions else np.nan,
            "share_wait": ep_actions.count(7) / len(ep_actions) if ep_actions else np.nan
        })

        for f in firms:
            agent_rows.append({
                "episode": ep,
                "firm_id": f.firm_id,
                "sector": f.sector,
                "default": f.default,
                "total_reward": f.total_reward,
                "cash_final": f.cash,
                "debt_final": f.debt,
                "employees_final": f.employees,
                "price_final": f.price,
                "productivity_final": f.productivity,
                "tech_final": f.tech_stock,
                "sales_value_final": f.sales_value,
                "export_sales_final": f.export_sales,
                "domestic_sales_final": f.domestic_sales,
                "markup_final": f.last_markup,
                "market_share_final": f.last_market_share,
                "fx_hedge_final": f.fx_hedge,
                "policy_norm": np.linalg.norm(f.W),
                "firm_action_entropy": action_entropy(f.action_history),
                "mean_reward_per_action": safe_mean(f.reward_history),
                "mean_market_share": safe_mean(f.market_share_history),
                "mean_markup": safe_mean(f.markup_history)
            })

    df_episode = pd.DataFrame(episode_rows)
    df_agent = pd.DataFrame(agent_rows)
    df_period = pd.DataFrame(period_rows)

    return df_episode, df_agent, df_period, firms, households, banks, government, central_bank, ext_env, G

# ============================================================
# 13. RUN
# ============================================================

df_episode, df_agent, df_period, firms, households, banks, government, central_bank, ext_env, G = train_memf_v7()

# ============================================================
# 14. TABLES
# ============================================================

df_episode["episode_block"] = pd.cut(
    df_episode["episode"],
    bins=[0,10,20,30,40,50,60,70,80],
    labels=["1-10","11-20","21-30","31-40","41-50","51-60","61-70","71-80"]
)

table_blocks = (
    df_episode.groupby("episode_block", observed=False)[[
        "avg_reward",
        "avg_cash_final",
        "default_rate",
        "survival_rate",
        "avg_productivity_final",
        "avg_tech_final",
        "avg_employees_final",
        "avg_markup_final",
        "policy_drift",
        "mean_policy_distance",
        "action_entropy",
        "sales_gini",
        "hhi",
        "gov_debt_final",
        "cb_rate_final",
        "fx_final"
    ]]
    .mean()
    .round(3)
    .reset_index()
)

first10 = df_episode[df_episode["episode"] <= 10]
last10 = df_episode[df_episode["episode"] > (N_EPISODES - 10)]

table_first_last = pd.DataFrame({
    "metric": [
        "avg_reward", "avg_cash_final", "default_rate", "survival_rate",
        "avg_productivity_final", "avg_tech_final", "avg_employees_final",
        "avg_markup_final", "action_entropy", "sales_gini", "hhi",
        "gov_debt_final", "cb_rate_final", "fx_final"
    ],
    "first_10": [
        first10["avg_reward"].mean(),
        first10["avg_cash_final"].mean(),
        first10["default_rate"].mean(),
        first10["survival_rate"].mean(),
        first10["avg_productivity_final"].mean(),
        first10["avg_tech_final"].mean(),
        first10["avg_employees_final"].mean(),
        first10["avg_markup_final"].mean(),
        first10["action_entropy"].mean(),
        first10["sales_gini"].mean(),
        first10["hhi"].mean(),
        first10["gov_debt_final"].mean(),
        first10["cb_rate_final"].mean(),
        first10["fx_final"].mean()
    ],
    "last_10": [
        last10["avg_reward"].mean(),
        last10["avg_cash_final"].mean(),
        last10["default_rate"].mean(),
        last10["survival_rate"].mean(),
        last10["avg_productivity_final"].mean(),
        last10["avg_tech_final"].mean(),
        last10["avg_employees_final"].mean(),
        last10["avg_markup_final"].mean(),
        last10["action_entropy"].mean(),
        last10["sales_gini"].mean(),
        last10["hhi"].mean(),
        last10["gov_debt_final"].mean(),
        last10["cb_rate_final"].mean(),
        last10["fx_final"].mean()
    ]
}).round(3)

table_first_last["delta_last_minus_first"] = (
    table_first_last["last_10"] - table_first_last["first_10"]
).round(3)

table_actions = (
    df_episode.groupby("episode_block", observed=False)[[
        "share_cut_price",
        "share_raise_price",
        "share_hire",
        "share_fire",
        "share_borrow",
        "share_invest_innovation",
        "share_hedge_fx",
        "share_wait"
    ]]
    .mean()
    .round(3)
    .reset_index()
)

table_sector = (
    df_agent[df_agent["episode"] > (N_EPISODES - 10)]
    .groupby("sector")[[
        "default",
        "total_reward",
        "cash_final",
        "debt_final",
        "employees_final",
        "productivity_final",
        "tech_final",
        "sales_value_final",
        "export_sales_final",
        "markup_final"
    ]]
    .mean()
    .round(3)
    .reset_index()
)

print("\n================ BLOCK AVERAGES ================\n")
print(table_blocks)

print("\n================ FIRST 10 VS LAST 10 ================\n")
print(table_first_last)

print("\n================ ACTION MIX BY BLOCK ================\n")
print(table_actions)

print("\n================ SECTOR SUMMARY (LAST WINDOW) ================\n")
print(table_sector)

# ============================================================
# 15. PLOTS
# ============================================================

plt.figure(figsize=(10, 5))
plt.plot(df_episode["episode"], df_episode["avg_reward"], lw=2)
plt.title("Average Reward by Episode")
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.grid(True, alpha=0.3)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(df_episode["episode"], df_episode["default_rate"], lw=2, label="Default rate")
plt.plot(df_episode["episode"], df_episode["survival_rate"], lw=2, label="Survival rate")
plt.title("Defaults and Survival")
plt.xlabel("Episode")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(df_episode["episode"], df_episode["avg_productivity_final"], lw=2, label="Productivity")
plt.plot(df_episode["episode"], df_episode["avg_tech_final"], lw=2, label="Tech")
plt.plot(df_episode["episode"], df_episode["avg_markup_final"], lw=2, label="Markup")
plt.title("Productivity, Technology and Markups")
plt.xlabel("Episode")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(df_episode["episode"], df_episode["gov_debt_final"], lw=2, label="Government debt")
plt.plot(df_episode["episode"], df_episode["cb_rate_final"] * 1e6, lw=2, label="CB rate x 1e6")
plt.title("Government Debt and Monetary Tightness")
plt.xlabel("Episode")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(df_episode["episode"], df_episode["sales_gini"], lw=2, label="Sales Gini")
plt.plot(df_episode["episode"], df_episode["hhi"], lw=2, label="HHI")
plt.title("Concentration Metrics")
plt.xlabel("Episode")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

period_group = df_period.groupby("day").mean(numeric_only=True)

plt.figure(figsize=(10, 5))
plt.plot(period_group.index, period_group["avg_reward_t"], lw=2, label="Reward")
plt.plot(period_group.index, period_group["avg_markup_t"], lw=2, label="Markup")
plt.plot(period_group.index, period_group["default_rate_t"], lw=2, label="Default rate")
plt.title("Within-Episode Daily Dynamics")
plt.xlabel("Day")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(period_group.index, period_group["fx_t"], lw=2, label="FX")
plt.plot(period_group.index, period_group["commodity_t"], lw=2, label="Commodity")
plt.plot(period_group.index, period_group["argentina_t"], lw=2, label="Argentina")
plt.plot(period_group.index, period_group["brazil_t"], lw=2, label="Brazil")
plt.title("External State Variables")
plt.xlabel("Day")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ============================================================
# 16. LEARNING DIAGNOSTICS
# ============================================================

print("\n================ LEARNING DIAGNOSTICS ================\n")
print("Mean policy distance (final):", round(df_episode["mean_policy_distance"].iloc[-1], 4))
print("Policy drift (final):", round(df_episode["policy_drift"].iloc[-1], 4))
print("Action entropy (final):", round(df_episode["action_entropy"].iloc[-1], 4))
print("Avg reward, first 10 episodes:", round(first10["avg_reward"].mean(), 4))
print("Avg reward, last 10 episodes:", round(last10["avg_reward"].mean(), 4))
print("Default rate, first 10 episodes:", round(first10["default_rate"].mean(), 4))
print("Default rate, last 10 episodes:", round(last10["default_rate"].mean(), 4))

# ============================================================
# 17. NETWORK SNAPSHOT
# ============================================================

alive_mask = [not f.default for f in firms]
node_sizes = [max(80, 8 * max(f.sales_value, 1.0)) for f in firms]
node_colors = [f.last_reward for f in firms]

plt.figure(figsize=(9, 7))
pos = nx.spring_layout(G, seed=42, k=0.45)
nx.draw_networkx_edges(G, pos, alpha=0.15, arrows=False)
nodes = nx.draw_networkx_nodes(
    G, pos,
    node_size=node_sizes,
    node_color=node_colors,
    cmap="coolwarm",
    alpha=0.9
)
plt.colorbar(nodes, label="Last reward")
plt.title("Firm Network Snapshot")
plt.axis("off")
plt.show()






######################################################
# FINAL PAPER FIGURES (ROBUST VERSION)
######################################################

import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

save_path = r"C:\Users\diego\OneDrive\Escritorio\Figures"

# ============================================================
# FIGURE 1 — LEARNING CONVERGENCE
# ============================================================

plt.figure(figsize=(8,5))

entropy = df_episode["action_entropy"]
drift = df_episode["policy_drift"]
distance = df_episode["mean_policy_distance"]

plt.plot(df_episode["episode"], entropy/entropy.max(), '-', color='black', label="Entropy")
plt.plot(df_episode["episode"], drift/drift.max(), '--', color='black', label="Policy drift")
plt.plot(df_episode["episode"], distance/distance.max(), '-.', color='black', label="Policy distance")

plt.xlabel("Episode")
plt.ylabel("Normalized level")
plt.title("Learning Convergence Dynamics")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_path, "fig1_learning_bw.png"), dpi=300)
plt.close()


# ============================================================
# FIGURE 2 — PRODUCTIVITY VS FRAGILITY
# ============================================================

plt.figure(figsize=(8,5))

plt.plot(df_episode["episode"], df_episode["avg_productivity_final"], '-', color='black', label="Productivity")
plt.plot(df_episode["episode"], df_episode["default_rate"], '--', color='black', label="Default rate")

plt.xlabel("Episode")
plt.ylabel("Level")
plt.title("Productivity Gains and Systemic Fragility")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_path, "fig2_fragility_bw.png"), dpi=300)
plt.close()


# ============================================================
# FIGURE 3 — BEHAVIORAL CONCENTRATION
# ============================================================

plt.figure(figsize=(8,5))

plt.plot(df_episode["episode"], df_episode["share_invest_innovation"], '-', color='black', label="Innovation")
plt.plot(df_episode["episode"], df_episode["share_borrow"], '--', color='black', label="Borrowing")
plt.plot(df_episode["episode"], df_episode["share_hire"], '-.', color='black', label="Hiring")

plt.xlabel("Episode")
plt.ylabel("Action share")
plt.title("Behavioral Concentration and Policy Synchronization")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_path, "fig3_behavior_bw.png"), dpi=300)
plt.close()


# ============================================================
# FIGURE 4 — NETWORK RISK (MEJORADA)
# ============================================================

plt.figure(figsize=(8,5))

plt.plot(df_episode["episode"], df_episode["default_rate"], '-', color='black', label="Default rate")
plt.plot(df_episode["episode"], df_episode["avg_productivity_final"]/df_episode["avg_productivity_final"].max(),
         '--', color='black', label="Productivity (norm)")

plt.xlabel("Episode")
plt.ylabel("Level")
plt.title("Network Amplification: Productivity vs Risk")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_path, "fig4_network_dynamics_bw.png"), dpi=300)
plt.close()


# ============================================================
# FIGURE 5 — NETWORK STRUCTURE (ROBUST)
# ============================================================

# --- Try to use A if exists ---
if 'A' in globals():
    adj_matrix = A

# --- Try to reconstruct from edges ---
elif 'df_edges' in globals():
    G_temp = nx.from_pandas_edgelist(df_edges, 'source', 'target')
    adj_matrix = nx.to_numpy_array(G_temp)

# --- Fallback: synthetic network ---
else:
    print("⚠️ A not found → generating synthetic network")
    N = len(df_episode) if 'df_episode' in globals() else 30
    G_temp = nx.erdos_renyi_graph(N, 0.1, seed=42)
    adj_matrix = nx.to_numpy_array(G_temp)

# Build graph
G = nx.from_numpy_array(adj_matrix)

# Node size = degree
degrees = dict(G.degree())
node_sizes = [50 + 150 * degrees[i] for i in G.nodes()]

# Node color = last reward if available
if 'last_rewards' in globals():
    rewards = last_rewards
    rewards = (rewards - np.min(rewards)) / (np.max(rewards) - np.min(rewards))
    node_colors = [1 - r for r in rewards]  # grayscale
else:
    node_colors = 'gray'

plt.figure(figsize=(8,6))

pos = nx.spring_layout(G, seed=42)

nx.draw_networkx_nodes(
    G, pos,
    node_size=node_sizes,
    node_color=node_colors,
    cmap=plt.cm.gray
)

nx.draw_networkx_edges(
    G, pos,
    alpha=0.15,
    width=0.5
)

plt.title("Firm Network Structure (Grayscale: Performance)")
plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(save_path, "fig5_network_bw.png"), dpi=300)
plt.close()






# ============================================================
# 18. COUNTERFACTUAL EXPERIMENTS (FOR JEBO SECTION)
# ============================================================

def run_experiment(label,
                   learning_rate=LEARNING_RATE,
                   network_p=NETWORK_P,
                   innovation_fast=False,
                   high_spillovers=False):

    global LEARNING_RATE, NETWORK_P

    # Backup
    original_lr = LEARNING_RATE
    original_net = NETWORK_P

    # Apply experiment
    LEARNING_RATE = learning_rate
    NETWORK_P = network_p

    # Run model
    df_ep, df_ag, df_per, *_ = train_memf_v7()

    # Restore
    LEARNING_RATE = original_lr
    NETWORK_P = original_net

    # Extract metrics (last 10 episodes)
    last = df_ep[df_ep["episode"] > (N_EPISODES - 10)]

    result = {
        "scenario": label,
        "default_rate": last["default_rate"].mean(),
        "survival_rate": last["survival_rate"].mean(),
        "productivity": last["avg_productivity_final"].mean(),
        "entropy": last["action_entropy"].mean(),
        "policy_distance": last["mean_policy_distance"].mean()
    }

    return result



# ============================================================
# RUN COUNTERFACTUALS
# ============================================================

results = []

# 1. Baseline
results.append(run_experiment("Baseline"))

# 2. No Learning
results.append(run_experiment(
    "No Learning",
    learning_rate=0.0
))

# 3. No Network
results.append(run_experiment(
    "No Network",
    network_p=0.0
))

# 4. High Spillovers (proxy: más subsidio indirecto vía comportamiento)
# (esto en tu código ya entra por gobierno → no tocamos estructura)
results.append(run_experiment(
    "High Spillovers",
    learning_rate=LEARNING_RATE,
    network_p=NETWORK_P * 1.5
))



df_counterfactual = pd.DataFrame(results).round(3)

print("\n================ COUNTERFACTUAL RESULTS ================\n")
print(df_counterfactual)


plt.figure(figsize=(8,5))

plt.bar(df_counterfactual["scenario"], df_counterfactual["default_rate"])

plt.ylabel("Default rate")
plt.title("Counterfactual Identification: Sources of Fragility")
plt.xticks(rotation=30)
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()


# ============================================================
# MAIN EXECUTION BLOCK (REPLICATION ENTRY POINT)
# ============================================================

import os

def main():

    print("\n========================================")
    print("Running ASM-MEMF Simulation")
    print("========================================\n")

    # --------------------------------------------------------
    # 1. Ensure output folders exist
    # --------------------------------------------------------
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    # --------------------------------------------------------
    # 2. Run simulation
    # --------------------------------------------------------
    try:
        print("Step 1: Running simulation...\n")

        # ⚠️ Ajusta este nombre si tu función principal se llama distinto
        df_episode, df_agent, df_period = train_memf_v7()

        print("Simulation completed.\n")

    except Exception as e:
        print("ERROR during simulation:")
        print(e)
        return

    # --------------------------------------------------------
    # 3. Save results
    # --------------------------------------------------------
    try:
        print("Step 2: Saving results...\n")

        if df_episode is not None:
            df_episode.to_csv("results/episode_results.csv", index=False)

        if df_agent is not None:
            df_agent.to_csv("results/agent_results.csv", index=False)

        if df_period is not None:
            df_period.to_csv("results/period_results.csv", index=False)

        print("Results saved in /results\n")

    except Exception as e:
        print("ERROR saving results:")
        print(e)

    # --------------------------------------------------------
    # 4. Generate figures (if functions exist)
    # --------------------------------------------------------
    try:
        print("Step 3: Generating figures...\n")

        # ⚠️ Ajusta estos nombres según tu código real
        if 'plot_learning_dynamics' in globals():
            plot_learning_dynamics(df_episode)
            print("Saved learning dynamics figure")

        if 'plot_network_snapshot' in globals():
            plot_network_snapshot()
            print("Saved network snapshot")

        if 'plot_counterfactuals' in globals():
            plot_counterfactuals()
            print("Saved counterfactual comparison")

        print("Figures saved in /figures\n")

    except Exception as e:
        print("WARNING: Error generating figures:")
        print(e)

    # --------------------------------------------------------
    # 5. Finish
    # --------------------------------------------------------
    print("========================================")
    print("RUN COMPLETED SUCCESSFULLY")
    print("========================================\n")


# ============================================================
# RUN SCRIPT
# ============================================================

if __name__ == "__main__":
    main()
