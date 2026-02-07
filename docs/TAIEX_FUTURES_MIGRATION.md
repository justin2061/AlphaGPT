# AlphaGPT 移植台指期 (TAIEX Futures) 腦力激盪

## 一、核心差異對照表

| 維度 | 現有系統 (Solana Meme) | 台指期目標系統 |
|------|----------------------|-------------|
| **標的** | 數百種 Meme 代幣 (多標的) | 台指期 TX / 小台 MTX (單一或少數標的) |
| **方向** | 僅做多 (Long-only) | 多空皆可 (Long / Short) |
| **槓桿** | 無槓桿 (現貨) | 保證金制度，約 7~12 倍槓桿 |
| **交易時段** | 24/7 全天候 | 日盤 8:45-13:45 / 夜盤 15:00-05:00 |
| **結算** | 即時鏈上結算 | 每日結算 + 每月第三週三結算日 |
| **手續費** | DEX 費用 ~0.6% + 滑點 | 期交稅萬分之二 + 券商手續費 (~$30-60/口) |
| **資料來源** | Birdeye API / DexScreener | Shioaji / 元大 API / Yahoo Finance / TAIFEX |
| **執行** | Solana 鏈上交易 (Jupiter) | 券商 API (Shioaji/元大/凱基) |
| **流動性風控** | 蜜罐偵測、最低流動性 | 漲跌停限制、保證金追繳、跳空風險 |
| **資料頻率** | 1 分鐘 K 線 | 1 分鐘 / 5 分鐘 / Tick-by-Tick |

---

## 二、可直接復用的核心模組 (約 70%)

### 2.1 Transformer 公式生成引擎 — 完全復用

`model_core/alphagpt.py` 的核心架構幾乎不需要修改：

- **LoopedTransformer**: 公式生成的 Transformer 結構與標的無關
- **MTPHead**: 多任務學習頭可直接復用
- **LoRD 正規化**: Newton-Schulz 低秩衰減與市場無關
- **StableRankMonitor**: 穩定秩監控可直接復用

唯一需要調整的是 `vocab_size`（因為因子數量和運算子可能改變）。

### 2.2 StackVM 虛擬機 — 完全復用

`model_core/vm.py` 的堆疊式虛擬機是通用的公式執行引擎，只要更新 `feat_offset` 對應新的因子數量即可。

### 2.3 REINFORCE 訓練引擎 — 大致復用

`model_core/engine.py` 的策略梯度訓練邏輯可直接復用，只需替換：
- `CryptoDataLoader` → `TaiexDataLoader`
- `MemeBacktest` → `FuturesBacktest`

### 2.4 Dashboard — 大致復用

Streamlit 監控面板架構可復用，需調整顯示內容（保證金、多空方向等）。

---

## 三、需要重新設計的模組

### 3.1 因子工程 — 重大改造

**現有 6 因子（Meme 專用）:**

```
ret       → 對數報酬率          ✅ 保留
liq_score → 流動性/FDV 健康度    ❌ 移除 (期貨不適用)
pressure  → 買賣壓力            ✅ 改造
fomo      → 成交量加速度         ✅ 保留
dev       → 偏離均線             ✅ 保留
log_vol   → 對數成交量           ✅ 保留
```

**台指期建議新增因子 (12~16 維):**

```python
# === 價格動量類 ===
ret_1m      # 1 分鐘對數報酬率
ret_5m      # 5 分鐘報酬率
ret_30m     # 30 分鐘報酬率 (趨勢)
overnight   # 隔夜跳空 = 今開 - 昨收

# === 波動率類 ===
vol_cluster # 波動率聚集 (GARCH-like)
atr_ratio   # ATR / Close，標準化真實波幅
intraday_rng# (High - Low) / Close 日內振幅

# === 量能類 ===
log_vol     # 對數成交量
vol_accel   # 成交量加速度 (FOMO 的期貨版)
vol_trend   # 量比 = 當前量 / N期均量

# === 期貨專屬類 ===
oi_chg      # 未平倉量變化率 (Open Interest Change)
basis       # 基差 = 期貨價 - 現貨價 (台指現貨 vs 期貨)
settlement  # 距結算日天數 (歸一化 0~1)
put_call    # 選擇權 Put/Call Ratio (若可取得)

# === 技術指標類 ===
rsi         # RSI(14)
pressure    # K 棒實體佔比 = (Close-Open)/(High-Low)
close_pos   # 收盤位置 = (Close-Low)/(High-Low)
```

**實作範例：**

```python
class TaiexFactorEngineer:
    INPUT_DIM = 14  # 14 維因子

    @staticmethod
    def compute_features(raw_dict):
        c = raw_dict['close']
        o = raw_dict['open']
        h = raw_dict['high']
        l = raw_dict['low']
        v = raw_dict['volume']
        oi = raw_dict['open_interest']    # 新增
        spot = raw_dict['spot_index']     # 新增：台指現貨

        # 多時間尺度報酬率
        ret_1 = torch.log(c / (torch.roll(c, 1, dims=1) + 1e-9))
        ret_5 = torch.log(c / (torch.roll(c, 5, dims=1) + 1e-9))
        ret_30 = torch.log(c / (torch.roll(c, 30, dims=1) + 1e-9))

        # 隔夜跳空 (需日頻對齊)
        overnight = o - torch.roll(c, 1, dims=1)

        # 波動率聚集
        vol_cluster = MemeIndicators.volatility_clustering(c)

        # ATR
        atr = compute_atr(h, l, c, window=14)
        atr_ratio = atr / (c + 1e-9)

        # 量能
        log_vol = torch.log1p(v)
        vol_accel = MemeIndicators.fomo_acceleration(v)

        # 期貨專屬
        oi_chg = (oi - torch.roll(oi, 1, dims=1)) / (torch.roll(oi, 1, dims=1) + 1e-9)
        basis = (c - spot) / (spot + 1e-9)  # 基差率

        # 技術指標
        rsi = MemeIndicators.relative_strength(c, h, l)
        pressure = MemeIndicators.buy_sell_imbalance(c, o, h, l)
        close_pos = (c - l) / (h - l + 1e-9)

        features = torch.stack([
            robust_norm(ret_1),
            robust_norm(ret_5),
            robust_norm(ret_30),
            robust_norm(overnight),
            robust_norm(vol_cluster),
            robust_norm(atr_ratio),
            robust_norm(log_vol),
            robust_norm(vol_accel),
            robust_norm(oi_chg),
            robust_norm(basis),
            robust_norm(rsi),
            pressure,
            close_pos,
            robust_norm(oi_chg),
        ], dim=1)

        return features
```

### 3.2 運算子擴展

現有 12 個運算子可完全保留，建議新增期貨適用運算子：

```python
# 新增運算子
TAIEX_OPS_EXTRA = [
    # 時間序列
    ('DELTA5',  lambda x: x - _ts_delay(x, 5), 1),        # 5 期差分
    ('MA20',    lambda x: _ts_decay_linear(x, 20), 1),     # 20 期加權均線
    ('STD20',   lambda x: _ts_zscore(x, 20), 1),           # 20 期 Z-Score
    ('RANK20',  lambda x: _ts_rank(x, 20), 1),             # 20 期排名
    ('CORR',    lambda x, y: _ts_corr(x, y, 20), 2),       # 滾動相關

    # 條件邏輯 (已有 GATE)
    ('CLAMP',   lambda x: torch.clamp(x, -3, 3), 1),       # 截斷極端值
    ('SMOOTH',  lambda x: _ts_decay_linear(x, 5), 1),      # 平滑
]
```

> **注意**：`times.py` 中已有 `DELTA5`, `MA20`, `STD20` 等運算子的實現，可直接參考移植。

### 3.3 回測引擎 — 重大改造

**現有 `MemeBacktest` 的問題：**
- 僅支持做多 (position > 0)
- 費率模型為 DEX 滑點 (0.6%)
- 不考慮保證金和槓桿

**台指期回測需求：**

```python
class FuturesBacktest:
    def __init__(self):
        self.point_value = 200          # 大台每點 200 元
        self.mini_point_value = 50      # 小台每點 50 元
        self.margin_per_lot = 184_000   # 大台原始保證金 (2024/2025)
        self.commission = 60            # 來回手續費 (券商)
        self.tax_rate = 0.00002         # 期交稅 (萬分之二)
        self.slippage_ticks = 1         # 滑點 1 跳 (1 點)

    def evaluate(self, factors, raw_data, target_ret):
        signal = torch.tanh(factors)  # tanh → [-1, 1] 支持多空

        # 倉位：+1 做多, -1 做空, 0 空倉
        position = torch.sign(signal)

        # 換手成本
        turnover = torch.abs(position - torch.roll(position, 1, dims=1))
        turnover[:, 0] = 0

        # 單邊成本 = 手續費 + 稅 + 滑點
        cost_per_point = (
            self.commission / self.point_value +
            self.slippage_ticks
        )
        cost_rate = cost_per_point / raw_data['close'].mean() * 2  # 歸一化

        # 淨 PnL
        gross_pnl = position * target_ret
        net_pnl = gross_pnl - turnover * cost_rate

        # Sharpe-like 評分
        cum_ret = net_pnl.sum(dim=1)
        mu = net_pnl.mean(dim=1)
        std = net_pnl.std(dim=1) + 1e-6
        sharpe = mu / std * (252 ** 0.5)

        # 最大回撤懲罰
        cum_curve = net_pnl.cumsum(dim=1)
        running_max = torch.cummax(cum_curve, dim=1)[0]
        drawdown = running_max - cum_curve
        max_dd = drawdown.max(dim=1)[0]

        # 過度交易懲罰
        avg_turnover = turnover.mean(dim=1)
        turnover_penalty = torch.where(avg_turnover > 0.5, avg_turnover * 2, torch.zeros_like(avg_turnover))

        score = sharpe - max_dd * 2 - turnover_penalty

        # 最低活動度
        activity = (position != 0).float().sum(dim=1)
        score = torch.where(activity < 10, torch.tensor(-10.0), score)

        return torch.median(score), cum_ret.mean().item()
```

### 3.4 資料管道 — 完全重寫

**方案 A：Shioaji (永豐金證券 API) — 推薦**

```python
# 優點：免費、穩定、支持即時 + 歷史、Python 原生
import shioaji as sj

class TaiexDataLoader:
    def __init__(self):
        self.api = sj.Shioaji()
        self.api.login(api_key="...", secret_key="...")

    def fetch_historical(self, contract, start, end):
        """取得歷史 K 線"""
        kbars = self.api.kbars(
            contract=self.api.Contracts.Futures.TXF.TXFR1,  # 近月台指期
            start=start,
            end=end
        )
        df = pd.DataFrame({**kbars})
        df.ts = pd.to_datetime(df.ts)
        return df  # columns: ts, Open, High, Low, Close, Volume

    def subscribe_realtime(self, callback):
        """即時報價訂閱"""
        self.api.quote.subscribe(
            self.api.Contracts.Futures.TXF.TXFR1,
            quote_type=sj.constant.QuoteType.Tick
        )
```

**方案 B：Yahoo Finance (免費歷史資料)**

```python
import yfinance as yf

# 台指期 (近月連續)
df = yf.download("^TWII", start="2020-01-01")  # 台灣加權指數
# 或使用 TAIFEX 開放資料
```

**方案 C：TAIFEX 官方開放資料**

```
https://www.taifex.com.tw/cht/3/futContractsDate  # 每日行情
https://www.taifex.com.tw/cht/3/futDataDown        # 下載專區
```

**額外資料源（因子用）：**

| 資料 | 來源 | 用途 |
|------|------|------|
| 台指現貨 (TWII) | Yahoo / TWSE | 計算基差 |
| 未平倉量 (OI) | TAIFEX 官網 / Shioaji | OI 變化因子 |
| 三大法人買賣超 | TWSE 開放資料 | 籌碼因子 |
| Put/Call Ratio | TAIFEX | 選擇權情緒 |
| VIX (台指選 VIX) | TAIFEX | 波動率指標 |

### 3.5 執行層 — 完全重寫

**現有：Solana RPC + Jupiter DEX → 替換為：券商 API**

```python
class TaiexTrader:
    def __init__(self):
        self.api = sj.Shioaji()
        self.api.login(api_key="...", secret_key="...")
        self.contract = self.api.Contracts.Futures.TXF.TXFR1  # 近月

    async def buy(self, lots=1):
        """做多下單"""
        order = self.api.Order(
            action=sj.constant.Action.Buy,
            price=0,  # 市價
            quantity=lots,
            price_type=sj.constant.FuturesPriceType.MKT,
            order_type=sj.constant.OrderType.IOC,
        )
        trade = self.api.place_order(self.contract, order)
        return trade

    async def sell(self, lots=1):
        """做空下單"""
        order = self.api.Order(
            action=sj.constant.Action.Sell,
            price=0,
            quantity=lots,
            price_type=sj.constant.FuturesPriceType.MKT,
            order_type=sj.constant.OrderType.IOC,
        )
        trade = self.api.place_order(self.contract, order)
        return trade

    async def close_position(self, direction, lots=1):
        """平倉"""
        action = sj.constant.Action.Sell if direction == "long" else sj.constant.Action.Buy
        order = self.api.Order(
            action=action,
            price=0,
            quantity=lots,
            price_type=sj.constant.FuturesPriceType.MKT,
            order_type=sj.constant.OrderType.IOC,
        )
        return self.api.place_order(self.contract, order)

    def get_margin_status(self):
        """查詢保證金餘額"""
        return self.api.margin()

    def get_positions(self):
        """查詢持倉"""
        return self.api.list_positions(self.api.futopt_account)
```

### 3.6 策略管理器 — 中度改造

**關鍵差異：**

| 項目 | Meme 版 | 台指期版 |
|------|--------|---------|
| 掃描標的 | 300 種代幣 | 1 種期貨 (大台/小台) |
| 倉位方向 | 僅多 | 多空皆可 |
| 最大倉位 | 3 個幣 | N 口合約 (依保證金) |
| 出場信號 | 停損 -5% / 止盈 +10% | 可能更小，例如 -2% / +3% |
| 循環週期 | 60 秒 | 1~5 分鐘 (或 Tick 驅動) |

```python
class TaiexStrategyRunner:
    def __init__(self):
        self.vm = StackVM()
        self.trader = TaiexTrader()
        self.current_direction = 0  # +1=多, -1=空, 0=空倉
        self.entry_price = 0
        self.lots = 0

    async def run_loop(self):
        while self.is_trading_hours():
            # 1. 取得最新資料
            features = self.compute_latest_features()

            # 2. 執行公式
            signal = self.vm.execute(self.formula, features)
            if signal is None:
                continue

            score = float(torch.tanh(signal[0, -1]).item())  # [-1, 1]

            # 3. 產生交易信號
            desired_direction = 0
            if score > 0.3:
                desired_direction = 1   # 做多
            elif score < -0.3:
                desired_direction = -1  # 做空

            # 4. 執行交易
            if desired_direction != self.current_direction:
                await self.change_position(desired_direction)

            await asyncio.sleep(60)

    def is_trading_hours(self):
        """判斷是否在交易時段"""
        now = datetime.now(tz=timezone(timedelta(hours=8)))
        t = now.time()
        # 日盤 8:45~13:45 或 夜盤 15:00~05:00
        day_session = time(8, 45) <= t <= time(13, 45)
        night_session = t >= time(15, 0) or t <= time(5, 0)
        return day_session or night_session
```

### 3.7 風控引擎 — 中度改造

```python
class TaiexRiskEngine:
    def __init__(self):
        self.max_lots = 5                   # 最大口數
        self.margin_per_lot = 184_000       # 原始保證金/口
        self.maintenance_margin = 141_000   # 維持保證金/口
        self.max_daily_loss = 50_000        # 每日最大虧損
        self.max_drawdown_pct = 0.10        # 最大回撤 10%

    def calculate_position_size(self, account_equity):
        """依資金計算可開口數"""
        max_by_margin = int(account_equity * 0.5 / self.margin_per_lot)
        return min(max_by_margin, self.max_lots)

    def check_margin_call(self, equity, margin_used):
        """保證金追繳檢查"""
        ratio = equity / (margin_used + 1e-9)
        if ratio < 1.0:  # 低於維持保證金
            return "MARGIN_CALL"
        if ratio < 1.25:
            return "WARNING"
        return "OK"

    def check_daily_limit(self, daily_pnl):
        """每日虧損上限"""
        if daily_pnl < -self.max_daily_loss:
            return False  # 停止交易
        return True

    def check_limit_move(self, price, prev_close):
        """漲跌停檢查 (期貨漲跌幅限制 ±10%)"""
        change_pct = abs(price - prev_close) / prev_close
        if change_pct > 0.09:  # 接近漲跌停
            return False
        return True
```

---

## 四、新增架構圖

```
┌─────────────────────────────────────────────────────────┐
│              Dashboard (Streamlit 監控面板)                │
│   [保證金餘額] [持倉方向] [即時PnL] [日績效] [緊急平倉]      │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│            TaiexStrategyRunner (策略管理器)                │
│  載入最佳公式 → StackVM 執行 → 多空信號 → 風控 → 下單       │
│  [交易時段控制] [信號平滑] [倉位管理]                       │
└───┬────────────────┬──────────────┬─────────────────────┘
    │                │              │
┌───▼──────┐  ┌──────▼─────┐  ┌────▼──────────┐
│Model Core│  │TaiexData   │  │TaiexTrader    │
│(AI 公式  │  │Loader      │  │(Shioaji API)  │
│生成引擎) │  │(Shioaji +  │  │  做多/做空     │
│          │  │ TAIFEX)    │  │  平倉/查詢     │
└──────────┘  └──────┬─────┘  └───────────────┘
                     │
              ┌──────▼───────┐     ┌──────────────┐
              │  PostgreSQL  │     │ TaiexRisk    │
              │  台指期 OHLCV │     │ Engine       │
              │  + OI + 基差  │     │ 保證金/漲跌停  │
              └──────────────┘     └──────────────┘
```

---

## 五、移植步驟建議 (Phase Plan)

### Phase 1：離線回測驗證 (2~3 週)

**目標**：確認 AlphaGPT 因子挖掘架構對台指期有效

1. **建立 `TaiexDataLoader`**
   - 從 TAIFEX 或 Yahoo Finance 下載台指期歷史 K 線 (至少 5 年)
   - 儲存至 PostgreSQL（或直接用 Parquet 檔案，參考 `times.py`）
   - 取得未平倉量 (OI)、台指現貨做基差計算

2. **建立 `TaiexFactorEngineer`**
   - 實現 14~16 維因子
   - 先以 `times.py` 的單一時間序列架構為基礎（非多標的）

3. **建立 `FuturesBacktest`**
   - 支持多空
   - 使用 Sortino Ratio 作為獎勵（參考 `times.py` 中的實現）
   - 合理的費率和滑點模型

4. **執行訓練**
   - 用 `AlphaEngine` 搭配新的 DataLoader 和 Backtest
   - 80/20 劃分訓練/測試集
   - 執行 OOS (Out-of-Sample) 回測驗證

> **重要提示**：`times.py` 已經是一個針對中國 A 股的回測原型，其架構 (單標的、Sortino 獎勵、Action Masking) 更適合作為台指期移植的起點，而非多標的的 Meme 版本。

### Phase 2：模擬交易 (2~3 週)

**目標**：驗證即時信號生成和交易邏輯

1. **接入 Shioaji API 模擬帳戶**
2. **建立 `TaiexStrategyRunner`**（事件驅動 / 定時輪詢）
3. **實現交易時段控制和保證金管理**
4. **Dashboard 調整**：顯示保證金、多空方向、即時損益

### Phase 3：實盤小額測試 (持續)

**目標**：用小台 (MTX) 進行實盤驗證

1. **小台期貨**：保證金僅 46,000/口，風險更可控
2. **每日最大虧損限制**
3. **完善監控告警系統**

---

## 六、技術要點與挑戰

### 6.1 單標的 vs 多標的架構轉換

現有系統設計為多標的 (`feat_tensor` shape: `[N_tokens, N_features, T_time]`)，台指期本質上是**單一標的**。

**解法 A — 直接用 `times.py` 的架構**：
- `feat_tensor` shape: `[N_features, T_time]`（2D 而非 3D）
- StackVM 直接操作 1D 時間序列
- 更簡單、更快速

**解法 B — 多合約維度**：
- 將不同月份合約 (近月、次月、季月) 視為不同「標的」
- 或將大台、小台、台指選擇權同時納入
- 保持 3D Tensor 架構

**建議**：Phase 1 先用解法 A（參考 `times.py`），Phase 2 再考慮擴展為多合約。

### 6.2 交易信號從 [0,1] 到 [-1,1]

現有 Meme 版用 `sigmoid` 只產生做多信號 (0~1)，台指期需要：

```python
# Meme 版 (做多)
score = torch.sigmoid(signal)  # [0, 1]
if score > 0.85: buy()

# 台指期版 (多空)
score = torch.tanh(signal)     # [-1, 1]
if score > 0.3: go_long()
elif score < -0.3: go_short()
else: flat()
```

### 6.3 時間序列對齊

期貨有盤前盤後、夜盤等概念，需要處理：

- **隔夜跳空**：夜盤收盤 → 日盤開盤
- **休市期間**：週末、國定假日
- **結算日效應**：每月第三週三（結算日）波動率異常
- **到期轉倉**：近月 → 次月的連續化處理

```python
def continuous_contract(df_near, df_next, rollover_date):
    """合約連續化處理（比率法）"""
    ratio = df_next.loc[rollover_date, 'close'] / df_near.loc[rollover_date, 'close']
    df_near.loc[:rollover_date, ['open','high','low','close']] *= ratio
    return pd.concat([df_near.loc[:rollover_date], df_next.loc[rollover_date:]])
```

### 6.4 保證金動態管理

```python
def dynamic_position_sizing(equity, margin_per_lot, volatility):
    """波動率調整的倉位管理"""
    # Kelly Criterion 簡化版
    base_lots = int(equity * 0.3 / margin_per_lot)

    # 高波動時減倉
    vol_scalar = 1.0 / (1.0 + volatility * 10)
    adjusted_lots = max(1, int(base_lots * vol_scalar))

    return adjusted_lots
```

### 6.5 times.py 作為移植起點

`times.py` 已包含多項可直接用於台指期的元素：

| 元素 | times.py 現有 | 台指期適用性 |
|------|-------------|------------|
| `DataEngine` | Tushare → ETF/指數 | 替換為 TAIFEX/Shioaji |
| `FEATURES` 5 因子 | RET, RET5, VOL_CHG, V_RET, TREND | 可直接擴展 |
| `OPS_CONFIG` 11 運算子 | DELTA5, MA20, STD20 等 | 完全適用 |
| `Action Masking` | 嚴格公式合法性檢查 | 直接復用 |
| `Sortino Reward` | 多空、下行風險 | 直接復用 |
| `OOS Backtest` | 80/20 劃分、Calmar | 直接復用 |
| `decode()` | 公式轉人類可讀 | 直接復用 |

---

## 七、台指期相較 Meme 幣的優勢

1. **雙向交易**：不只做多，放空也能獲利 → 策略空間更大
2. **標準化合約**：無蜜罐風險、無 Rug Pull
3. **低交易成本**：期交稅萬分之二 vs DEX 0.6%
4. **高流動性**：台指期日均量 10~20 萬口，幾乎無滑點
5. **槓桿效應**：用較少資金撬動更大部位
6. **歷史資料豐富**：20+ 年歷史數據可供訓練

---

## 八、預計依賴套件

```txt
# 核心 (不變)
torch>=2.0.0
pandas>=2.0.0
numpy>=1.24.0
sqlalchemy>=2.0.0
loguru>=0.7.0
tqdm

# 台指期專用
shioaji>=1.0.0          # 永豐金 API (推薦)
yfinance>=0.2.0         # 備用歷史資料
requests>=2.31.0        # TAIFEX 公開資料爬取

# 監控
streamlit>=1.28.0
plotly>=5.17.0

# 可選
matplotlib>=3.7.0       # 回測圖表
```

---

## 九、風險提醒

1. **槓桿風險**：台指期自帶槓桿，回撤會被放大
2. **跳空風險**：台股容易受國際盤影響開盤跳空
3. **結算風險**：結算日價格波動大，需特殊處理
4. **過度擬合**：AI 生成的公式可能在歷史數據上表現好但實盤失效
5. **系統風險**：網路中斷、API 故障時需要有緊急處理機制
6. **法規風險**：確認是否符合期交所自動交易相關規範

---

## 十、結論

AlphaGPT 的核心價值——**用 Transformer 生成可解釋交易公式 + StackVM 執行 + REINFORCE 訓練**——完全適用於台指期。約 **70% 的核心程式碼**（模型、VM、訓練引擎）可直接復用，需要重寫的主要是**資料管道**和**執行層**。

**最佳起點是 `times.py`**，因為它已經是單標的、多空、Sortino 獎勵的架構，與台指期的需求更為吻合。從 `times.py` 出發，加入期貨專屬因子（OI、基差、保證金）和 Shioaji 執行層，就是一套完整的台指期自動交易系統。
