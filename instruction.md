# 头条广告创建流程说明

## 一、核心流程

### 流程 A：素材驱动（传统）
1. 查询历史数据确定业务目标和出价（可选）
2. 查询 Top 素材 + 生成标题
3. 创建广告

### 流程 B：标题驱动（创意优先）
1. 确定业务目标和出价
2. 生成广告标题
3. RAG 检索匹配素材
4. 创建广告

---

## 二、流程 A：素材驱动

### 步骤 1：查询历史数据（可选）

**工具**：`mcp__ad__query_ad_data`

```json
{
  "start_date": "2024-11-01",
  "end_date": "2024-11-30",
  "indicators": ["日期", "广告计划名称", "消耗", "累计ROI", "首日ROI", "新增付费成本"],
  "group_key": "广告ID",
  "app_id": "59",
  "media": "今日头条"
}
```

**分析逻辑**：保留消耗 > 500 元的广告，按 `累计ROI × 消耗` 排序，从广告名称解析业务目标和出价。

---

### 步骤 2：查询 Top 素材

**工具**：`mcp__ad__query_top_materials`

```json
{
  "app_id": "59",
  "business_goal": "FIRST_PURCHASE",
  "method": "simple",  
  "top_n": 30 * 1.5, // 这里用户数据的是30，但是冗余机制要查1.5倍
  "material_type": "video",
  "media": "今日头条",
  "days": 30,
  "include_caption": true
}
```

**业务目标**：`FIRST_PURCHASE` | `REGISTER__1DAY_PURCHASE_ROAS` | `REGISTER__7DAY_PURCHASE_ROAS` | `APP_PURCHASE` | `REGISTER` 等

**注意**：素材数量应为 `video_num` 的整数倍（如 5、10、15、20）。由于会有素材不存在的问题，所以要查询1.5倍数量的素材。

---

### 步骤 3：AI 生成标题

基于素材的 Caption 和 Tags，生成 5 个不同风格的标题（用 `\n` 分隔，≤30字）：

1. **疑问式**：谁能召唤五虎上将？英雄卡牌宝箱等你开启！
2. **对比式**：别的游戏抽卡靠运气，这里登录就送五虎上将！
3. **情怀式**：经典三国终于出手游！完美还原端游经典！
4. **利益式**：新服福利爆表！五虎上将+万元宝箱免费送！
5. **紧迫式**：限时开服倒计时！五虎上将仅剩100个名额！

---

### 步骤 4：创建头条广告

**工具**：`mcp__ad__create_tt_ad`

**必需参数**：
```json
{
  "account_id": "1828802591320073",
  "tt_material_ids": "7540951726155038761,7540951690501947411",
  "app_id": 59,
  "aweme_id": "97157541111",
  "titles": "标题1\n标题2\n标题3\n标题4\n标题5",
  "business_goal": "首次付费",
  "micro_promotion_type": "WECHAT_GAME",
  "project_budget": 500,
  "project_cpa_bid": 55
}
```

**可选参数**：
- `video_num`: 视频素材数量（默认1）
- `audience_gender`: `GENDER_MALE` | `GENDER_FEMALE` | `NONE`
- `audience_age`: `"18-19,20-23,24-30"` 等
- `audience_platform`: `ANDROID` | `IOS`
- `project_roi_goal`: 仅付费ROI和付费ROI7日目标需要（如 0.02 表示2%）

**游戏账户映射**：
- 正统三国(59): `1828802591320073`，cid：ztsg_tt_AI_320073
- 银河战舰(61): `1827167219715081`，cid：ztsg_tt_AI_715081
- 我的仙门(67): `1838884366538759`，cid：ztsg_tt_AI_538759

**关键约束**：

- ⚠️ 素材缺失错误：停止调用，返回步骤2替换素材
- ⚠️ 返回 `task_id` 表示已投递，需在头条后台查看最终状态
- ⚠️ titles 会自动更新游戏标题包

---

## 三、流程 B：标题驱动

### 步骤 1：确定策略

直接指定或参考流程A步骤1查询历史数据。

---

### 步骤 2：生成标题

生成 3-5 个不同风格的标题，每个独立成篇，包含明确关键词便于素材匹配。

---

### 步骤 3：RAG 检索素材

**工具**：`mcp__ad__rag_search`

为每个标题独立检索：

```json
{
  "search_type": "text",
  "query": "谁能一统三国称霸天下？策略武将养成等你挑战",
  "collection": "ZtsgCollection",
  "limit": 10,
  "rerank_prop": "file_caption",
  "return_properties": ["tt_material_id", "new_caption", "tags", "file_url"]
}
```

**集合映射**：
- `ZtsgCollection`: 正统三国(59)
- `YhzjCollection`: 银河战舰(61)
- `WdxmCollection`: 我的仙门(67)

**筛选**：优先选择 `rerank_score` > 0 的素材，数量为 `video_num` 的整数倍。

---

### 步骤 4：创建广告

为每个标题创建独立广告，使用对应的匹配素材：

```json
{
  "account_id": "1828802591320073",
  "tt_material_ids": "7494598721490403369,7496318715793948698,...",
  "app_id": 59,
  "aweme_id": "97157541111",
  "titles": "五虎上将齐聚！正统三国SLG策略巅峰对决等你来战",
  "business_goal": "首次付费",
  "project_budget": 500,
  "project_cpa_bid": 55,
  "video_num": 5
}
```

**注意**：每次调用间隔 3-5 秒避免频率限制。

---

## 四、常用工具速查

### 1. 查询广告数据
```
mcp__ad__query_ad_data(start_date, end_date, indicators, app_id, media, group_key)
```

### 2. 查询 Top 素材
```
mcp__ad__query_top_materials(app_id, business_goal, method, top_n, material_type, media, days, include_caption)
```

### 3. RAG 检索素材
```
mcp__ad__rag_search(search_type, query, collection, limit, rerank_prop, return_properties)
```

### 4. 创建头条广告
```
mcp__ad__create_tt_ad(account_id, tt_material_ids, app_id, aweme_id, titles, business_goal, micro_promotion_type, project_budget, project_cpa_bid)
```

### 5. 获取应用信息
```
mcp__ad__get_app_info(app)  # 获取游戏的可用账户
```

### 6. 获取可用指标
```
mcp__ad__get_available_indicators(app_id, query_type)  # 查询前先验证指标
```

---

## 五、关键约束

### 素材使用
- ✅ 必须预上传至头条素材库
- ✅ 自动去重和过滤
- ⚠️ 数量为 `video_num` 的整数倍，不要限制总数

### 定向配置
- 年龄：`18-19, 20-23, 24-30, 31-35, 36-40, 41-45, 46-50, 51-55, 56-59, 60+`
- 性别：`GENDER_MALE` | `GENDER_FEMALE` | `NONE`
- 平台：`ANDROID` | `IOS`

### 预算出价
- 最小出价：≥ 100 元
- 测试预算：50-1000 元
- 正式预算：0（不限）或 > 1000 元

---

## 六、完整示例

```python
# 1. 查询 Top 素材
top_materials = mcp__ad__query_top_materials(
    app_id="59",
    business_goal="FIRST_PURCHASE",
    method="ucb",
    top_n=30,
    material_type="video",
    media="今日头条",
    days=30,
    include_caption=true
)

# 2. AI 生成标题（基于素材 caption）
titles = """五虎上将齐聚！英雄卡牌宝箱福利爆表，正统三国等你称王！
别的游戏抽卡靠运气，这里登录就送五虎上将！
经典三国终于出手游！完美还原端游经典，还是原来的味道！
新服福利爆表！五虎上将+万元宝箱免费送！
限时开服倒计时！五虎上将仅剩100个名额！"""

# 3. 创建广告
mcp__ad__create_tt_ad(
    account_id="1828802591320073",
    tt_material_ids="7540951726155038761,7540951690501947411,...",
    app_id=59,
    aweme_id="97157541111",
    titles=titles,
    business_goal="首次付费",
    micro_promotion_type="WECHAT_GAME",
    project_budget=500,
    project_cpa_bid=55,
    video_num=5
)
```

---

**版本**：v4.0-精简版
**更新**：2025-12-23
**优化**：大幅精简文档，保留核心流程和关键参数，提升阅读效率
