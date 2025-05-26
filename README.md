# CP_scheduling

🔧 Overview
This CP-SAT model is built using Google OR-Tools and generates an optimized schedule for manufacturing batches across multiple lines. It manages:

Production task scheduling

Changeovers between SKUs

Raw material availability and FIFO usage

Maintenance (windowed & optional)

Campaign constraints (batch grouping by SKU)

Makespan and unmet demand minimization

🧱 COMPONENT BREAKDOWN
1. Initialization
model = cp_model.CpModel() initializes the constraint programming model.

Inputs:

selected_lines: available production lines

skus: list of SKU demands and associated batch info

working_intervals, changeover, pre_tasks: line-specific metadata

raw_material_replenishments, rm_usage: material availability

2. Maintenance Intervals
Windowed Maintenance: Must occur once within a working interval.

Optional Maintenance: Can be skipped or scheduled within working intervals.

Constraints:
Let m_start be the start of maintenance, and duration the fixed maintenance time.

𝑤
𝑠
≤
𝑚
start
≤
𝑤
𝑒
−
𝑑
𝑢
𝑟
𝑎
𝑡
𝑖
𝑜
𝑛
w 
s
​
 ≤m 
start
​
 ≤w 
e
​
 −duration for selected working window 
(
𝑤
𝑠
,
𝑤
𝑒
)
(w 
s
​
 ,w 
e
​
 )

Windowed: scheduled once

Optional: if scheduled, fits in exactly one working window

3. Production Task Creation
For each SKU batch:

Assign it to a line (binary assign[line])

Define start, end, and duration variables per line

Compute raw material needs using pre-defined BOM

Handle:

Assignment constraints (at most 1 line)

Raw material usage constraints

Time windows (start ≥ recipe start, < recipe end)

Lead time & expiry for each RM cycle

4. Working Interval Enforcement
For each task assigned to a line:

Ensure start and end fall within a working interval

If 
𝑆
S is the task start, 
𝐸
E is the task end:

𝑤
𝑠
≤
𝑆
 and 
𝐸
≤
𝑤
𝑒
for some 
(
𝑤
𝑠
,
𝑤
𝑒
)
w 
s
​
 ≤S and E≤w 
e
​
 for some (w 
s
​
 ,w 
e
​
 )
5. Raw Material Constraints
Let 
𝑄
𝑖
𝑗
𝑘
Q 
ijk
​
  be quantity from RM 
𝑖
i used in task 
𝑗
j from cycle 
𝑘
k

If task is not assigned → 
𝑄
𝑖
𝑗
𝑘
=
0
Q 
ijk
​
 =0

If assigned → 
∑
𝑘
𝑄
𝑖
𝑗
𝑘
=
required quantity
∑ 
k
​
 Q 
ijk
​
 =required quantity

FIFO constraint: if cycle 
𝑘
k has leftover > 0, next cycle 
𝑘
+
1
k+1 can’t be used

6. Changeover Constraints
For two consecutive tasks 
𝑖
→
𝑗
i→j on a line:

Add binary immediate[i][j] variable

Let 
𝐶
𝑂
𝑖
𝑗
CO 
ij
​
  be the changeover time between SKUs:

start
𝑗
≥
end
𝑖
+
𝐶
𝑂
𝑖
𝑗
start 
j
​
 ≥end 
i
​
 +CO 
ij
​
 
Add OptionalIntervalVar for changeover

Also handle pre-production → first production task changeovers similarly.

7. Campaign Length Constraints
For same SKU tasks in a chain:

Sum of demands \leq \text{max_campaign_length}

Recursive constraint: 
campaign
𝑗
=
campaign
𝑖
+
demand
𝑗
campaign 
j
​
 =campaign 
i
​
 +demand 
j
​
 

8. Objective Function
Minimize:

Objective
=
total_unmet_demand
+
makespan
Objective=total_unmet_demand+makespan
Unmet demand: tasks not assigned to any line

Makespan: latest end time among all tasks

9. Solution Callback: KPI Tracking
Captures:

Makespan

Idle Time per line (sum of gaps)

Utilization = total production time / horizon

Total production per line

Stores all KPIs in df_kpis DataFrame

10. Schedule Reporting
The function build_schedule_report:

Converts solver outputs into structured DataFrame with:

Start/End times (converted to datetime)

Type: Production, Changeover, Maintenance, Buffer, Downtime

RM usage per task

Detailed labels
