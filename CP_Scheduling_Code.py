from ortools.sat.python import cp_model
import math
import random
import datetime
import copy
from itertools import combinations
import matplotlib.dates as mdates

#############################################
# 1. User Input: Selected Lines
#############################################
# The user can choose to schedule only line 0, only line 1, or both lines [0, 1].
selected_lines = [0, 1]  # Example; change as needed.

#############################################
# Helper Functions (Modified)
#############################################
def generate_sunday_downtimes(start_datetime, horizon, day_length, lines):
    """Generates downtime intervals (in hours) for each Sunday on the given lines."""
    downtimes = {line: [] for line in lines}
    num_days = horizon // day_length
    for line in lines:
        for d in range(num_days):
            current_date = start_datetime + datetime.timedelta(days=d)
            if current_date.weekday() == 6:  # Sunday
                start_offset = d * day_length
                end_offset = min((d + 1) * day_length, horizon)
                downtimes[line].append((start_offset, end_offset))
    return downtimes

def generate_monthly_buffers(start_datetime, horizon, buffer_duration, day_length, downtimes, lines):
    """Generates monthly buffer windows (per line) that avoid overlapping with downtimes."""
    buffers = {line: [] for line in lines}
    current = start_datetime
    end_datetime = start_datetime + datetime.timedelta(hours=horizon)
    while current < end_datetime:
        if current.month == 12:
            next_month = datetime.datetime(current.year + 1, 1, 1, current.hour, current.minute)
        else:
            next_month = datetime.datetime(current.year, current.month + 1, 1, current.hour, current.minute)
        proposed_buffer_end = next_month
        proposed_buffer_start = next_month - datetime.timedelta(hours=buffer_duration)
        prop_start = (proposed_buffer_start - start_datetime).total_seconds() / 3600.0
        prop_end = (proposed_buffer_end - start_datetime).total_seconds() / 3600.0
        adjusted_start = prop_start
        # Clip the buffer start to avoid overlapping any downtime across all lines.
        for line in lines:
            for (dt_start, dt_end) in downtimes.get(line, []):
                if dt_start < prop_end and dt_end > prop_start:
                    adjusted_start = max(adjusted_start, dt_end)
        if prop_end - adjusted_start >= buffer_duration:
            final_start = adjusted_start
            final_end = final_start + buffer_duration
        else:
            final_start = adjusted_start
            final_end = prop_end
        if final_start < horizon:
            for line in lines:
                buffers[line].append((int(final_start), int(min(final_end, horizon))))
        current = next_month
    return buffers

def merge_intervals(intervals):
    """Merge overlapping intervals (each as (s, e))."""
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = []
    current_start, current_end = intervals[0]
    for s, e in intervals[1:]:
        if s <= current_end:
            current_end = max(current_end, e)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = s, e
    merged.append((current_start, current_end))
    return merged

def generate_working_intervals_full(horizon, fixed_intervals):
    """Returns intervals [0, horizon] not covered by any fixed interval."""
    merged_fixed = merge_intervals(fixed_intervals)
    working = []
    prev_end = 0
    for s, e in merged_fixed:
        if s > prev_end:
            working.append((prev_end, s))
        prev_end = max(prev_end, e)
    if prev_end < horizon:
        working.append((prev_end, horizon))
    return working

def hour_to_datenum(hour, start_datetime):
    """Converts an hour offset to a matplotlib date number."""
    dt = start_datetime + datetime.timedelta(hours=hour)
    return mdates.date2num(dt)

class ObjectiveEvolutionCallback(cp_model.CpSolverSolutionCallback):
    def __init__(self):
        super().__init__()
        self.objective_values = []
        self.iterations = []
        self.solution_count = 0
    def OnSolutionCallback(self):
        self.solution_count += 1
        obj_val = self.ObjectiveValue()
        self.objective_values.append(obj_val)
        self.iterations.append(self.solution_count)
        print(f"Iteration {self.solution_count}: objective = {obj_val}")

#############################################
# 2. Parameters and Data Generation
#############################################
day_length = 24
horizon = 2100  # in hours (~87.5 days)
start_datetime = datetime.datetime(2023, 3, 8, 11, 0, 0)

# Generate downtimes and monthly buffers for the selected lines.
downtimes = generate_sunday_downtimes(start_datetime, horizon, day_length, selected_lines)
buffer_duration = 24  # 24-hour buffer at end of each month
monthly_buffers = generate_monthly_buffers(start_datetime, horizon, buffer_duration, day_length, downtimes, selected_lines)

# Sample fixed intervals = downtimes + maintenance + buffers.
user_maintenance = {
    0: [(300, 350), (700, 750), (960, 1000), (1800, 1900)],
    1: [(500, 550), (624, 650), (960, 1000), (1800, 1900)]
}
user_maintenance_window = {
    0: [(100, 150, 5)],
    1: [(200, 300, 12)]
}
user_optional_maintenance_window = {
    0: [(100, 150, 5)],
    1: [(200, 300, 12)]
}

F_line = {line: [] for line in selected_lines}
for line in selected_lines:
    F_line[line].extend(downtimes.get(line, []))
    F_line[line].extend(user_maintenance.get(line, []))
    F_line[line].extend(monthly_buffers.get(line, []))
    F_line[line] = merge_intervals(F_line[line])

# Working intervals are the complement of F_line.
working_intervals = {
    line: generate_working_intervals_full(horizon, F_line[line])
    for line in selected_lines
}

# Create some raw materials (IDP + RM) for demonstration:
raw_material_replenishments = {}
num_idp = 5
num_rm = 45

# IDP materials
for i in range(1, num_idp + 1):
    idp_key = f"IDP{i}"
    time1 = 0
    expiration1 = random.randint(50, 3000)
    avail_qty1 = random.randint(10000, 20000)
    cycle1 = {
        'batch_id': 1,
        'time': time1,
        'avail_qty': avail_qty1,
        'expiration': expiration1,
        'material_type': 'IDP'
    }
    time2 = 0
    expiration2 = horizon
    avail_qty2 = random.randint(10000, 20000)
    cycle2 = {
        'batch_id': 2,
        'time': time2,
        'avail_qty': avail_qty2,
        'expiration': expiration2,
        'material_type': 'IDP'
    }
    raw_material_replenishments[idp_key] = [cycle1, cycle2]

# RM materials
for i in range(1, num_rm + 1):
    rm_key = f"RM{i}"
    time1 = 0
    expiration1 = random.randint(50, 3000)
    avail_qty1 = random.randint(10000, 20000)
    cycle1 = {
        'batch_id': 1,
        'time': time1,
        'avail_qty': avail_qty1,
        'expiration': expiration1,
        'material_type': 'RM'
    }
    time2 = 0
    expiration2 = horizon
    avail_qty2 = random.randint(10000, 20000)
    cycle2 = {
        'batch_id': 2,
        'time': time2,
        'avail_qty': avail_qty2,
        'expiration': expiration2,
        'material_type': 'RM'
    }
    raw_material_replenishments[rm_key] = [cycle1, cycle2]

# Track usage
rm_usage = {
    rm: {k: [] for k in range(len(raw_material_replenishments[rm]))}
    for rm in raw_material_replenishments
}

# Generate SKUs - Remove the Throughput calculation, RM and IDP Usage. They will now be dynamic based on the BOM
skus = []
selected_lines = [0, 1]  # allowed production lines
for i in range(1, 21):
    sku_key = f"SKU{i}"
    # Remove static throughput since it will be derived from the recipe.
    allowed_lines = [l for l in selected_lines]
    batches = []
    for b in [1, 2, 3, 4]:
        demand = random.randint(50, 500)
        # Remove static raw_required assignment as well.
        batches.append({
            'batch_id': b,
            'demand': demand
        })
    skus.append({
        'sku': sku_key,
        'allowed_lines': allowed_lines,
        'batches': batches,
    })

##########################
# Generate Recipes for Each SKU
##########################
sku_recipes = {}  # Dictionary to hold recipes keyed by SKU name

# Use the keys from raw_material_replenishments as the pool of materials.
possible_materials = list(raw_material_replenishments.keys())

for sku in skus:
    sku_name = sku["sku"]
    recipes = []
    
    # Choose a random split point between 1 and horizon-1.
    split = random.randint(1, horizon - 1)
    
    # ---- Recipe 1 ----
    recipe1 = {
        "recipe_id": 1,
        "start": 0,          # Recipe 1 is valid starting at time 0
        "end": split,        # Ends at the random split point
        "throughput": {      # Throughput values (units per hour) for each production line
            0: random.randint(10, 30),
            1: random.randint(10, 30)
        },
        "bom": {}            # Bill-of-Materials: to be defined below
    }
    # Generate a random BOM for Recipe 1.
    # Randomly select between 1 and 3 materials from the inventory pool.
    selected_mats1 = random.sample(possible_materials, random.randint(1, 3))
    bom1 = {}
    for mat in selected_mats1:
        # Assign a random usage factor between 0.5 and 2.0 (units of material per unit SKU)
        bom1[mat] = round(random.uniform(0.5, 2.0), 2)
    recipe1["bom"] = bom1

    # ---- Recipe 2 ----
    recipe2 = {
        "recipe_id": 2,
        "start": split,      # Recipe 2 starts at the split point
        "end": horizon,      # Ends at the overall horizon
        "throughput": {      # Throughput values for each production line
            0: random.randint(10, 30),
            1: random.randint(10, 30)
        },
        "bom": {}            # BOM for Recipe 2
    }
    selected_mats2 = random.sample(possible_materials, random.randint(1, 3))
    bom2 = {}
    for mat in selected_mats2:
        bom2[mat] = round(random.uniform(0.5, 2.0), 2)
    recipe2["bom"] = bom2

    # Append both recipes for this SKU.
    recipes.append(recipe1)
    recipes.append(recipe2)
    sku_recipes[sku_name] = recipes

# Changeover matrix
changeover = {}
for i in range(1, 21):
    sku_i = f"SKU{i}"
    changeover[sku_i] = {}
    for j in range(1, 21):
        sku_j = f"SKU{j}"
        changeover[sku_i][sku_j] = 4 if i == j else 6

# --- Generate Max Campaign Dictionary ---
max_campaign = {}
for sku in skus:
    sku_key = sku['sku']
    # Use a dynamic value if provided, otherwise default to 1000.
    max_campaign[sku_key] = sku.get('max_campaign', 1000)


#############################################
# 3. CP-SAT Model Building
#############################################
model = cp_model.CpModel()

# Intervals storage
intervals_per_line = {line: [] for line in selected_lines}

# Maintenance intervals (fixed)
for line in selected_lines:
    for idx, (s, e) in enumerate(downtimes.get(line, [])):
        dur = e - s
        dt_int = model.NewIntervalVar(s, dur, e, f"dt_line{line}_{idx}")
        intervals_per_line[line].append(dt_int)
    for idx, (s, e) in enumerate(user_maintenance.get(line, [])):
        dur = e - s
        m_int = model.NewIntervalVar(s, dur, e, f"maint_line{line}_{idx}")
        intervals_per_line[line].append(m_int)
    for idx, (s, e) in enumerate(monthly_buffers.get(line, [])):
        dur = e - s
        b_int = model.NewIntervalVar(s, dur, e, f"buffer_line{line}_{idx}")
        intervals_per_line[line].append(b_int)

# Windowed maintenance intervals
windowed_maint_intervals = {line: [] for line in selected_lines}
for line in selected_lines:
    for idx, (earliest, latest, duration) in enumerate(user_maintenance_window.get(line, [])):
        m_start = model.NewIntVar(earliest, latest, f"windowed_maint_{line}_{idx}_start")
        m_end   = model.NewIntVar(earliest + duration, latest + duration, f"windowed_maint_{line}_{idx}_end")
        m_interval = model.NewIntervalVar(m_start, duration, m_end, f"windowed_maint_{line}_{idx}")
        intervals_per_line[line].append(m_interval)
        windowed_maint_intervals[line].append({
            'interval': m_interval,
            'm_start': m_start,
            'duration': duration
        })

# Optional maintenance intervals
optional_maint_intervals = {line: [] for line in selected_lines}
for line in selected_lines:
    for idx, (earliest, latest, duration) in enumerate(user_optional_maintenance_window.get(line, [])):
        scheduled = model.NewBoolVar(f"optional_maint_{line}_{idx}_scheduled")
        m_start = model.NewIntVar(earliest, latest, f"optional_maint_{line}_{idx}_start")
        m_end   = model.NewIntVar(earliest + duration, latest + duration, f"optional_maint_{line}_{idx}_end")
        m_interval = model.NewOptionalIntervalVar(m_start, duration, m_end, scheduled, f"optional_maint_{line}_{idx}")
        intervals_per_line[line].append(m_interval)
        optional_maint_intervals[line].append({
            'interval': m_interval,
            'm_start': m_start,
            'duration': duration,
            'scheduled': scheduled
        })

#############################################
# CP-SAT Production Task Creation (Updated with BOM)
#############################################
# Assume rm_usage has been defined for each material:
# rm_usage = { material: { cycle_index: [] for each cycle } for material in raw_material_replenishments }

# Create Production Tasks (with Recipe Selection and Dynamic Throughput/Material Usage)
tasks = {}
for sku in skus:
    sku_name = sku['sku']
    allowed_lines = sku['allowed_lines']
    # Get the list of recipes for this SKU (each recipe has its own throughput, BOM, and valid time window)
    recipes_for_sku = sku_recipes[sku_name]  
    for batch in sku['batches']:
        batch_id = batch['batch_id']
        task_key = f"{sku_name}_batch{batch_id}"
        batch_demand = batch['demand']

        # --- Recipe Selection Setup ---
        # Create a Boolean for each recipe (one must be chosen if the batch is scheduled)
        recipe_bools = []
        # For candidate durations, create a dictionary keyed by line.
        candidate_durations = {line: [] for line in selected_lines}
        # Also, store the raw material usage info for each recipe in a temporary dictionary.
        raw_mapping_recipe = {}  # key: recipe index, value: dict mapping material -> usage info
        
        for idx, rcp in enumerate(recipes_for_sku):
            r_bool = model.NewBoolVar(f"{task_key}_recipe_{rcp['recipe_id']}")
            recipe_bools.append(r_bool)
            # Enforce that if this recipe is chosen and the batch is scheduled on a given line,
            # the start time must fall within the recipe's valid window.
            # (We will apply these constraints later, after creating start variables.)
            
            # For each allowed production line, compute candidate duration based on this recipe’s throughput.
            for line in selected_lines:
                if line in allowed_lines:
                    # Candidate duration = ceil(demand / recipe_throughput for that line)
                    cand_dur = math.ceil(batch_demand / rcp['throughput'][line])
                    candidate_durations[line].append(cand_dur)
                else:
                    candidate_durations[line].append(0)
            
            # Now, for each material in the recipe’s BOM, build usage variables.
            mapping = {}
            for material, usage_per_unit in rcp['bom'].items():
                # Calculate the required amount for this batch:
                required_qty = int(round(batch_demand * usage_per_unit))
                cycles = raw_material_replenishments[material]
                Q_vars = []
                used_flags = []
                for k in range(len(cycles)):
                    Q = model.NewIntVar(0, required_qty, f"{task_key}_{material}_rcp{rcp['recipe_id']}_cycle_{k}_qty")
                    Y = model.NewBoolVar(f"{task_key}_{material}_rcp{rcp['recipe_id']}_cycle_{k}_used")
                    # When this recipe is not chosen, enforce Q == 0.
                    model.Add(Q == 0).OnlyEnforceIf(r_bool.Not())
                    # (Also, if the batch is not scheduled, Q == 0 will be enforced below.)
                    model.Add(Q >= 1).OnlyEnforceIf(Y)
                    model.Add(Q <= cycles[k]['avail_qty']).OnlyEnforceIf(Y)
                    model.Add(Q == 0).OnlyEnforceIf(Y.Not())
                    Q_vars.append(Q)
                    used_flags.append(Y)
                    # Add Q to the global usage aggregator.
                    rm_usage[material][k].append(Q)
                # When this recipe is active, the sum of Q_vars must equal the required quantity.
                model.Add(sum(Q_vars) == required_qty).OnlyEnforceIf(r_bool)
                model.Add(sum(Q_vars) == 0).OnlyEnforceIf(r_bool.Not())
                mapping[material] = {
                    'Q_vars': Q_vars,
                    'used_vars': used_flags,
                    'cycles': cycles
                }
            raw_mapping_recipe[idx] = mapping

        # --- Assignment Variables and Time Variables ---
        assign = {}
        start = {}
        end = {}
        duration = {}
        interval = {}
        for line in selected_lines:
            if line in allowed_lines:
                assign[line] = model.NewBoolVar(f"{task_key}_on_line{line}")
                start[line] = model.NewIntVar(0, horizon, f"start_{task_key}_line{line}")
                end[line] = model.NewIntVar(0, horizon, f"end_{task_key}_line{line}")
                # Compute dynamic duration for this line as the weighted sum of candidate durations.
                # (Since exactly one recipe is chosen, the sum picks out that recipe's candidate.)
                dur_var = model.NewIntVar(0, horizon, f"duration_{task_key}_line{line}")
                model.Add(dur_var == sum(candidate_durations[line][i] * recipe_bools[i]
                                          for i in range(len(recipes_for_sku))))
                duration[line] = dur_var
                interval[line] = model.NewOptionalIntervalVar(start[line], dur_var, end[line],
                                                               assign[line], f"interval_{task_key}_line{line}")
            else:
                assign[line] = model.NewConstant(0)
        assign_none = model.NewBoolVar(f"{task_key}_not_scheduled")
        model.Add(sum(assign[line] for line in selected_lines) + assign_none == 1)

        # --- Link Recipe Selection with Scheduling ---
        # Enforce that exactly one recipe is chosen when the batch is scheduled.
        model.Add(sum(recipe_bools) == (sum(assign[line] for line in selected_lines)))
        
        # --- Enforce Recipe Time Windows and Lead-Time for Material Usage ---
        # For each recipe, if it is chosen, enforce the recipe's valid time window.
        for idx, rcp in enumerate(recipes_for_sku):
            # For line 0:
            model.Add(start[0] >= rcp['start']).OnlyEnforceIf([recipe_bools[idx], assign[0]])
            model.Add(start[0] <  rcp['end']).OnlyEnforceIf([recipe_bools[idx], assign[0]])
            # For line 1:
            model.Add(start[1] >= rcp['start']).OnlyEnforceIf([recipe_bools[idx], assign[1]])
            model.Add(start[1] <  rcp['end']).OnlyEnforceIf([recipe_bools[idx], assign[1]])
            
            # For each material used in this recipe, enforce lead time and expiration.
            mapping = raw_mapping_recipe[idx]
            for material, map_info in mapping.items():
                cycles = map_info['cycles']
                Q_vars = map_info['Q_vars']
                used_flags = map_info['used_vars']
                for k, cycle in enumerate(cycles):
                    lead_time = 120 if cycle['material_type'] == 'IDP' else 72
                    # For line 0:
                    model.Add(start[0] >= cycle['time'] + lead_time).OnlyEnforceIf([used_flags[k], recipe_bools[idx], assign[0]])
                    model.Add(start[0] < cycle['expiration']).OnlyEnforceIf([used_flags[k], recipe_bools[idx], assign[0]])
                    # For line 1:
                    model.Add(start[1] >= cycle['time'] + lead_time).OnlyEnforceIf([used_flags[k], recipe_bools[idx], assign[1]])
                    model.Add(start[1] < cycle['expiration']).OnlyEnforceIf([used_flags[k], recipe_bools[idx], assign[1]])

        # --- Store the Task ---
        tasks[task_key] = {
            'sku': sku_name,
            'batch_id': batch_id,
            'allowed_lines': allowed_lines,
            'assign': assign,
            'assign_none': assign_none,
            'start': start,
            'end': end,
            'duration': duration,
            'interval': interval,
            'demand': batch_demand,
            'recipe_bools': recipe_bools,
            'recipe_list': recipes_for_sku,
            'raw_mapping_recipe': raw_mapping_recipe  # stored for post-processing if needed
        }
        for line in selected_lines:
            if line in allowed_lines:
                intervals_per_line[line].append(interval[line])


# Restrict Production Tasks to Working Hours
bigM = horizon
work_assign = {t: {} for t in tasks}
for t, task in tasks.items():
    for line in selected_lines:
        work_assign[t][line] = []
        if line in task['assign']:
            for idx, (w_s, w_e) in enumerate(working_intervals[line]):
                W = model.NewBoolVar(f"work_{t}_line{line}_{idx}")
                work_assign[t][line].append((W, w_s, w_e))
                model.Add(task['start'][line] >= w_s - bigM*(1 - W)).OnlyEnforceIf(task['assign'][line])
                model.Add(task['end'][line]   <= w_e + bigM*(1 - W)).OnlyEnforceIf(task['assign'][line])
            # Exactly one working interval if assigned on that line
            model.Add(sum(W for (W, _, _) in work_assign[t][line]) == task['assign'][line])\
                 .OnlyEnforceIf(task['assign'][line])

# Raw Material Expiry & Lead Time Constraints (Updated for Recipe-Based Structure)
for task_key, task in tasks.items():
    # Loop over each recipe (by index) in the task's raw mapping.
    for recipe_idx, mapping_dict in task['raw_mapping_recipe'].items():
        # mapping_dict: keys are material names, values are the material mapping info.
        for rm, mapping in mapping_dict.items():
            cycles = mapping['cycles']
            for k in range(len(cycles)):
                cycle = cycles[k]
                # Determine lead time based on material type.
                lead_time = 120 if cycle['material_type'] == 'IDP' else 72
                for line in selected_lines:
                    # Ensure that the assignment exists for the line.
                    if line in task['assign']:
                        # Add constraints so that if:
                        # - mapping['used_vars'][k] is active,
                        # - the recipe (with index recipe_idx) is chosen (i.e. task['recipe_bools'][recipe_idx] is True),
                        # - and the task is scheduled on this line (task['assign'][line] is True),
                        # then the task's start time on that line must be at least cycle['time']+lead_time
                        # and less than the cycle's expiration.
                        model.Add(task['start'][line] >= cycle['time'] + lead_time).OnlyEnforceIf(
                            [mapping['used_vars'][k], task['recipe_bools'][recipe_idx], task['assign'][line]])
                        model.Add(task['start'][line] < cycle['expiration']).OnlyEnforceIf(
                            [mapping['used_vars'][k], task['recipe_bools'][recipe_idx], task['assign'][line]])


# Global Raw Material Capacity
for rm in rm_usage:
    for k, Q_list in rm_usage[rm].items():
        if Q_list:
            model.Add(sum(Q_list) <= raw_material_replenishments[rm][k]['avail_qty'])

# Global Sequencing for Changeovers
# We'll create "immediate" for each line, and "is_first", "is_last" for each line.
immediate = {}
is_first = {}
is_last  = {}
for line in selected_lines:
    task_keys = list(tasks.keys())
    for i in task_keys:
        is_first[(line, i)] = model.NewBoolVar(f"is_first_{i}_line{line}")
        is_last[(line, i)]  = model.NewBoolVar(f"is_last_{i}_line{line}")
        for j in task_keys:
            if i == j:
                continue
            imm_ij = model.NewBoolVar(f"immediate_{i}_{j}_line{line}")
            immediate[(line, i, j)] = imm_ij
            both_assigned = model.NewBoolVar(f"both_{i}_{j}_line{line}")
            model.AddBoolAnd([tasks[i]['assign'][line], tasks[j]['assign'][line]]).OnlyEnforceIf(both_assigned)
            model.AddBoolOr([tasks[i]['assign'][line].Not(), tasks[j]['assign'][line].Not()]).OnlyEnforceIf(both_assigned.Not())
            model.Add(imm_ij <= both_assigned)
            # j starts after i + changeover
            ch_time = changeover[tasks[i]['sku']][tasks[j]['sku']]
            model.Add(tasks[j]['start'][line] >= tasks[i]['end'][line] + ch_time).OnlyEnforceIf(imm_ij)
            # If i or j is not scheduled, imm_ij = 0
            model.Add(imm_ij == 0).OnlyEnforceIf(tasks[i]['assign_none'])
            model.Add(imm_ij == 0).OnlyEnforceIf(tasks[j]['assign_none'])
    # Predecessor/successor sums
    for tkey in task_keys:
        pred_sum = [immediate[(line, i, tkey)] for i in task_keys if i != tkey]
        succ_sum = [immediate[(line, tkey, j)] for j in task_keys if j != tkey]
        model.Add(sum(pred_sum) + is_first[(line, tkey)] == tasks[tkey]['assign'][line])
        model.Add(sum(succ_sum) + is_last[(line, tkey)]  == tasks[tkey]['assign'][line])
    model.Add(sum(is_first[(line, t)] for t in task_keys) <= 1)
    model.Add(sum(is_last[(line, t)] for t in task_keys)  <= 1)

# Effective Changeover Gap Constraint Using Working Intervals
changeover_start_vars = {}
Z_vars = {}
for line in selected_lines:
    task_keys = list(tasks.keys())
    for i in task_keys:
        for j in task_keys:
            if i == j:
                continue
            ch = changeover[tasks[i]['sku']][tasks[j]['sku']]
            T_ij = model.NewIntVar(0, horizon, f"T_{i}_{j}_line{line}")
            changeover_start_vars[(line, i, j)] = T_ij

            imm_ij = immediate[(line, i, j)]

            # 1) Force T_ij >= end of task i, tasks[j].start >= T_ij + ch
            model.Add(T_ij >= tasks[i]['end'][line]).OnlyEnforceIf(imm_ij)
            model.Add(tasks[j]['start'][line] >= T_ij + ch).OnlyEnforceIf(imm_ij)

            # 2) Create an end variable for the changeover
            co_end_ij = model.NewIntVar(0, horizon, f"co_end_{i}_{j}_line{line}")
            # If imm_ij is True, then co_end_ij = T_ij + ch
            model.Add(co_end_ij == T_ij + ch).OnlyEnforceIf(imm_ij)
            # If imm_ij is False, we can force co_end_ij to 0 (or T_ij) as well, or just let it be.
            # For clarity:
            model.Add(co_end_ij == 0).OnlyEnforceIf(imm_ij.Not())

            # 3) Create an OptionalIntervalVar for the changeover
            co_interval = model.NewOptionalIntervalVar(
                T_ij, ch, co_end_ij, imm_ij,
                f"co_int_{i}_{j}_line{line}"
            )

            # 4) Add that interval to intervals_per_line so that NoOverlap sees it
            intervals_per_line[line].append(co_interval)

            # 5) Keep your "candidate_Z" constraints if you want the changeover 
            #    to fall in a working interval. (Optional if you rely on NoOverlap)
            candidate_Z = []
            for idx, (w_s, w_e) in enumerate(working_intervals[line]):
                Z = model.NewBoolVar(f"Z_{i}_{j}_{idx}_line{line}")
                Z_vars[(line, i, j, idx)] = Z
                candidate_Z.append(Z)
                model.Add(T_ij >= w_s).OnlyEnforceIf(imm_ij).OnlyEnforceIf(Z)
                model.Add(T_ij + ch <= w_e).OnlyEnforceIf(imm_ij).OnlyEnforceIf(Z)
                # etc.
            model.Add(sum(candidate_Z) == imm_ij)

# NoOverlap for each line
for line in selected_lines:
    model.AddNoOverlap(intervals_per_line[line])

# Makespan Calculation
end_times = []
for t in tasks:
    task_end = model.NewIntVar(0, horizon, f"global_end_{t}")
    # If assigned on line => end = tasks[t]['end'][line]
    for line in selected_lines:
        model.Add(task_end == tasks[t]['end'][line]).OnlyEnforceIf(tasks[t]['assign'][line])
    # If not scheduled => end=0
    model.Add(task_end == 0).OnlyEnforceIf(tasks[t]['assign_none'])
    end_times.append(task_end)

makespan = model.NewIntVar(0, horizon, "makespan")
model.AddMaxEquality(makespan, end_times)

# Unmet Demand
unmet_vars = []
for t in tasks:
    unmet = model.NewIntVar(0, tasks[t]['demand'], f"unmet_{t}")
    model.Add(unmet == tasks[t]['demand']).OnlyEnforceIf(tasks[t]['assign_none'])
    for line in selected_lines:
        model.Add(unmet == 0).OnlyEnforceIf(tasks[t]['assign'][line])
    unmet_vars.append(unmet)
total_unmet_demand = model.NewIntVar(0, sum(tasks[t]['demand'] for t in tasks), "total_unmet_demand")
model.Add(total_unmet_demand == sum(unmet_vars))


# --- Create campaign_sum variable for each task ---
# This must be done before adding any constraints that refer to campaign_sum.
campaign_sum = {}  # campaign_sum[t] will be an IntVar for task t.
for t in tasks:
    sku = tasks[t]['sku']
    # Create an IntVar with an upper bound set by the max campaign for that SKU.
    campaign_sum[t] = model.NewIntVar(0, max_campaign[sku], f"campaign_sum_{t}")

# --- Max Campaign Constraint for Line 0 ---
for t in tasks:
    sku = tasks[t]['sku']
    demand = tasks[t]['demand']  # Production quantity for task t

    # Identify candidate predecessor tasks on line 0: tasks with the same SKU and where an immediate precedence link exists.
    preds_line0 = [
        j for j in tasks 
        if j != t and tasks[j]['sku'] == sku and ((0, j, t) in immediate)
    ]

    if preds_line0:
        # Create a Boolean variable that indicates if task t has no active immediate predecessor on line 0.
        no_pred = model.NewBoolVar(f"no_pred_{t}_line0")
        # Enforce: if no_pred is true, then none of the immediate links from any candidate predecessor are active.
        model.Add(sum(immediate[(0, j, t)] for j in preds_line0) == 0).OnlyEnforceIf(no_pred)
        # Conversely, if no_pred is false then at least one immediate link is active.
        model.Add(sum(immediate[(0, j, t)] for j in preds_line0) >= 1).OnlyEnforceIf(no_pred.Not())

        # If there is no active immediate predecessor, then the campaign for task t equals its own demand.
        model.Add(campaign_sum[t] == demand).OnlyEnforceIf(no_pred)

        # Otherwise, for each candidate predecessor j, if the immediate link is active then accumulate.
        for j in preds_line0:
            imm_key = (0, j, t)
            imm_var = immediate.get(imm_key)  # Safely get the immediate variable.
            # Only add the constraint if the immediate variable exists and campaign_sum[j] is available.
            if imm_var is not None and j in campaign_sum:
                model.Add(campaign_sum[t] == campaign_sum[j] + demand).OnlyEnforceIf(imm_var)
    else:
        # If no candidate predecessor exists, task t starts a new campaign.
        model.Add(campaign_sum[t] == demand)

    # Enforce that the campaign sum does not exceed the SKU's allowed maximum,
    # but only when task t is assigned to line 0.
    model.Add(campaign_sum[t] <= max_campaign[sku]).OnlyEnforceIf(tasks[t]['assign'][0])

# --- Max Campaign Constraint for Line 1 ---
for t in tasks:
    sku = tasks[t]['sku']
    demand = tasks[t]['demand']  # Production quantity for task t

    # Identify candidate predecessor tasks on line 1.
    preds_line1 = [
        j for j in tasks 
        if j != t and tasks[j]['sku'] == sku and ((1, j, t) in immediate)
    ]

    if preds_line1:
        # Create a Boolean variable that indicates if task t has no active immediate predecessor on line 1.
        no_pred = model.NewBoolVar(f"no_pred_{t}_line1")
        model.Add(sum(immediate[(1, j, t)] for j in preds_line1) == 0).OnlyEnforceIf(no_pred)
        model.Add(sum(immediate[(1, j, t)] for j in preds_line1) >= 1).OnlyEnforceIf(no_pred.Not())

        # If no active predecessor exists, campaign equals the task's own demand.
        model.Add(campaign_sum[t] == demand).OnlyEnforceIf(no_pred)

        # Otherwise, for each candidate predecessor j, accumulate the campaign sum.
        for j in preds_line1:
            imm_key = (1, j, t)
            imm_var = immediate.get(imm_key)
            if imm_var is not None and j in campaign_sum:
                model.Add(campaign_sum[t] == campaign_sum[j] + demand).OnlyEnforceIf(imm_var)
    else:
        model.Add(campaign_sum[t] == demand)

    # Enforce that the campaign sum does not exceed the maximum allowed for the SKU,
    # but only if task t is assigned to line 1.
    model.Add(campaign_sum[t] <= max_campaign[sku]).OnlyEnforceIf(tasks[t]['assign'][1])

# Objective
penalty_weight = 1000
model.Minimize(total_unmet_demand + makespan)

#############################################
# 4. Solve the Model
#############################################
user_time_limit = 200.0
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = user_time_limit
solver.parameters.log_search_progress = True

callback = ObjectiveEvolutionCallback()
status = solver.Solve(model, callback)

#############################################
# 5. Extract the Solution
#############################################
if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
    print("Solution found:")
    print("Makespan =", solver.Value(makespan))

    # Build schedule dictionary keyed by line
    schedule = {line: [] for line in selected_lines}
    for t in tasks:
        assigned_line = None
        for line in selected_lines:
            if solver.BooleanValue(tasks[t]['assign'][line]):
                assigned_line = line
                break
        sku_name = tasks[t]['sku']
        batch_id = tasks[t]['batch_id']
        if assigned_line is not None:
            s_val = solver.Value(tasks[t]['start'][assigned_line])
            e_val = solver.Value(tasks[t]['end'][assigned_line])
            dur   = tasks[t]['duration'][assigned_line]
            label = f"{sku_name}_B{batch_id}_{s_val}"
            schedule[assigned_line].append((s_val, dur, label, t))
            print(f"{t}: {sku_name} Batch {batch_id} on Line {assigned_line}, "
                  f"Start {s_val}, End {e_val}, Duration {dur}")
        else:
            print(f"{t}: {sku_name} Batch {batch_id} not scheduled.")
else:
    print("No solution found.")

# Print out some changeover details if desired
for (line, i, j, idx), Z in Z_vars.items():
    if solver.BooleanValue(Z):
        if solver.BooleanValue(immediate[(line, i, j)]):
            print(f"Changeover from {i} to {j} on line {line} uses working interval index {idx} "
                  f"with bounds {working_intervals[line][idx]}, "
                  f"CO start time {solver.Value(changeover_start_vars[(line, i, j)])}")

# --- Print Windowed Maintenance Timings ---
print("\nWindowed Maintenance Timings:")
for line, maint_list in windowed_maint_intervals.items():
    for idx, maint in enumerate(maint_list):
        m_start_val = solver.Value(maint['m_start'])
        duration = maint['duration']
        m_end_val = m_start_val + duration
        print(f"Line {line} - Windowed Maintenance {idx+1}: Start = {m_start_val}, End = {m_end_val}")

# --- Print Optional Maintenance Timings ---
print("\nOptional Maintenance Timings:")
for line, maint_list in optional_maint_intervals.items():
    for idx, maint in enumerate(maint_list):
        if solver.BooleanValue(maint['scheduled']):
            m_start_val = solver.Value(maint['m_start'])
            duration = maint['duration']
            m_end_val = m_start_val + duration
            print(f"Line {line} - Optional Maintenance {idx+1}: Scheduled, Start = {m_start_val}, End = {m_end_val}")
        else:
            print(f"Line {line} - Optional Maintenance {idx+1}: Not scheduled")




import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import pandas as pd

# --- Assume these variables come from the model solution:
# schedule: dictionary {line: [(start, duration, label, task_key), ...]}
# changeover_start_vars: dictionary with keys (line, i, j) and values T_ij
# immediate: dictionary keyed by (line, i, j) with Boolean values indicating immediate precedence
# solver: the CpSolver() instance after solving the model
# tasks: dictionary with task information (each task’s 'assign' is now a dict keyed by line)
# changeover: dictionary-of-dictionaries with changeover durations
# working_intervals: dictionary {line: [(w_s, w_e), ...]}
# user_maintenance: dictionary {line: [(s, e), ...]}
# windowed_maint_intervals: dictionary {line: [ { 'm_start': ..., 'duration': ... }, ... ]}
# optional_maint_intervals: dictionary {line: [ { 'm_start': ..., 'duration': ..., 'scheduled': ... }, ... ]}
# monthly_buffers: dictionary {line: [(s, e), ...]}
# start_datetime: production start datetime
# horizon: scheduling horizon in hours

# Visualization parameters:
line_base   = {0: 10, 1: 40}  # adjust or expand as needed
height_prod = 9
height_maint = 3
height_buff  = 3
height_co   = 3

def convert_hour(hour):
    dt = start_datetime + datetime.timedelta(hours=hour)
    return mdates.date2num(dt)

fig, ax = plt.subplots(figsize=(15, 10))

# --- Plot Active Working Intervals (Background) ---
for line in working_intervals:
    base_y = line_base[line]  # vertical position for production tasks
    for (w_s, w_e) in working_intervals[line]:
        s_num = convert_hour(w_s)
        dur_days = (w_e - w_s) / 24.0
        ax.broken_barh([(s_num, dur_days)],
                       (base_y, height_prod),
                       facecolors='grey', alpha=0.3, edgecolor='none')

# --- Plot Production Tasks ---
for line in schedule:
    for (s, dur, label, tkey) in schedule[line]:
        s_num = convert_hour(s)
        ext_dur = solver.Value(dur) / 24.0
        ax.broken_barh([(s_num, ext_dur)],
                       (line_base[line], height_prod),
                       facecolors='aqua')
        x_center = s_num + ext_dur/2
        ax.text(x_center, line_base[line] + height_prod/2, label,
                ha='center', va='center', fontsize=8, rotation=90)

# --- Plot Maintenance Intervals ---
for line, m_list in user_maintenance.items():
    base_y = line_base[line] + height_prod + 3
    for idx, (s, e) in enumerate(m_list):
        s_num = convert_hour(s)
        dur_days = (e - s) / 24.0
        ax.broken_barh([(s_num, dur_days)],
                       (base_y, height_maint),
                       facecolors='purple')
        ax.text(s_num + dur_days/2, base_y + height_maint/2,
                f"Maint {idx+1}",
                ha='center', va='center', fontsize=8, rotation=90)

# --- Plot Windowed Maintenance Intervals ---
windowed_offset = 10  # vertical offset for windowed maintenance
for line, maint_list in windowed_maint_intervals.items():
    base_y = line_base[line] + height_prod + windowed_offset
    for idx, maint in enumerate(maint_list):
        m_start_val = solver.Value(maint['m_start'])
        duration = maint['duration']
        s_num = convert_hour(m_start_val)
        dur_days = duration / 24.0
        ax.broken_barh([(s_num, dur_days)],
                       (base_y, height_maint),
                       facecolors='pink')
        ax.text(s_num + dur_days/2, base_y + height_maint/2,
                f"Windowed Maint {idx+1}",
                ha='center', va='center', fontsize=8, rotation=90)

# --- Plot Optional Maintenance Intervals ---
optional_offset = 15  # vertical offset for optional maintenance
for line, maint_list in optional_maint_intervals.items():
    base_y = line_base[line] + height_prod + optional_offset
    for idx, maint in enumerate(maint_list):
        if solver.BooleanValue(maint['scheduled']):
            m_start_val = solver.Value(maint['m_start'])
            duration = maint['duration']
            s_num = convert_hour(m_start_val)
            dur_days = duration / 24.0
            ax.broken_barh([(s_num, dur_days)],
                           (base_y, height_maint),
                           facecolors='red')
            ax.text(s_num + dur_days/2, base_y + height_maint/2,
                    f"Optional Maint {idx+1}",
                    ha='center', va='center', fontsize=8, rotation=90)

# --- Plot Buffer Intervals ---
for line, b_list in monthly_buffers.items():
    base_y = line_base[line] + height_prod + 7
    for idx, (s, e) in enumerate(b_list):
        s_num = convert_hour(s)
        dur_days = (e - s) / 24.0
        ax.broken_barh([(s_num, dur_days)],
                       (base_y, height_buff),
                       facecolors='blue')
        ax.text(s_num + dur_days/2, base_y + height_buff/2,
                f"Buffer {idx+1}",
                ha='center', va='center', fontsize=8, rotation=90)

# --- Plot Changeover Intervals Using Actual T_ij Values ---
# Loop over each line in schedule.
for line in schedule:
    base_y_co = line_base[line] - 5  # place changeovers above production tasks
    tasks_line = [t for t in tasks if solver.Value(tasks[t]['assign'][line]) == 1]
    for i in tasks_line:
        for j in tasks_line:
            if i == j:
                continue
            if solver.BooleanValue(immediate[(line, i, j)]):
                T_ij = solver.Value(changeover_start_vars[(line, i, j)])
                if 0 <= T_ij <= horizon:
                    s_num = convert_hour(T_ij)
                    co_duration = changeover[tasks[i]['sku']][tasks[j]['sku']]
                    dur_days = co_duration / 24.0
                    ax.broken_barh([(s_num, dur_days)],
                                   (base_y_co, height_co),
                                   facecolors='orange')
                    ax.text(s_num + dur_days/2, base_y_co + height_co/2,
                            f"CO: {co_duration}",
                            ha='center', va='center', fontsize=8, rotation=90)
                else:
                    print(f"Skipping changeover {i}->{j} on line {line}: T_ij out-of-range: {T_ij}")

# --- Configure the Chart ---
ax.set_xlabel("Time")
ax.set_ylabel("Production Lines")
ax.set_title("Gantt Chart with Working, Maintenance, Buffer, and Changeover Intervals")
ax.set_ylim(0, 100)
ax.set_xlim(mdates.date2num(start_datetime),
            mdates.date2num(start_datetime + datetime.timedelta(hours=horizon)))
ax.xaxis_date()
date_format = mdates.DateFormatter('%Y-%m-%d %H:%M')
ax.xaxis.set_major_formatter(date_format)
plt.xticks(rotation=45)
# Set y-ticks based on the available lines.
yticks = [line_base[line] + height_prod/2 for line in sorted(schedule.keys())]
yticklabels = [f"Line {line}" for line in sorted(schedule.keys())]
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)
ax.grid(True)
plt.tight_layout()
plt.show()



#############################################
# Build a DataFrame Report of the Schedule
#############################################
rows = []

# --- (1) Production Tasks (only scheduled ones) ---
for line in schedule:
    for (start, duration, label, tkey) in schedule[line]:
        end = start + duration
        task = tasks[tkey]
        sku = task['sku']
        batch_id = task['batch_id']
        demand = task['demand']

        rm_usage_details = []
        used_rm_batches = {}

        # Determine which recipe was chosen for this task.
        active_recipe_idx = None
        for idx, r_bool in enumerate(task['recipe_bools']):
            if solver.BooleanValue(r_bool):
                active_recipe_idx = idx
                break

        # If an active recipe is found, use its raw mapping.
        if active_recipe_idx is not None:
            active_mapping = task['raw_mapping_recipe'][active_recipe_idx]
            for rm, mapping in active_mapping.items():
                sub_entries = []
                for k, Q in enumerate(mapping['Q_vars']):
                    q_val = solver.Value(Q)
                    if q_val > 0:
                        rm_batch_key = f"{rm}_B{mapping['cycles'][k]['batch_id']}"
                        sub_entries.append(f"{rm_batch_key}: {q_val}")
                        used_rm_batches[rm_batch_key] = q_val
                if sub_entries:
                    rm_usage_details.append(f"{rm} (" + ", ".join(sub_entries) + ")")
        else:
            rm_usage_details.append("No recipe selected")

        rm_usage_text = " | ".join(rm_usage_details)
        details = f"{label}\nDemand: {demand} | RM Usage: {rm_usage_text}"
        rows.append({
            "Activity_Type": "Production",
            "Line": line,
            "Start_Time": start,
            "End_Time": end,
            "Duration": duration,
            "Task_Details": details,
            "Task_Key": tkey,
            "Used_RM": used_rm_batches
        })

# --- (2) Downtime Intervals ---
for line, dt_list in downtimes.items():
    for idx, (s, e) in enumerate(dt_list):
        dur = e - s
        rows.append({
            "Activity_Type": "Downtime",
            "Line": line,
            "Start_Time": s,
            "End_Time": e,
            "Duration": dur,
            "Task_Details": f"Downtime {idx+1}",
            "Task_Key": None,
            "Used_RM": {}
        })

# --- (3) Maintenance Intervals ---
for line, m_list in user_maintenance.items():
    for idx, (s, e) in enumerate(m_list):
        dur = e - s
        rows.append({
            "Activity_Type": "Maintenance",
            "Line": line,
            "Start_Time": s,
            "End_Time": e,
            "Duration": dur,
            "Task_Details": f"Maintenance {idx+1}",
            "Task_Key": None,
            "Used_RM": {}
        })

# --- (4) Buffer Intervals ---
for line, b_list in monthly_buffers.items():
    for idx, (s, e) in enumerate(b_list):
        dur = e - s
        rows.append({
            "Activity_Type": "Buffer",
            "Line": line,
            "Start_Time": s,
            "End_Time": e,
            "Duration": dur,
            "Task_Details": f"Buffer {idx+1}",
            "Task_Key": None,
            "Used_RM": {}
        })

# --- (5) Windowed Maintenance Intervals ---
for line, w_list in windowed_maint_intervals.items():
    for idx, w_maint in enumerate(w_list):
        # Because these are mandatory intervals, we can just read their start/end.
        w_start = solver.Value(w_maint['m_start'])
        w_dur = w_maint['duration']
        w_end = w_start + w_dur
        rows.append({
            "Activity_Type": "Windowed Maintenance",
            "Line": line,
            "Start_Time": w_start,
            "End_Time": w_end,
            "Duration": w_dur,
            "Task_Details": f"Windowed Maintenance {idx+1}",
            "Task_Key": None,
            "Used_RM": {}
        })

# --- (6) Optional Maintenance Intervals ---
for line, o_list in optional_maint_intervals.items():
    for idx, o_maint in enumerate(o_list):
        # Only add to the schedule if the maintenance was actually scheduled.
        if solver.BooleanValue(o_maint['scheduled']):
            o_start = solver.Value(o_maint['m_start'])
            o_dur = o_maint['duration']
            o_end = o_start + o_dur
            rows.append({
                "Activity_Type": "Optional Maintenance",
                "Line": line,
                "Start_Time": o_start,
                "End_Time": o_end,
                "Duration": o_dur,
                "Task_Details": f"Optional Maintenance {idx+1}",
                "Task_Key": None,
                "Used_RM": {}
            })

# --- (7) Changeover Intervals ---
for line in schedule:
    tasks_line = [t for t in tasks if solver.Value(tasks[t]['assign'][line]) == 1]
    for i in tasks_line:
        for j in tasks_line:
            if i == j:
                continue
            if solver.BooleanValue(immediate[(line, i, j)]):
                co_start = solver.Value(changeover_start_vars[(line, i, j)])
                co_duration = changeover[tasks[i]['sku']][tasks[j]['sku']]
                co_end = co_start + co_duration
                label = f"CO_{i}_to_{j}"
                rows.append({
                    "Activity_Type": "Changeover",
                    "Line": line,
                    "Start_Time": co_start,
                    "End_Time": co_end,
                    "Duration": co_duration,
                    "Task_Details": label,
                    "Task_Key": None,
                    "Used_RM": {}
                })

# --- Convert numeric times to datetimes ---
rows_sorted = sorted(rows, key=lambda r: solver.Value(r["Start_Time"]))
for row in rows_sorted:
    # Extract the solved integer values for start and end times.
    start_val = solver.Value(row["Start_Time"])
    end_val = solver.Value(row["End_Time"])
    start_dt = start_datetime + datetime.timedelta(hours=start_val)
    end_dt = start_datetime + datetime.timedelta(hours=end_val)
    row["Start_Datetime"] = start_dt
    row["End_Datetime"] = end_dt
    del row["Start_Time"]
    del row["End_Time"]

df_schedule = pd.DataFrame(rows_sorted)
df_schedule = df_schedule.sort_values(by=["Start_Datetime"]).reset_index(drop=True)

cols = list(df_schedule.columns)
for col in ["Start_Datetime", "End_Datetime"]:
    cols.remove(col)
df_schedule = df_schedule[["Start_Datetime", "End_Datetime"] + cols]

# Example of how to inspect the final schedule for each line:
# for line in sorted(df_schedule["Line"].unique()):
#     df_line = df_schedule[df_schedule["Line"] == line].sort_values(by=["Start_Datetime"])
#     print(f"Schedule for Line {line}:")
#     print(df_line)


